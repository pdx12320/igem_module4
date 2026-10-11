#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import random
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file, save_file

from transformers import (
    AutoConfig,
    AutoTokenizer,
    set_seed as hf_set_seed,
    get_cosine_schedule_with_warmup,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

LAMAR_ROOT = SCRIPT_DIR

if LAMAR_ROOT not in sys.path:
    sys.path.insert(0, LAMAR_ROOT)

from LAMAR.sequence_classification_patch import EsmForSequenceClassification


TRAIN_JSON = os.path.join(ROOT_DIR, "data",  "101nt_dataset", "orca_puf12_train.json")
VALID_JSON = os.path.join(ROOT_DIR, "data",  "101nt_dataset", "orca_puf12_valid.json")

INIT_MODEL = os.path.join(
    ROOT_DIR, 
    "model_continual_pretrain", 
    "output",
    "transcriptome_pretrain",
    "checkpoint-100000",
    "model.safetensors"
)

TOKENIZER_PATH = os.path.join(ROOT_DIR, "tokenizer")
MODEL_CONFIG_PATH = os.path.join(ROOT_DIR, "config_150M.json")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


SEED = 42
WINDOW_LENGTH = 101
HALF_WINDOW = WINDOW_LENGTH // 2
MODEL_MAX_LENGTH = 103
CENTER_TOKEN_INDEX = 51

HIDDEN_SIZE = 768
INTERMEDIATE_SIZE = 3072
NUM_ATTENTION_HEADS = 12
NUM_HIDDEN_LAYERS = 12
NUM_LABELS = 1
POSITIONAL_EMBEDDING_TYPE = "rotary"
FREEZE_ENCODER = False
NUM_EPOCHS = 8
LEARNING_RATE = 5e-6
WEIGHT_DECAY = 0.01
GRADIENT_ACCUMULATION = 8
BATCH_SIZE_PER_GPU = 1
NUM_WORKERS = 4
WARMUP_RATIO = 0.05
SAVE_BEST_METRIC = "hit10"


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


set_all_seeds(SEED)


def check_paths():
    paths = {
        "TRAIN_JSON": TRAIN_JSON,
        "VALID_JSON": VALID_JSON,
        "INIT_MODEL": INIT_MODEL,
        "TOKENIZER_PATH": TOKENIZER_PATH,
        "MODEL_CONFIG_PATH": MODEL_CONFIG_PATH,
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} missing: {path}")
    print("All paths exist.")


check_paths()
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())


print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH,
    model_max_length=MODEL_MAX_LENGTH,
)
print("Vocabulary size:", len(tokenizer))


def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    print(path, "samples:", len(data))
    return data


train_records = load_json(TRAIN_JSON)
valid_records = load_json(VALID_JSON)


def normalize_sequence(seq):
    seq = seq.upper()
    seq = seq.replace("U", "T")
    return seq


def get_center_window(sequence, center_position):
    start = center_position - HALF_WINDOW
    end = center_position + HALF_WINDOW + 1

    left_pad = 0
    right_pad = 0

    if start < 0:
        left_pad = -start
        start = 0

    if end > len(sequence):
        right_pad = end - len(sequence)
        end = len(sequence)

    window = "N" * left_pad + sequence[start:end] + "N" * right_pad
    assert len(window) == WINDOW_LENGTH
    return window


def extract_candidates(record):
    seq = normalize_sequence(record["seq"])
    labels = record["label"]

    assert len(seq) == len(labels), (len(seq), len(labels))

    candidates = []
    for pos, base in enumerate(seq):
        if base != "C":
            continue
        candidates.append(
            {
                "position": pos,
                "window": get_center_window(seq, pos),
                "raw_label": float(labels[pos]),
                "label": 1.0 if labels[pos] > 0 else 0.0,
            }
        )
    return candidates


class OrcaRankingDataset(Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        candidates = extract_candidates(record)
        if len(candidates) == 0:
            return None

        windows = [x["window"] for x in candidates]
        labels = torch.tensor([x["label"] for x in candidates], dtype=torch.float)
        positions = [x["position"] for x in candidates]

        return {
            "windows": windows,
            "labels": labels,
            "positions": positions,
        }


def ranking_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None
    return batch


def tokenize_windows(windows, device):
    encoded = tokenizer(
        windows,
        padding=True,
        truncation=True,
        max_length=MODEL_MAX_LENGTH,
        return_tensors="pt",
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    return encoded


class CenterTokenScoreHead(nn.Module):
    def __init__(self, config, center_index):
        super().__init__()
        self.center_index = center_index
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 1)

    def forward(self, hidden_states):
        x = hidden_states[:, self.center_index, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        score = self.out_proj(x)
        return score.squeeze(-1)


def build_model():
    config = AutoConfig.from_pretrained(
        MODEL_CONFIG_PATH,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        num_labels=1,
        problem_type="regression",
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        positional_embedding_type=POSITIONAL_EMBEDDING_TYPE,
    )

    print("Building LAMAR model...")
    model = EsmForSequenceClassification(
        config,
        head_type="Linear",
        freeze=FREEZE_ENCODER,
        kernel_sizes=[2, 3, 5],
        ocs=32,
    )

    model.classifier = CenterTokenScoreHead(config, CENTER_TOKEN_INDEX)

    print("Loading previous ORCA binary model:")
    print(INIT_MODEL)

    state_dict = load_file(INIT_MODEL, device="cpu")
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    state_dict = {
        k: v for k, v in state_dict.items()
        if not k.startswith("classifier.")
    }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    encoder_missing = [k for k in missing if not k.startswith("classifier.")]

    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    if len(encoder_missing) > 0:
        print("Warning: missing encoder keys:")
        print(encoder_missing[:20])
    else:
        print("Encoder weights loaded successfully.")

    return model, config


class MultiPositiveRankingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, scores, labels):
        positive_mask = (labels > 0)
        if positive_mask.sum() == 0:
            return None
        positive_scores = scores[positive_mask]
        all_logsumexp = torch.logsumexp(scores, dim=0)
        positive_logsumexp = torch.logsumexp(positive_scores, dim=0)
        loss = -(positive_logsumexp - all_logsumexp)
        return loss


def calculate_rank_metrics(scores, labels):
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    positive_indices = np.where(sorted_labels > 0)[0]
    if len(positive_indices) == 0:
        return None

    best_rank = positive_indices[0] + 1
    hit = {}
    for k in [1, 5, 10, 20]:
        hit[k] = int(np.any(sorted_labels[:k] > 0))

    recall = {}
    total_positive = labels.sum()
    for k in [5, 10, 20]:
        recall[k] = sorted_labels[:k].sum() / max(total_positive, 1)

    mrr = 1.0 / best_rank
    return {
        "rank": best_rank,
        "hit": hit,
        "recall": recall,
        "mrr": mrr,
    }


def forward_transcript(model, transcript, device):
    windows = transcript["windows"]
    labels = transcript["labels"].to(device)

    encoded = tokenize_windows(windows, device)

    outputs = model(**encoded, output_hidden_states=True)

    if isinstance(model, torch.nn.DataParallel):
        classifier = model.module.classifier
    else:
        classifier = model.classifier

    hidden = outputs.hidden_states[-1]
    scores = classifier(hidden)

    return scores, labels


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    model.train()
    loss_function = MultiPositiveRankingLoss()

    total_loss = 0.0
    valid_steps = 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        if batch is None:
            continue

        batch_loss = 0.0
        transcript_count = 0

        for transcript in batch:
            scores, labels = forward_transcript(model, transcript, device)
            loss = loss_function(scores, labels)
            if loss is None:
                continue
            batch_loss += loss
            transcript_count += 1

        if transcript_count == 0:
            continue

        batch_loss = batch_loss / transcript_count
        log_loss = batch_loss.item()
        (batch_loss / GRADIENT_ACCUMULATION).backward()

        if (step + 1) % GRADIENT_ACCUMULATION == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += log_loss
        valid_steps += 1

        if step % 20 == 0:
            print(f"Epoch {epoch} Step {step} Loss {log_loss:.5f}")

    return total_loss / max(valid_steps, 1)


@torch.no_grad()
def calculate_transcript_metrics(model, transcript, device):
    scores, labels = forward_transcript(model, transcript, device)
    scores = scores.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    return calculate_rank_metrics(scores, labels)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_metrics = []

    for batch in loader:
        if batch is None:
            continue
        for transcript in batch:
            result = calculate_transcript_metrics(model, transcript, device)
            if result is not None:
                all_metrics.append(result)

    if len(all_metrics) == 0:
        return {}

    metrics = {}
    for k in [1, 5, 10, 20]:
        metrics[f"hit{k}"] = float(np.mean([x["hit"][k] for x in all_metrics]))
    for k in [5, 10, 20]:
        metrics[f"recall{k}"] = float(np.mean([x["recall"][k] for x in all_metrics]))
    metrics["mrr"] = float(np.mean([x["mrr"] for x in all_metrics]))

    ranks = [x["rank"] for x in all_metrics]
    metrics["mean_best_rank"] = float(np.mean(ranks))
    metrics["median_best_rank"] = float(np.median(ranks))
    metrics["num_transcripts"] = len(all_metrics)

    return metrics


def build_optimizer(model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=LEARNING_RATE
    )
    return optimizer


def save_checkpoint(
    model,
    config,
    tokenizer,
    optimizer,
    scheduler,
    output_dir,
    epoch,
    best_metric,
    global_step,
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    model_to_save = model
    if isinstance(model, torch.nn.DataParallel):
        model_to_save = model.module

    state_dict = {k: v.cpu() for k, v in model_to_save.state_dict().items()}
    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))

    config.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    torch.save(
        optimizer.state_dict(),
        os.path.join(output_dir, "optimizer.pt"),
    )
    torch.save(
        scheduler.state_dict(),
        os.path.join(output_dir, "scheduler.pt"),
    )

    training_state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "save_best_metric": SAVE_BEST_METRIC,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "num_epochs": NUM_EPOCHS,
        "seed": SEED,
        "window_length": WINDOW_LENGTH,
        "model_max_length": MODEL_MAX_LENGTH,
        "center_token_index": CENTER_TOKEN_INDEX,
        "init_model": INIT_MODEL,
        "train_json": TRAIN_JSON,
        "valid_json": VALID_JSON,
    }
    with open(os.path.join(output_dir, "training_state.json"), "w") as f:
        json.dump(training_state, f, indent=2)

    print("Saved checkpoint:", output_dir)


def main():
    print("\nCreating datasets...")

    train_dataset = OrcaRankingDataset(train_records)
    valid_dataset = OrcaRankingDataset(valid_records)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_PER_GPU,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=ranking_collate_fn,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=ranking_collate_fn,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\nBuilding model...")
    model, config = build_model()
    model.to(device)

    if torch.cuda.device_count() > 1:
        print("Using DataParallel:", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)

    optimizer = build_optimizer(model)

    total_steps = len(train_loader) * NUM_EPOCHS // GRADIENT_ACCUMULATION
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print("Total optimizer steps:", total_steps)
    print("Warmup steps:", warmup_steps)

    best_metric = -1
    global_step = 0

    print("\nStart training...")

    for epoch in range(1, NUM_EPOCHS + 1):
        print("\n====================")
        print("Epoch", epoch)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        global_step = epoch * (len(train_loader) // GRADIENT_ACCUMULATION)
        print("Training loss:", train_loss)

        valid_metrics = evaluate(model, valid_loader, device)
        print("\nValidation metrics:")
        for k, v in valid_metrics.items():
            print(f"{k}: {v}")

        current_metric = valid_metrics.get(SAVE_BEST_METRIC, 0)
        if current_metric > best_metric:
            best_metric = current_metric
            print("New best model:", best_metric)
            save_checkpoint(
                model,
                config,
                tokenizer,
                optimizer,
                scheduler,
                os.path.join(OUTPUT_DIR, "best_model"),
                epoch,
                best_metric,
                global_step,
            )

        save_checkpoint(
            model,
            config,
            tokenizer,
            optimizer,
            scheduler,
            os.path.join(OUTPUT_DIR, "last_model"),
            epoch,
            best_metric,
            global_step,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nTraining finished.")
    print("Best", SAVE_BEST_METRIC, ":", best_metric)


if __name__ == "__main__":
    main()