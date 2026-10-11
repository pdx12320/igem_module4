#!/usr/bin/env python3

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import AutoConfig, AutoTokenizer
from safetensors.torch import load_file

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

LAMAR_ROOT = SCRIPT_DIR

if LAMAR_ROOT not in sys.path:
    sys.path.insert(0, LAMAR_ROOT)

from LAMAR.sequence_classification_patch import EsmForSequenceClassification


TEST_CSV = os.path.join(ROOT_DIR, "data", "1024nt_dataset", "orca_puf12_test.csv")

MODEL_WEIGHTS_PATH = os.path.join(
    SCRIPT_DIR,
    "output",
    "final_model",
    "checkpoint-240",
    "model.safetensors",
)

TOKENIZER_PATH = os.path.join(ROOT_DIR, "tokenizer")
MODEL_CONFIG_PATH = os.path.join(ROOT_DIR, "config_150M.json")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output", "test_evaluation")


MODEL_MAX_LENGTH = 1026
POSITIONAL_EMBEDDING_TYPE = "rotary"
HIDDEN_SIZE = 768
INTERMEDIATE_SIZE = 3072
NUM_ATTENTION_HEADS = 12
NUM_HIDDEN_LAYERS = 12
NUM_LABELS = 2
HEAD_TYPE = "Linear"
FREEZE_ENCODER = False
KERNEL_SIZES = [2, 3, 5]
OCS = 32

BATCH_SIZE = 32
SEQ_COLUMN = "seq"
LABEL_COLUMN = "label"


Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def check_paths():
    required_paths = {
        "TEST_CSV": TEST_CSV,
        "MODEL_WEIGHTS_PATH": MODEL_WEIGHTS_PATH,
        "TOKENIZER_PATH": TOKENIZER_PATH,
        "MODEL_CONFIG_PATH": MODEL_CONFIG_PATH,
    }
    for name, path in required_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} does not exist: {path}")
    print("All required paths exist.")


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        model_max_length=MODEL_MAX_LENGTH,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Tokenizer has no pad_token and no eos_token")

    print("Tokenizer vocabulary size:", len(tokenizer))
    print("Tokenizer pad_token_id:", tokenizer.pad_token_id)
    print("Tokenizer model_max_length:", tokenizer.model_max_length)

    return tokenizer


def build_model(tokenizer, device):
    config_kwargs = dict(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        num_labels=NUM_LABELS,
        positional_embedding_type=POSITIONAL_EMBEDDING_TYPE,
        hidden_size=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        num_attention_heads=NUM_ATTENTION_HEADS,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
    )

    mask_token_id = getattr(tokenizer, "mask_token_id", None)
    if mask_token_id is not None:
        config_kwargs["mask_token_id"] = mask_token_id

    config = AutoConfig.from_pretrained(MODEL_CONFIG_PATH, **config_kwargs)

    model = EsmForSequenceClassification(
        config,
        head_type=HEAD_TYPE,
        freeze=FREEZE_ENCODER,
        kernel_sizes=KERNEL_SIZES,
        ocs=OCS,
    )

    print("Loading weights from:", MODEL_WEIGHTS_PATH)

    state_dict = load_file(MODEL_WEIGHTS_PATH, device="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print("Missing keys:", len(missing_keys))
    print("Unexpected keys:", len(unexpected_keys))

    if len(missing_keys) > 0:
        print("Missing key names (first 20):", missing_keys[:20])
    if len(unexpected_keys) > 0:
        print("Unexpected key names (first 20):", unexpected_keys[:20])

    model.to(device)
    model.eval()

    return model


def load_test_dataframe():
    df = pd.read_csv(TEST_CSV)

    if SEQ_COLUMN not in df.columns:
        raise ValueError(
            f"CSV must contain column '{SEQ_COLUMN}'. "
            f"Available columns: {list(df.columns)}"
        )

    if LABEL_COLUMN not in df.columns:
        raise ValueError(
            f"CSV must contain column '{LABEL_COLUMN}'. "
            f"Available columns: {list(df.columns)}"
        )

    df = df[[SEQ_COLUMN, LABEL_COLUMN]].copy()
    df = df.dropna(subset=[SEQ_COLUMN, LABEL_COLUMN])
    df[SEQ_COLUMN] = df[SEQ_COLUMN].astype(str)
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    print("Test dataframe shape:", df.shape)
    print("Test label distribution:",
          df[LABEL_COLUMN].value_counts().to_dict())

    return df


@torch.no_grad()
def run_inference(model, tokenizer, sequences, device):
    all_logits = []

    total = len(sequences)

    for start in range(0, total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total)
        batch_sequences = sequences[start:end]

        encoded = tokenizer(
            batch_sequences,
            truncation=True,
            max_length=MODEL_MAX_LENGTH,
            padding=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        all_logits.append(logits.detach().cpu())

        print(f"Inference progress: {end}/{total}")

    return torch.cat(all_logits, dim=0)


def numpy_softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def compute_metrics_from_logits(logits, labels):
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()

    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)

    probabilities = numpy_softmax(logits)
    positive_probabilities = probabilities[:, 1]
    predicted_classes = np.argmax(probabilities, axis=1)

    metrics = {
        "accuracy": float(accuracy_score(labels, predicted_classes)),
        "precision": float(precision_score(labels, predicted_classes, zero_division=0)),
        "recall": float(recall_score(labels, predicted_classes, zero_division=0)),
        "f1": float(f1_score(labels, predicted_classes, zero_division=0)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, positive_probabilities))
    except Exception:
        metrics["roc_auc"] = float("nan")

    try:
        metrics["pr_auc"] = float(
            average_precision_score(labels, positive_probabilities)
        )
    except Exception:
        metrics["pr_auc"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(labels, predicted_classes, labels=[0, 1]).ravel()

    metrics["true_negative"] = int(tn)
    metrics["false_positive"] = int(fp)
    metrics["false_negative"] = int(fn)
    metrics["true_positive"] = int(tp)

    metrics["num_samples"] = int(len(labels))
    metrics["num_positive"] = int((labels == 1).sum())
    metrics["num_negative"] = int((labels == 0).sum())

    return metrics, probabilities, predicted_classes


def main():
    check_paths()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    tokenizer = load_tokenizer()
    model = build_model(tokenizer, device)

    df = load_test_dataframe()

    sequences = df[SEQ_COLUMN].tolist()
    labels = df[LABEL_COLUMN].to_numpy(dtype=np.int64)

    logits = run_inference(model, tokenizer, sequences, device)

    metrics, probabilities, predicted_classes = compute_metrics_from_logits(
        logits,
        labels,
    )

    print("\n" + "=" * 80)
    print("Test metrics")
    print("=" * 80)
    print(json.dumps(metrics, indent=2))

    logits_np = logits.detach().cpu().numpy()

    predictions_df = pd.DataFrame(
        {
            "seq": sequences,
            "label": labels,
            "predicted_class": predicted_classes,
            "prob_class_0": probabilities[:, 0],
            "prob_class_1": probabilities[:, 1],
            "logit_class_0": logits_np[:, 0],
            "logit_class_1": logits_np[:, 1],
        }
    )

    predictions_path = Path(OUTPUT_DIR) / "test_predictions.csv"
    metrics_path = Path(OUTPUT_DIR) / "test_metrics.json"

    predictions_df.to_csv(predictions_path, index=False)

    with open(metrics_path, "w") as handle:
        json.dump(metrics, handle, indent=2)

    print("\nSaved predictions to:", predictions_path)
    print("Saved metrics to:", metrics_path)

    print("\nConfusion matrix:")
    print(f"  TN={metrics['true_negative']}  FP={metrics['false_positive']}")
    print(f"  FN={metrics['false_negative']}  TP={metrics['true_positive']}")


if __name__ == "__main__":
    main()