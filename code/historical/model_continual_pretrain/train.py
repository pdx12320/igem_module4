import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("TORCH_DISTRIBUTED_DEBUG", "DETAIL")

import sys
import re
import argparse
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

LAMAR_ROOT = SCRIPT_DIR

from LAMAR.modeling_nucESM2 import EsmForMaskedLM

DEFAULT_TOKENIZER_PATH = os.path.join(ROOT_DIR, "tokenizer")
DEFAULT_MODEL_NAME = os.path.join(ROOT_DIR, "config_150M.json")

DEFAULT_PRETRAINED_MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "pretrained_base",
    "model.safetensors",
)

DEFAULT_TRAIN_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "pretrain_dataset\human_transcriptome_pretraining_train.txt",
)

DEFAULT_VALID_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "pretrain_dataset\human_transcriptome_pretraining_valid.txt",
)

DEFAULT_OUTPUT_DIR = os.path.join(
    SCRIPT_DIR,
    "output",
)


def normalize_seq(seq: str) -> str:
    seq = seq.strip().upper().replace("U", "T")
    seq = re.sub(r"[^ATCGN]", "N", seq)
    return seq


def load_pretrained_model(pretrained_model_path: str, config):
    load_dir = pretrained_model_path
    if os.path.isfile(pretrained_model_path):
        load_dir = os.path.dirname(pretrained_model_path)

    try:
        model = EsmForMaskedLM.from_pretrained(load_dir, config=config)
        return model
    except Exception as e:
        pass

    if not os.path.isfile(pretrained_model_path):
        raise FileNotFoundError(f"Pretrained model file not found: {pretrained_model_path}")

    if not pretrained_model_path.endswith(".safetensors"):
        raise RuntimeError(
            "Fallback only supports .safetensors file. "
            "Please provide checkpoint dir or model.safetensors."
        )

    try:
        from safetensors.torch import load_file as load_safetensors
    except Exception as e:
        raise ImportError("safetensors is required to load model.safetensors manually.") from e

    model = EsmForMaskedLM(config)
    state_dict = load_safetensors(pretrained_model_path)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    return model


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    preds, labels = eval_pred
    mask = labels != -100
    if mask.sum() == 0:
        return {"masked_accuracy": 0.0}
    acc = (preds[mask] == labels[mask]).mean()
    return {"masked_accuracy": float(acc)}


class UnifiedMlmCollator:
    def __init__(self, tokenizer, mlm_probability=0.15):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability

    def _pad_labels(self, labels_list, max_len):
        padded = []
        for lab in labels_list:
            padded.append(lab + [-100] * (max_len - len(lab)))
        return torch.tensor(padded, dtype=torch.long)

    def _pad_attention_mask(self, attn_list, max_len):
        padded = []
        for am in attn_list:
            padded.append(am + [0] * (max_len - len(am)))
        return torch.tensor(padded, dtype=torch.long)

    def _pad_input_ids(self, ids_list, max_len):
        pad_id = self.tokenizer.pad_token_id
        padded = []
        for ids in ids_list:
            padded.append(ids + [pad_id] * (max_len - len(ids)))
        return torch.tensor(padded, dtype=torch.long)

    def _pad_special_tokens_mask(self, stm_list, max_len):
        padded = []
        for stm in stm_list:
            padded.append(stm + [1] * (max_len - len(stm)))
        return torch.tensor(padded, dtype=torch.long)

    def __call__(self, features):
        if "labels" in features[0]:
            input_ids = [f["input_ids"] for f in features]
            attention_mask = [f.get("attention_mask", [1] * len(f["input_ids"])) for f in features]
            labels = [f["labels"] for f in features]

            max_len = max(len(x) for x in input_ids)

            batch = {
                "input_ids": self._pad_input_ids(input_ids, max_len),
                "attention_mask": self._pad_attention_mask(attention_mask, max_len),
                "labels": self._pad_labels(labels, max_len),
            }
            return batch

        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f.get("attention_mask", [1] * len(f["input_ids"])) for f in features]
        special_tokens_mask = [f.get("special_tokens_mask", [0] * len(f["input_ids"])) for f in features]

        max_len = max(len(x) for x in input_ids)

        input_ids = self._pad_input_ids(input_ids, max_len)
        attention_mask = self._pad_attention_mask(attention_mask, max_len)
        special_tokens_mask = self._pad_special_tokens_mask(special_tokens_mask, max_len)

        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        probability_matrix.masked_fill_(special_tokens_mask.bool(), value=0.0)

        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            low=0,
            high=len(self.tokenizer),
            size=labels.shape,
            dtype=torch.long,
        )
        input_ids[indices_random] = random_words[indices_random]

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return batch


def tokenize_train_batch(examples, tokenizer, model_max_length):
    seqs = [normalize_seq(s) for s in examples["text"]]
    return tokenizer(
        seqs,
        truncation=True,
        max_length=model_max_length,
        return_special_tokens_mask=True,
    )


def tokenize_and_mask_eval_batch(
    examples,
    indices,
    tokenizer,
    model_max_length,
    mlm_probability,
    seed,
):
    seqs = [normalize_seq(s) for s in examples["text"]]
    encodings = tokenizer(
        seqs,
        truncation=True,
        max_length=model_max_length,
        return_special_tokens_mask=True,
    )

    special_ids = set(tokenizer.all_special_ids)
    candidate_random_token_ids = [
        i for i in range(len(tokenizer)) if i not in special_ids
    ]
    if len(candidate_random_token_ids) == 0:
        candidate_random_token_ids = [tokenizer.mask_token_id]

    out_input_ids = []
    out_attention_mask = []
    out_labels = []

    for i, input_ids in enumerate(encodings["input_ids"]):
        special_tokens_mask = encodings["special_tokens_mask"][i]
        idx = indices[i]
        rng = np.random.RandomState(seed + int(idx))

        labels = [-100] * len(input_ids)

        for pos, is_special in enumerate(special_tokens_mask):
            if is_special:
                continue

            if rng.rand() < mlm_probability:
                labels[pos] = input_ids[pos]
                prob = rng.rand()
                if prob < 0.8:
                    input_ids[pos] = tokenizer.mask_token_id
                elif prob < 0.9:
                    input_ids[pos] = int(
                        candidate_random_token_ids[rng.randint(len(candidate_random_token_ids))]
                    )
                else:
                    pass

        out_input_ids.append(input_ids)
        out_attention_mask.append([1] * len(input_ids))
        out_labels.append(labels)

    return {
        "input_ids": out_input_ids,
        "attention_mask": out_attention_mask,
        "labels": out_labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Continue pretraining LAMAR on human 3'UTR")

    parser.add_argument("--tokenizer_path", type=str, default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--pretrained_model_path", type=str, default=DEFAULT_PRETRAINED_MODEL_PATH)
    parser.add_argument("--data_for_pretrain_path", type=str, default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--validation_data_path", type=str, default=DEFAULT_VALID_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--model_max_length", type=int, default=1026)
    parser.add_argument("--token_dropout", action="store_true")
    parser.add_argument("--positional_embedding_type", type=str, default="rotary")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    parser.add_argument("--num_attention_heads", type=int, default=12)
    parser.add_argument("--num_hidden_layers", type=int, default=12)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--peak_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--grad_clipping_norm", type=float, default=1.0)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--max_eval_samples", type=int, default=10000)

    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mlm_probability", type=float, default=0.15)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    if "LOCAL_RANK" not in os.environ:
        print(
            "[WARN] Not launched via torchrun. "
            "To use all 3 GPUs (0,1,2) in parallel, run: "
            "torchrun --nproc_per_node 3 pretrain_continue.py"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        model_max_length=args.model_max_length,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.sep_token is not None:
            tokenizer.pad_token = tokenizer.sep_token
        else:
            raise ValueError(
                "Tokenizer has no pad/eos/sep token. "
                "Please check tokenizer/single_nucleotide."
            )

    config = AutoConfig.from_pretrained(
        args.model_name,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.mask_token_id,
        token_dropout=args.token_dropout,
        positional_embedding_type=args.positional_embedding_type,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
    )

    if hasattr(config, "max_position_embeddings"):
        config.max_position_embeddings = args.model_max_length

    raw_datasets = load_dataset(
        "text",
        data_files={
            "train": args.data_for_pretrain_path,
            "validation": args.validation_data_path,
        },
    )

    raw_datasets = raw_datasets.filter(lambda x: len(x["text"].strip()) >= 50)

    if args.max_eval_samples is not None and len(raw_datasets["validation"]) > args.max_eval_samples:
        raw_datasets["validation"] = raw_datasets["validation"].shuffle(seed=args.seed).select(
            range(args.max_eval_samples)
        )

    def train_tokenize_fn(examples):
        return tokenize_train_batch(examples, tokenizer, args.model_max_length)

    train_dataset = raw_datasets["train"].map(
        train_tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing train dataset",
    )

    def valid_tokenize_mask_fn(examples, indices):
        return tokenize_and_mask_eval_batch(
            examples=examples,
            indices=indices,
            tokenizer=tokenizer,
            model_max_length=args.model_max_length,
            mlm_probability=args.mlm_probability,
            seed=args.seed,
        )

    eval_dataset = raw_datasets["validation"].map(
        valid_tokenize_mask_fn,
        batched=True,
        with_indices=True,
        remove_columns=["text"],
        desc="Tokenizing and masking validation dataset",
    )

    collator = UnifiedMlmCollator(
        tokenizer=tokenizer,
        mlm_probability=args.mlm_probability,
    )

    model = load_pretrained_model(args.pretrained_model_path, config)

    try:
        from LAMAR.flash_attn_patch import EsmSelfAttentionAddFlashAttnPatch
        for i in range(config.num_hidden_layers):
            model.esm.encoder.layer[i].attention.self = EsmSelfAttentionAddFlashAttnPatch(
                config, position_embedding_type=args.positional_embedding_type
            )
    except Exception as e:
        pass

    train_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.accum_steps,
        learning_rate=args.peak_lr,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-8,
        warmup_ratio=args.warmup_ratio,
        max_steps=args.max_steps,
        max_grad_norm=args.grad_clipping_norm,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="masked_accuracy",
        greater_is_better=True,
        dataloader_drop_last=True,
        fp16=args.fp16,
        report_to="none",
        disable_tqdm=args.disable_tqdm,
        save_safetensors=True,
        remove_unused_columns=False,
        ddp_find_unused_parameters=True,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()