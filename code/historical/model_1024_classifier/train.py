#!/usr/bin/env python3

import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gc
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import optuna

from datasets import load_from_disk, Dataset, DatasetDict

from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback,
    set_seed as hf_set_seed,
)

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


TRAIN_PATH = os.path.join(ROOT_DIR, "data", "seq_1024_orca_puf12", "train")
VALIDATION_PATH = os.path.join(ROOT_DIR, "data", "seq_1024_orca_puf12", "valid")
PRETRAIN_STATE_PATH = os.path.join(
    ROOT_DIR, 
    "model_continual_pretrain", 
    "output",
    "transcriptome_pretrain",
    "checkpoint-100000",
    "model.safetensors"
)
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "output")
TOKENIZER_PATH = os.path.join(ROOT_DIR, "tokenizer")
MODEL_CONFIG_PATH = os.path.join(ROOT_DIR, "config_150M.json")


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
FLASH_ATTENTION = False

RANDOM_SEED = 42
N_TRIALS = 24
MAX_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 10
BEST_METRIC = "pr_auc"

BATCH_CONFIGS = {
    "batch4_accum2": {
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 2,
    },
    "batch8_accum1": {
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 1,
    },
    "batch8_accum2": {
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 2,
    },
    "batch16_accum1": {
        "per_device_train_batch_size": 16,
        "gradient_accumulation_steps": 1,
    },
}


Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    hf_set_seed(seed)


set_all_seeds(RANDOM_SEED)


def check_paths():
    required_paths = {
        "TRAIN_PATH": TRAIN_PATH,
        "VALIDATION_PATH": VALIDATION_PATH,
        "PRETRAIN_STATE_PATH": PRETRAIN_STATE_PATH,
        "TOKENIZER_PATH": TOKENIZER_PATH,
        "MODEL_CONFIG_PATH": MODEL_CONFIG_PATH,
    }
    for name, path in required_paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} does not exist: {path}")
    print("All required paths exist.")


check_paths()

print("CUDA available:", torch.cuda.is_available())
print("Visible GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for gpu_index in range(torch.cuda.device_count()):
        print(f"GPU {gpu_index}: {torch.cuda.get_device_name(gpu_index)}")


print("\nLoading tokenizer...")

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

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding=True,
    return_tensors="pt",
)


def unwrap_dataset(dataset_object, preferred_split):
    if isinstance(dataset_object, Dataset):
        return dataset_object

    if isinstance(dataset_object, DatasetDict):
        if preferred_split in dataset_object:
            return dataset_object[preferred_split]
        available_splits = list(dataset_object.keys())
        if not available_splits:
            raise ValueError(f"No dataset split found for {preferred_split}")
        print(f"Warning: split '{preferred_split}' not found. Using '{available_splits[0]}' instead.")
        return dataset_object[available_splits[0]]

    raise TypeError(f"Unsupported dataset object type: {type(dataset_object)}")


print("\nLoading training dataset...")
train_object = load_from_disk(TRAIN_PATH)
train_dataset = unwrap_dataset(train_object, "train")

print("\nLoading validation dataset...")
validation_object = load_from_disk(VALIDATION_PATH)
validation_dataset = unwrap_dataset(validation_object, "valid")

print("\nDataset summary:")
print("Train size:", len(train_dataset))
print("Validation size:", len(validation_dataset))
print("Train columns:", train_dataset.column_names)
print("Validation columns:", validation_dataset.column_names)


def normalize_label_column(dataset, dataset_name):
    columns = dataset.column_names
    if "labels" in columns:
        return dataset
    if "label" in columns:
        print(f"Renaming '{dataset_name}.label' to '{dataset_name}.labels'")
        return dataset.rename_column("label", "labels")
    raise ValueError(f"{dataset_name} must contain 'labels' or 'label'. Available columns: {columns}")


train_dataset = normalize_label_column(train_dataset, "train")
validation_dataset = normalize_label_column(validation_dataset, "validation")


def print_label_distribution(dataset, dataset_name):
    labels = np.asarray(dataset["labels"], dtype=np.int64)
    unique, counts = np.unique(labels, return_counts=True)
    distribution = {int(label): int(count) for label, count in zip(unique, counts)}
    print(f"{dataset_name} label distribution:", distribution)
    if len(unique) < 2:
        raise ValueError(f"{dataset_name} contains only one class.")


print_label_distribution(train_dataset, "Train")
print_label_distribution(validation_dataset, "Validation")


def softmax_numpy(logits):
    logits = np.asarray(logits)
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def compute_metrics(eval_prediction):
    predictions = eval_prediction.predictions
    labels = eval_prediction.label_ids

    if isinstance(predictions, tuple):
        predictions = predictions[0]

    predictions = np.asarray(predictions)
    labels = np.asarray(labels).reshape(-1)

    probabilities = softmax_numpy(predictions)
    positive_probabilities = probabilities[:, 1]
    predicted_classes = np.argmax(predictions, axis=1)

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
        metrics["pr_auc"] = float(average_precision_score(labels, positive_probabilities))
    except Exception:
        metrics["pr_auc"] = float("nan")

    tn, fp, fn, tp = confusion_matrix(labels, predicted_classes, labels=[0, 1]).ravel()

    metrics["true_negative"] = int(tn)
    metrics["false_positive"] = int(fp)
    metrics["false_negative"] = int(fn)
    metrics["true_positive"] = int(tp)

    return metrics


def build_model(token_dropout_value):
    config_kwargs = dict(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        num_labels=NUM_LABELS,
        token_dropout=token_dropout_value,
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

    print(f"Creating model. token_dropout={token_dropout_value}")

    model = EsmForSequenceClassification(
        config,
        head_type=HEAD_TYPE,
        freeze=FREEZE_ENCODER,
        kernel_sizes=KERNEL_SIZES,
        ocs=OCS,
    )

    if FLASH_ATTENTION:
        from flash_attn_patch import EsmSelfAttentionAddFlashAttnPatch
        for layer_index in range(config.num_hidden_layers):
            model.esm.encoder.layer[layer_index].attention.self = (
                EsmSelfAttentionAddFlashAttnPatch(
                    config,
                    position_embedding_type="rotary",
                )
            )

    print("Loading pretrained weights:", PRETRAIN_STATE_PATH)

    state_dict = load_file(PRETRAIN_STATE_PATH, device="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print("Pretrained missing keys:", len(missing_keys))
    print("Pretrained unexpected keys:", len(unexpected_keys))

    if len(missing_keys) > 30:
        print("Warning: many parameters were not loaded. Please verify model architecture compatibility.")

    return model


class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial):
        self.trial = trial
        self.report_step = 0

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return control

        current_value = metrics.get(f"eval_{BEST_METRIC}", None)
        if current_value is None:
            return control

        if not np.isfinite(current_value):
            current_value = 0.0

        self.report_step += 1

        self.trial.report(float(current_value), step=self.report_step)

        if self.trial.should_prune():
            print(
                f"Trial {self.trial.number} pruned at "
                f"report_step {self.report_step}, "
                f"eval_{BEST_METRIC}={current_value:.6f}"
            )
            raise optuna.TrialPruned()

        return control


def sample_hyperparameters(trial):
    batch_config_name = trial.suggest_categorical(
        "batch_config",
        list(BATCH_CONFIGS.keys()),
    )
    batch_config = BATCH_CONFIGS[batch_config_name]

    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "token_dropout": trial.suggest_float("token_dropout", 0.0, 0.05),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.1),
        "batch_config": batch_config_name,
        "per_device_train_batch_size": batch_config["per_device_train_batch_size"],
        "gradient_accumulation_steps": batch_config["gradient_accumulation_steps"],
        "num_train_epochs": MAX_EPOCHS,
    }

    return params


def get_visible_gpu_count():
    if torch.cuda.is_available():
        return max(1, torch.cuda.device_count())
    return 1


def compute_eval_steps(train_size, effective_batch_size):
    updates_per_epoch = max(1, math.ceil(train_size / effective_batch_size))
    eval_steps = max(10, updates_per_epoch // 2)
    return updates_per_epoch, eval_steps


def build_training_args(output_dir, params):
    gpu_count = get_visible_gpu_count()
    per_device_batch_size = params["per_device_train_batch_size"]
    accumulation_steps = params["gradient_accumulation_steps"]
    effective_batch_size = per_device_batch_size * gpu_count * accumulation_steps

    updates_per_epoch, eval_steps = compute_eval_steps(
        len(train_dataset),
        effective_batch_size,
    )

    print("GPU count:", gpu_count)
    print("Per-device batch size:", per_device_batch_size)
    print("Gradient accumulation steps:", accumulation_steps)
    print("Estimated effective batch size:", effective_batch_size)
    print("Estimated updates per epoch:", updates_per_epoch)
    print("Evaluation steps:", eval_steps)

    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=accumulation_steps,
        learning_rate=params["learning_rate"],
        weight_decay=params["weight_decay"],
        warmup_ratio=params["warmup_ratio"],
        num_train_epochs=params["num_train_epochs"],
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        dataloader_drop_last=False,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        logging_strategy="steps",
        logging_steps=max(1, eval_steps // 2),
        load_best_model_at_end=True,
        metric_for_best_model=BEST_METRIC,
        greater_is_better=True,
        fp16=False,
        bf16=False,
        report_to="none",
        save_safetensors=True,
        remove_unused_columns=True,
        seed=RANDOM_SEED,
        data_seed=RANDOM_SEED,
    )


def run_trial(trial):
    params = sample_hyperparameters(trial)

    trial_name = f"trial_{trial.number:04d}"
    trial_output_dir = Path(OUTPUT_ROOT) / "optuna_trials" / trial_name

    if trial_output_dir.exists():
        shutil.rmtree(trial_output_dir)
    trial_output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Starting Optuna trial {trial.number}")
    print(json.dumps(params, indent=2))
    print("=" * 80)

    set_all_seeds(RANDOM_SEED + trial.number)

    model = build_model(token_dropout_value=params["token_dropout"])
    training_args = build_training_args(trial_output_dir, params)

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
        OptunaPruningCallback(trial),
    ]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    try:
        trainer.train()

        metrics = trainer.evaluate(eval_dataset=validation_dataset)

        objective_value = metrics.get(f"eval_{BEST_METRIC}", 0.0)
        if not np.isfinite(objective_value):
            objective_value = 0.0

        with open(trial_output_dir / "trial_params.json", "w") as handle:
            json.dump(params, handle, indent=2)

        with open(trial_output_dir / "validation_metrics.json", "w") as handle:
            json.dump(metrics, handle, indent=2)

        log_history = trainer.state.log_history
        with open(trial_output_dir / "log_history.json", "w") as handle:
            json.dump(log_history, handle, indent=2)

        print(
            f"Trial {trial.number} finished. "
            f"Validation {BEST_METRIC.upper()}={objective_value:.6f}"
        )

        return float(objective_value)

    except optuna.TrialPruned:
        print(f"Trial {trial.number} was pruned.")
        raise

    finally:
        del trainer
        del model

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def convert_best_params(best_trial):
    params = dict(best_trial.params)

    batch_config_name = params["batch_config"]
    batch_config = BATCH_CONFIGS[batch_config_name]

    params["per_device_train_batch_size"] = batch_config["per_device_train_batch_size"]
    params["gradient_accumulation_steps"] = batch_config["gradient_accumulation_steps"]
    params["num_train_epochs"] = MAX_EPOCHS

    return params


def train_final_model(best_trial):
    best_params = convert_best_params(best_trial)

    print("\n" + "=" * 80)
    print("Training final model")
    print("=" * 80)

    print(f"Best validation {BEST_METRIC.upper()} from search:", best_trial.value)
    print("Final training parameters:")
    print(json.dumps(best_params, indent=2))

    final_output_dir = Path(OUTPUT_ROOT) / "final_model"

    if final_output_dir.exists():
        shutil.rmtree(final_output_dir)
    final_output_dir.mkdir(parents=True, exist_ok=True)

    set_all_seeds(RANDOM_SEED + 10000)

    model = build_model(token_dropout_value=best_params["token_dropout"])

    per_device_batch_size = best_params["per_device_train_batch_size"]
    accumulation_steps = best_params["gradient_accumulation_steps"]
    gpu_count = get_visible_gpu_count()
    effective_batch_size = per_device_batch_size * accumulation_steps * gpu_count

    updates_per_epoch, eval_steps = compute_eval_steps(
        len(train_dataset),
        effective_batch_size,
    )

    final_args = TrainingArguments(
        output_dir=str(final_output_dir),
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=accumulation_steps,
        learning_rate=best_params["learning_rate"],
        weight_decay=best_params["weight_decay"],
        warmup_ratio=best_params["warmup_ratio"],
        num_train_epochs=MAX_EPOCHS,
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        dataloader_drop_last=False,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        logging_strategy="steps",
        logging_steps=max(1, eval_steps // 2),
        load_best_model_at_end=True,
        metric_for_best_model=BEST_METRIC,
        greater_is_better=True,
        fp16=False,
        bf16=False,
        report_to="none",
        save_safetensors=True,
        remove_unused_columns=True,
        seed=RANDOM_SEED + 10000,
        data_seed=RANDOM_SEED + 10000,
    )

    trainer = Trainer(
        model=model,
        args=final_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
        ],
    )

    print("Final model effective batch size:", effective_batch_size)
    print("Final model updates per epoch:", updates_per_epoch)
    print("Final model evaluation steps:", eval_steps)

    trainer.train()

    print("\nEvaluating final model on validation dataset...")
    validation_metrics = trainer.evaluate(eval_dataset=validation_dataset)

    with open(final_output_dir / "best_hyperparameters.json", "w") as handle:
        json.dump(best_params, handle, indent=2)

    with open(final_output_dir / "validation_metrics.json", "w") as handle:
        json.dump(validation_metrics, handle, indent=2)

    with open(final_output_dir / "log_history.json", "w") as handle:
        json.dump(trainer.state.log_history, handle, indent=2)

    print("\nFinal validation metrics:")
    print(json.dumps(validation_metrics, indent=2))

    print("Best checkpoint:", trainer.state.best_model_checkpoint)
    print(f"Best validation {BEST_METRIC.upper()}:", trainer.state.best_metric)

    return validation_metrics


def main():
    study_db_path = Path(OUTPUT_ROOT) / "optuna_seq1026.db"
    study_database = "sqlite:///" + str(study_db_path)

    print("\nOptuna database:", study_database)

    study = optuna.create_study(
        study_name="lamar_seq1026_classification",
        storage=study_database,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=1,
            max_resource=MAX_EPOCHS * 2,
            reduction_factor=3,
        ),
    )

    completed_or_pruned = len(
        [
            trial
            for trial in study.trials
            if trial.state in (
                optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.PRUNED,
            )
        ]
    )

    remaining_trials = max(0, N_TRIALS - completed_or_pruned)

    print("Existing completed/pruned trials:", completed_or_pruned)
    print("Remaining trials:", remaining_trials)

    if remaining_trials > 0:
        study.optimize(
            run_trial,
            n_trials=remaining_trials,
            gc_after_trial=True,
        )

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]

    if not completed_trials:
        raise RuntimeError("No completed Optuna trial found.")

    best_trial = study.best_trial

    print("\n" + "=" * 80)
    print("Optuna search finished")
    print("=" * 80)

    print("Best trial number:", best_trial.number)
    print(f"Best validation {BEST_METRIC.upper()}:", best_trial.value)
    print("Best raw Optuna parameters:")
    print(json.dumps(best_trial.params, indent=2))

    trials_dataframe = study.trials_dataframe()
    trials_dataframe.to_csv(Path(OUTPUT_ROOT) / "optuna_trials.csv", index=False)

    with open(Path(OUTPUT_ROOT) / "best_trial.json", "w") as handle:
        json.dump(
            {
                "trial_number": best_trial.number,
                f"validation_{BEST_METRIC}": best_trial.value,
                "params": best_trial.params,
            },
            handle,
            indent=2,
        )

    train_final_model(best_trial)

    print("\nAll training finished.")
    print("Results saved to:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()