# -*- coding: utf-8 -*-
from transformers import AutoTokenizer
from datasets import load_dataset
import multiprocessing as mp
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

tokenizer = AutoTokenizer.from_pretrained(
    os.path.join(ROOT_DIR, "tokenizer"),
    model_max_length=103
)

def group_texts(examples):
    tokenized_inputs = tokenizer(
        examples["seq"],
        truncation=True,
        max_length=tokenizer.model_max_length
    )
    tokenized_inputs["labels"] = [int(x) for x in examples["label"]]
    return tokenized_inputs

train_set_path = os.path.join(ROOT_DIR, "data", "1024nt_dataset", "orca_puf12_train.csv")
valid_set_path = os.path.join(ROOT_DIR, "data", "1024nt_dataset", "orca_puf12_valid.csv")

data_path = os.path.join(ROOT_DIR, "data", "seq_1024_orca_puf12")

dataset = load_dataset("csv", data_files={'train': train_set_path, 'valid': valid_set_path})

data = dataset.map(group_texts, batched=True, num_proc=mp.cpu_count())

data.save_to_disk(data_path, num_proc=mp.cpu_count())

print(data)