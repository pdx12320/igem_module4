#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation script for LAMAR ORCA ranking model.

Outputs:
    hit@1
    hit@5
    hit@10
    hit@20
    recall@5
    recall@10
    recall@20
    MRR
    mean best rank
    median best rank
"""

import os
import sys
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from safetensors.torch import load_file

from transformers import (
    AutoConfig,
    AutoTokenizer,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

LAMAR_ROOT = SCRIPT_DIR

if LAMAR_ROOT not in sys.path:
    sys.path.insert(0, LAMAR_ROOT)

from LAMAR.sequence_classification_patch import (
    EsmForSequenceClassification,
)


# ============================================================
# PATHS
# ============================================================

TEST_JSON = os.path.join(ROOT_DIR, "data", "101nt_dataset", "orca_puf12_test.json")

MODEL_PATH = os.path.join(SCRIPT_DIR, "output", "best_model", "model.safetensors")

TOKENIZER_PATH = os.path.join(ROOT_DIR, "tokenizer")

MODEL_CONFIG_PATH = os.path.join(ROOT_DIR, "config_150M.json")


# ============================================================
# PARAMETERS
# ============================================================

WINDOW_LENGTH = 101
HALF_WINDOW = 50

MODEL_MAX_LENGTH = 103

CENTER_TOKEN_INDEX = 51


HIDDEN_SIZE = 768
INTERMEDIATE_SIZE = 3072
NUM_ATTENTION_HEADS = 12
NUM_HIDDEN_LAYERS = 12

POSITIONAL_EMBEDDING_TYPE = "rotary"


NUM_WORKERS = 4



# ============================================================
# SEED
# ============================================================

def seed_everything(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



seed_everything()



# ============================================================
# LOAD DATA
# ============================================================


def load_json(path):

    with open(path,"r") as f:
        data=json.load(f)

    print(
        "Loaded:",
        path,
        "samples:",
        len(data)
    )

    return data



records = load_json(TEST_JSON)



# ============================================================
# TOKENIZER
# ============================================================


print("Loading tokenizer")


tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH,
    model_max_length=MODEL_MAX_LENGTH
)


print(
    "vocab:",
    len(tokenizer)
)



# ============================================================
# SEQUENCE PROCESS
# ============================================================


def normalize_sequence(seq):

    seq = seq.upper()
    seq = seq.replace(
        "U",
        "T"
    )

    return seq



def get_center_window(
        sequence,
        center_position
):

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



    window = (
        "N"*left_pad
        +
        sequence[start:end]
        +
        "N"*right_pad
    )


    assert len(window)==WINDOW_LENGTH


    return window




# ============================================================
# CANDIDATE EXTRACTION
# ============================================================


def extract_candidates(record):

    seq = normalize_sequence(
        record["seq"]
    )

    labels = record["label"]


    candidates=[]


    for pos,base in enumerate(seq):

        if base!="C":
            continue


        candidates.append(
            {

            "position":pos,

            "window":
                get_center_window(
                    seq,
                    pos
                ),

            "label":
                float(
                    labels[pos]>0
                ),

            "raw_label":
                float(
                    labels[pos]
                )

            }
        )


    return candidates




# ============================================================
# DATASET
# ============================================================


class RankingDataset(Dataset):

    def __init__(self,records):

        self.records=records



    def __len__(self):

        return len(self.records)



    def __getitem__(self,index):

        candidates = extract_candidates(
            self.records[index]
        )


        if len(candidates)==0:

            return None



        return {

            "windows":
                [
                    x["window"]
                    for x in candidates
                ],

            "labels":
                torch.tensor(
                    [
                    x["label"]
                    for x in candidates
                    ],
                    dtype=torch.float
                ),

            "positions":
                [
                    x["position"]
                    for x in candidates
                ],

            "raw_labels":
                [
                    x["raw_label"]
                    for x in candidates
                ]

        }




def collate_fn(batch):

    batch=[
        x for x in batch
        if x is not None
    ]

    if len(batch)==0:

        return None


    return batch




# ============================================================
# MODEL HEAD
# ============================================================


class CenterTokenScoreHead(nn.Module):


    def __init__(
        self,
        config,
        center_index
    ):

        super().__init__()

        self.center_index=center_index


        self.dropout=nn.Dropout(
            config.hidden_dropout_prob
        )


        self.dense=nn.Linear(
            config.hidden_size,
            config.hidden_size
        )


        self.out_proj=nn.Linear(
            config.hidden_size,
            1
        )



    def forward(
        self,
        hidden_states
    ):


        x=hidden_states[:,
            self.center_index,:
        ]


        x=self.dropout(x)

        x=self.dense(x)

        x=torch.tanh(x)

        x=self.dropout(x)

        score=self.out_proj(x)


        return score.squeeze(-1)




# ============================================================
# BUILD MODEL
# ============================================================


def build_model():


    config=AutoConfig.from_pretrained(

        MODEL_CONFIG_PATH,

        vocab_size=len(tokenizer),

        pad_token_id=
        tokenizer.pad_token_id,


        num_labels=1,

        problem_type="regression",


        hidden_size=HIDDEN_SIZE,

        intermediate_size=
        INTERMEDIATE_SIZE,


        num_attention_heads=
        NUM_ATTENTION_HEADS,


        num_hidden_layers=
        NUM_HIDDEN_LAYERS,


        positional_embedding_type=
        POSITIONAL_EMBEDDING_TYPE

    )



    model=EsmForSequenceClassification(

        config,

        head_type="Linear",

        freeze=False,

        kernel_sizes=[2,3,5],

        ocs=32

    )



    model.classifier=CenterTokenScoreHead(

        config,

        CENTER_TOKEN_INDEX

    )



    print(
        "Loading model:",
        MODEL_PATH
    )



    state_dict=load_file(
        MODEL_PATH,
        device="cpu"
    )


    model.load_state_dict(
        state_dict,
        strict=True
    )


    print(
        "Model loaded"
    )


    return model




# ============================================================
# TOKENIZE
# ============================================================


def tokenize_windows(
        windows,
        device
):


    encoded=tokenizer(

        windows,

        padding=True,

        truncation=True,

        max_length=
        MODEL_MAX_LENGTH,

        return_tensors="pt"

    )


    return {
        k:v.to(device)
        for k,v in encoded.items()
    }




# ============================================================
# FORWARD
# ============================================================


@torch.no_grad()

def predict_scores(
        model,
        transcript,
        device
):


    encoded=tokenize_windows(
        transcript["windows"],
        device
    )


    outputs=model(

        **encoded,

        output_hidden_states=True

    )



    hidden=outputs.hidden_states[-1]


    scores=model.classifier(
        hidden
    )


    return (

        scores.cpu().numpy(),

        transcript["labels"].numpy()

    )




# ============================================================
# METRICS
# ============================================================


def calculate_metrics(
        scores,
        labels
):


    order=np.argsort(
        -scores
    )


    sorted_labels=labels[order]



    positive=np.where(
        sorted_labels>0
    )[0]


    if len(positive)==0:

        return None



    best_rank=positive[0]+1



    result={}


    for k in [1,5,10,20]:

        result[f"hit{k}"]=int(
            np.any(
                sorted_labels[:k]>0
            )
        )



    total_positive=labels.sum()


    for k in [5,10,20]:

        result[f"recall{k}"]=(
            sorted_labels[:k].sum()
            /
            max(total_positive,1)
        )



    result["mrr"]=1.0/best_rank

    result["rank"]=best_rank


    return result




# ============================================================
# EVALUATION
# ============================================================


@torch.no_grad()

def evaluate(
        model,
        loader,
        device
):


    model.eval()


    all_results=[]



    for batch in loader:


        if batch is None:
            continue


        for transcript in batch:


            scores,labels=predict_scores(

                model,

                transcript,

                device

            )


            metric=calculate_metrics(
                scores,
                labels
            )


            if metric:

                all_results.append(
                    metric
                )



    final={}


    for k in [1,5,10,20]:

        final[f"hit{k}"]=np.mean(

            [
            x[f"hit{k}"]
            for x in all_results
            ]

        )



    for k in [5,10,20]:

        final[f"recall{k}"]=np.mean(

            [
            x[f"recall{k}"]
            for x in all_results
            ]

        )



    final["MRR"]=np.mean(

        [
        x["mrr"]
        for x in all_results
        ]

    )


    ranks=[x["rank"] for x in all_results]


    final["mean_best_rank"]=np.mean(ranks)

    final["median_best_rank"]=np.median(ranks)


    final["num_transcripts"]=len(all_results)



    return final




# ============================================================
# MAIN
# ============================================================


def main():


    device=torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print(
        "Device:",
        device
    )



    dataset=RankingDataset(
        records
    )



    loader=DataLoader(

        dataset,

        batch_size=1,

        shuffle=False,

        num_workers=NUM_WORKERS,

        collate_fn=collate_fn

    )



    model=build_model()

    model.to(device)



    metrics=evaluate(

        model,

        loader,

        device

    )



    print("\n======================")
    print("FINAL RESULTS")
    print("======================")


    for k,v in metrics.items():

        print(
            f"{k}: {v:.6f}"
            if isinstance(v,float)
            else
            f"{k}: {v}"
        )




if __name__=="__main__":

    main()