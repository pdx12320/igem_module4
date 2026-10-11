\# ORCA\_EditPredict



基于 LAMAR 的 RNA 编辑预测项目，包含：



1\. \*\*Stage 1 – Continual Pretrain\*\*：在转录组数据上继续预训练 LAMAR 基础模型。

2\. \*\*Stage 2 – 1024nt Classifier\*\*：使用 1024 nt 窗口进行二分类（编辑 vs 未编辑）。

3\. \*\*Stage 3 – 101nt Ranking\*\*：使用 101 nt 窗口进行候选位点的排序（ranking），输出 hit@k、recall@k、MRR 等指标。



\---



\## 目录结构



```text

ORCA\_EditPredict/

├── README.md

├── tokenizer/                         ← LAMAR single\_nucleotide 分词器（共享）

├── config\_150M.json                   ← LAMAR 150M 模型配置（共享）

├── data/

│   ├── 1024nt\_dataset/                ← Stage 2 数据集

│   │   ├── orca\_puf12\_train.csv

│   │   ├── orca\_puf12\_valid.csv

│   │   ├── orca\_puf12\_test\_1.csv

│   │   └── orca\_puf12\_test\_2.csv

│   └── 101nt\_dataset/                 ← Stage 3 数据集

│       ├── orca\_puf12\_train.json

│       ├── orca\_puf12\_valid.json

│       └── orca\_puf12\_test.json

├── model\_continual\_pretrain/          ← Stage 1

│   ├── train.py

│   └── output/

│       └── transcriptome\_pretrain/

│           └── checkpoint-100000/

│               └── model.safetensors  ← 继续预训练权重

├── model\_1024\_classifier/             ← Stage 2

│   ├── build\_dataset.py               ← 构建 1024 nt 分类数据集

│   ├── train.py                       ← Optuna 超参搜索 + final 训练

│   └── output/                        ← 训练产出（checkpoints、final\_model）

├── model\_101\_classifier/              ← Stage 3

│   ├── train\_101\_ranking.py           ← 101 nt 排序训练

│   ├── eval\_101\_ranking.py            ← 101 nt 排序评估

│   └── output/                        ← 训练产出（best\_model、last\_model）

└── assets/                            ← 图片等资源

```

