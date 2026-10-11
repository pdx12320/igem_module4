# Historical model scripts

The supplied scripts document the model implementation used by the project. They are retained for inspection, with their original scientific parameters and behavior. They are **not a ready-to-run training package**.

| Script | Role |
|---|---|
| `model_continual_pretrain/train.py` | Continued masked-language-model training |
| `model_1024_classifier/tokenize_data.py` | Historical preparation of tokenized region data |
| `model_1024_classifier/train.py` | Stage I region classification and Optuna search |
| `model_1024_classifier/evaluation_1024.py` | Stage I saved-model evaluation |
| `model_101_classifier/train_orca_ranking_101.py` | Stage II multi-positive ranking training |
| `model_101_classifier/evaluation_top_ranking.py` | Stage II ranking evaluation within positive regions |

## Requirements for a future rerun

1. Recover the exact compatible `LAMAR` source package. Imports such as `LAMAR.sequence_classification_patch` and `LAMAR.modeling_nucESM2` are not bundled. The upstream project is [rnasys/LAMAR](https://github.com/rnasys/LAMAR), but the archive does not identify its commit.
2. Recover the original weights and pretraining inputs from the source archive. All `model.safetensors` files are excluded from Git and listed in the [archive inventory](../../provenance/archive-inventory.json).
3. Reconcile expected `output/` paths with the actual checkpoint locations, and configure data paths explicitly. Moving the historical code into this directory intentionally separates it from the published data and results.
4. Resolve the tokenizer-length discrepancy before recomputing any scientific results: the supplied Stage I tokenization script uses `model_max_length=103`, whereas the classifier configures 1,026 tokens. No cached tokenized dataset is supplied, so the effective training context cannot be verified.
5. Recover the training environment and set bounded CPU/GPU resources. The scripts import PyTorch, Transformers, Datasets, NumPy, pandas, Optuna, scikit-learn and safetensors. The archive supplies no environment lock; a guessed dependency list is not a reconstruction.

The pretraining script also contains Windows-style backslashes inside default data paths and a default multi-GPU setting. The tokenization script uses all detected CPU cores. Do not execute these scripts as part of a publication check.

The raw configuration filename `config_150M.json` is not a reliable description of the downstream architecture. Training scripts override the template to hidden size 768, 12 layers and 12 attention heads. The source scalar-ranking checkpoint also uses these dimensions.

The malformed Stage II training metadata is preserved as [`training_state.json.txt`](../../results/model_101_classifier/best_model/training_state.json.txt). Its trailing comma is documented rather than silently repaired. Scientific parameters, labels and model code were not changed during repository preparation.
