# Saved results

This directory holds the supplied training histories, checkpoint metadata, selected hyperparameters, Stage I test predictions and evaluation summaries. It contains no neural-network weights.

| Result | Location |
|---|---|
| Continued-pretraining history | [trainer_state.json](model_continual_pretrain/transcriptome_pretrain/checkpoint-100000/trainer_state.json) |
| Stage I five-condition comparison | `model_1024_classifier/1024_*/final_model/` |
| Best Optuna trial | [best_trial.json](model_1024_classifier/1024_orca_puf12/best_trial.json) |
| Final validation summary | [validation_metrics.json](model_1024_classifier/1024_orca_puf12/final_model/validation_metrics.json) |
| Test metrics | [test_metrics.json](model_1024_classifier/1024_orca_puf12/test_evaluation/test_metrics.json) |
| Test row-level predictions | [test_predictions.csv](model_1024_classifier/1024_orca_puf12/test_evaluation/test_predictions.csv) |
| Stage II selected-checkpoint metadata | [training_state.json.txt](model_101_classifier/best_model/training_state.json.txt) |

The original key `pr_auc` is preserved. In the supplied Stage I code it is computed with `average_precision_score`, so publication text names it **average precision (AP)**. Search-best validation AP, final-model validation AP and test AP are different records.

Only machine-specific path metadata was sanitized for publication. All metric values and log entries remain unchanged; [import-manifest.json](../provenance/import-manifest.json) records both source and publication hashes.

Stage II per-region scores and the detailed ranking-evaluation output are absent. Values transcribed from the original Wiki tables are explicitly described as draft-reported rather than recomputed.
