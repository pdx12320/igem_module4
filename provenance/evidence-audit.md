# Evidence audit

This audit reads the supplied data, scripts, saved logs and predictions. It does not train a model, change labels or evaluate biological performance. Run `python3 scripts/validate_repository.py` to reproduce the checks using Python's standard library. The initial report is recorded in [evidence-audit.json](evidence-audit.json); reruns may include more files in syntax counts as documentation and plotting scripts are added.

## What is recoverable

The region CSVs contain 3,835 training, 213 validation and 212 test rows, with 2,643, 150 and 140 positive regions, respectively. The JSON files under `data/101nt_dataset/` contain the same region sequences in the same order, with one value per base; the files do not store pre-extracted 101 nt examples. Positive values occur only at cytidines. The ranking code extracts 101 nt C-centered contexts and adds N padding at region boundaries. Region lengths vary up to 1,024 nt, including shorter records.

There are 722,477 / 39,344 / 39,381 cytidine candidate occurrences and 7,130 / 402 / 330 positive site occurrences in training / validation / test. These are occurrences within region records, not deduplicated genomic sites. The positive-site TSV has 12,908 unique chromosome-position-strand tuples, 6,320 plus-strand C>T and 6,588 minus-strand G>A entries; all provided sequence contexts are 101 nt with central C. Raw depth/supporting-read columns and a mapping from this TSV to split records are absent, so the draft filtering thresholds cannot be reconstructed.

Saved Stage I test predictions align row-for-row with the test CSV. Recomputing their statistics recovers:

| Quantity | Value |
|---|---:|
| Average precision (historical key `pr_auc`) | 0.7222672657 |
| ROC AUC | 0.5602182540 |
| Positive prevalence | 140/212 = 0.6603773585 |
| TP / FP / FN / TN | 100 / 42 / 40 / 30 |
| Accuracy | 0.6132075472 |
| Precision / Recall / F1 | 0.7042253521 / 0.7142857143 / 0.7092198582 |

The source computes `pr_auc` with `average_precision_score`, which is non-interpolated average precision (AP), not trapezoidal area under the precision-recall curve. The audit implements the same threshold-grouped treatment of tied scores. These are metrics reconstructed from saved predictions, not new model inference. Unobserved labels remain uncertain biological negatives.

## Split overlap and evaluation unit

| Split pair | Shared unique region sequences | Shared unique C-centered 101 nt contexts |
|---|---:|---:|
| Training–validation | 1 | 10,312 |
| Training–test | 2 | 9,919 |
| Validation–test | 0 | 5,471 |

The one shared training–validation sequence occurs twice in training and once in validation. The two shared training–test sequences each occur once in each split. Context counts use the supplied ranking code's U-to-T normalization and N-padding convention; they count unique exact contexts, not genomic loci. Overlap may involve isoforms or related regions, but its cause cannot be determined without identifiers and construction code. Exact overlap means that the evaluation is not sequence-disjoint. The drafts' transcript-level split assertion cannot be verified from records containing only `seq` and `label`. Do not call the test set independent or interpret these scores as established transcript-disjoint generalization.

Four unique 1,024-nt sequences have both binary labels across their occurrences: three conflicts occur within training, and one occurs between training and test. One of the three within-training cases is also the sequence shared with validation. The validator reports sequence hashes and source CSV line numbers for review. Identical sequence alone cannot explain the label disagreement; missing locus/isoform provenance prevents resolving its cause. No duplicate was removed and no label was changed.

The ranking evaluator skips records without a labelled positive and averages metrics across remaining records. Its 150 validation and 140 test evaluation units are positive **regions**, not verified unique transcripts. It reads the entire corresponding JSON split; it does not filter regions using Stage I predictions. Therefore the reported ranking metrics describe conditional ranking within known positive regions, not an end-to-end two-stage screening evaluation.

## Draft contradictions resolved in the presentation

- Stage I best Optuna trial 17 has validation AP 0.8134355865. The separately saved final model has validation AP 0.8100144359. They should not be conflated.
- `best_hyperparameters.json` records learning rate `6.706824153824951e-05`, weight decay `7.919413614819866e-06`, token dropout `0.005515807069728134`, warmup ratio `0.05338263866669417`, device batch size 4 and accumulation 2. Draft learning rate `5e-4` and dropout `0.0` conflict with those saved settings.
- Stage II is a scalar center-token ranking head with a multi-positive log-sum-exp ranking loss. Its supplied script uses learning rate `5e-6`, weight decay `0.01`, accumulation 8, and up to 8 epochs; it selects by validation Hit@10. This differs from the drafts' two-class/Optuna model description. The saved state reports epoch 2, global step 958 and best Hit@10 0.8466666667, but has a trailing comma that makes it invalid JSON. The original is retained as `.json.txt` without editing its content.
- The strongest saved checkpoint validation AP is 0.8180507779 for original LAMAR with frozen encoder; adapted fine-tuning reaches 0.8100144359, original fine-tuning 0.7975657688, adapted frozen encoder 0.7812724672 and random initialization 0.7416366054. These are single recorded runs with different saved history lengths, not matched replicated comparisons. They do not establish uniform superiority of continued pretraining.
- Continued-pretraining logs contain 100 evaluation points from step 1,000 to 100,000. The first saved masked accuracy is 0.5320836602 and final accuracy is 0.5754675750. The claimed original-model baseline 0.505 is present only in the drafts. A gain from 0.505 to 0.575 is therefore not independently verifiable from saved evaluations.
- Stage II Hit@K, Recall@K, MRR, mean/median best rank and AP around 0.2 are reported in the drafts, but per-candidate scores and machine-readable ranking evaluations are absent. Preserve attribution rather than presenting them as revalidated outcomes.

## Reproducibility gaps

The archived Stage I `tokenize_data.py` sets a maximum of 103 tokens, while training and evaluation declare 1,026 tokens. The saved tokenized training cache is absent. This discrepancy could materially change the input context and cannot be resolved by silently changing one value; the effective context of the historical training run remains unknown.

This publication repository omits large model weights and pretraining inputs retained in the original archive. The [archive inventory](archive-inventory.json) records eight `.safetensors` files, the GENCODE annotation and transcriptome FASTA, and train/validation/test pretraining text files; those members were not extracted for this lightweight audit. The imported local `LAMAR` implementation, the dataset construction/splitting program, the tokenized cache and a pinned training environment are missing from the supplied archive. Several scripts expect an `output/` layout not present in the source archive. The `config_150M.json` name is not evidence of the trained parameter count: it is a 30-layer/640-hidden base configuration that the scripts override with 12 layers and hidden size 768. Model training and inference are therefore not advertised as runnable from this publication repository alone.

Draft transcriptome counts and read/depth thresholds are source-reported background, not independently reconstructed measurements. No biological replication, experimental validation of new predictions, calibration assessment, uncertainty estimate or transcriptome-wide safety test was performed by this audit. Labels, thresholds, reference versions, splits, statistical methods and source data were left unchanged.

## Integrity and limits of the validator

The validator checks imported file digests against the publication manifest, parses JSON and Python syntax without executing training code, validates fixed dataset counts and schemas, checks exact cross-split overlaps, validates TSV context/strand conventions, and recomputes Stage I metrics from saved predictions. Known scientific limitations produce warnings; altered frozen counts, inconsistent labels, corrupted files or metric disagreements produce errors and a nonzero exit status. `--output PATH` creates a new report and refuses to overwrite an existing file. The manifest records source and published digests; independently verifying transformed source bytes requires the original archive.
