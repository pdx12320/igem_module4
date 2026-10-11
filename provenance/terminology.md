# Terminology ledger

| Canonical term | Definition and editorial decision |
|---|---|
| Module 4 / ORCA_EditPredict | The supplied LAMAR-based editing-candidate prioritization project; the archive name is retained for provenance. |
| PUF12 | Editor dataset identifier in the supplied files. Do not generalize its performance to other PUF or APOBEC variants. |
| LAMAR | Foundation language model for RNA regulation described by Zhou et al. (2025). Credit the original model separately from this project’s adaptation. |
| Continued pretraining | Masked language modelling on human transcript sequences, preceding the two downstream stages. It is not called Stage I in the public narrative. |
| Stage I | Intended region classifier using up to 1,024-nt inputs. The 103-token preprocessing limit conflicts with the 1,026-token training/evaluation declarations; historical effective context is unverified. Source directory: `model_1024_classifier`. |
| Stage II | Cytidine ranking using 101-nt contexts and a scalar centre-token score. Source directory `model_101_classifier` is historical; it does not establish binary classification as the retained objective. |
| RNA-seq-supported positive | A supplied positive label associated with detected editing evidence. It is not a complete census of editable cytidines. |
| Nominal negative | No positive label in the supplied record; not an experimentally confirmed non-editable site. |
| Average precision (AP) | The quantity computed by `average_precision_score`; saved JSON keys use `pr_auc`. The public description specifies AP rather than implying trapezoidal precision–recall area. |
| Best trial validation AP | The largest stored Optuna trial value, 0.813435586. Distinct from the final validation report, 0.810014436. |
| Region / window / record | A stored sequence record, generally 1,024 nt. Transcript identities are absent, so record counts must not be called unique-transcript counts. |
| Hit@K | Fraction of eligible positive regions with at least one labelled positive among the top K scored candidate cytidines. |
| Recall@K | Mean fraction of labelled positive candidates recovered among the top K within each eligible positive region. |
| Mean reciprocal rank (MRR) | Mean reciprocal rank of the highest-ranked labelled positive in each eligible positive region. |
| Ranking score | An uncalibrated scalar used to order candidates; not an editing probability or editing fraction. |
| Draft-reported | A value available in the original narrative but not recomputed from the required prediction outputs in this publication audit. |
| Recomputed from saved predictions | A consistency check against archived predictions, not a new inference run or independent biological validation. |
| GENCODE release 50 | Source reference version retained unchanged. The release date is June 2026, as listed by GENCODE; the source’s 2025 attribution is corrected. |

The original README numbers continued pretraining, region classification and ranking as stages 1–3. The English wiki uses “continued pretraining”, “Stage I” and “Stage II” consistently with the source scientific narrative. This is terminology harmonization, not a change in the pipeline.

The source expression `PUF-LIKER-APOBEC` is not expanded or silently interpreted as a verified construct specification. The public account uses the supplied PUF12 dataset identifier and PUF–APOBEC context without inventing a linker identity.
