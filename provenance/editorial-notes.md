# Editorial and structural revision notes

The public pages were rewritten from the supplied `ORCA_EditPredict` manuscripts, using Modules 1–3 as organization and language references. Original manuscripts are retained in [source-documents](source-documents/). Their embedded instructions and figure placeholders were treated as source material, not as additional user requests.

## Main revisions

- Separated the concise scientific narrative from the DBTL record. The main page now follows background, model overview, results, discussion and availability; the DBTL page organizes actual source-reported decisions separately from the later repository audit.
- Replaced inflated or ambiguous validation language with evidence-specific wording. Saved outputs, recomputation from archived predictions and draft-only summaries are distinguished throughout.
- Corrected method descriptions against supplied code and metadata, while leaving source labels, thresholds, datasets and saved outputs unchanged.
- Shortened figure captions, added meaningful alternative text, and linked publication figures to source data and provenance rather than using figure-generation instructions as captions.
- Harmonized names and units through the [terminology ledger](terminology.md), including AP, region counts and the two downstream stages.

## Substantive corrections and retained uncertainties

| Source claim or conflict | Public treatment | Supporting source or limitation |
|---|---|---|
| Test set is independent; split prevents related transcript windows crossing partitions | Use “test partition”; transcript-disjoint independence is unverified | Transcript IDs and builder absent; exact region and 101-nt candidate contexts overlap between partitions |
| 150 validation and 140 test “transcripts” | Use positive regions evaluated conditionally | JSON records are region sequences; evaluation code skips records without positive candidates |
| `pr_auc` described generically as PR-AUC | Specify average precision (AP) | Stage I code calls `average_precision_score` |
| Best validation value 0.813 presented as final-model value | Separate best Optuna trial 0.813435586 from final validation AP 0.810014436 | `best_trial.json` and `final_model/validation_metrics.json` |
| Stage I uses 1,024 nt throughout | Preserve the intended regional task but flag the tokenization inconsistency | `tokenize_data.py` limits inputs to 103 tokens; training/evaluation declare 1,026; the tokenized training cache is absent |
| Stage I learning rate 5e-4 and token dropout 0 | Report saved values 6.706824e-5 and 0.005515807 | `final_model/best_hyperparameters.json` |
| Both downstream models use binary heads and Optuna | Describe Stage II scalar centre-token scoring and multi-positive ranking loss | `train_orca_ranking_101.py`; supplied script has no Stage II Optuna search |
| Stage II learning rate 1.0139e-5 and other trial settings | Use retained state: learning rate 5e-6, weight decay 0.01, accumulation eight, epoch two | `best_model/training_state.json.txt`; the original invalid-JSON formatting is preserved without changing its content |
| Adapted LAMAR consistently improves validation performance | Report five descriptive histories without a general superiority claim | Frozen original-LAMAR logged maximum AP 0.818050778 exceeds adapted retained-history maximum 0.810014436 |
| Baseline masked accuracy 0.505; corpus 644,292 transcripts and 1,443,754,799 nt | Treat baseline and corpus counts as draft-reported; omit unverified corpus counts from the main text | Baseline evaluation absent; large FASTA/partition members remain in the original archive and their corpus counts were not recomputed; final masked accuracy 0.575467575 is logged |
| Stage II test ranking establishes practical model benefit | Present the original numbers as draft-reported, conditional summaries | Full score outputs absent and no new inference performed; weights remain in the original archive; checkpoint metadata supports validation Hit@10 only |
| Cytidine classification AP around 0.2 | Mention as source-reported motivation in DBTL only | No corresponding predictions or metric output provided |
| Two-stage pipeline reduces the experimental search space while retaining sites | Describe the intended use and require end-to-end evaluation | No archived evaluation after an explicit Stage I selection threshold |
| GENCODE v50 cited as 2025 | Correct release date to June 2026 without changing reference version | [Official GENCODE release history](https://www.gencodegenes.org/human/releases.html) |
| Root README advertises a complete runtime layout | Replace with an accurate publication map and reuse boundary | Large weights and corpus members remain in the original archive; dataset-generation code, transcript IDs and the local LAMAR implementation are unavailable |

## Editorial scope

No source labels, read thresholds, sample inclusion rules, reference versions, statistical methods or frozen splits were changed. Correct metric naming describes the supplied computation; it does not replace it. Missing outputs were not reconstructed from plot pixels, imputed or invented.

The revised main wiki does not claim prospective experimental validation, calibrated editing probabilities, transcriptome-wide specificity or superiority across editor variants. An overlap audit checks a limitation of the existing partition; it does not quantify the effect of that overlap on model performance.

The Nature-polishing workflow was used with an algorithmic paper type, introduction/methods/results/discussion sections, English source language and generic journal framing. The reference structure came chiefly from Module 3, with Module 1's evidence-aware DBTL style. The language revision does not imply journal submission or acceptance.
