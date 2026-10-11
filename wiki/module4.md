# Module 4 | LAMAR-guided prioritization of RNA editing candidates

## Background

RNA sequencing (RNA-seq) identifies editing events under the conditions measured, but does not exhaustively establish which cytidines can be edited. An undetected event may reflect limited coverage, transcript abundance or editing frequency. Consequently, unobserved cytidines provide uncertain negative labels for sequence-based prediction.

Module 4 examines a two-stage workflow for prioritizing candidate C-to-U editing sites from the PUF12 dataset. Stage I scores transcript regions; Stage II ranks cytidines within regions. This connects the RNA-seq evidence in [Module 1](https://github.com/pdx12320/igem_module1) with candidate selection for subsequent editor-engineering experiments.

The workflow builds on LAMAR, a Transformer-based RNA foundation model pretrained through masked nucleotide prediction. Its published applications motivate adaptation to RNA sequence tasks, without establishing performance for this project's editing labels. [Zhou et al., 2025](https://doi.org/10.1186/s13059-025-03752-x).

## Model overview

### Sequence evidence and evaluation units

The source narrative defines positive sites using at least 20 editing-supporting reads and total depth of at least 100. These criteria are retained as the source description, rather than reconstructed from raw sequencing reads. Nominal negatives indicate absence of a positive label, not confirmed resistance to editing.

The supplied region datasets contain the following records. Short terminal regions are retained alongside full 1,024-nt inputs.

| Partition | Regions | Positive regions | Nominal negative regions |
|---|---:|---:|---:|
| Training | 3,835 | 2,643 | 1,192 |
| Validation | 213 | 150 | 63 |
| Test | 212 | 140 | 72 |

The original description specifies 128-nt overlap and a transcript-level split. However, transcript identifiers and the dataset-generation script are absent. The audit finds one shared full region sequence between training and validation, and two between training and test. These partitions therefore cannot support an independently verified transcript-disjoint evaluation claim.

### Continued pretraining and region classification

The source describes continued masked language modelling using GRCh38 transcripts derived from GENCODE release 50. GENCODE dates this release to June 2026. The transcriptome FASTA and pretraining partitions remain in the original source archive and are omitted from this publication repository. Their corpus counts were not recomputed. [GENCODE release history](https://www.gencodegenes.org/human/releases.html).

The supplied downstream scripts configure a 12-layer encoder with 768 hidden units, 12 attention heads and rotary positional embeddings. Stage I adds a binary classification head and selects configurations using validation average precision (AP). The saved key `pr_auc` denotes `average_precision_score`, not a separately computed trapezoidal precision–recall area.

The supplied Stage I tokenization script limits inputs to 103 tokens, whereas training and evaluation declare 1,026. Without the tokenized cache, the historical training context length cannot be confirmed.

The selected configuration records a learning rate of 6.7068 × 10⁻⁵, weight decay of 7.9194 × 10⁻⁶ and token dropout of 0.00552. It uses batches of four, gradient accumulation of two and up to 15 epochs. These saved settings supersede the conflicting values in the original prose.

### Cytidine ranking

Stage II extracts 101-nt contexts centred on candidate cytidines, padding sequence boundaries as implemented in the archived code. A scalar head scores the central token representation. A multi-positive ranking loss compares the aggregate exponentiated score of labelled positives with that of all candidates in each region.

This implementation differs from the draft's binary-classifier description. The saved training record specifies a learning rate of 5 × 10⁻⁶, weight decay of 0.01 and gradient accumulation of eight. The selected checkpoint is from epoch two, using validation Hit@10 as the selection metric.

Hit@K measures whether a region has a labelled positive among its top K candidates. Recall@K averages the fraction of labelled positives recovered within each region. Mean reciprocal rank (MRR) uses the rank of the first labelled positive. The evaluator skips regions without positive candidates; ranking summaries are therefore conditional on positive regions.

## Results

### Logged masked accuracy increases during continued pretraining

The logged masked nucleotide accuracy increases from **0.5321 at step 1,000** to **0.5755 at step 100,000**. The curves below reproduce 100 recorded evaluation points. The original draft reports an initial-model accuracy of 0.505, but its separate evaluation output is absent. The archive therefore supports the final logged value without independently establishing the stated before-and-after comparison.

![Training loss and masked nucleotide accuracy during continued pretraining.](figures/pretraining-curves/figure@2x.png)

**Figure 1.** Continued-pretraining history reconstructed from the supplied checkpoint log. Masked nucleotide accuracy evaluates sequence reconstruction, not editing prediction.

### Stage I performance depends on the comparison and partition

The best Optuna trial reached validation AP **0.8134**. The final model's separate validation report records **0.8100**, and the saved test predictions reproduce AP **0.7223** and ROC AUC **0.5602**. The test partition contains 140 positive regions among 212 records, giving a prevalence reference of **0.6604**.

![Training-loss histories for five Stage I model conditions.](figures/stage1-training-loss/figure@2x.png)

**Figure 2.** Saved training-loss histories for adapted, original, encoder-frozen and randomly initialized conditions. Runs have different logged training lengths.

![Validation average precision histories for five Stage I model conditions.](figures/stage1-validation-ap/figure@2x.png)

**Figure 3.** Logged validation AP across five conditions. The frozen original-LAMAR run reaches 0.8181, compared with 0.8100 in the adapted model's retained history.

These records do not establish consistent superiority of transcriptome adaptation. The saved test probabilities provide a checkable regional evaluation, but sequence overlap and uncertain negative labels limit generalization claims.

![Precision–recall curve reconstructed from the saved Stage I test probabilities.](figures/stage1-test-pr/figure@2x.png)

**Figure 4.** Stage I test precision–recall curve for 212 regions. AP is 0.7223; the prevalence reference is 140/212. The curve uses archived predictions without new model inference.

### Stage II ranking remains partly documented by the draft

The source reports the following ranking results. Validation and test counts correspond to **150 and 140 positive regions**, respectively, rather than verified unique transcripts.

| Metric | Validation, draft-reported | Test, draft-reported |
|---|---:|---:|
| Hit@1 | 0.247 | 0.243 |
| Hit@5 | 0.607 | 0.607 |
| Hit@10 | 0.847 | 0.779 |
| Hit@20 | 0.953 | 0.921 |
| Recall@5 | 0.381 | 0.373 |
| Recall@10 | 0.635 | 0.573 |
| Recall@20 | 0.841 | 0.792 |
| MRR | 0.409 | 0.406 |
| Median best rank | 4 | 4 |

![Draft-reported Stage II Hit@K and Recall@K values for 140 positive test regions.](figures/stage2-ranking-reported/figure@2x.png)

**Figure 5.** Test ranking summaries for 140 positive regions, transcribed from the supplied draft. The package lacks per-candidate scores needed to recompute these values; no Recall@1 value is inferred.

The checkpoint metadata independently records validation Hit@10 of 0.8467, matching the rounded draft value. The remaining ranking summaries were not recomputed: candidate-score outputs are absent, and no new inference was run. The weights remain in the original archive. They do not measure end-to-end recovery after applying a Stage I selection threshold.

## Discussion

The workflow frames sequence prediction as candidate prioritization for experimental follow-up. Regional AP exceeds the test prevalence reference, but the current evidence does not establish a calibrated probability of editing. Conditional ranking results cannot establish transcriptome-wide specificity or quantify the benefit of the complete two-stage pipeline.

The split audit also identifies 10,312 shared unique 101-nt candidate contexts between training and validation, and 9,919 between training and test. Related transcripts and overlapping contexts require explicit control before future evaluation. These findings are documented without changing the frozen labels, sequences or partitions.

The next iteration should restore transcript identifiers, audit sequence-related groups and evaluate a locked pipeline on new data. Targeted assays should include high-ranked candidates and suitable low-ranked controls. Newly measured editing and non-editing evidence could then inform subsequent models, while keeping model development separate from prospective evaluation.

## Data and code availability

[Datasets](../data/), [saved results](../results/), [historical code](../code/historical/) and [provenance records](../provenance/) accompany this page. [Figure sources](../figure-sources/) retain redraw inputs and export metadata. The separate [DBTL record](DBTL.md) documents engineering decisions and their evidence.

The publication repository preserves a lightweight research handoff. Large model weights, transcriptome references and pretraining corpora remain in the original archive, as recorded in the [inventory](../provenance/archive-inventory.json). The local LAMAR implementation and dataset-generation code remain unavailable, so training and inference require reconstruction. No training, new prediction or wet-lab validation was performed for this publication revision.

## References

1. Zhou, H. et al. (2025). A foundation language model to decipher diverse regulation of RNAs. *Genome Biology*, 26, 301. [Article](https://doi.org/10.1186/s13059-025-03752-x). [LAMAR source repository](https://github.com/rnasys/LAMAR).
2. GENCODE. Human release 50, June 2026. [Human release history](https://www.gencodegenes.org/human/releases.html).
