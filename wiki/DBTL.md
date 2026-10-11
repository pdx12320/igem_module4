# Module 4 | Design–Build–Test–Learn

## Engineering objective

We aimed to prioritize PUF12-associated RNA editing candidates for targeted experimental follow-up. RNA-seq provided positive editing evidence, while LAMAR supplied sequence representations. Our workflow combines human transcriptome adaptation, regional classification and cytidine ranking.

The central challenge was to connect model outputs with a practical screening decision. Observed editing can support positive labels, but absence of detection does not establish that a cytidine cannot be edited.

## Cycle 1 | Adapt sequence representations to human transcripts

**Design.** We asked whether continued pretraining could adapt LAMAR representations to the human transcriptome before learning editing-associated labels. Masked nucleotide prediction provided an unsupervised objective for this step. [Zhou et al., 2025](https://doi.org/10.1186/s13059-025-03752-x).

**Build.** We continued masked language modelling on human transcript sequences derived from GRCh38 and GENCODE release 50. The model reconstructed masked nucleotides from their surrounding sequence context. [Mudge et al., 2025](https://doi.org/10.1093/nar/gkae1078); [GENCODE release history](https://www.gencodegenes.org/human/releases.html).

**Test.** Masked nucleotide accuracy increased from **0.5321 at step 1,000** to **0.5755 at step 100,000**, across 100 recorded evaluations.

**Learn.** Sequence reconstruction improved during adaptation, but this objective does not measure editing prediction. We therefore evaluated downstream classification separately rather than assuming that adaptation would improve every task.

**Evidence:** [Continued-pretraining curves](figures/pretraining-curves/figure@2x.png).

## Cycle 2 | Distinguish detected editing from uncertain negatives

**Design.** We required positive-site evidence supported by at least **20 editing reads** and **100 total reads**. The aim was to reduce uncertainty among positive labels while preserving the distinction between undetected and non-editable cytidines.

**Build.** We paired transcript-region sequences with binary region labels and position-level editing labels. A region was positive when it contained at least one labelled editing site. Other regions received nominal negative labels.

**Test.** The training, validation and test partitions contained **3,835, 213 and 212 regions**, respectively. Their positive counts were **2,643, 150 and 140**. Positive position-level labels occurred only at cytidines and agreed with the binary region labels.

**Learn.** Consistent labels make the computational task well defined, but do not remove uncertainty in biological negatives. Limited coverage, transcript abundance and editing frequency can all affect whether a site is detected.

## Cycle 3 | Screen regions before prioritizing individual cytidines

**Design.** We separated two questions: which regions are associated with detected editing, and which cytidines within those regions should be tested first? This allowed regional screening and local ranking to use different objectives.

**Build.** Stage I used a LAMAR-based binary region classifier, with Optuna selecting hyperparameters by validation average precision (AP). [Akiba et al., 2019](https://doi.org/10.1145/3292500.3330701). The intended regional input was up to **1,024 nt**. Stage II used **101-nt cytidine-centred contexts** and a scalar ranking head.

**Test.** Stage I reached test **AP 0.7223** and **ROC AUC 0.5602** across 212 regions. Positive prevalence was **140/212 = 0.6604**. The best search-trial validation AP was **0.8134**, while the final model's validation AP was **0.8100**.

**Learn.** The regional model showed modest discrimination on this partition. Its performance must be interpreted against the high positive prevalence and does not establish useful end-to-end recovery by itself. [Saito & Rehmsmeier, 2015](https://doi.org/10.1371/journal.pone.0118432).

A context-length inconsistency also remains unresolved: tokenization specifies 103 tokens, while training and evaluation specify 1,026. Reconciling this discrepancy is necessary before testing the contribution of long-range sequence context.

**Evidence:** [Stage I precision–recall curve](figures/stage1-test-pr/figure@2x.png).

## Cycle 4 | Test whether adaptation improves downstream performance

**Design.** We compared adapted and original LAMAR, their encoder-frozen variants and a randomly initialized model. These conditions tested the relationship between initialization, encoder training and regional prediction.

**Build.** We examined training loss and validation AP for all five conditions. These quantities describe different aspects of learning: fitting the training data and ranking labelled validation regions.

**Test.** Frozen original LAMAR reached validation **AP 0.8181**, compared with **0.8100** in the adapted model's retained history. The other maxima were **0.7976** for original fine-tuning, **0.7813** for frozen adapted LAMAR and **0.7416** for random initialization.

**Learn.** The comparisons did not establish consistent superiority of continued pretraining. Different training lengths and the absence of repeated runs limit the strength of between-condition conclusions. Training loss alone should not determine which model enters the next experimental cycle.

**Evidence:** [Training loss](figures/stage1-training-loss/figure@2x.png) and [validation AP](figures/stage1-validation-ap/figure@2x.png).

## Cycle 5 | Evaluate ranking against a limited candidate budget

**Design.** Experimental screening can test only a subset of cytidines. We therefore evaluated whether labelled editing sites appeared among the highest-scoring candidates using Hit@K, Recall@K and mean reciprocal rank.

**Build.** Stage II used a multi-positive ranking loss and selected its checkpoint by validation Hit@10. The selected checkpoint was from epoch two. Ranking evaluation excluded regions without labelled positives, leaving **150 validation and 140 test regions**.

**Test.** Checkpoint metadata records validation **Hit@10 of 0.8467**. The project summary reports test **Hit@5 of 0.607**, **Hit@10 of 0.779** and **Hit@20 of 0.921**, with median best rank **4**. These test summaries remain provisional because the underlying candidate scores are unavailable for independent reproduction.

**Learn.** Conditional ranking provides an initial basis for selecting candidates within positive regions. It does not establish performance across all transcriptomic regions or after Stage I selection. A complete assessment must include regions that Stage I misses and candidates that experiments do not confirm.

**Evidence:** [Reported Stage II ranking performance](figures/stage2-ranking-reported/figure@2x.png).

## Next cycle | Establish independent end-to-end performance

**Design.** The next evaluation must separate related sequences and test the full screening procedure. Current training data share one complete region with validation and two with test, alongside extensive local-context overlap. Transcript-disjoint performance remains unestablished. Similarity-aware separation of biomolecular sequences provides a rationale for the next split design. [Joeres et al., 2025](https://doi.org/10.1038/s41467-025-58606-8).

**Build.** We will restore transcript provenance, reconcile input tokenization and retain all candidate scores. Regional selection and the number of candidates tested per region should be fixed before evaluating new data.

**Test.** Planned assays will compare high-ranked candidates with appropriate low-ranked controls under defined editor and cell conditions. End-to-end recovery and experimental precision will connect computational scores with the intended screening use.

**Learn.** Newly confirmed editing and non-editing outcomes can improve subsequent training labels. Prospective results will determine whether the workflow reduces experimental effort while retaining relevant candidate sites.

## Data and code availability

Datasets, analysis code, training records and reproducibility notes are available in the original [Module 4 GitHub repository](https://github.com/pdx12320/igem_module4). The [main wiki](module4.md) presents the model design, results and interpretation.
