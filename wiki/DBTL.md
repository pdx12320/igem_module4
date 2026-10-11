# Module 4 | Design–Build–Test–Learn

## Engineering objective

Prioritize PUF12-associated RNA editing candidates for targeted follow-up using sequence evidence and RNA-seq-derived labels. Continued pretraining precedes a region classifier (Stage I) and a cytidine ranker (Stage II).

This record separates source-reported development, saved computational outputs and checks performed during repository preparation. No model was trained and no new biological experiment was performed for this revision.

## Cycle 1 | Adapt RNA sequence representations to the human transcriptome

**Design.** The source proposed adapting LAMAR to human transcripts before fitting editing labels. Masked sequence reconstruction supplied an unsupervised objective, distinct from predicting whether editing occurs. [Zhou et al., 2025](https://doi.org/10.1186/s13059-025-03752-x).

**Build.** The source describes transcripts generated from GRCh38 and GENCODE release 50, followed by continued masked language modelling. The publication repository includes the training script and checkpoint history. Large transcriptome, pretraining-partition and weight files remain in the original source archive, as recorded in the inventory.

**Test.** The saved log records masked nucleotide accuracy of **0.5755 at step 100,000**. The baseline accuracy of 0.505 occurs in the draft without a corresponding baseline evaluation file.

**Learn.** The available history documents sequence reconstruction during adaptation. It does not independently establish the reported baseline comparison or guarantee improved editing prediction.

**Evidence:** [Figure 1](figures/pretraining-curves/figure@2x.png), [continued-pretraining records](../results/model_continual_pretrain/).

## Cycle 2 | Separate observed editing from uncertain negative labels

**Design.** The original draft sought positive labels supported by at least 20 editing reads and total depth of at least 100. These criteria describe the supplied source; the repository audit does not regenerate labels from raw sequencing reads.

**Build.** The archived datasets pair transcript-region sequences with binary region labels and position-level editing labels. Nominal negative regions contain no recorded positive, rather than experimentally established non-editable cytidines.

**Test.** The supplied partitions contain **3,835 training, 213 validation and 212 test regions**. Their positive counts are **2,643, 150 and 140**, respectively. The position-label records agree with the corresponding binary region labels and sequence order.

**Learn.** Label consistency supports reuse of the frozen files, while biological interpretation still depends on how editing was detected. Unobserved sites cannot automatically provide definitive negative evidence.

**Evidence:** [Region datasets](../data/1024nt_dataset/), [position-labelled datasets](../data/101nt_dataset/), [provenance](../provenance/).

## Cycle 3 | Introduce region scoring before cytidine prioritization

**Design.** The draft reports that direct cytidine classification performed poorly and motivated a regional screening step. Its cytidine-level AP of approximately 0.2 lacks the prediction files required for verification.

**Build.** Stage I uses a LAMAR-based binary region classifier with Optuna selection by validation AP. Stage II uses a centre-token scalar head and multi-positive ranking loss, rather than the binary head described in the draft.

**Test.** The archived Stage I test predictions reproduce **AP 0.7223 and ROC AUC 0.5602** over 212 regions. Positive prevalence is **140/212 = 0.6604**. The best Optuna validation AP, 0.8134, is distinct from final validation AP, 0.8100.

**Learn.** The regional classifier supplies a checkable screening result with limited separation on this partition. The tokenizer limits inputs to 103 tokens, while training and evaluation specify 1,026; the missing cache prevents resolving historical context length. The archive does not establish the complete pipeline's recall after a Stage I threshold, so regional and conditional ranking metrics remain separate.

**Evidence:** [Figure 4](figures/stage1-test-pr/figure@2x.png), [saved test outputs](../results/model_1024_classifier/1024_orca_puf12/test_evaluation/), [Stage II implementation](../code/historical/model_101_classifier/train_orca_ranking_101.py).

## Cycle 4 | Compare training conditions without overstating adaptation

**Design.** The source compared adapted and original LAMAR, their encoder-frozen variants and a randomly initialized condition. These conditions probe different initialization and optimization choices.

**Build.** Five checkpoint histories preserve training loss and validation AP. The retained logs have different lengths, and the bundle does not provide repeated runs establishing uncertainty for between-condition comparisons.

**Test.** The frozen original-LAMAR history reaches **validation AP 0.8181**, while the retained adapted-model history reaches **0.8100**. The other logged maxima are 0.7976 for original fine-tuning, 0.7813 for frozen adapted LAMAR and 0.7416 for random initialization.

**Learn.** Lower training loss and higher validation AP answer different questions. These records do not establish consistent superiority of continued pretraining, and the comparison remains descriptive.

**Evidence:** [Training loss](figures/stage1-training-loss/figure@2x.png), [validation AP](figures/stage1-validation-ap/figure@2x.png), [saved Stage I histories](../results/model_1024_classifier/).

## Cycle 5 | Align ranking metrics with their actual evaluation population

**Design.** Candidate ranking asks whether labelled editing sites appear early enough for a limited experimental screen. The archived evaluator reports Hit@K, Recall@K and mean reciprocal rank within each eligible region.

**Build.** Stage II scores 101-nt cytidine contexts and selects the checkpoint using validation Hit@10. Regions without labelled positive candidates are skipped. The retained checkpoint state records epoch two, learning rate 5 × 10⁻⁶ and accumulation of eight.

**Test.** Checkpoint metadata records validation Hit@10 of **0.8467**. The draft reports test Hit@5, Hit@10 and Hit@20 of **0.607, 0.779 and 0.921**, respectively, over 140 positive regions. Missing candidate-score outputs prevent recomputation of the full ranking summaries.

**Learn.** Ranking results concern regions already known to contain positive labels. They do not quantify performance on all transcriptomic regions, prove calibrated editing probabilities or replace prospective experimental confirmation.

**Evidence:** [Figure 5](figures/stage2-ranking-reported/figure@2x.png), [saved Stage II metadata](../results/model_101_classifier/), [evaluation code](../code/historical/model_101_classifier/evaluation_top_ranking.py).

## Repository audit | Check the claimed evaluation boundary

**Question.** The original draft calls the test set independent and describes a transcript-level split. We checked whether the supplied files support those descriptions without modifying any partition.

**Check.** Transcript identifiers and the dataset builder are absent. Exact sequence comparison identifies one shared full region between training and validation and two between training and test. The corresponding sets share **10,312 and 9,919 unique 101-nt candidate contexts**.

**Consequence.** The public narrative uses “test partition” and “regions”, avoiding an independently verified transcript-disjoint claim. These checks assess archive consistency and overlap; they do not create a new benchmark or repair the frozen split.

## Next iteration

The next model-development cycle requires restored transcript provenance, an explicitly grouped split and complete per-candidate predictions. New experiments should evaluate high-ranked candidates alongside appropriate controls under defined editor and cell conditions.

A future test should lock the Stage I selection rule and Stage II ranking budget before evaluating new data. Reporting end-to-end recovery and experimental precision would connect computational scores to the proposed screening use. Changes to labels, reference versions or partitions require a separately documented analysis; none were made here.
