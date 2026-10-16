# iGEM Module 4 · LAMAR-guided RNA editing prioritization
The main Wiki text and figures are synchronized from the current [WIKI_SUSTech-BIO Model page source](https://github.com/pdx12320/WIKI_SUSTech-BIO/tree/main/wiki/components/model). See [source and figure checksums](wiki/CURRENT_MODEL_SOURCE.json).


**Sequence models propose candidates; experiments establish whether those candidates are edited.**

This repository organizes the ORCA_EditPredict archive for the PUF12 RNA editing dataset. The workflow combines continued pretraining of LAMAR, region classification and cytidine ranking. English wiki pages and ORCA figures distinguish saved evidence from claims that remain documented only in the original draft.

## Two English Markdown documents

1. **[Main wiki](wiki/module4.md)**: background, model overview, results, discussion, data availability and references.
2. **[DBTL record](wiki/DBTL.md)**: separate design, build, test and learn decisions, with links to the supporting evidence.

![Module 4 training and proposed screening workflow](wiki/figures/current-model/model-figure-33-panel1.png)

The workflow separates model development from the proposed screening application. Download the editable [PowerPoint source](wiki/figures/workflow/figure.pptx) or [vector PDF](wiki/figures/workflow/figure.pdf).

![Stage I test precision–recall curve](wiki/figures/stage1-test-pr/figure@2x.png)

The curve is reconstructed from archived test probabilities. Average precision is **0.7223** over **212 regions**, with positive prevalence **0.6604**. No model inference was needed to redraw it.

## Data and code

| Directory | Contents |
|---|---|
| `wiki/` | English main wiki, separate DBTL record and selected publication figures |
| `data/` | Supplied region datasets, position-labelled records and positive-site table |
| `results/` | Archived training histories, model-selection records, Stage I predictions and Stage II checkpoint metadata |
| `code/historical/` | Supplied training, tokenization and evaluation scripts, tokenizer and model configuration |
| `figure-sources/` | ORCA redraw scripts, plot inputs and export metadata |
| `scripts/` | Lightweight repository checks |
| `provenance/` | Terminology, editorial corrections, historical source hashes and validation records |

## Evidence and reuse scope

The Stage I metric saved as `pr_auc` is **average precision (AP)**. The best Optuna validation value is 0.8134; the final validation report records 0.8100. Stage II's validation Hit@10 of 0.8467 appears in checkpoint metadata, while its complete ranking summaries remain draft-reported.

The split audit finds one shared full region sequence between training and validation and two between training and test. Transcript identities are absent, and 101-nt candidate contexts also overlap between partitions. Evaluation is therefore described as development evidence, without an independently verified transcript-disjoint test claim.

**This is a lightweight research handoff.** Large model weights, transcriptome references and pretraining corpora remain in the original source archive and are omitted from ordinary Git. The [archive inventory](provenance/archive-inventory.json) identifies those members. Transcript identities, dataset-generation code and the local LAMAR implementation remain unavailable. The historical scripts contain a context-length inconsistency and require dependency and path reconstruction before training or inference; they are not a ready-to-run prediction package. Read the [main wiki](wiki/module4.md) and [editorial notes](provenance/editorial-notes.md) before reuse.

## Reproduction and provenance

Run the lightweight archive checks from the repository root:

```bash
python3 scripts/validate_repository.py
```

Publication work preserves supplied scientific labels, thresholds, sequences and partitions. Checks and redraws operate on the stored files; no training, new inference, full sequencing reanalysis or wet-lab validation was performed.

Historical source-document hashes and their Git revision remain traceable in the [import manifest](provenance/import-manifest.json); obsolete draft copies have been removed from the current tree. [ORCA figure sources](figure-sources/) provide the redraw inputs and plotting code. [Validation and audit records](provenance/) record the checks actually executed and their limitations.

The existing repository [MIT license](LICENSE) is retained. LAMAR and externally sourced material retain their respective provenance and applicable rights. See the original [LAMAR paper](https://doi.org/10.1186/s13059-025-03752-x) and [source repository](https://github.com/rnasys/LAMAR).
