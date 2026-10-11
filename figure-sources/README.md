# Reproducible ORCA figures

These figures use the user-supplied ORCA scientific illustrator's real-data workflow, `orca_plot.py` helper, and `orca-style.json` tokens. The helper and default tokens are preserved without changes. The project configuration fixes group colors, markers, line styles, and display size. Figures target a 960 CSS px wiki width, with 1920 px PNGs and vector SVG/PDF exports. SVG glyphs are paths; edit labels in Python. No commercial font files are distributed.

Use an existing environment containing the versions in `requirements.txt`, or an isolated environment you manage. The figures were generated with Python 3.12.14 and installed Futura; the configured font fallback is Arial, then DejaVu Sans. A fallback changes typography and is recorded in each manifest.

From the repository root, regenerate into a **new directory**:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=work/mplconfig \
  python figure-sources/plot.py --output-root work/regenerated-figures
```

The default destination is `wiki/figures/`. Existing figure directories are never overwritten. Input files under `results/` and `provenance/source-documents/` are read-only. The script checks source schemas, preserves all logged observations, and verifies recomputed test average precision against the saved metric. It does not train or rerun a model.

Each figure directory contains PNG, SVG, PDF, a caption, alt text, plotted data, a style snapshot, and a manifest with repository-relative sources and SHA-256 hashes. Summaries use stored precision; captions round only for readability. No observations, error bars, confidence intervals, significance tests, smoothing, or biological replicates are synthesized.

- `pretraining-curves`: all 1,000 logged training losses and 100 validation evaluations, with separate axes for loss and masked-token accuracy.
- `stage1-training-loss`: all available logged training losses for five model variants. The adapted model uses its longer final-run history.
- `stage1-validation-ap`: chronological validation average precision (AP) for the same five variants. The final reevaluation of the restored adapted checkpoint is retained in the data with a separate role, not treated as an epoch-7 training update. Runs differ in length and configuration; these are single runs, not controlled replicate estimates.
- `stage1-test-pr`: an empirical precision–recall staircase from all 212 saved predictions, plus the positive-label prevalence reference. AP uses the stepwise average-precision definition used by the historical evaluation script, not trapezoidal area.
- `stage2-ranking-reported`: discrete Hit@K and Recall@K values transcribed programmatically from the archived draft test table. These are **draft-reported**, not recomputed; no site-level scores or ranking output were supplied. Recall@1 is absent and remains absent. Evaluation conditions on 140 regions with at least one positive site and excludes 72 regions without a positive site; this is not an end-to-end Stage I/II test.

See each caption and the repository's evidence audit for dataset overlap and other interpretation limits. A figure export is not experimental or scientific validation.
