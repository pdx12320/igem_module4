# Figure validation — 11 October 2026

All five figures were regenerated from the repository sources using `plot.py`. No model training or inference was run. The export script completed all five figures, including an average-precision recomputation of 0.7222672656649612 versus the saved 0.7222672656649609 (absolute tolerance 1e-12).

## Source and numeric checks

- Source files and the plotting script match the SHA-256 values in every figure manifest.
- Pretraining: 1,000 training-loss records and 100 evaluations; both validation loss and accuracy are retained (1,200 plotted-data rows).
- Stage I: 200 training-loss observations across five recorded runs. The validation table has 100 chronological evaluations plus one separately labelled reevaluation of the restored adapted checkpoint. That last record is not connected as a training-trajectory point.
- Test precision–recall: all 212 saved predictions; 140 positive and 72 negative labels; 212 distinct score thresholds plus one labelled plotting endpoint.
- Stage II: seven draft-reported values (four Hit@K, three Recall@K); no Recall@1 imputation. The figure describes 140 positive-containing test regions and is not presented as an end-to-end system evaluation.

## Format and visual checks

All final PNGs are 1920 px wide with opaque white backgrounds. The exported SVGs parse successfully and contain no raster image elements. Text is converted to vector paths. The PDFs have one page and contain embedded TrueType font resources (`FontFile2`/`CIDFontType2`). Exact dimensions and metadata are recorded in [qa-report.json](qa-report.json).

The final PNGs were viewed, and every PDF was rendered with `pdftoppm` at 960 px width and visually inspected. Labels and legends were legible, without observed clipping or overlap. Model variants have redundant marker and line-style encodings. The source data tables and alt text provide non-image alternatives. This review is not a formal accessibility certification.

All five SVGs loaded in a browser at a natural width of 960 px. Full browser screenshots of pretraining, validation AP, test precision–recall, and reported ranking were visually inspected with no clipping or label overlap. The training-loss SVG loaded successfully; its PNG and PDF were fully reviewed, without a separate SVG screenshot review. Quick Look generated square thumbnails that cropped the right edge of the pretraining and ranking SVGs, so those thumbnails were not used to judge SVG page boundaries. The browser and PDF views preserve the intended rectangular canvas.

No missing values were imputed, no observations were removed for visual reasons, and no smoothing, uncertainty, or significance tests were added. Source-data limitations remain scientific limitations after visual polishing.
