# Publication validation

Checks performed on 2026-10-11 are recorded in [publication-validation.json](publication-validation.json) and the [figure QA report](../figure-sources/qa-report.md).

- `python3 scripts/validate_repository.py` completed with zero errors and documented scientific limitations. It checked imported hashes, archive inventory, input schemas, fixed counts, row alignment, sequence overlap, conflicting labels and saved Stage I metrics. It parsed Python without executing model code.
- All five figures were regenerated into a fresh directory using `python figure-sources/plot.py --output-root <new-directory>`. The five PNGs and all plotted data, summaries, captions and alt-text files matched the publication copies byte for byte: 24 comparisons passed.
- Local Markdown targets were checked, with no missing files or directories. Historical source drafts were excluded because they intentionally preserve the original formatting.
- All five final PNG and PDF figures were visually reviewed. SVG browser rendering, module-icon readability and a rendered README preview were checked. The plotting QA report records export dimensions and font handling.
- Publication text was scanned for host-specific paths and common private-key/token prefixes, with none found. The largest published file is approximately 21.1 MB; model weights and large reference/corpus files are excluded.
- Git whitespace checks on authored documentation and scripts passed. An unrestricted check also reports original CRLF/trailing whitespace in imported source files and generated SVG/CSV formatting. Those bytes are intentionally retained rather than changing immutable input copies for formatting alone. `.gitattributes` prevents newline conversion of imported scientific files.

The recorded checks establish file consistency and reproducibility of presentation outputs. They do not validate the biological claims, resolve the historical tokenization discrepancy, prove independent test performance or provide a ready-to-run trained predictor.
