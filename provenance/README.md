# Provenance and publication scope

This publication copy was organized on 2026-10-11 from `ORCA_EditPredict.zip`, using the supplied module1–3 archives as structural and language references. Module3 provided the primary pattern: a short repository guide, a focused English Wiki page, separate DBTL cycles, selected figures, data, historical code and reproducible figure sources.

## Evidence trail

- [Import manifest](import-manifest.json): SHA-256 hashes before and after publication, member-to-repository paths and transformations.
- [Archive inventory](archive-inventory.json): every supplied archive member, its uncompressed size, archive CRC32 and publication status.
- [Evidence audit](evidence-audit.md): metric definitions, dataset checks, contradictions and limitations.
- [Publication validation](validation.md): executed checks, figure regeneration and display review.
- [Source documents](source-documents/): byte-identical original drafts retained for comparison. Their prose is historical source material, not current instructions or independently verified evidence.
- [Editorial notes](editorial-notes.md): changes in structure and wording, including supported corrections.
- [Terminology](terminology.md): names and units used across the Wiki and DBTL.

## Preservation

Original ZIP files and extracted inputs were not overwritten. Dataset labels, split membership, scientific thresholds, model weights and statistical methods were not changed. No training, inference or wet-lab validation was performed during publication preparation.

Checkpoint metadata containing machine-specific absolute paths was copied with only those paths reduced to basenames. Numeric records were retained. The Stage II metadata with a trailing comma was preserved as `.json.txt`; its original bytes remain available for inspection.

The source archive contains 10.69 GB of uncompressed material. Large GENCODE reference files, pretraining corpora and model weights remain outside Git. Their omission is explicit; the repository is a compact publication and audit package, not a complete model distribution.

## External sources and reuse

LAMAR was developed by Zhou and colleagues. See [the primary article](https://doi.org/10.1186/s13059-025-03752-x) and [upstream repository](https://github.com/rnasys/LAMAR). GENCODE v50 is documented in [the official release history](https://www.gencodegenes.org/human/releases.html). These pages were checked on 2026-10-11.

The original Figure 1 is described in the source draft as adapted from the LAMAR paper. It is not republished here; the Wiki cites the primary article. Source Figures 2–4 are superseded in the publication by independently regenerated plots of the supplied logs and reported values. Their archive entries remain in the inventory.

The repository's existing [MIT license](../LICENSE) is retained. It does not change the licenses of upstream models, reference resources or third-party material. ORCA plotting helper attribution and style provenance are recorded alongside the figure sources.
