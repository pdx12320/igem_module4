# Data used in Module 4

These files are byte-identical copies from the supplied `ORCA_EditPredict.zip`. The publication preparation did not change labels, splits, reference versions or sample inclusion. SHA-256 hashes are recorded in [the import manifest](../provenance/import-manifest.json).

| File group | Contents | Interpretation |
|---|---|---|
| `1024nt_dataset/orca_puf12_{train,valid,test}.csv` | `seq,label` | Transcript-region sequences and binary region labels; 3,835 / 213 / 212 rows |
| `101nt_dataset/orca_puf12_{train,valid,test}.json` | Records with `seq` and per-nucleotide `label` | Full regions, despite the directory name; historical code extracts centered 101-nt contexts when scoring cytidines |
| `PUF12_positive_with_strand.tsv` | Genomic position, strand, ref/alt, gene, 101-nt sequence, editing rate, background and pooled-read FDR | Source positive-site table; coordinate convention is not explicitly documented in the archive |

The JSON records and CSV rows retain their original ordering. The genomic table contains strand-specific reference alleles; do not reinterpret every genomic `G` as an invalid C-to-U event. Transcript-oriented sequence context and genomic coordinates serve different purposes.

## Evaluation boundaries

The region files lack transcript IDs, gene IDs and window coordinates. Consequently, transcript-level separation and the claimed window overlap cannot be reconstructed from these files alone. Exact region sequences and centered sequence contexts overlap across some supplied splits. See the [evidence audit](../provenance/evidence-audit.md) for counts and scope.

Unobserved cytidines and regions without a detected event are operational comparison labels. They are not experimentally established non-editable sites. The publication copy retains this distinction without relabeling any observations.

## Large inputs retained outside Git

GENCODE v50 annotation, the generated transcriptome FASTA and continued-pretraining text datasets remain in the original archive. They are listed with sizes and archive CRC32 values in [archive-inventory.json](../provenance/archive-inventory.json). CRC32 identifies archive members; it is not a cryptographic integrity check.

The source draft names `GRCh38.primary_assembly.genome.fa`, but that genome file is absent from the archive. The dataset-generation scripts and a transcript/window mapping are also absent. This repository does not claim to reconstruct the data preparation from scratch.
