# Code availability

[historical/](historical/) contains the six supplied Python scripts, shared tokenizer and model configuration. Python scripts and tokenizer files are preserved byte for byte. Host-specific configuration metadata is sanitized as recorded in the [import manifest](../provenance/import-manifest.json).

The executable publication workflow is the lightweight [repository validator](../scripts/validate_repository.py) and the [figure-source workflow](../figure-sources/). These inspect existing files and redraw saved results; they do not train or evaluate neural-network checkpoints.

For historical training-code status, read [historical/README.md](historical/README.md) before running any script.
