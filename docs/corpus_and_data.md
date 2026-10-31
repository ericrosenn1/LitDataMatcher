# Corpus and data requirements

The public package does not bundle production literature corpora, private
datasets, trained model weights, or user-specific caches. The offline demo is a
small synthetic fixture that exercises the software path; it is not scientific
evidence or a substitute for a source-qualified corpus.

For a real run, retain the input literature JSONL, source identifiers, source
URLs or local paths, acquisition time, parser/version information, and any
dataset-catalog inputs. Archive these with the exact Git commit, Python
environment, run directory, SQLite output, metrics, and expert-review labels.

The software can rank metadata compatibility and record caveats. It cannot
silently convert missing metadata into negative evidence or determine that a
candidate dataset is scientifically sufficient without review.
