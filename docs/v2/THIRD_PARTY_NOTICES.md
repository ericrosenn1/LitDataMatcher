# Third-party and data notices

The LitDataMatcher source package is MIT licensed. The delivered source archive
and wheel do not contain model weights, raw article corpora, processed study
matrices, credentials, or verbose runtime logs.

The qualified local extractor is Qwen/Qwen2.5-7B-Instruct under Apache-2.0. The
qualified retrieval encoder is sentence-transformers/all-MiniLM-L6-v2 under
Apache-2.0. Model revision and file hashes are recorded in external local model
manifests; users obtain weights under their upstream terms.

The real-data alpha queried Europe PMC/PMC, NCBI GEO, ENA, and UniProtKB. Source
snapshots remain outside the delivery archive. Individual article and dataset
reuse terms continue to apply. The selected UniProtKB records and license page
were captured with release/version provenance under CC BY 4.0. A code license
does not grant redistribution rights for upstream scientific data.

Python dependencies and exact tested versions are listed in
`requirements-v2.lock`. Their upstream licenses apply. The package disables
paid API use in its tested configuration.
