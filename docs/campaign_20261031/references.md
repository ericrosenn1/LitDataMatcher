# Primary sources and attribution

These are the upstream models, research datasets, and software used by the
campaign. Model revisions below match the recorded local manifests. Dataset
selection and split restrictions are specified in the
[retrieval protocol](retrieval_protocol.md); experiment outcomes are reported
separately in the [model card](../model_card.md). Links checked 31 October 2026.

- **Qwen team — Qwen2.5-Instruct.** The extraction experiments use the
  [7B model card at revision `a09a35458c70`](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/a09a35458c702b33eeacc393d103063234e8bc28/README.md)
  and compare the
  [1.5B model card at revision `989aa7980e4c`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/README.md).
  These are upstream model references, not evidence of biomedical accuracy.

- **Sentence Transformers — all-MiniLM-L6-v2.** The pretrained retrieval encoder
  is recorded in its
  [model card at revision `1110a243fdf4`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/README.md).
  See the [Sentence Transformers documentation](https://www.sbert.net/) and
  Reimers and Gurevych (2019),
  [*Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*](https://aclanthology.org/D19-1410/),
  for the embedding framework. Local fine-tuning is a separate project experiment.

- **Evidence Inference 2.0.** DeYoung et al. (2020),
  [*Evidence Inference 2.0: More Data, Better Models*](https://aclanthology.org/2020.bionlp-1.13/),
  and the [authors' data/code repository](https://github.com/jayded/evidence-inference).
  This project uses selected source-linked questions under its recorded split
  restrictions, not the upstream test partition.

- **PubMedQA.** Jin et al. (2019),
  [*PubMedQA: A Dataset for Biomedical Research Question Answering*](https://aclanthology.org/D19-1259/),
  and the [official repository](https://github.com/pubmedqa/pubmedqa).
  The retrieval experiment uses the PQA-L collection and its recorded grouped
  allocation to recover original paper links; it is not a PubMedQA answer benchmark.

- **NCBI / U.S. National Library of Medicine — PubMed.** The
  [official download guide](https://pubmed.ncbi.nlm.nih.gov/download/) and
  [baseline/daily-update README and terms](https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/README.txt)
  describe the citation/abstract XML, annual baseline, and updates containing new,
  revised, and deleted records. The retrieval benchmark's frozen annual-baseline
  scope is distinct from later update-ledger processing.

- **LM Format Enforcer.** The
  [upstream implementation](https://github.com/noamgat/lm-format-enforcer) and
  [version 0.11.3 package](https://pypi.org/project/lm-format-enforcer/0.11.3/)
  provide the constrained-decoding library used by the optional profiles.
  Format constraints remain separate from LitDataMatcher's source-validation guard.

LitDataMatcher's own software is covered by the [MIT license](../../LICENSE).
That license does not relicense upstream model weights, datasets, or source
papers. Consult the linked model cards, repositories, and NCBI terms for their
respective conditions; metadata availability does not establish unrestricted
reuse of source text. Cite this project separately using [CITATION.cff](../../CITATION.cff).
