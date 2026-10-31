# LitDataMatcher documentation

LitDataMatcher is a reproducible research-software pipeline that turns
scientific literature into reviewable question-data opportunities. It extracts
open questions, represents their data needs, discovers candidate public data,
and produces explainable rankings for expert review.

The primary research path is the local `litdatamatcher-v2` workflow, with
configurable source discovery, pretrained inference, typed question compilation,
dataset assessment, checkpoints, and source-linked review artifacts. A small
deterministic demo and optional web workspace are also included.

The software does not claim
that a ranking proves a dataset can answer a question, establish causality, or
replace domain-expert review.

## Start here

- Start with the [research workflow and evidence contracts](research_workflow.md).
- Configure [source discovery](campaign_20261031/discovery.md) and inspect the
  [local model profile](campaign_20261031/model_source_schema.md).
- Review the [retrieval protocol](campaign_20261031/retrieval_protocol.md) and
  [source-backed cases](campaign_20261031/research_cases.md) for evaluation scope.
- Follow [Installation and CLI](installation.md) for the offline deterministic
  demo.
- Review [Architecture](architecture.md) and [Node contracts](node_contracts.md)
  before extending the pipeline.
- Read [Corpus and data](corpus_and_data.md), [Scientific limitations](limitations.md),
  and [Reproducibility](reproducibility.md) before interpreting an output.
- Use [Citation](citation.md) when referring to this software.
- Explore the optional [web workspace](web-app.md) for visual review.

The source repository, issue tracker, and development history are available at
[GitHub](https://github.com/ericrosenn1/LitDataMatcher).
