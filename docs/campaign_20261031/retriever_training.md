# Local retriever experiment

`scripts/campaign_20261031/train_retriever.py` implements the fixed protocol in
`retrieval_protocol.md`. It does not update the incumbent or the deployed app.
The output is a separately versioned, experimental source-link retriever.

## Execute

Use a new output directory and the existing qualified runtime environment:

```powershell
$env:PYTHONPATH = (Get-Location).Path
& 'C:\Codex\LitDataMatcher-v2\runtime-env\Scripts\python.exe' scripts/campaign_20261031/train_retriever.py --benchmark 'C:\Codex\LitDataMatcher-v2\data\campaign_20261031\retrieval-data-v1' --model 'C:\Codex\LitDataMatcher-v2\data\models\all-MiniLM-L6-v2\1110a243fdf4706b3f48f1d95db1a4f5529b4d41' --out 'C:\Codex\LitDataMatcher-v2\data\campaign_20261031\retriever-training-v1' --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Retriever experiment failed; inspect its TRAINING_STATE.json' }
```

To continue an interrupted identical experiment, repeat the command with
`--resume`. Changing inputs, configuration, implementation, runtime versions,
device, or parent model requires a different run directory. No dependency is
installed and no model or tokenizer is downloaded by this command.

## Input and training contracts

The loader verifies the `BENCHMARK_MANIFEST.json` file receipts, row counts,
canonical source-paper IDs and aliases, exact normalized questions, and
cross-split document text. Every positive paper must occur in the fixed document
pool. It rejects duplicate source papers in a split, cross-split overlaps,
missing provenance, and unexpected split labels. Numeric local IDs are not
silently interpreted as PubMed identifiers. Unicode separators inside source
text are preserved during JSONL parsing.

Training uses the fixed two epochs, batch size 16, learning rate 0.00002,
weight decay 0.01, temperature 0.05, 256-token maximum, seed 1729, and gradient
clipping at 1.0. A shared AutoModel encodes questions and linked source text with
attention-masked mean pooling and unit normalization. Loss is the mean of
question-to-document and document-to-question in-batch cross entropy. Training
uses float32 and deterministic PyTorch algorithms. CUDA availability is checked;
there is no silent device or precision fallback.

Batch permutations are deterministic per epoch. A last batch of one example is
skipped because it supplies no contrastive negative; the receipt reports that
count. Other linked papers in a batch are **unjudged negatives**. A known source
link does not establish that alternatives are irrelevant or that the paper can
answer the question.

## Persistence and evaluation

Each completed epoch is an atomic new directory containing safe-tensor weights,
tokenizer files, `MODEL_MANIFEST.json`, optimizer state, and `EPOCH_RECEIPT.json`.
The manifest records every model/tokenizer file hash, license, immediate parent,
training fingerprint, and a 40-character content-derived local revision. The
revision is explicitly typed `local_training_content_fingerprint`; it is not
represented as an upstream Git commit. Optimizer restoration uses
`torch.load(..., weights_only=True)` after checksum validation.

Completed valid epochs and evaluation receipts are reused. An interrupted epoch
restarts from the last completed epoch with its optimizer state and deterministic
permutation. Staging directories remain available for diagnosis. Corrupt model,
optimizer, or evaluation artifacts fail closed and preserve prior outputs.

Evaluation compares BM25, pretrained MiniLM, their reciprocal-rank fusion, and
each trained epoch on exactly the same development queries and candidate pool.
It records recall@1/10, MRR@10, nDCG@10, source strata, per-query outcomes, paired
development changes, descriptive bootstrap intervals, and index/query timing.
The initial and warm vector-cache origins are recorded explicitly; a resumed
cache hit is not relabeled as a cold encoding.

Experimental selection requires MRR@10 improvement of at least 0.01 with
recall@10 loss no greater than 0.005 relative to the pretrained model. Ties prefer
greater MRR improvement, then recall, then the earlier epoch. Otherwise the
incumbent is retained. This is development selection, not independent testing,
expert validation, or a probability of scientific truth.

Unit tests use tiny local synthetic CPU models. They verify actual weight updates,
safe manifests, checksum rejection, and identical final safe-tensor weights after
optimizer-state resume versus uninterrupted training. Those tests are engineering
evidence only. Actual source-backed experiment results are stored separately
outside Git and must be reported with their observed outcomes, including no gain.
