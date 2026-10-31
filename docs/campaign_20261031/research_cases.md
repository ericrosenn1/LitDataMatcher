# Frozen research cases and processed-data inspection

## Case selection

Thirty unique development paper/query pairs were frozen before examining end-to-end model results. Selection uses SHA-256 ordering with seed `research-cases-20261031-v1` and a round-robin allocation across source-origin strata: 15 Evidence Inference 2.0 cases and 15 PubMedQA cases.

`research-cases-v1/cases.jsonl` preserves each original question, source title and abstract, PMID/PMCID/DOI, baseline shard SHA-256 and row locator. `pipeline_inputs.jsonl` supplies the same source documents for the pipeline. `CASES_MANIFEST.json` records selection, input/output hashes, and limitations. These are source-linked development cases for checking execution, abstention, and evidence grounding; no expert correctness labels were fabricated.

Case directory: `C:\Codex\LitDataMatcher-v2\data\campaign_20261031\research-cases-v1`.

```bash
python scripts/campaign_20261031/build_research_cases.py --benchmark PATH_TO_RETRIEVAL_DATA --out NEW_CASE_DIRECTORY
```

## Actual processed tables

Two processed biomedical datasets were retrieved from the original UCI repository with their data dictionaries and repository pages. The following values were computed from the downloaded CSV files:

| Dataset | Rows | Columns | Source-defined features | Missing cells | Nonfinite cells |
| --- | ---: | ---: | ---: | ---: | ---: |
| [Wisconsin Diagnostic Breast Cancer](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) | 569 | 32 | 30 | 0 | 0 |
| [Heart Disease — processed Cleveland table](https://archive.ics.uci.edu/dataset/45/heart+disease) | 303 | 14 | 13 | 6 | 0 |

The repository-reported row and feature counts agree with these measurements. WDBC includes an identifier, 30 nuclear-image features, and a diagnosis target. The Cleveland table includes 13 source-defined features and the `num` target; available fields include age, sex, chest-pain category, blood pressure, cholesterol, ECG and exercise-related measurements. Exact source definitions, units when reported, per-column missing counts, ranges, and categorical counts are retained in the inspection JSON. Unreported units remain null.

Dataset DOIs are [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B) and [10.24432/C52P4X](https://doi.org/10.24432/C52P4X). Both captured repository pages report CC BY 4.0. License evidence is preserved with each page; this is not a separate consent or clinical-use determination.

Output directory: `C:\Codex\LitDataMatcher-v2\data\campaign_20261031\processed-inspection-v1\uci-tabular`. Original CSV bytes, metadata responses, repository HTML, SHA-256 hashes, timestamps and computed summaries are retained. The six downloaded source files total 528,211 bytes, below the 50 MiB ceiling. No rows were imputed, normalized, or used for model fitting. Rows are reported instances; subject independence and statistical power were not inferred.

```bash
python scripts/campaign_20261031/inspect_uci_data.py --out NEW_INSPECTION_DIRECTORY
```

## Retained negative result

The initial GEO series-matrix attempt failed with an incomplete transport response before any body bytes arrived. The failure receipt remains at `processed-inspection-v1/INSPECTION_MANIFEST.json`. A bounded range retry also failed. Several other queried GEO matrix directories were empty. These retrieval failures were not converted into biological absence or successful matrix inspection.

The GEO parser remains available for valid local/remote series matrices through `inspect_processed_data.py`, with checks for table completeness, row width, sample identity, missing and nonfinite values. The successful UCI inspection is a separate artifact and does not overwrite the failed GEO receipt.

## Validation boundary

Fixture tests check reproducible case selection, source-hash validation, raw-data preservation, missing/nonfinite separation, malformed or inaccessible content, and unsupported-download rejection. These checks validate the software and the reported table measurements. They do not supply an end-to-end pipeline result or scientific acceptance for the frozen cases; the lead runs that evaluation separately.
