# Independent final functional review

Status: **PASS_WITH_LIMITATIONS**. Six accepted findings (one high, five medium); **0 unresolved** after the recorded revalidation.

Review base `f89d4b4ff64a80e3abca052556476b7bc83ec304`. Root repair source `d84e439bb35362784073fb80392aa990d587d35a`. Executed writer HEAD `f9ecd2a57df10aef8b8a5f9312efb6160461a1b2`; the JSON companion records actual file hashes, Git blob hashes and the complete source fingerprint.

The review inspected the root's analyze integration, source adapters/lifecycle/page provenance and the separate calibration repair. The reviewer authored regression fixtures and receipts; the lead retained core implementation ownership and all pushes. Earlier cache/acquisition work by this reviewer is outside the independence claim.

## Findings and closure

| ID | Severity | Reproduced defect | Revalidation |
|---|---|---|---|
| IR-01 | MEDIUM | Declared Python 3.10 runtime uses a Python 3.11 hashing helper | VERIFIED_FIXED; 1 targeted case |
| IR-02 | MEDIUM | Biological unit metadata crosses observed, unknown and absent states | VERIFIED_FIXED; 10 targeted cases |
| IR-03 | HIGH | Explicit provider retraction signals were discarded before lifecycle selection | VERIFIED_FIXED; 6 targeted cases |
| IR-04 | MEDIUM | PubMed EFetch failure was hidden behind successful summary metadata | VERIFIED_FIXED; 1 targeted case |
| IR-05 | MEDIUM | Provider error objects became successful empty searches | VERIFIED_FIXED; 12 targeted cases |
| IR-06 | MEDIUM | Known canonical ontology identifiers failed their own normalization round trip | VERIFIED_FIXED; 10 targeted cases |

IR-02's first repair exposed two further regressions: source-confirmed absence became unknown, and legacy `known` capabilities were normalized after the biological contract had already been derived. Both red cases remain in the receipt history and regression suite. They are part of the same finding's closure.

## Executed evidence

- Independent adversarial suite: **45/45 passed**.
- Related existing regression suite: **174/174 passed**.
- Retained initial base receipt: 11 failed tests (five initial findings). Expanded committed receipt: 20 passed / 13 failed, including additional source lifecycle and canonical-ID repros. Targeted biological-repair receipt: eight passed / two failed.
- Transport was denied at requests and socket layers for the final tests. Each run records attempted calls, exact command, UTC times, Python version, log/XML hashes and exit status.
- Actual `analyze` orchestration was exercised with explicit synthetic extraction/index stubs: lifecycle/reserved-record filtering, exact parent spans, hard species eligibility before scores, context-only dossiers, artifact hashes and partial inference-failure receipts.
- Independent grouped-isotonic probes verified equal group weighting under group replication and rejected altered probabilities even after recomputing the claimed model hash. Existing repair tests cover split leakage, provenance, invalid labels, synthetic/expert separation and acceptance wiring.

```powershell
& 'C:\Codex\LitDataMatcher-v2\env\Scripts\python.exe' -B -m pytest tests/test_final_independent_review.py -q -p no:cacheprovider
```

Run from the recorded review worktree. The external receipt runner adds the process-wide transport guard and JUnit/log capture.

## Source semantics and limits

`hashlib.file_digest` was introduced in Python 3.11; a streaming SHA256 implementation removes this precise 3.10 incompatibility. [Python documentation](https://docs.python.org/3.11/library/hashlib.html#hashlib.file_digest).

PubMed's direct relation type identifies the related correction/retraction; OpenAlex exposes an explicit retraction Boolean; Crossref exposes retraction updates in `update-to`. These source fields require typed handling before evidence selection. [PubMed DTD](https://dtd.nlm.nih.gov/ncbi/pubmed/doc/out/230101/el-CommentsCorrections.html), [OpenAlex attributes](https://help.openalex.org/data/works/attributes/), [Crossref retraction metadata](https://www.crossref.org/documentation/retrieve-metadata/retraction-watch/).

- Fixtures establish program behavior, not scientific validity, expert calibration, or performance on real cases.
- No sealed holdout, frozen evaluation, new acquisition, source corpus, model files, or GPU inference was executed or read by these tests.
- The 3.10 compatibility probe removes the unavailable helper on Python 3.12.13; the AST grammar check covers 67 modules. No actual Python 3.10 interpreter was installed.
- Ontology coverage remains limited to declared local identifiers and spellings. Unknown inputs remain unresolved.
- Calibration provenance and declared pre-registration are externally supplied. Numerical validation does not authenticate reviewer credentials or turn synthetic fixture labels into expert evidence.
- Review covers the recorded source fingerprints. Later model-case and package changes require the separate follow-up assigned by the lead.
- All functional package module blobs match root d84e439. Only package-version declarations differ: reviewer 0.2.0 versus root 0.3.0 in __init__.py and pyproject.toml; requires-python remains >=3.10.

External receipts: `C:\Codex\LitDataMatcher-v2\data\final_campaign_20260913\independent_review`. The compact handoff includes this report, JSON, regression code, final and red receipts, fingerprints, state and a verified manifest. No scientific corpus or model data are packaged.
