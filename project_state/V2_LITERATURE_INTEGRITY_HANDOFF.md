# Literature integrity tranche

Normalized PubMed, Europe PMC, OpenAlex, and Crossref rows now retain source-specific snapshots/status, explicit version relationships, full-text state, a deterministic dedup group, lifecycle status, and a derivation invalidation key. Retraction, correction, version, duplicate, missing-fulltext, and retrieval/schema uncertainty never become automatic clean evidence or independent support.

Receipt:

```powershell
C:\Codex\LitDataMatcher-v2\runtime-env\Scripts\python.exe scripts\v2\literature_integrity_receipt.py --out C:\Codex\LitDataMatcher-v2\data\phase2\literature_integrity\receipt.json
```
