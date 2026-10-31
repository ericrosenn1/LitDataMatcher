# Web app validation

Validated locally on October 31, 2026, using Node.js 24.16.0 and Python 3.13.15. The browser checks below ran against the **production Vite build**, not the development server.

| Check | Result | Scope |
| --- | --- | --- |
| TypeScript | Passed | Strict type checking of application and data engine |
| Production build | Passed | Static assets, locally bundled fonts, no runtime backend required |
| Browser engine tests | 22 passed | Imports, export round trips, formula injection, source failures, cancellation, scoring and provenance |
| Browser workflow checks | 22 passed | Eleven workflows at desktop and mobile viewport sizes |
| Automated accessibility checks | 12 passed | Five pages and the evidence-review dialog at both viewport sizes; no detected WCAG A/AA violations in these states |
| Existing Python core tests | 11 passed | Lightweight `tests/` suite; optional model-dependent root tests excluded |
| Existing Python demo | Passed | Three questions, five illustrative source collections, 15 matches; JSONL, review exports and SQLite integrity checked |
| Dependency audit | No known vulnerabilities reported | `npm audit`, including development dependencies, at the validation time |

## Browser scenarios

The Playwright suite covers the responsive demo; evidence and component inspection; saved assessments and notes after reload; JSON export/import; dataset metadata and links; filtering and empty states; local text analysis; archived library retention after replacing the active workspace; total and partial source failures; and keyboard search dialogs.

The test viewport sizes are 1440 × 1100 and 390 × 844. Automated accessibility checks are useful regression checks, not certification of every interaction, browser, or assistive technology.

A separate, unmocked browser search for `sleep cognition` returned 24 literature records, 30 dataset records and 72 lexical candidate matches. Three literature records lacked abstracts and were explicitly identified as such. These counts describe one observed response, not a fixed corpus size, scientific benchmark, or performance claim.

## Reproduce

```bash
npm --prefix web ci
npm --prefix web run typecheck
npm --prefix web test
cd web
npx playwright install chromium
npm run test:e2e
```

The last command builds the application and starts its local production preview for testing. Browser results are written under `local/app-validation/`, which is ignored by Git. Set `LDM_BASE_URL` to test a separately running instance; set `LDM_CAPTURE_SCREENSHOTS=1` to refresh documentation screenshots deliberately.

## Boundaries

- Demo literature is authored illustration. Catalog collections, their assumed variable availability, and scoring inputs are not measured research results.
- Live matching uses documented lexical overlap. It does not establish feasibility, novelty, causality, access permission, or scientific validity.
- Public source availability and search results can change. Missing fields remain unknown, and failures are shown rather than replaced by demo data.
- Saved work is local to the browser. Export JSON to back it up; there is no cloud synchronization or account service.
- No deployment, release, or default-branch merge is implied by local validation.
