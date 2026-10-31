# Web app guide

LitDataMatcher’s web app is an evidence-review workspace. It connects candidate research questions with dataset metadata and keeps the reason for a match available for inspection.

## Start locally

Install Node.js 22.12 or newer. From the repository root:

```bash
npm --prefix web ci
npm --prefix web run dev
```

Open the address printed in the terminal. Python is not required to explore the app. The initial demonstration workspace is bundled into the build and works offline.

## Three kinds of workspace

| Mode | Origin | Interpretation |
| --- | --- | --- |
| Demo | Bundled, reproducible example built from the Python processing path | Illustrative literature and curated catalog summaries; not a real study or verified cohort inventory. |
| Live | Metadata returned by Europe PMC and DataCite | Source-linked records retrieved for your query; matching is a browser heuristic. |
| Imported | Python `matches.jsonl` or an app workspace export | Results to inspect in the app; validity depends on the inputs and originating method. |

Provenance is part of the workspace. Loading a demonstration does not turn its records into live search results.

## Search and review

Enter a research topic to search Europe PMC literature and DataCite dataset metadata. A focused query usually produces a more useful shortlist than a broad discipline name. Search results depend on provider coverage, metadata quality, and availability.

Open a candidate pair to inspect its source text, the matched dataset, score components, and caveats. Follow the original record before deciding whether a dataset is suitable. You can save a pair, record a review assessment, and add notes.

Use the dataset view to inspect the available candidate data independently of question rankings. Counts describe the records in the current workspace, not the total scientific literature or all datasets available on a topic.

If a provider fails or returns no results, the app reports that state. An unavailable response is not evidence that relevant literature or data do not exist.

## Analyze pasted text

The app can identify candidate statements in pasted text and compare them with the current workspace’s datasets. Extraction uses explicit textual cues and preserves the supporting text. Broader topic matching is labeled separately from an extracted open question.

Pasted text is processed in the browser. Avoid interpreting rule-based extraction as comprehensive reading, expert synthesis, or a demonstration of novelty.

## Import an existing run

To create a Python run:

```bash
python -m pip install -e .
python -m litdatamatcher.cli run --input papers.jsonl --out run/research
```

Import `run/research/matches.jsonl` into the app. This file contains the question, dataset, evidence, and score components for each ranked pair. The importer also accepts a JSON array or single Python match object, a raw app workspace, or a JSON workspace exported by the web app. Files are limited to 10 MB and 5,000 records per collection. Malformed references and invalid scores are rejected; unsafe URL schemes are removed.

An import is an inspection workflow. The web app does not silently rerun or scientifically validate the originating Python analysis.

## Export and retain work

JSON is the workspace format for reopening records and associated review information. CSV is convenient for a spreadsheet or external review. CSV exports escape cells that could otherwise be interpreted as spreadsheet formulas.

Saved pairs and review notes persist in the current browser through local storage. The research library archives saved matches together with their source records, so they remain available when you switch to another workspace or reload the app. Export from the library page to export the saved collection; exports from other views contain the active workspace.

Browser settings, private browsing, storage eviction, or clearing site data can remove local work. Export important work. There is no cloud account, shared review database, or cross-device synchronization.

## Methodology and limits

The browser app and the Python pipeline are related tools with distinct computational methods:

- **Browser matching** uses lexical overlap between a question or topic and dataset metadata, with displayed score components. It does not run a language model or embedding model.
- **Python processing** performs deterministic question extraction, variable normalization, candidate discovery, evidence synthesis, and heuristic ranking. Imported outputs retain their originating component values.
- **The demonstration** illustrates these mechanics. Synthetic source text, curated collection summaries, and example counts must not be cited as measured scientific evidence.

Scores are ranking aids, not probabilities. Textual relevance does not establish that individual-level variables are present, that study populations are compatible, that files can be accessed, or that a proposed design is valid. Metadata can be incomplete or outdated. Unknown values remain unknown rather than being presented as verified measurements.

The browser score is `0.45 × title coverage + 0.40 × description coverage + 0.15 × query coverage`, expressed on a 0–100 scale. Title and description coverage use the distinct tokens in the question and paper title; query coverage uses the search terms. Tokens are lowercased, short tokens and stop words are removed, and matching uses exact terms without stemming. Shared vocabulary can still be conceptually irrelevant, while synonyms can be missed.

Live search requests at most 24 literature records and 30 dataset records. Matching retains up to three datasets per extracted question and up to 100 pairs overall. These bounds make a query manageable; they are not a systematic or exhaustive literature search. Requests have a 25-second deadline, and a provider failure is reported rather than replaced with demonstration data.

The bundled example contains six authored research briefs, five illustrative source collections, and 30 ranked pairs generated by the Python core. The underlying inputs and original match objects are retained in its provenance. Collection counts are not displayed as verified sample sizes.

Neither workflow establishes causal identification, novelty, statistical power, clinical utility, or scientific acceptance. An expert must assess source records, data dictionaries, license and access conditions, cohort structure, confounding, and endpoints before using any match for research. See also [scientific limitations](limitations.md) and the [model robustness plan](model_robustness_plan.md).

## Privacy and network behavior

The static app has no application server or login service. Imports, pasted text, and review notes are processed locally. Fonts are bundled with the app using Fontsource packages rather than requested from an external font service. A live search sends the query to the public Europe PMC and DataCite APIs; those services apply their own privacy policies and request limits. Following an external source link opens that source’s website.

Live search needs an internet connection and browser access to the providers. Offline operation supports the bundled demo and locally imported workspaces. Software checks use deterministic examples and mocked provider responses where applicable; they do not depend on unrestricted provider uptime.

## Build and validation

```bash
npm --prefix web run typecheck
npm --prefix web test
npm --prefix web run build
npm --prefix web run preview
```

The static output is `web/dist/`. The web CI workflow checks TypeScript, runs tests, builds the app, and stores the build as a downloadable artifact. It does not publish the app or change the existing documentation deployment.

Install the test browser once, then run the browser suite:

```bash
cd web
npx playwright install chromium
npm run test:e2e
```

Playwright exercises desktop and mobile layouts and key user workflows; axe checks selected interface states for accessibility violations. Automated checks cover a bounded set of states, not every browser, assistive technology, or scientific use case. The [validation record](app-validation.md) reports the measured results.

Python tests remain separate:

```bash
python -m pytest
```

Passing these checks validates software behavior within the tested cases. It does not validate the scientific usefulness of a particular ranking.
