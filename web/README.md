# LitDataMatcher web app

A browser research workspace for literature-derived questions and candidate public datasets. Built with React, TypeScript, and Vite; paired with the Python package in the repository root.

For the complete **Scientific workflow**, build this frontend and start the
Python application service from the repository root. The
[integrated application guide](../docs/full_platform.md) provides installation,
model options and restart commands. It serves this build at
`http://127.0.0.1:8765` and connects source extraction, requirements, dataset
inspection, evidence and persistent jobs. The standalone Vite commands below
provide the lightweight Discover/review workspace; the scientific view requires
the Python service.

From this directory, with Node.js 22.12 or newer:

```bash
npm ci
npm run dev
```

Open the URL printed by Vite. The offline demo is available immediately. Live search calls Europe PMC and DataCite from the browser.

Fonts are bundled locally. The research library keeps saved matches and notes across workspace changes and reloads. Exporting from the library exports that saved collection; exporting from other views exports the active workspace.

```bash
npm run typecheck
npm test
npm run build
npm run preview
```

For desktop/mobile browser tests and axe accessibility checks:

```bash
npx playwright install chromium
npm run test:e2e
```

`npm run build` creates `dist/`. Preview serves that build locally; neither command publishes it.

For user workflows, import/export behavior, provenance, and scientific limits, read the [app guide](../docs/web-app.md). The [validation record](../docs/app-validation.md) reports measured checks and their limits. For the Python CLI and project overview, read the [root README](../README.md).
