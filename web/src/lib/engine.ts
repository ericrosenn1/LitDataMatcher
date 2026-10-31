import bundledDemo from "../data/demo.json";
import type { Dataset, Opportunity, Paper, Review, Workspace } from "./types";

export const demoWorkspace = bundledDemo as Workspace;
export const MAX_IMPORT_BYTES = 10 * 1024 * 1024;
const MAX_RECORDS = 5000;
// These are source-collection examples from CuratedBiomedicalCatalogAdapter,
// not accession-level datasets with validated cohort measurements.
const CURATED_DATASET_IDS = new Set([
  "qiita_microbiome_antibiotics_longitudinal",
  "mgnify_human_gut_metagenomics",
  "geo_ibd_transcriptomics",
  "clinicaltrials_ibd_interventions",
  "metabolomics_workbench_gut_inflammation",
]);
const STOP_WORDS = new Set(
  "a an and are as at be been by can could data dataset datasets did do does for from had has have how if in into is it its may more of on or our should study studies than that the their these this to using was we were what whether which will with would".split(
    " ",
  ),
);
const FUTURE =
  /\b(future (?:work|research|studies)|further (?:research|studies)|remain(?:s)? (?:unknown|unclear)|is unclear|not yet (?:understood|examined)|warrants investigation|should be (?:investigated|examined|tested)|open question)\b/i;
const LIMITATION =
  /\b(limitation|limited by|small sample|lack of|could not|did not|missing data|incomplete data|not generalizable)\b/i;

type Row = Record<string, unknown>;
const row = (value: unknown): Row =>
  typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Row)
    : {};
const list = (value: unknown): unknown[] => (Array.isArray(value) ? value : []);
const text = (value: unknown, fallback = ""): string =>
  typeof value === "string" ? value : fallback;
const strings = (value: unknown): string[] =>
  list(value).filter((item): item is string => typeof item === "string");
const score100 = (value: unknown, scale = 1): number => {
  if (typeof value !== "number" || !Number.isFinite(value))
    throw new Error("Scores must be finite numbers.");
  if (value < 0 || value > 100 / scale)
    throw new Error(`Score must be between 0 and ${100 / scale}.`);
  return Math.round(value * scale * 10) / 10;
};
const optionalYear = (value: unknown): number | undefined => {
  const year = Number(value);
  return Number.isInteger(year) && year >= 1000 && year <= 3000
    ? year
    : undefined;
};

/** Only HTTP(S) links become navigable, including imported content. */
export function safeUrl(value: unknown): string {
  if (typeof value !== "string" || !value.trim()) return "";
  try {
    const url = new URL(value);
    return ["https:", "http:"].includes(url.protocol) ? url.href : "";
  } catch {
    return "";
  }
}

function plainText(value: unknown): string {
  return text(value)
    .replace(/<[^>]*>/g, " ")
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\s+/g, " ")
    .trim();
}

function id(prefix: string, value: string): string {
  let hash = 2166136261;
  for (const char of value)
    hash = Math.imul(hash ^ char.charCodeAt(0), 16777619);
  return `${prefix}_${(hash >>> 0).toString(16)}`;
}

function terms(value: string): string[] {
  return [
    ...new Set(
      (value.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []).filter(
        (term) => term.length > 2 && !STOP_WORDS.has(term),
      ),
    ),
  ];
}

function overlap(needle: string[], haystack: string): number {
  if (!needle.length) return 0;
  const available = new Set(terms(haystack));
  return needle.filter((term) => available.has(term)).length / needle.length;
}

function extract(
  paper: Paper,
  fallback: string,
): { question: string; evidence: string; type: string }[] {
  const sentences =
    paper.abstract.match(/[^.!?]+[.!?]+(?:[”'"])?|[^.!?]+$/g) ?? [];
  const found = sentences
    .map((value) => value.trim())
    .filter(Boolean)
    .flatMap((sentence) => {
      const type = FUTURE.test(sentence)
        ? "future_direction"
        : LIMITATION.test(sentence)
          ? "limitation"
          : sentence.endsWith("?")
            ? "research_question"
            : "";
      if (!type) return [];
      // Quoting an extracted sentence avoids inventing a new scientific claim.
      return [{ question: sentence, evidence: sentence, type }];
    });
  if (found.length) return found.slice(0, 3);
  return [
    {
      question: `Which datasets could support further investigation of ${fallback}?`,
      evidence: paper.abstract,
      type: "topic_match",
    },
  ];
}

function rank(
  papers: Paper[],
  datasets: Dataset[],
  query: string,
): Opportunity[] {
  const matches: Opportunity[] = [];
  for (const paper of papers) {
    for (const question of extract(paper, query)) {
      const questionTerms = terms(`${question.question} ${paper.title}`);
      const candidates = datasets
        .map((dataset) => {
          const components = [
            {
              label: "Title overlap",
              score: overlap(questionTerms, dataset.title) * 100,
              weight: 0.45,
            },
            {
              label: "Description overlap",
              score: overlap(questionTerms, dataset.description) * 100,
              weight: 0.4,
            },
            {
              label: "Search-topic overlap",
              score:
                overlap(
                  terms(query),
                  `${dataset.title} ${dataset.description}`,
                ) * 100,
              weight: 0.15,
            },
          ].map((component) => ({
            ...component,
            score: Math.round(component.score * 10) / 10,
          }));
          const score =
            Math.round(
              components.reduce(
                (sum, component) => sum + component.score * component.weight,
                0,
              ) * 10,
            ) / 10;
          const datasetTerms = new Set(
            terms(`${dataset.title} ${dataset.description}`),
          );
          const shared = questionTerms.filter((term) => datasetTerms.has(term));
          return {
            id: id("match", `${paper.id}|${question.question}|${dataset.id}`),
            question: question.question,
            evidence: question.evidence,
            extractionType: question.type,
            paperId: paper.id,
            datasetId: dataset.id,
            score,
            components,
            rationale: [
              shared.length
                ? `Shared metadata terms: ${shared.join(", ")}.`
                : "No shared metadata terms.",
              "Weighted lexical overlap; this is not a probability or an assessment of scientific feasibility.",
            ],
            caveats: [
              "Dataset variables, population, sample size, and reuse permissions require source-level review.",
              ...(question.type === "topic_match"
                ? [
                    "No explicit open-question cue was found in this abstract; this is a topic-level suggestion.",
                  ]
                : []),
            ],
            terms: shared,
          } satisfies Opportunity;
        })
        .filter((match) => match.score > 0)
        .sort((a, b) => b.score - a.score)
        .slice(0, 3);
      matches.push(...candidates);
    }
  }
  return matches.sort((a, b) => b.score - a.score).slice(0, 100);
}

async function getJson(url: URL, signal: AbortSignal): Promise<Row> {
  const response = await fetch(url, {
    signal,
    headers: { Accept: "application/json" },
  });
  if (!response.ok)
    throw new Error(
      `The source returned HTTP ${response.status}. Try again later.`,
    );
  return row(await response.json());
}

/** Retrieve public metadata. Failure of one source is recorded, never replaced by demo data. */
export async function searchWorkspace(
  query: string,
  signal?: AbortSignal,
): Promise<Workspace> {
  query = query.trim();
  if (query.length < 3 || query.length > 300)
    throw new Error("Enter a research topic between 3 and 300 characters.");
  const deadline = AbortSignal.timeout(25000);
  const activeSignal = signal ? AbortSignal.any([signal, deadline]) : deadline;
  const literatureUrl = new URL(
    "https://www.ebi.ac.uk/europepmc/webservices/rest/search",
  );
  literatureUrl.search = new URLSearchParams({
    query,
    format: "json",
    resultType: "core",
    pageSize: "24",
  }).toString();
  const datasetUrl = new URL("https://api.datacite.org/dois");
  datasetUrl.search = new URLSearchParams({
    query,
    "resource-type-id": "dataset",
    "page[size]": "30",
  }).toString();
  const results = await Promise.allSettled([
    getJson(literatureUrl, activeSignal),
    getJson(datasetUrl, activeSignal),
  ]);
  if (activeSignal.aborted)
    throw new Error(
      signal?.aborted
        ? "Search cancelled."
        : "The search timed out. Please try again.",
    );
  const warnings: string[] = [];
  const papers: Paper[] = [];
  const datasets: Dataset[] = [];
  const literature = results[0];
  if (literature.status === "fulfilled") {
    for (const raw of list(row(literature.value.resultList).result)) {
      const item = row(raw);
      const title = plainText(item.title);
      if (!title || !item.id) continue;
      const doi = text(item.doi);
      papers.push({
        id: `epmc_${text(item.source)}_${text(item.id)}`,
        title,
        abstract: plainText(item.abstractText),
        doi: doi || undefined,
        url: doi
          ? safeUrl(`https://doi.org/${doi}`)
          : `https://europepmc.org/article/${encodeURIComponent(text(item.source))}/${encodeURIComponent(text(item.id))}`,
        year: optionalYear(item.pubYear),
        source: "Europe PMC",
      });
    }
  } else
    warnings.push(
      `Europe PMC unavailable: ${literature.reason instanceof Error ? literature.reason.message : "Network error."}`,
    );
  const datasetResult = results[1];
  if (datasetResult.status === "fulfilled") {
    for (const raw of list(datasetResult.value.data)) {
      const outer = row(raw);
      const item = row(outer.attributes);
      const title = plainText(row(list(item.titles)[0]).title);
      const doi = text(item.doi, text(outer.id));
      if (!title || !doi) continue;
      datasets.push({
        id: `datacite_${doi}`,
        title,
        description: list(item.descriptions)
          .map((value) => plainText(row(value).description))
          .filter(Boolean)
          .join(" ")
          .slice(0, 15000),
        url: safeUrl(`https://doi.org/${doi}`),
        source: "DataCite",
        year: optionalYear(item.publicationYear),
        sampleSize: null,
        variables: [],
        population: [],
        access: "Check source",
        license:
          list(item.rightsList)
            .map((value) => text(row(value).rights))
            .filter(Boolean)
            .join("; ") || "Not reported",
      });
    }
  } else
    warnings.push(
      `DataCite unavailable: ${datasetResult.reason instanceof Error ? datasetResult.reason.message : "Network error."}`,
    );
  if (literature.status === "rejected" && datasetResult.status === "rejected")
    throw new Error(
      "Both metadata sources are currently unavailable. Check your connection or try the demo.",
    );
  if (!papers.length)
    warnings.push("No literature records were returned. Try a broader topic.");
  if (!datasets.length)
    warnings.push(
      "No dataset records were returned. No matches can be inferred without dataset metadata.",
    );
  warnings.push(
    "Live suggestions use lexical metadata overlap. Dataset suitability and scientific validity have not been established.",
  );
  const missingAbstracts = papers.filter((paper) => !paper.abstract).length;
  if (missingAbstracts)
    warnings.push(
      `${missingAbstracts} literature record(s) have no abstract; topic suggestions use titles only.`,
    );
  return {
    id: id("live", `${query}|${Date.now()}`),
    name: query,
    mode: "live",
    createdAt: new Date().toISOString(),
    papers,
    datasets,
    opportunities: rank(papers, datasets, query),
    warnings,
    provenance: {
      algorithm: "browser-lexical-v1",
      literatureUrl: literatureUrl.href,
      datasetUrl: datasetUrl.href,
      retrievedAt: new Date().toISOString(),
      limits: { literature: 24, datasets: 30, matches: 100 },
      scoreFormula:
        "0.45 × title overlap + 0.40 × description overlap + 0.15 × search-topic overlap (percent)",
    },
  };
}

/** Analyze local text only. No text is sent to a server by this function. */
export function analyzeText(
  title: string,
  sourceText: string,
  workspace: Workspace,
): Workspace {
  title = title.trim();
  sourceText = sourceText.trim();
  if (!title || title.length > 300)
    throw new Error("Enter a document title of up to 300 characters.");
  if (sourceText.length < 40 || sourceText.length > 200000)
    throw new Error("Paste between 40 and 200,000 characters of source text.");
  const paper: Paper = {
    id: id("local", `${title}|${sourceText}`),
    title,
    abstract: sourceText,
    source: "Local text",
  };
  return {
    id: id("local_workspace", `${workspace.id}|${paper.id}`),
    name: title,
    mode: "import",
    createdAt: new Date().toISOString(),
    papers: [paper],
    datasets: workspace.datasets,
    opportunities: rank([paper], workspace.datasets, title),
    warnings: [
      ...new Set([
        "Local rule-based extraction; scientific suitability requires review.",
        ...workspace.warnings,
      ]),
    ],
    provenance: {
      algorithm: "browser-lexical-v1",
      input: "User-pasted text",
      datasetSourceWorkspace: workspace.id,
      localOnly: true,
    },
  };
}

function requiredString(value: unknown, field: string): string {
  if (typeof value !== "string" || !value.trim())
    throw new Error(`Missing or invalid ${field}.`);
  return value;
}

function uniqueIds(values: { id: string }[], label: string): void {
  if (new Set(values.map((value) => value.id)).size !== values.length)
    throw new Error(
      `Duplicate ${label} identifiers in the imported workspace.`,
    );
}

function reviewMap(value: unknown): Record<string, Review> {
  const result: Record<string, Review> = Object.create(null) as Record<
    string,
    Review
  >;
  for (const [key, raw] of Object.entries(row(value))) {
    const review = row(raw);
    result[key] = {
      saved: review.saved === true,
      status: text(review.status, "unreviewed"),
      note: text(review.note).slice(0, 20000),
    };
  }
  return result;
}

function importAppWorkspace(raw: Row, filename: string): Workspace {
  const source = row(raw.workspace ?? raw);
  if (
    !Array.isArray(source.papers) ||
    !Array.isArray(source.datasets) ||
    !Array.isArray(source.opportunities)
  )
    throw new Error(
      "Expected an exported workspace or Python matches.jsonl file.",
    );
  const allLists = [source.papers, source.datasets, source.opportunities];
  if (allLists.some((items) => list(items).length > MAX_RECORDS))
    throw new Error(
      `Import at most ${MAX_RECORDS.toLocaleString()} records per collection.`,
    );
  const papers = list(source.papers).map((rawPaper) => {
    const paper = row(rawPaper);
    return {
      id: requiredString(paper.id, "paper id"),
      title: requiredString(paper.title, "paper title"),
      abstract: text(paper.abstract),
      source: text(paper.source, "Imported"),
      doi: text(paper.doi) || undefined,
      url: safeUrl(paper.url) || undefined,
      year: optionalYear(paper.year),
      isIllustrative: paper.isIllustrative === true,
    } satisfies Paper;
  });
  const datasets = list(source.datasets).map((rawDataset) => {
    const dataset = row(rawDataset);
    const samples = dataset.sampleSize;
    if (
      samples !== null &&
      samples !== undefined &&
      (typeof samples !== "number" || !Number.isInteger(samples) || samples < 0)
    )
      throw new Error("Sample size must be a non-negative integer or null.");
    return {
      id: requiredString(dataset.id, "dataset id"),
      title: requiredString(dataset.title, "dataset title"),
      description: text(dataset.description),
      url: safeUrl(dataset.url),
      source: text(dataset.source, "Imported"),
      year: optionalYear(dataset.year),
      sampleSize: typeof samples === "number" ? samples : null,
      variables: strings(dataset.variables),
      population: strings(dataset.population),
      access: text(dataset.access, "Not reported"),
      license: text(dataset.license, "Not reported"),
      isIllustrative: dataset.isIllustrative === true,
    } satisfies Dataset;
  });
  uniqueIds(papers, "paper");
  uniqueIds(datasets, "dataset");
  const paperIds = new Set(papers.map((paper) => paper.id));
  const datasetIds = new Set(datasets.map((dataset) => dataset.id));
  const opportunities = list(source.opportunities).map((rawOpportunity) => {
    const item = row(rawOpportunity);
    const paperId = requiredString(item.paperId, "match paperId");
    const datasetId = requiredString(item.datasetId, "match datasetId");
    if (!paperIds.has(paperId) || !datasetIds.has(datasetId))
      throw new Error("A match references a missing paper or dataset.");
    return {
      id: requiredString(item.id, "match id"),
      question: requiredString(item.question, "match question"),
      evidence: text(item.evidence),
      extractionType: text(item.extractionType, "imported"),
      paperId,
      datasetId,
      score: score100(item.score),
      components: list(item.components).map((rawComponent) => {
        const component = row(rawComponent);
        return {
          label: requiredString(component.label, "score component label"),
          score: score100(component.score),
          weight: score100(component.weight, 100) / 100,
        };
      }),
      rationale: strings(item.rationale),
      caveats: strings(item.caveats),
      terms: strings(item.terms),
    } satisfies Opportunity;
  });
  uniqueIds(opportunities, "match");
  return {
    id: text(source.id, id("import", filename)),
    name: text(source.name, filename),
    mode: "import",
    createdAt: text(source.createdAt, new Date().toISOString()),
    papers,
    datasets,
    opportunities,
    warnings: [
      ...new Set([
        ...strings(source.warnings),
        "Imported scores and annotations are retained from their source; import does not validate scientific claims.",
      ]),
    ],
    provenance: row(source.provenance),
    reviews: reviewMap(raw.reviews ?? source.reviews),
  };
}

function importPythonMatches(values: unknown[], filename: string): Workspace {
  if (!values.length || values.length > MAX_RECORDS)
    throw new Error(
      `Import between 1 and ${MAX_RECORDS.toLocaleString()} match records.`,
    );
  const papers = new Map<string, Paper>();
  const datasets = new Map<string, Dataset>();
  const opportunities: Opportunity[] = [];
  for (const raw of values) {
    const match = row(raw);
    const question = row(match.question);
    const dataset = row(match.dataset);
    const scores = row(match.score);
    const evidence = row(list(question.evidence)[0]);
    const questionText = requiredString(
      question.question,
      "Python question.question",
    );
    const datasetId = requiredString(
      dataset.dataset_id,
      "Python dataset.dataset_id",
    );
    const sourceId =
      text(evidence.source_id) ||
      text(list(question.source_ids)[0]) ||
      id("paper", text(evidence.title, questionText));
    const doi = text(evidence.doi);
    const isDemoPaper = doi.startsWith("10.0000/");
    const isCuratedCollection = CURATED_DATASET_IDS.has(datasetId);
    const previousSource = papers.get(sourceId);
    const excerpts = [
      ...new Set(
        [
          ...(previousSource?.abstract ? [previousSource.abstract] : []),
          text(evidence.text),
        ].filter(Boolean),
      ),
    ].join("\n\n");
    papers.set(sourceId, {
      id: sourceId,
      title: text(evidence.title, "Imported source document"),
      abstract: excerpts,
      doi: doi && !isDemoPaper ? doi : undefined,
      url: doi && !isDemoPaper ? safeUrl(`https://doi.org/${doi}`) : undefined,
      source: "Python pipeline evidence",
      isIllustrative: isDemoPaper || previousSource?.isIllustrative === true,
    });
    datasets.set(datasetId, {
      id: datasetId,
      title: requiredString(dataset.title, "Python dataset.title"),
      description: text(dataset.description),
      url: safeUrl(dataset.url),
      source: text(dataset.source, "Imported"),
      sampleSize:
        !isCuratedCollection &&
        typeof dataset.sample_size === "number" &&
        Number.isInteger(dataset.sample_size) &&
        dataset.sample_size > 0
          ? dataset.sample_size
          : null,
      variables: list(dataset.variables)
        .map((value) => text(row(value).name))
        .filter(Boolean),
      population: strings(dataset.populations),
      access: text(dataset.access_type, "Not reported"),
      license: text(dataset.license, "Not reported"),
      isIllustrative: isCuratedCollection,
    });
    const components = [
      ["Significance", "significance", 0.32],
      ["Feasibility heuristic", "feasibility", 0.28],
      ["Variable overlap", "variable_overlap", 0.16],
      ["Lexical relevance", "semantic_relevance", 0.1],
      ["Catalog quality heuristic", "data_quality", 0.08],
      ["Sample-size heuristic", "sample_adequacy", 0.06],
      ["Uncertainty penalty (subtracted)", "uncertainty_penalty", 0.18],
    ] as const;
    opportunities.push({
      id: requiredString(match.match_id, "Python match_id"),
      question: questionText,
      evidence: text(evidence.text),
      extractionType: text(question.extraction_type, "imported"),
      paperId: sourceId,
      datasetId,
      score: score100(scores.combined, 100),
      components: components
        .filter(([, key]) => typeof scores[key] === "number")
        .map(([label, key, weight]) => ({
          label,
          score: score100(scores[key], 100),
          weight,
        })),
      rationale: strings(match.rationale),
      caveats: [
        ...strings(row(row(match.assessments).feasibility).caveats),
        ...(strings(match.missing_variables).length
          ? [
              `Missing required variables: ${strings(match.missing_variables).join(", ")}.`,
            ]
          : []),
        "Imported Python heuristic scores are not calibrated probabilities.",
      ],
      terms: strings(question.domain_terms),
    });
  }
  uniqueIds(opportunities, "match");
  return {
    id: id(
      "python_import",
      `${filename}|${JSON.stringify(values).slice(0, 1000)}`,
    ),
    name: filename.replace(/\.[^.]+$/, ""),
    mode: "import",
    createdAt: new Date().toISOString(),
    papers: [...papers.values()],
    datasets: [...datasets.values()],
    opportunities: opportunities.sort((a, b) => b.score - a.score),
    warnings: [
      "Imported from the Python pipeline. Check the source run for catalog provenance and scientific assumptions.",
      "Curated catalog entries may describe source collections rather than accession-level datasets. Verify sample counts and variable availability before use.",
    ],
    provenance: {
      algorithm: "python-pipeline-import",
      inputFile: filename,
      rawMatches: values,
    },
  };
}

export function importWorkspace(contents: string, filename: string): Workspace {
  if (new TextEncoder().encode(contents).length > MAX_IMPORT_BYTES)
    throw new Error("Choose a JSON or JSONL file smaller than 10 MB.");
  const clean = contents.replace(/^\uFEFF/, "").trim();
  if (!clean) throw new Error("The selected file is empty.");
  let value: unknown;
  try {
    value = JSON.parse(clean);
  } catch {
    const lines = clean.split(/\r?\n/).filter((line) => line.trim());
    try {
      value = lines.map((line) => JSON.parse(line));
    } catch {
      throw new Error(
        "The file is not valid JSON or JSONL. Export matches.jsonl from the Python pipeline or a workspace JSON from this app.",
      );
    }
  }
  if (Array.isArray(value)) return importPythonMatches(value, filename);
  const object = row(value);
  if (object.match_id && object.question && object.dataset)
    return importPythonMatches([object], filename);
  return importAppWorkspace(object, filename);
}

function csvCell(value: unknown): string {
  const raw = value == null ? "" : String(value);
  // Spreadsheet applications interpret these prefixes as formulas, even after CSV quoting.
  const safe =
    /^[\s]*[=+@-]/.test(raw) || /^[\t\r\n]/.test(raw) ? `'${raw}` : raw;
  return `"${safe.replace(/"/g, '""')}"`;
}

export function exportWorkspace(
  workspace: Workspace,
  reviews: Record<string, Review>,
  format: "json" | "csv",
): string {
  if (format === "json")
    return JSON.stringify(
      {
        format: "litdatamatcher-workspace",
        version: 1,
        exportedAt: new Date().toISOString(),
        workspace,
        reviews,
      },
      null,
      2,
    );
  const header = [
    "match_id",
    "question",
    "evidence",
    "paper",
    "paper_url",
    "dataset",
    "dataset_url",
    "source",
    "score",
    "score_method",
    "saved",
    "review_status",
    "review_note",
    "caveats",
  ];
  const rows = workspace.opportunities.map((opportunity) => {
    const paper = workspace.papers.find(
      (value) => value.id === opportunity.paperId,
    );
    const dataset = workspace.datasets.find(
      (value) => value.id === opportunity.datasetId,
    );
    const review = reviews[opportunity.id];
    return [
      opportunity.id,
      opportunity.question,
      opportunity.evidence,
      paper?.title,
      safeUrl(paper?.url),
      dataset?.title,
      safeUrl(dataset?.url),
      dataset?.source,
      opportunity.score,
      text(workspace.provenance?.algorithm, "See source provenance"),
      review?.saved ?? false,
      review?.status ?? "unreviewed",
      review?.note ?? "",
      opportunity.caveats.join(" | "),
    ]
      .map(csvCell)
      .join(",");
  });
  return "\uFEFF" + [header.map(csvCell).join(","), ...rows].join("\r\n");
}
