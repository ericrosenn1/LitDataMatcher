export interface Requirement {
  field: string;
  expected: unknown;
  essential: boolean;
  source_locator?: string;
}

export interface CompiledQuestion {
  compilation_status: string;
  question_purpose?: string;
  matching_requirements: Requirement[];
  ambiguities?: { message?: string; code?: string }[];
  proposed_designs?: {
    design_id: string;
    description: string;
    status: string;
    assumptions_requiring_review: string[];
  }[];
}

export interface ResearchDocument {
  document_id?: string;
  id?: string;
  title?: string;
  text?: string;
  abstract?: string;
  url?: string;
  source_locator?: string;
}

export interface DiscoveredQuestion {
  question_id: string;
  question: string;
  text?: string;
  category?: string;
  source_document_id?: string;
  source_locator?: string;
  unanswered_status?: string;
}

export interface ScientificCandidate {
  dataset_id: string;
  dataset?: { title?: string; url?: string; source?: string };
  assessment: {
    eligibility: string;
    compatibility_status?: string;
    requirements: {
      field: string;
      expected: unknown;
      status: string;
      essential?: boolean;
      observation?: {
        value: unknown;
        source_locator?: string;
        reason?: string;
      };
    }[];
  };
  explanation?: unknown;
}

export interface ResearchResult {
  question?:
    | string
    | {
        question: string;
        question_id?: string;
        source_locator?: string;
        novelty_status?: string;
      };
  compilation?: CompiledQuestion;
  candidates: ScientificCandidate[];
  questions?: {
    questions: DiscoveredQuestion[];
    coverage?: { failed_documents?: number; truncated?: boolean };
  };
  documents?: ResearchDocument[];
  scientific_status?: string;
  status?: string;
  sources?: unknown;
}

export interface ResearchJob {
  id: string;
  status: string;
  progress?: { stage?: string };
  error?: unknown;
  mode?: string;
  result?: ResearchResult;
  retryable?: boolean;
}

export interface EngineHealth {
  mode: string;
  worker_active: boolean;
  model_available: boolean;
  semantic_available: boolean;
  paused?: boolean;
  sources: { literature: string[]; datasets: string[] };
}

export interface ResearchRequest {
  query: string;
  question: string;
  requirements: Requirement[];
  literature_sources: string[];
  dataset_sources: string[];
  limit: number;
  inspection_limit: number;
  extraction_mode: "rules" | "model";
  ranking: "bm25" | "dense" | "hybrid";
  documents?: ResearchDocument[];
}

export async function scientificRequest<T>(
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  const response = await fetch(`/api/${path}`, {
    method: body === undefined ? "GET" : "POST",
    headers:
      body === undefined
        ? { Accept: "application/json" }
        : { Accept: "application/json", "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    credentials: "same-origin",
    signal,
  });
  const type = response.headers.get("content-type") || "";
  if (!type.includes("application/json")) {
    throw new Error(
      "The research engine did not return a usable response. Start the local Python server and reconnect.",
    );
  }
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(
      typeof payload.error === "string"
        ? payload.error
        : `Research engine request failed (${response.status}).`,
    );
  }
  return payload as T;
}

export function scientificExportUrl(
  jobId: string,
  filename: "result.json" | "dossier.md" | "candidates.csv",
) {
  return `/api/jobs/${encodeURIComponent(jobId)}/${filename}`;
}

export function parseRequirementValue(value: string): unknown {
  const clean = value.trim();
  if (!clean) throw new Error("Each requirement needs an expected value.");
  if (clean === "true") return true;
  if (clean === "false") return false;
  if (/^-?(?:0|[1-9]\d*)(?:\.\d+)?$/.test(clean)) {
    const number = Number(clean);
    if (!Number.isFinite(number))
      throw new Error("Requirement values must be finite.");
    return number;
  }
  return clean;
}

export function safeScientificUrl(value?: string) {
  return value && /^https?:\/\//i.test(value) ? value : undefined;
}
