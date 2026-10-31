import { useEffect, useRef, useState } from "react";
import {
  ArrowDownToLine,
  ChevronRight,
  FlaskConical,
  LoaderCircle,
  Plus,
  RefreshCw,
  Search,
  X,
} from "lucide-react";
import {
  parseRequirementValue,
  safeScientificUrl,
  scientificExportUrl,
  scientificRequest,
  scientificSourcePayload,
} from "../lib/scientificApi";
import type {
  CompiledQuestion,
  EngineHealth,
  ResearchDocument,
  ResearchJob,
  ResearchRequest,
  ResearchResult,
  SourceQuestionReference,
} from "../lib/scientificApi";
import "./ScientificWorkflow.css";

const ACTIVE = new Set([
  "QUEUED",
  "RUNNING",
  "PENDING",
  "WAITING",
  "PAUSED",
  "RETRYING",
]);
const text = (value: unknown) =>
  value === undefined || value === null
    ? "Not reported"
    : typeof value === "object"
      ? JSON.stringify(value)
      : String(value);
const label = (value?: string) =>
  (value || "unknown").replaceAll("_", " ").toLowerCase();
const fields = [
  "species",
  "tissue",
  "intervention",
  "comparator",
  "outcome",
  "assay",
  "modality",
  "paired",
  "time",
  "joint_observation",
  "outcome_ascertainment",
  "temporal_order",
  "repeated_measure_unit",
  "validation_partition",
  "study_design",
  "feature",
  "covariate",
  "sex",
  "age",
];
type EditorRequirement = { field: string; value: string; essential: boolean };

export function ScientificWorkflow() {
  const [health, setHealth] = useState<EngineHealth | null>(null);
  const [query, setQuery] = useState("");
  const [question, setQuestion] = useState("");
  const [questionReference, setQuestionReference] =
    useState<SourceQuestionReference | null>(null);
  const [originalQuestion, setOriginalQuestion] = useState("");
  const [requirements, setRequirements] = useState<EditorRequirement[]>([]);
  const [compiled, setCompiled] = useState<CompiledQuestion | null>(null);
  const [documents, setDocuments] = useState<ResearchDocument[]>([]);
  const [literature, setLiterature] = useState<ResearchDocument[]>([]);
  const [paperText, setPaperText] = useState("");
  const [paperTitle, setPaperTitle] = useState("");
  const [paperUrl, setPaperUrl] = useState("");
  const [sources, setSources] = useState<{
    literature: string[];
    datasets: string[];
  }>({ literature: [], datasets: [] });
  const [limit, setLimit] = useState(5);
  const [inspect, setInspect] = useState(3);
  const [extraction, setExtraction] = useState<"rules" | "model">("rules");
  const [ranking, setRanking] = useState<"bm25" | "dense" | "hybrid">("bm25");
  const [jobs, setJobs] = useState<ResearchJob[]>([]);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<ResearchJob | null>(null);
  const [filter, setFilter] = useState("ALL");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [result, setResult] = useState<ResearchResult | null>(null);
  const [refresh, setRefresh] = useState(0);
  const alive = useRef(true);

  async function connect() {
    setBusy("connect");
    setError("");
    try {
      const info = await scientificRequest<EngineHealth>("health");
      if (!alive.current) return;
      setHealth({
        ...info,
        paused: info.paused ?? info.mode.toUpperCase().includes("PAUSED"),
      });
      setSources((previous) => ({
        literature: previous.literature.length
          ? previous.literature
          : info.sources.literature
              .filter((name) => ["europepmc", "pubmed"].includes(name))
              .slice(0, 1),
        datasets: previous.datasets.length
          ? previous.datasets
          : info.sources.datasets
              .filter((name) =>
                ["geo", "pride", "metabolomics_workbench"].includes(name),
              )
              .slice(0, 2),
      }));
      const listed = await scientificRequest<
        { jobs: ResearchJob[] } | ResearchJob[]
      >("jobs");
      if (alive.current) {
        setJobs(Array.isArray(listed) ? listed : listed.jobs);
        setRefresh((previous) => previous + 1);
      }
    } catch (reason) {
      if (alive.current) {
        setHealth(null);
        setError(text(reason instanceof Error ? reason.message : reason));
      }
    } finally {
      if (alive.current) setBusy("");
    }
  }

  useEffect(() => {
    alive.current = true;
    void connect();
    return () => {
      alive.current = false;
    };
  }, []);

  useEffect(() => {
    if (!jobId) {
      setJob(null);
      setResult(null);
      return;
    }
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    setResult(null);
    setFilter("ALL");
    setError("");
    async function poll() {
      try {
        const value = await scientificRequest<ResearchJob>(
          `jobs/${encodeURIComponent(jobId!)}`,
          undefined,
          controller.signal,
        );
        if (controller.signal.aborted) return;
        setJob(value);
        setJobs((previous) => [
          value,
          ...previous.filter((item) => item.id !== value.id),
        ]);
        if (value.result) setResult(value.result);
        if (ACTIVE.has(value.status.toUpperCase()))
          timer = setTimeout(poll, 1800);
        else if (
          !value.result &&
          !["FAILED", "CANCELLED"].includes(value.status.toUpperCase())
        ) {
          const completed = await scientificRequest<ResearchResult>(
            `jobs/${encodeURIComponent(jobId!)}/result.json`,
            undefined,
            controller.signal,
          );
          if (!controller.signal.aborted) setResult(completed);
        }
      } catch (reason) {
        if (!controller.signal.aborted)
          setError(text(reason instanceof Error ? reason.message : reason));
      }
    }
    void poll();
    return () => {
      controller.abort();
      if (timer) clearTimeout(timer);
    };
  }, [jobId, refresh]);

  function editQuestion(value: string) {
    setQuestion(value);
    setCompiled(null);
  }

  function activeRequirements() {
    const seen = new Set<string>();
    return requirements.map((row) => {
      if (seen.has(row.field))
        throw new Error(
          `Use one explicit requirement per field: ${row.field}.`,
        );
      seen.add(row.field);
      return {
        field: row.field,
        expected: parseRequirementValue(row.value),
        essential: row.essential,
      };
    });
  }

  async function preview() {
    setBusy("requirements");
    setError("");
    try {
      setCompiled(
        await scientificRequest<CompiledQuestion>("requirements", {
          question,
          requirements: activeRequirements(),
          ...(questionReference
            ? { question_reference: questionReference }
            : {}),
        }),
      );
    } catch (reason) {
      setError(text(reason instanceof Error ? reason.message : reason));
    } finally {
      setBusy("");
    }
  }

  async function launch(mode: "full" | "questions" = "full") {
    setBusy(mode === "questions" ? "questions" : "launch");
    setError("");
    setNotice("");
    try {
      if (
        !Number.isInteger(limit) ||
        limit < 1 ||
        limit > 20 ||
        !Number.isInteger(inspect) ||
        inspect < 0 ||
        inspect > 10
      ) {
        throw new Error(
          "Use 1–20 results per source and 0–10 datasets to inspect.",
        );
      }
      if (
        (extraction === "model" &&
          !(mode === "full" && questionReference) &&
          !health?.model_available) ||
        (mode === "full" && ranking !== "bm25" && !health?.semantic_available)
      ) {
        throw new Error(
          "The selected model is unavailable. Choose an available extraction or retrieval option.",
        );
      }
      const body: ResearchRequest = {
        workflow_mode: mode,
        query: query || (mode === "full" ? question : ""),
        question: mode === "questions" ? "" : question,
        requirements: mode === "questions" ? [] : activeRequirements(),
        literature_sources:
          mode === "questions" && !query.trim() ? [] : sources.literature,
        dataset_sources: mode === "questions" ? [] : sources.datasets,
        limit,
        inspection_limit: mode === "questions" ? 0 : inspect,
        extraction_mode: extraction,
        ranking,
        ...(mode === "full" && questionReference
          ? { question_reference: questionReference }
          : {}),
        ...scientificSourcePayload(documents),
      };
      const value = await scientificRequest<ResearchJob>("jobs", body);
      setJob(value);
      setJobId(value.id);
      setNotice(
        mode === "questions"
          ? "Literature extraction queued. Select a source question from the results before finding datasets."
          : "Research job queued. Completed stages are retained by the local engine.",
      );
    } catch (reason) {
      setError(text(reason instanceof Error ? reason.message : reason));
    } finally {
      setBusy("");
    }
  }

  async function searchLocal() {
    setBusy("literature");
    setError("");
    setNotice("");
    try {
      const response = await scientificRequest<
        | {
            records?: ResearchDocument[];
            results?: ResearchDocument[];
            documents?: ResearchDocument[];
          }
        | ResearchDocument[]
      >(`literature?q=${encodeURIComponent(query)}`);
      const rows = Array.isArray(response)
        ? response
        : response.records || response.results || response.documents || [];
      setLiterature(rows);
      if (!rows.length)
        setNotice(
          "No records returned from the local literature index. A discovery job can also search the selected public sources.",
        );
    } catch (reason) {
      setError(text(reason instanceof Error ? reason.message : reason));
    } finally {
      setBusy("");
    }
  }

  async function control(action: "pause" | "resume" | "retry") {
    setBusy(action);
    setError("");
    try {
      const value = await scientificRequest<ResearchJob>(
        action === "retry"
          ? `jobs/${encodeURIComponent(jobId!)}/retry`
          : action,
        {},
      );
      if (action === "retry") {
        setJobId(null);
        setTimeout(() => {
          if (alive.current) setJobId(value.id || jobId);
        }, 0);
      } else {
        setHealth((previous) =>
          previous ? { ...previous, paused: action === "pause" } : previous,
        );
        setNotice(
          action === "pause"
            ? "Active and queued execution is paused; completed stages are retained."
            : "Job execution resumed.",
        );
      }
    } catch (reason) {
      setError(text(reason instanceof Error ? reason.message : reason));
    } finally {
      setBusy("");
    }
  }

  function toggleSource(kind: "literature" | "datasets", name: string) {
    setSources((previous) => ({
      ...previous,
      [kind]: previous[kind].includes(name)
        ? previous[kind].filter((item) => item !== name)
        : [...previous[kind], name],
    }));
  }

  const candidates = (result?.candidates || []).filter(
    (candidate) =>
      filter === "ALL" || candidate.assessment.eligibility === filter,
  );
  const categories = [
    ...new Set(
      (result?.candidates || []).map(
        (candidate) => candidate.assessment.eligibility,
      ),
    ),
  ];

  return (
    <section className="scientific-workflow" aria-labelledby="scientific-title">
      <div className="scientific-heading">
        <div>
          <p className="scientific-eyebrow">Python research engine</p>
          <h1 id="scientific-title">Scientific discovery</h1>
          <p>
            Build an explicit question, inspect its requirements, and examine
            dataset compatibility with source evidence.
          </p>
        </div>
        <button
          className="secondary-button"
          onClick={() => void connect()}
          disabled={!!busy}
        >
          <RefreshCw size={16} /> Reconnect
        </button>
      </div>
      <div className="scientific-connection" role="status">
        <span className={`status-dot ${health ? "" : "warning-dot"}`} />
        {health
          ? `Engine connected · ${health.paused ? "queue paused" : health.worker_active ? "worker active" : "worker idle"}`
          : "Research engine is not connected"}
        <span>
          {health?.model_available
            ? "Local model available"
            : "Source-sentence extraction available when connected"}
        </span>
      </div>
      {error && (
        <div className="alert alert-error" role="alert">
          {error}
        </div>
      )}
      {notice && (
        <div className="alert" role="status">
          {notice}
        </div>
      )}

      <div className="scientific-grid">
        <section className="scientific-panel">
          <h2>1. Choose a question</h2>
          <label htmlFor="scientific-query">Literature search</label>
          <div className="scientific-input-action">
            <input
              id="scientific-query"
              placeholder="e.g. insulin glucose uptake human muscle"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
            <button
              className="secondary-button"
              disabled={!health || !!busy || !query.trim()}
              onClick={() => void searchLocal()}
            >
              <Search size={16} /> Search local corpus
            </button>
          </div>
          <button
            className="primary-button"
            disabled={!health || !!busy || (!query.trim() && !documents.length)}
            onClick={() => void launch("questions")}
          >
            {busy === "questions" && (
              <LoaderCircle size={16} className="spin" />
            )}
            Find questions
          </button>
          <p>
            Extract source questions from selected text and literature sources.
            Dataset search starts after you choose a question.
          </p>
          {literature.length > 0 && (
            <ul className="scientific-literature">
              {literature.map((paper, index) => (
                <li key={paper.document_id || paper.id || index}>
                  <div>
                    <strong>
                      {paper.title || paper.document_id || "Literature record"}
                    </strong>
                    <p>{(paper.abstract || paper.text || "").slice(0, 240)}</p>
                  </div>
                  <button
                    className="secondary-button"
                    onClick={() => {
                      setDocuments((previous) => [...previous, paper]);
                      setLiterature((previous) =>
                        previous.filter((_, i) => i !== index),
                      );
                    }}
                  >
                    Use source
                  </button>
                </li>
              ))}
            </ul>
          )}
          <label htmlFor="scientific-question">Research question</label>
          <textarea
            id="scientific-question"
            rows={4}
            placeholder="State the organism, exposure or comparison, and measured outcome where known."
            value={question}
            onChange={(event) => editQuestion(event.target.value)}
          />
          {questionReference && (
            <div className="scientific-coverage" role="status">
              <p>
                {question === originalQuestion
                  ? "Selected source passage: citation and context will be reused from the verified saved job."
                  : "User-modified question: the original source passage remains linked as derivation, not as this wording's evidence."}
              </p>
              <button
                className="secondary-button"
                onClick={() => {
                  setQuestionReference(null);
                  setOriginalQuestion("");
                  setCompiled(null);
                }}
              >
                Use as a new question
              </button>
            </div>
          )}
          <details>
            <summary>Add source text</summary>
            <div className="scientific-source-fields">
              <label htmlFor="scientific-paper-title">Source title</label>
              <input
                id="scientific-paper-title"
                value={paperTitle}
                onChange={(event) => setPaperTitle(event.target.value)}
              />
              <label htmlFor="scientific-paper-url">
                Source URL (optional)
              </label>
              <input
                id="scientific-paper-url"
                value={paperUrl}
                onChange={(event) => setPaperUrl(event.target.value)}
              />
              <label htmlFor="scientific-paper-text">
                Abstract or source passage
              </label>
              <textarea
                id="scientific-paper-text"
                rows={5}
                value={paperText}
                onChange={(event) => setPaperText(event.target.value)}
              />
              <button
                className="secondary-button"
                disabled={!paperText.trim()}
                onClick={() => {
                  if (paperUrl.trim() && !safeScientificUrl(paperUrl.trim())) {
                    setError("Use an HTTP(S) source URL or leave it empty.");
                    return;
                  }
                  setDocuments((previous) => [
                    ...previous,
                    {
                      title: paperTitle,
                      text: paperText,
                      url: safeScientificUrl(paperUrl.trim()),
                    },
                  ]);
                  setPaperText("");
                  setPaperTitle("");
                  setPaperUrl("");
                }}
              >
                <Plus size={15} /> Add source
              </button>
            </div>
          </details>
          {documents.length > 0 && (
            <ul className="scientific-attached">
              {documents.map((paper, index) => (
                <li key={index}>
                  <span>
                    {paper.title || paper.document_id || `Source ${index + 1}`}
                  </span>
                  <button
                    className="icon-button"
                    aria-label={`Remove source ${index + 1}`}
                    onClick={() =>
                      setDocuments((previous) =>
                        previous.filter((_, i) => i !== index),
                      )
                    }
                  >
                    <X size={15} />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="scientific-panel">
          <h2>2. Review requirements</h2>
          <p>
            Explicit requirements override inferred values. Preferred fields
            remain preferences; unresolved information stays unknown.
          </p>
          {requirements.map((row, index) => (
            <div className="scientific-requirement-editor" key={index}>
              <label>
                <span className="sr-only">Requirement field {index + 1}</span>
                <select
                  value={row.field}
                  onChange={(event) => {
                    setRequirements((previous) =>
                      previous.map((item, i) =>
                        i === index
                          ? { ...item, field: event.target.value }
                          : item,
                      ),
                    );
                    setCompiled(null);
                  }}
                >
                  {fields.map((field) => (
                    <option key={field} value={field}>
                      {label(field)}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <span className="sr-only">Expected value {index + 1}</span>
                <input
                  placeholder="Expected value"
                  value={row.value}
                  onChange={(event) => {
                    setRequirements((previous) =>
                      previous.map((item, i) =>
                        i === index
                          ? { ...item, value: event.target.value }
                          : item,
                      ),
                    );
                    setCompiled(null);
                  }}
                />
              </label>
              <label className="scientific-check">
                <input
                  type="checkbox"
                  checked={row.essential}
                  onChange={(event) => {
                    setRequirements((previous) =>
                      previous.map((item, i) =>
                        i === index
                          ? { ...item, essential: event.target.checked }
                          : item,
                      ),
                    );
                    setCompiled(null);
                  }}
                />{" "}
                Required
              </label>
              <button
                className="icon-button"
                aria-label={`Remove requirement ${index + 1}`}
                onClick={() => {
                  setRequirements((previous) =>
                    previous.filter((_, i) => i !== index),
                  );
                  setCompiled(null);
                }}
              >
                <X size={15} />
              </button>
            </div>
          ))}
          <div className="scientific-actions">
            <button
              className="secondary-button"
              onClick={() => {
                setRequirements((previous) => [
                  ...previous,
                  { field: "species", value: "", essential: true },
                ]);
                setCompiled(null);
              }}
            >
              <Plus size={15} /> Add requirement
            </button>
            <button
              className="secondary-button"
              disabled={!health || !question.trim() || !!busy}
              onClick={() => void preview()}
            >
              {busy === "requirements" ? (
                <LoaderCircle size={15} className="spin" />
              ) : (
                <FlaskConical size={15} />
              )}{" "}
              Preview requirements
            </button>
          </div>
          {compiled && (
            <div className="scientific-contract">
              <strong>{label(compiled.compilation_status)}</strong>
              {compiled.ambiguities?.map((ambiguity, index) => (
                <p key={index}>{ambiguity.message || label(ambiguity.code)}</p>
              ))}
              <ul>
                {compiled.matching_requirements.map((row, index) => (
                  <li key={index}>
                    <span>
                      {label(row.field)}: <strong>{text(row.expected)}</strong>
                    </span>
                    <small>{row.essential ? "Required" : "Preferred"}</small>
                  </li>
                ))}
              </ul>
              {compiled.proposed_designs?.length ? (
                <details>
                  <summary>Possible analysis designs to review</summary>
                  {compiled.proposed_designs.map((plan) => (
                    <div key={plan.design_id}>
                      <strong>{plan.description}</strong>
                      <p>{plan.assumptions_requiring_review.join(" ")}</p>
                    </div>
                  ))}
                  <p>
                    These options have not been assessed or applied to the
                    active requirements.
                  </p>
                </details>
              ) : null}
            </div>
          )}
        </section>
      </div>

      <section className="scientific-panel">
        <h2>3. Discover and inspect datasets</h2>
        <div className="scientific-options">
          <label>
            Question extraction
            <select
              value={extraction}
              onChange={(event) =>
                setExtraction(event.target.value as "rules" | "model")
              }
            >
              <option value="rules">Source-sentence rules</option>
              <option value="model" disabled={!health?.model_available}>
                Local model with source guard
              </option>
            </select>
          </label>
          <label>
            Candidate retrieval
            <select
              value={ranking}
              onChange={(event) =>
                setRanking(event.target.value as ResearchRequest["ranking"])
              }
            >
              <option value="bm25">Lexical baseline (BM25)</option>
              <option value="dense" disabled={!health?.semantic_available}>
                Semantic retrieval
              </option>
              <option value="hybrid" disabled={!health?.semantic_available}>
                Hybrid retrieval
              </option>
            </select>
          </label>
          <label>
            Results per source
            <input
              type="number"
              min={1}
              max={20}
              value={limit}
              onChange={(event) => setLimit(Number(event.target.value))}
            />
          </label>
          <label>
            Datasets to inspect
            <input
              type="number"
              min={0}
              max={10}
              value={inspect}
              onChange={(event) => setInspect(Number(event.target.value))}
            />
          </label>
        </div>
        <details>
          <summary>Public sources</summary>
          <div className="scientific-source-groups">
            {(["literature", "datasets"] as const).map((kind) => (
              <fieldset key={kind}>
                <legend>
                  {kind === "datasets"
                    ? "Dataset repositories"
                    : "Literature repositories"}
                </legend>
                {(health?.sources[kind] || []).map((name) => (
                  <label className="scientific-check" key={name}>
                    <input
                      type="checkbox"
                      checked={sources[kind].includes(name)}
                      onChange={() => toggleSource(kind, name)}
                    />{" "}
                    {label(name)}
                  </label>
                ))}
              </fieldset>
            ))}
          </div>
        </details>
        <div className="scientific-actions">
          <button
            className="primary-button"
            disabled={!health || !!busy || !question.trim()}
            onClick={() => void launch()}
          >
            {busy === "launch" ? (
              <LoaderCircle size={16} className="spin" />
            ) : (
              <FlaskConical size={16} />
            )}{" "}
            Find datasets <ChevronRight size={16} />
          </button>
          <span>
            Compatibility uses explicit requirements and observed evidence.
            Ranking is not a probability of scientific success.
          </span>
        </div>
      </section>

      <section className="scientific-panel">
        <div className="scientific-panel-heading">
          <h2>Research jobs</h2>
          <div className="scientific-actions">
            <button
              className="secondary-button"
              disabled={!health || !!busy}
              onClick={() => void control(health?.paused ? "resume" : "pause")}
            >
              {health?.paused ? "Resume jobs" : "Pause jobs"}
            </button>
          </div>
        </div>
        {jobs.length ? (
          <label>
            Saved job
            <select
              aria-label="Saved research job"
              value={jobId || ""}
              onChange={(event) => setJobId(event.target.value || null)}
            >
              <option value="">Select a saved job</option>
              {jobs.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.label || item.id} · {label(item.status)}
                </option>
              ))}
            </select>
          </label>
        ) : (
          <p>No jobs have been created in this engine workspace.</p>
        )}
        {job && (
          <div className="scientific-job" role="status">
            <strong>{label(job.status)}</strong>
            <span>{label(job.progress?.stage)}</span>
            {ACTIVE.has(job.status.toUpperCase()) && (
              <LoaderCircle size={17} className="spin" />
            )}
            {!!job.error && <p>{text(job.error)}</p>}
            {(job.retryable || job.status.toUpperCase() === "FAILED") && (
              <button
                className="secondary-button"
                disabled={!!busy}
                onClick={() => void control("retry")}
              >
                Retry unfinished stages
              </button>
            )}
          </div>
        )}
      </section>

      {result && (
        <section
          className="scientific-panel"
          aria-labelledby="scientific-results-title"
        >
          <div className="scientific-panel-heading">
            <div>
              <h2 id="scientific-results-title">
                {result.workflow_mode === "questions"
                  ? "Source questions"
                  : "Evidence and compatibility"}
              </h2>
              <p>
                {typeof result.question === "string"
                  ? result.question
                  : result.question?.question}
              </p>
              {typeof result.question === "object" &&
                result.question?.selection_reference && (
                  <p>
                    {result.question.derived_from
                      ? "User-modified wording; original source question retained as derivation."
                      : "Source passage and context retained from a verified saved job."}
                    {safeScientificUrl(
                      result.question.derived_from?.source_locator ||
                        result.question.source_locator,
                    ) && (
                      <>
                        {" "}
                        <a
                          href={safeScientificUrl(
                            result.question.derived_from?.source_locator ||
                              result.question.source_locator,
                          )}
                          target="_blank"
                          rel="noreferrer"
                        >
                          View original source
                        </a>
                      </>
                    )}
                  </p>
                )}
            </div>
            <div className="scientific-exports">
              {(["dossier.md", "result.json", "candidates.csv"] as const).map(
                (filename) => (
                  <a
                    key={filename}
                    className="secondary-button"
                    href={scientificExportUrl(jobId!, filename)}
                    download
                  >
                    <ArrowDownToLine size={15} />{" "}
                    {filename === "dossier.md"
                      ? "Dossier"
                      : filename === "result.json"
                        ? "JSON"
                        : "CSV"}
                  </a>
                ),
              )}
            </div>
          </div>
          <p className="scientific-result-status">
            {label(result.scientific_status || result.status)}
          </p>
          {(result.processing_status || result.questions?.coverage) && (
            <div
              className="scientific-coverage"
              role="status"
              aria-label="Research coverage"
            >
              {result.processing_status && (
                <p>
                  {result.processing_status === "COMPLETE"
                    ? "All requested processing stages completed."
                    : result.processing_status === "PARTIAL_RETRYABLE"
                      ? "Some processing stages are incomplete and can be retried."
                      : `Processing status: ${label(result.processing_status)}.`}
                </p>
              )}
              {typeof result.questions?.coverage?.processed_documents ===
                "number" && (
                <p>
                  Documents with source text processed:{" "}
                  {result.questions.coverage.processed_documents}.
                </p>
              )}
              {!!result.questions?.coverage?.reused_documents && (
                <p>
                  Verified source contexts reused without extraction:{" "}
                  {result.questions.coverage.reused_documents}.
                </p>
              )}
              {!!result.questions?.coverage?.metadata_only_documents && (
                <p>
                  Metadata-only documents without source text to extract:{" "}
                  {result.questions.coverage.metadata_only_documents}.
                </p>
              )}
              {!!result.questions?.coverage?.excluded_documents && (
                <p>
                  Documents excluded by source lifecycle or version review:{" "}
                  {result.questions.coverage.excluded_documents}.
                </p>
              )}
              {!!result.questions?.coverage?.failed_documents && (
                <p>
                  Documents that could not be processed:{" "}
                  {result.questions.coverage.failed_documents}. Their failures
                  remain in the dossier.
                </p>
              )}
              {result.questions?.coverage?.truncated && (
                <p>The retained question list reached its configured limit.</p>
              )}
            </div>
          )}
          {!!result.limitations?.length && (
            <details>
              <summary>Coverage and interpretation limits</summary>
              <ul className="scientific-limitations">
                {result.limitations.map((limitation, index) => (
                  <li key={index}>{limitation}</li>
                ))}
              </ul>
            </details>
          )}
          {!!result.questions?.questions.length && (
            <details
              className="scientific-extracted"
              open={result.workflow_mode === "questions" ? true : undefined}
            >
              <summary>
                Source-linked questions ({result.questions.questions.length})
              </summary>
              {result.questions.questions.map((item) => (
                <article key={item.question_id}>
                  <small>
                    {label(item.category)} · {label(item.unanswered_status)}
                  </small>
                  <p>{item.question}</p>
                  <div className="scientific-actions">
                    <button
                      className="secondary-button"
                      onClick={() => {
                        editQuestion(item.question);
                        setQuestionReference({
                          job_id: jobId!,
                          question_id: item.question_id,
                        });
                        setOriginalQuestion(item.question);
                        setNotice(
                          "Source question selected. Review or refine its requirements before launching a new job.",
                        );
                      }}
                    >
                      Use this question
                    </button>
                    {safeScientificUrl(item.source_locator) && (
                      <a
                        href={safeScientificUrl(item.source_locator)}
                        target="_blank"
                        rel="noreferrer"
                      >
                        View source
                      </a>
                    )}
                  </div>
                </article>
              ))}
            </details>
          )}
          {result.addressed_evidence && (
            <details>
              <summary>Bounded addressing evidence</summary>
              <p>{label(result.addressed_evidence.status)}</p>
              <p>
                Checked{" "}
                {result.addressed_evidence.searched_document_ids?.length ?? 0}{" "}
                supplied documents. Related result passages require review; this
                check establishes neither an answer nor global novelty.
              </p>
              {result.addressed_evidence.candidates.map((item, index) => (
                <article key={index}>
                  <p>{item.evidence_span.text}</p>
                  {safeScientificUrl(item.source_locator) ? (
                    <a
                      href={safeScientificUrl(item.source_locator)}
                      target="_blank"
                      rel="noreferrer"
                    >
                      View related evidence
                    </a>
                  ) : (
                    <p>{item.source_locator || item.document_id}</p>
                  )}
                </article>
              ))}
            </details>
          )}
          {result.workflow_mode !== "questions" && (
            <label className="scientific-filter">
              Compatibility filter
              <select
                value={filter}
                onChange={(event) => setFilter(event.target.value)}
              >
                <option value="ALL">All candidates</option>
                {categories.map((category) => (
                  <option key={category} value={category}>
                    {label(category)}
                  </option>
                ))}
              </select>
            </label>
          )}
          {!candidates.length && result.workflow_mode === "questions" && (
            <p>
              Select a source question above, review its requirements, then
              choose Find datasets. No dataset discovery, inspection, or
              qualification was requested.
            </p>
          )}
          {!candidates.length && result.workflow_mode !== "questions" && (
            <p>
              No candidates in this view. Missing evidence, source failures, and
              lack of a compatible dataset are different outcomes; inspect the
              dossier for recorded coverage.
            </p>
          )}
          <div className="scientific-candidates">
            {candidates.map((candidate) => (
              <article key={candidate.dataset_id}>
                <div className="scientific-candidate-heading">
                  <div>
                    <small>
                      {candidate.dataset_id} · {candidate.dataset?.source}
                    </small>
                    <h3>{candidate.dataset?.title || candidate.dataset_id}</h3>
                  </div>
                  <span
                    className={`scientific-eligibility ${candidate.assessment.eligibility === "DIRECT_FIT" ? "fit" : ""}`}
                  >
                    {label(candidate.assessment.eligibility)}
                  </span>
                </div>
                <p>{label(candidate.assessment.compatibility_status)}</p>
                <p>
                  Statistical adequacy:{" "}
                  {label(
                    candidate.assessment.statistical_adequacy || "UNKNOWN",
                  )}
                </p>
                {safeScientificUrl(candidate.dataset?.url) && (
                  <a
                    className="scientific-source-link"
                    href={safeScientificUrl(candidate.dataset?.url)}
                    target="_blank"
                    rel="noreferrer"
                  >
                    Dataset source
                  </a>
                )}
                <details>
                  <summary>Inspect requirement evidence</summary>
                  <div className="scientific-table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Requirement</th>
                          <th>Expected</th>
                          <th>Observed</th>
                          <th>Decision</th>
                          <th>Evidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        {candidate.assessment.requirements.map((row, index) => (
                          <tr key={index}>
                            <td>
                              {label(row.field)}
                              <small>
                                {row.essential === false
                                  ? "Preferred"
                                  : "Required"}
                              </small>
                            </td>
                            <td>{text(row.expected)}</td>
                            <td>
                              {text(row.observation?.value)}
                              {row.observation?.reason && (
                                <small>{row.observation.reason}</small>
                              )}
                            </td>
                            <td>{label(row.status)}</td>
                            <td>
                              {safeScientificUrl(
                                row.observation?.source_locator,
                              ) ? (
                                <a
                                  href={safeScientificUrl(
                                    row.observation?.source_locator,
                                  )}
                                  target="_blank"
                                  rel="noreferrer"
                                >
                                  Source
                                </a>
                              ) : (
                                text(row.observation?.source_locator)
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </details>
              </article>
            ))}
          </div>
        </section>
      )}
    </section>
  );
}

export default ScientificWorkflow;
