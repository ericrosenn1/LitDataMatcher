import { useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent } from "react";
import {
  ArrowDownToLine,
  ArrowRight,
  ArrowUpRight,
  Asterisk,
  BookOpen,
  Bookmark,
  Check,
  CheckCheck,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Database,
  FileJson,
  FileText,
  FlaskConical,
  Github,
  LayoutDashboard,
  Link2,
  LoaderCircle,
  Menu,
  Microscope,
  Network,
  Plus,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
import {
  analyzeText,
  demoWorkspace,
  exportWorkspace,
  importWorkspace,
  searchWorkspace,
} from "./lib/engine";
import type { Dataset, Opportunity, Workspace } from "./lib/types";
import { Score, Modal, EvidenceMap } from "./components/ResearchPrimitives";
import ScientificWorkflow from "./components/ScientificWorkflow";

type Page =
  "overview" | "scientific" | "discover" | "datasets" | "library" | "methods";
type Review = { saved: boolean; status: string; note: string };
type Reviews = Record<string, Review>;
const emptyReview: Review = { saved: false, status: "unreviewed", note: "" };
const emptyLibrary: Workspace = {
  id: "saved-library",
  name: "Saved library",
  mode: "import",
  createdAt: new Date().toISOString(),
  papers: [],
  datasets: [],
  opportunities: [],
  warnings: [
    "Saved matches retain their original scores and source notes. They may come from different searches or scoring methods.",
  ],
  provenance: {
    algorithm: "Mixed original scoring methods; inspect each match.",
  },
};
const navItems = [
  { id: "overview" as Page, label: "Overview", icon: LayoutDashboard },
  { id: "scientific" as Page, label: "Scientific workflow", icon: Microscope },
  { id: "discover" as Page, label: "Discover", icon: Search },
  { id: "datasets" as Page, label: "Dataset catalog", icon: Database },
  { id: "library" as Page, label: "Saved library", icon: Bookmark },
];
const labels: Record<Page, string> = {
  overview: "Overview",
  scientific: "Scientific workflow",
  discover: "Discover",
  datasets: "Dataset catalog",
  library: "Saved library",
  methods: "How it works",
};

function readLocal<T>(key: string, fallback: T): T {
  try {
    const value = localStorage.getItem(key);
    return value ? (JSON.parse(value) as T) : fallback;
  } catch {
    return fallback;
  }
}
function safeUrl(url?: string) {
  return url && /^https?:\/\//i.test(url) ? url : undefined;
}
function shortSource(source: string) {
  return source.replace("Metabolomics Workbench", "Metabolomics WB");
}
function formatTerm(term: string) {
  return term.replaceAll("_", " ");
}

export default function App() {
  const [workspace, setWorkspace] = useState<Workspace>(() => {
    const stored = readLocal<Workspace | null>("ldm.workspace.v1", null);
    try {
      return stored
        ? {
            ...importWorkspace(JSON.stringify(stored), "workspace.json"),
            mode: ["demo", "live", "import"].includes(stored.mode)
              ? stored.mode
              : "import",
            warnings: Array.isArray(stored.warnings)
              ? stored.warnings.filter((item) => typeof item === "string")
              : [],
          }
        : demoWorkspace;
    } catch {
      return demoWorkspace;
    }
  });
  const [library, setLibrary] = useState<Workspace>(() => {
    const stored = readLocal<Workspace | null>("ldm.library.v1", null);
    try {
      return stored
        ? importWorkspace(JSON.stringify(stored), "library.json")
        : emptyLibrary;
    } catch {
      return emptyLibrary;
    }
  });
  const [reviews, setReviews] = useState<Reviews>(() => {
    const raw = readLocal<unknown>("ldm.reviews.v1", {});
    if (!raw || typeof raw !== "object" || Array.isArray(raw)) return {};
    return Object.fromEntries(
      Object.entries(raw).filter(
        ([, v]) =>
          v &&
          typeof v === "object" &&
          typeof v.saved === "boolean" &&
          typeof v.status === "string" &&
          typeof v.note === "string",
      ),
    );
  });
  const [page, setPage] = useState<Page>("overview");
  const [isNarrow, setIsNarrow] = useState(
    () => window.matchMedia("(max-width: 720px)").matches,
  );
  const [mobileNav, setMobileNav] = useState(false);
  const [filter, setFilter] = useState("");
  const [source, setSource] = useState("all");
  const [tab, setTab] = useState("all");
  const [minScore, setMinScore] = useState(0);
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [selected, setSelected] = useState<Opportunity | null>(null);
  const [datasetDetail, setDatasetDetail] = useState<Dataset | null>(null);
  const [query, setQuery] = useState("microbiome antibiotic recovery");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [commandOpen, setCommandOpen] = useState(false);
  const [exportOpen, setExportOpen] = useState(false);
  const [analyzeOpen, setAnalyzeOpen] = useState(false);
  const [textTitle, setTextTitle] = useState("");
  const [paperText, setPaperText] = useState("");
  const [storageOkay, setStorageOkay] = useState(true);
  const fileRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    try {
      localStorage.setItem("ldm.workspace.v1", JSON.stringify(workspace));
      localStorage.setItem("ldm.reviews.v1", JSON.stringify(reviews));
      localStorage.setItem("ldm.library.v1", JSON.stringify(library));
      setStorageOkay(true);
    } catch {
      setStorageOkay(false);
    }
  }, [workspace, reviews, library]);
  useEffect(() => {
    const saved = workspace.opportunities.filter(
      (item) =>
        reviews[item.id]?.saved &&
        !library.opportunities.some((old) => old.id === item.id),
    );
    if (!saved.length) return;
    // Snapshot source identifiers within the originating workspace. Imported
    // runs can reuse source IDs for different metadata; old reviews stay intact.
    const archived = saved.map((item) => ({
      ...item,
      paperId: `${item.id}::${item.paperId}`,
      datasetId: `${item.id}::${item.datasetId}`,
    }));
    const archivedPapers = saved.flatMap((item) => {
      const paper = workspace.papers.find((paper) => paper.id === item.paperId);
      return paper ? [{ ...paper, id: `${item.id}::${paper.id}` }] : [];
    });
    const archivedDatasets = saved.flatMap((item) => {
      const dataset = workspace.datasets.find(
        (dataset) => dataset.id === item.datasetId,
      );
      return dataset ? [{ ...dataset, id: `${item.id}::${dataset.id}` }] : [];
    });
    setLibrary((previous) => ({
      ...previous,
      opportunities: [
        ...new Map(
          [...previous.opportunities, ...archived].map((item) => [
            item.id,
            item,
          ]),
        ).values(),
      ],
      papers: [
        ...new Map(
          [...previous.papers, ...archivedPapers].map((item) => [
            item.id,
            item,
          ]),
        ).values(),
      ],
      datasets: [
        ...new Map(
          [...previous.datasets, ...archivedDatasets].map((item) => [
            item.id,
            item,
          ]),
        ).values(),
      ],
      provenance: {
        ...previous.provenance,
        sourceWorkspaces: {
          ...((previous.provenance?.sourceWorkspaces as Record<
            string,
            unknown
          >) ?? {}),
          [workspace.id]: {
            name: workspace.name,
            mode: workspace.mode,
            createdAt: workspace.createdAt,
            warnings: workspace.warnings,
            provenance: workspace.provenance,
          },
        },
      },
    }));
  }, [workspace, reviews, library]);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setCommandOpen((prev) => !prev);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  useEffect(() => {
    if (!notice) return;
    const timer = setTimeout(() => setNotice(""), 4500);
    return () => clearTimeout(timer);
  }, [notice]);
  useEffect(() => () => abortRef.current?.abort(), []);
  useEffect(() => {
    const media = window.matchMedia("(max-width: 720px)");
    const update = () => setIsNarrow(media.matches);
    media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  useEffect(() => {
    if (!mobileNav) return;
    const handleEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMobileNav(false);
    };
    window.addEventListener("keydown", handleEscape);
    return () => window.removeEventListener("keydown", handleEscape);
  }, [mobileNav]);

  const datasets = useMemo(
    () =>
      new Map(
        (page === "library" ? library.datasets : workspace.datasets).map(
          (item) => [item.id, item],
        ),
      ),
    [workspace, library, page],
  );
  const papers = useMemo(
    () =>
      new Map(
        (page === "library" ? library.papers : workspace.papers).map((item) => [
          item.id,
          item,
        ]),
      ),
    [workspace, library, page],
  );
  const sources = [
    ...new Set(
      (page === "library" ? library.datasets : workspace.datasets).map(
        (item) => item.source,
      ),
    ),
  ];
  const savedCount = [
    ...new Map(
      [...library.opportunities, ...workspace.opportunities].map((item) => [
        item.id,
        item,
      ]),
    ).values(),
  ].filter((item) => reviews[item.id]?.saved).length;
  const reviewedCount = workspace.opportunities.filter(
    (item) => reviews[item.id] && reviews[item.id].status !== "unreviewed",
  ).length;
  const questionCount = new Set(
    workspace.opportunities.map((item) => item.question),
  ).size;
  const filtered = (
    page === "library" ? library.opportunities : workspace.opportunities
  )
    .filter((item) => {
      const dataset = datasets.get(item.datasetId);
      return (
        (page !== "library" || reviews[item.id]?.saved) &&
        (tab !== "saved" || reviews[item.id]?.saved) &&
        (tab !== "reviewed" ||
          (reviews[item.id] && reviews[item.id].status !== "unreviewed")) &&
        (source === "all" || dataset?.source === source) &&
        item.score >= minScore &&
        `${item.question} ${dataset?.title} ${item.terms.join(" ")}`
          .toLowerCase()
          .includes(filter.toLowerCase())
      );
    })
    .sort((a, b) => b.score - a.score);
  const visible = page === "overview" ? filtered.slice(0, 4) : filtered;
  const navigate = (next: Page) => {
    setPage(next);
    setMobileNav(false);
    setFilter("");
    setSource("all");
    setTab("all");
    setMinScore(0);
    setError("");
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  const updateReview = (id: string, partial: Partial<Review>) =>
    setReviews((previous) => ({
      ...previous,
      [id]: { ...emptyReview, ...previous[id], ...partial },
    }));
  const toggleSave = (id: string) => {
    updateReview(id, { saved: !reviews[id]?.saved });
    setNotice(
      reviews[id]?.saved
        ? "Removed from your library"
        : "Saved to your library",
    );
  };
  const chooseSource = (value: string) => {
    navigate("discover");
    setSource(value);
  };
  const cancelPendingSearch = () => {
    const controller = abortRef.current;
    abortRef.current = null;
    controller?.abort();
    setBusy(false);
  };

  const runSearch = async (event?: FormEvent) => {
    event?.preventDefault();
    if (query.trim().length < 3) {
      setError("Enter a research topic with at least 3 characters.");
      return;
    }
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setBusy(true);
    setError("");
    setCommandOpen(false);
    setPage("discover");
    setFilter("");
    setSource("all");
    setTab("all");
    setMinScore(0);
    try {
      const result = await searchWorkspace(query.trim(), controller.signal);
      if (abortRef.current !== controller || controller.signal.aborted) return;
      setWorkspace(result);
      setNotice("Search complete. Source metadata is ready to explore.");
    } catch (err) {
      if (abortRef.current !== controller) return;
      setError(
        controller.signal.aborted
          ? "Search cancelled. Your previous workspace is still available."
          : err instanceof Error
            ? err.message
            : "Search failed. Please try again.",
      );
    } finally {
      if (abortRef.current === controller) setBusy(false);
    }
  };

  const importFile = async (file?: File) => {
    if (!file) return;
    try {
      if (file.size > 10 * 1024 * 1024)
        throw new Error(
          "Please choose a JSON or JSONL file smaller than 10 MB.",
        );
      const result = importWorkspace(await file.text(), file.name);
      cancelPendingSearch();
      setWorkspace(result);
      const imported = (result as Workspace & { reviews?: Reviews }).reviews;
      if (imported) setReviews((previous) => ({ ...previous, ...imported }));
      navigate("discover");
      setNotice(`Imported ${result.opportunities.length} opportunities`);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "This file could not be imported.",
      );
    }
    if (fileRef.current) fileRef.current.value = "";
  };
  const download = (format: "json" | "csv") => {
    const target =
      page === "library"
        ? {
            ...library,
            opportunities: library.opportunities.filter(
              (item) => reviews[item.id]?.saved,
            ),
          }
        : workspace;
    const text = exportWorkspace(target, reviews, format);
    const blob = new Blob([text], {
      type:
        format === "csv"
          ? "text/csv;charset=utf-8"
          : "application/json;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `litdatamatcher-${page === "library" ? "library" : workspace.mode}-${new Date().toISOString().slice(0, 10)}.${format}`;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setExportOpen(false);
    setNotice("Workspace exported with source references and review notes");
  };
  const analyze = (event: FormEvent) => {
    event.preventDefault();
    try {
      if (paperText.trim().length < 40)
        throw new Error("Add at least 40 characters of research text.");
      const result = analyzeText(
        textTitle.trim() || "Imported research text",
        paperText,
        workspace,
      );
      cancelPendingSearch();
      setWorkspace(result);
      setAnalyzeOpen(false);
      setPaperText("");
      setTextTitle("");
      navigate("discover");
      setNotice("Text analyzed against the current dataset catalog");
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "The text could not be analyzed.",
      );
    }
  };

  const opportunityCard = (item: Opportunity, index: number) => {
    const dataset = datasets.get(item.datasetId);
    const paper = papers.get(item.paperId);
    const review = reviews[item.id] ?? emptyReview;
    return (
      <article className="opportunity" key={item.id}>
        <div className="opportunity-top">
          <span className="match-number">
            {String(index + 1).padStart(2, "0")}
          </span>
          <span
            className={`tag ${item.extractionType.includes("limitation") ? "tag-amber" : "tag-green"}`}
          >
            {item.extractionType.includes("limitation")
              ? "Research gap"
              : item.extractionType === "topic_match"
                ? "Topic connection"
                : item.extractionType.includes("future")
                  ? "Future direction"
                  : "Research question"}
          </span>
          {review.status !== "unreviewed" && (
            <span className="review-tag">
              <CheckCheck size={12} />
              {formatTerm(review.status)}
            </span>
          )}
          <button
            className={`icon-button bookmark ${review.saved ? "is-saved" : ""}`}
            onClick={() => toggleSave(item.id)}
            aria-label={
              review.saved ? "Remove from library" : "Save to library"
            }
            aria-pressed={review.saved}
          >
            <Bookmark size={18} fill={review.saved ? "currentColor" : "none"} />
          </button>
        </div>
        <div className="opportunity-body">
          <div>
            <button className="question-link" onClick={() => setSelected(item)}>
              {item.question}
            </button>
            <p className="paper-ref">
              <BookOpen size={13} />
              <span>{paper?.title || "Source literature"}</span>
            </p>
            <div className="connection">
              <span className="connection-line" />
              <Database size={15} />
              <button onClick={() => dataset && setDatasetDetail(dataset)}>
                {dataset?.title || "Dataset metadata unavailable"}
              </button>
            </div>
          </div>
          <Score value={item.score} />
        </div>
        <div className="opportunity-bottom">
          <div className="term-list">
            {item.terms.slice(0, 3).map((term) => (
              <span key={term}>{formatTerm(term)}</span>
            ))}
          </div>
          <button className="text-button" onClick={() => setSelected(item)}>
            Review match <ArrowUpRight size={14} />
          </button>
        </div>
      </article>
    );
  };

  return (
    <div className="app-shell">
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>
      <aside
        inert={isNarrow && !mobileNav}
        className={`sidebar ${mobileNav ? "sidebar-open" : ""}`}
      >
        <button
          className="brand"
          onClick={() => navigate("overview")}
          aria-label="LitDataMatcher home"
        >
          <span className="brand-symbol">
            <Asterisk size={29} strokeWidth={1.8} />
          </span>
          <span>
            LitData<span className="brand-light">Matcher</span>
            <small>CONNECTING WHAT COMES NEXT</small>
          </span>
        </button>
        <div className="workspace-selector">
          <span className="workspace-avatar">
            <Microscope size={19} />
          </span>
          <span>
            Research workspace
            <small>
              {workspace.mode === "demo"
                ? "Example project"
                : workspace.mode === "live"
                  ? "Live source search"
                  : "Imported research"}
            </small>
          </span>
          <ChevronDown size={14} />
        </div>
        <p className="nav-label">WORKSPACE</p>
        <nav aria-label="Main navigation">
          {navItems.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              className={`nav-item ${page === id ? "active" : ""}`}
              onClick={() => navigate(id)}
              aria-current={page === id ? "page" : undefined}
            >
              <Icon size={18} />
              <span>{label}</span>
              {id === "library" && savedCount > 0 && (
                <span className="nav-count">{savedCount}</span>
              )}
              {page === id && <span className="active-dot" />}
            </button>
          ))}
        </nav>
        <div className="nav-divider" />
        <p className="nav-label">RESOURCES</p>
        <button
          className={`nav-item ${page === "methods" ? "active" : ""}`}
          onClick={() => navigate("methods")}
        >
          <CircleHelp size={18} />
          How it works
        </button>
        <a
          className="nav-item"
          href="https://github.com/ericrosenn1/LitDataMatcher"
          target="_blank"
          rel="noreferrer"
        >
          <Github size={18} />
          GitHub repository
          <ArrowUpRight size={13} className="nav-external" />
        </a>
        <div className="sidebar-bottom">
          <div className="sidebar-note">
            <span className="tiny-star">✳</span>
            <h3>
              Every discovery starts
              <br />
              with a good question.
            </h3>
            <p>
              Follow the evidence.
              <br />
              Find the next connection.
            </p>
            <button onClick={() => navigate("discover")}>
              Start exploring <ArrowRight size={15} />
            </button>
          </div>
          <div className="open-source">
            <span className="status-dot" />
            Open source. Open possibilities.
            <small>Built for curious minds.</small>
          </div>
        </div>
      </aside>
      {mobileNav && (
        <button
          className="mobile-backdrop"
          aria-label="Close navigation"
          onClick={() => setMobileNav(false)}
        />
      )}
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            <button
              className="icon-button mobile-menu"
              aria-label="Open navigation"
              onClick={() => setMobileNav(true)}
            >
              <Menu size={20} />
            </button>
            <span>Workspace</span>
            <ChevronRight size={13} />
            <strong>{labels[page]}</strong>
          </div>
          <div className="topbar-actions">
            <button
              className="command-button"
              aria-label="Search anything"
              onClick={() => setCommandOpen(true)}
            >
              <Search size={15} />
              <span>Search anything</span>
              <kbd>Ctrl K</kbd>
            </button>
            <span
              className="local-status"
              title="Workspace and notes are saved in this browser only"
            >
              <span
                className={`status-dot ${storageOkay ? "" : "warning-dot"}`}
              />
              {storageOkay ? "Saved locally" : "Not persisted"}
            </span>
            <button
              className="avatar"
              title="Your local research workspace"
              onClick={() => navigate("library")}
            >
              R
            </button>
          </div>
        </header>
        <main id="main-content" tabIndex={-1}>
          {error && (
            <div role="alert" className="alert alert-error">
              <CircleHelp size={18} />
              <span>{error}</span>
              <button
                className="icon-button"
                onClick={() => setError("")}
                aria-label="Dismiss error"
              >
                <X size={16} />
              </button>
            </div>
          )}
          {!storageOkay && (
            <div role="status" className="alert">
              Browser storage is unavailable or full. Export your workspace to
              keep your work.
            </div>
          )}
          {page === "scientific" && <ScientificWorkflow />}
          {page === "overview" && (
            <>
              <section className="page-intro">
                <div>
                  <div className="eyebrow">
                    <span className="status-dot" />
                    YOUR RESEARCH, CONNECTED
                  </div>
                  <h1>A workspace for what’s next.</h1>
                  <p>
                    Turn the gaps in literature into the beginnings of
                    discovery.
                  </p>
                </div>
                <button
                  className="button button-white"
                  onClick={() => setExportOpen(true)}
                >
                  <ArrowDownToLine size={16} />
                  Export workspace
                </button>
              </section>
              <section className="hero">
                <div className="hero-copy">
                  <span className="hero-kicker">
                    <Sparkles size={13} />
                    FROM QUESTION TO POSSIBILITY
                  </span>
                  <h2>
                    Good questions.
                    <br />
                    <em>Better connections.</em>
                  </h2>
                  <p>
                    Explore open research questions. Find the datasets
                    <br className="desktop-break" /> that could help answer
                    them. Keep the evidence in view.
                  </p>
                  <button
                    className="button button-dark"
                    onClick={() => navigate("discover")}
                  >
                    Explore a research question <ArrowUpRight size={17} />
                  </button>
                  <span className="hero-footnote">
                    Literature → Questions → Data → Your next idea
                  </span>
                </div>
                <div className="hero-visual">
                  <span className="hero-visual-label">
                    A WORLD OF CONNECTED EVIDENCE
                  </span>
                  <EvidenceMap
                    workspace={workspace}
                    onSelect={chooseSource}
                    large
                  />
                  <span className="visual-legend">
                    <i />
                    Literature
                    <span />
                    <i />
                    Public datasets
                  </span>
                </div>
              </section>
              <section className="stats-grid" aria-label="Workspace statistics">
                {[
                  {
                    title: "Research questions",
                    count: questionCount,
                    icon: BookOpen,
                    foot: "Open directions in the literature",
                    color: "green",
                  },
                  {
                    title: "Candidate datasets",
                    count: workspace.datasets.length,
                    icon: Database,
                    foot: `Across ${sources.length} source${sources.length === 1 ? "" : "s"}`,
                    color: "blue",
                  },
                  {
                    title: "Question–data matches",
                    count: workspace.opportunities.length,
                    icon: Link2,
                    foot: "Ranked, with evidence attached",
                    color: "amber",
                  },
                  {
                    title: "In your library",
                    count: savedCount,
                    icon: Bookmark,
                    foot: reviewedCount
                      ? `${reviewedCount} matches reviewed`
                      : "Make room for the good ideas",
                    color: "purple",
                  },
                ].map(({ title, count, icon: Icon, foot, color }) => (
                  <div className="stat-card" key={title}>
                    <div className="stat-label">
                      {title}
                      <span className={`stat-icon ${color}`}>
                        <Icon size={16} />
                      </span>
                    </div>
                    <strong>{count.toLocaleString()}</strong>
                    <p>{foot}</p>
                  </div>
                ))}
              </section>
            </>
          )}
          {page === "discover" && (
            <>
              <section className="page-intro">
                <div>
                  <div className="eyebrow">FOLLOW YOUR CURIOSITY</div>
                  <h1>Find your next connection.</h1>
                  <p>
                    Search the literature and public data, or bring your own
                    research.
                  </p>
                </div>
                <button
                  className="button button-white"
                  onClick={() => setExportOpen(true)}
                >
                  <ArrowDownToLine size={16} />
                  Export
                </button>
              </section>
              <section className="discovery-panel">
                <form onSubmit={runSearch}>
                  <label htmlFor="research-query">
                    What are you curious about?
                  </label>
                  <div className="search-input-wrap">
                    <Search size={20} />
                    <input
                      id="research-query"
                      value={query}
                      maxLength={300}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="e.g. microbiome recovery after antibiotics"
                    />
                    <button
                      className="button button-dark"
                      disabled={busy}
                      type="submit"
                    >
                      {busy ? (
                        <LoaderCircle className="spin" size={17} />
                      ) : (
                        <Sparkles size={16} />
                      )}
                      {busy ? "Searching sources…" : "Discover matches"}
                    </button>
                  </div>
                </form>
                <div className="discovery-options">
                  <span>
                    <span className="status-dot" />
                    Live metadata from Europe PMC + DataCite
                  </span>
                  <div>
                    <button
                      className="text-button"
                      onClick={() => {
                        setError("");
                        setAnalyzeOpen(true);
                      }}
                    >
                      <FileText size={14} />
                      Paste research text
                    </button>
                    <button
                      className="text-button"
                      onClick={() => fileRef.current?.click()}
                    >
                      <Upload size={14} />
                      Import results
                    </button>
                  </div>
                </div>
                {busy && (
                  <div role="status" className="search-progress">
                    <span>
                      Retrieving abstracts and dataset metadata. Your current
                      workspace stays available.
                    </span>
                    <button
                      className="text-button"
                      onClick={() => abortRef.current?.abort()}
                    >
                      Cancel
                    </button>
                  </div>
                )}
                <div className="query-suggestions">
                  <span>Try a direction</span>
                  {[
                    "gut microbiome",
                    "single cell atlas",
                    "sleep cognition",
                  ].map((text) => (
                    <button key={text} onClick={() => setQuery(text)}>
                      {text}
                      <ArrowUpRight size={11} />
                    </button>
                  ))}
                </div>
              </section>
            </>
          )}
          {page === "library" && (
            <section className="page-intro">
              <div>
                <div className="eyebrow">KEEP THE THREAD</div>
                <h1>Your saved library.</h1>
                <p>
                  A home for promising connections, considered questions, and
                  next steps.
                </p>
              </div>
              <button
                className="button button-white"
                onClick={() => setExportOpen(true)}
              >
                <ArrowDownToLine size={16} />
                Export workspace
              </button>
            </section>
          )}
          {["overview", "discover", "library"].includes(page) && (
            <>
              <div className="section-heading">
                <div>
                  <h2>
                    {page === "overview"
                      ? "Promising connections"
                      : page === "library"
                        ? "Your shortlist"
                        : "Research opportunities"}
                    <span className="count-pill">
                      {page === "library"
                        ? savedCount
                        : workspace.opportunities.length}
                    </span>
                  </h2>
                  <p>
                    {page === "overview"
                      ? "A closer look at what the evidence could make possible."
                      : `Exploring: ${workspace.name}`}
                  </p>
                </div>
                {page === "overview" && (
                  <button
                    className="text-button"
                    onClick={() => navigate("discover")}
                  >
                    View all matches <ArrowRight size={15} />
                  </button>
                )}
              </div>
              <div
                className={`research-grid ${page === "library" ? "full-width" : ""}`}
              >
                <div className="matches-column">
                  <div className="match-controls">
                    <div
                      className="tabs"
                      role="tablist"
                      aria-label="Filter opportunities"
                    >
                      {[
                        {
                          id: "all",
                          name:
                            page === "library"
                              ? "Saved matches"
                              : "All matches",
                        },
                        ...(page === "library"
                          ? []
                          : [{ id: "saved", name: "Saved" }]),
                        { id: "reviewed", name: "Reviewed" },
                      ].map((item) => (
                        <button
                          role="tab"
                          aria-selected={tab === item.id}
                          className={tab === item.id ? "selected" : ""}
                          key={item.id}
                          onClick={() => setTab(item.id)}
                        >
                          {item.name}
                        </button>
                      ))}
                    </div>
                    <button
                      className={`icon-button ${filtersOpen ? "filter-active" : ""}`}
                      title="Filter matches"
                      aria-label="Filter matches"
                      aria-expanded={filtersOpen}
                      onClick={() => setFiltersOpen(!filtersOpen)}
                    >
                      <SlidersHorizontal size={17} />
                    </button>
                  </div>
                  {filtersOpen && (
                    <div className="filters">
                      <label>
                        <Search size={14} />
                        <input
                          aria-label="Filter by keyword"
                          value={filter}
                          onChange={(e) => setFilter(e.target.value)}
                          placeholder="Filter by keyword"
                        />
                      </label>
                      <label>
                        Source
                        <select
                          value={source}
                          onChange={(e) => setSource(e.target.value)}
                        >
                          <option value="all">All sources</option>
                          {sources.map((value) => (
                            <option key={value}>{value}</option>
                          ))}
                        </select>
                      </label>
                      <label>
                        Minimum score
                        <select
                          value={minScore}
                          onChange={(e) => setMinScore(Number(e.target.value))}
                        >
                          <option value={0}>Any score</option>
                          <option value={25}>25+</option>
                          <option value={50}>50+</option>
                          <option value={75}>75+</option>
                        </select>
                      </label>
                    </div>
                  )}
                  {source !== "all" && !filtersOpen && (
                    <div className="source-filter">
                      Showing {source}
                      <button
                        onClick={() => setSource("all")}
                        aria-label="Clear source filter"
                      >
                        <X size={13} />
                      </button>
                    </div>
                  )}
                  {visible.length ? (
                    <div className="opportunity-list">
                      {visible.map(opportunityCard)}
                    </div>
                  ) : (
                    <div className="empty-state">
                      <span>
                        <Bookmark size={27} />
                      </span>
                      <h3>
                        {page === "library" || tab === "saved"
                          ? "Give a good idea a place to land."
                          : "No connections here yet."}
                      </h3>
                      <p>
                        {page === "library" || tab === "saved"
                          ? "Save a match using the bookmark icon. Your evidence and notes will be right here."
                          : source !== "all" || minScore > 0 || filter
                            ? "Try another keyword or clear your filters to see more matches."
                            : "Search a research topic or import pipeline results to start exploring."}
                      </p>
                      <button
                        className="button button-white"
                        onClick={() => {
                          navigate("discover");
                        }}
                      >
                        Explore matches <ArrowRight size={15} />
                      </button>
                    </div>
                  )}
                  {page === "overview" && filtered.length > 4 && (
                    <button
                      className="view-more"
                      onClick={() => navigate("discover")}
                    >
                      Explore all {filtered.length} connections{" "}
                      <ArrowRight size={15} />
                    </button>
                  )}
                  <div className="ranking-note">
                    <ShieldCheck size={15} />
                    <span>
                      Scores help prioritize review. They are not evidence of
                      scientific feasibility.
                    </span>
                    <button onClick={() => navigate("methods")}>
                      Our approach <ArrowUpRight size={12} />
                    </button>
                  </div>
                </div>
                {page !== "library" && (
                  <aside className="insight-column">
                    <section className="insight-card">
                      <div className="insight-title">
                        <h3>The connection map</h3>
                        <Network size={17} />
                      </div>
                      <p>Different sources. Shared possibilities.</p>
                      <EvidenceMap
                        workspace={workspace}
                        onSelect={chooseSource}
                      />
                      <div className="map-caption">
                        <span className="status-dot" />
                        Click a source to explore its matches
                      </div>
                    </section>
                    <section className="insight-card source-card">
                      <div className="insight-title">
                        <h3>Behind the evidence</h3>
                        <Database size={16} />
                      </div>
                      <p>
                        {workspace.mode === "demo"
                          ? "Curated source collections in this example."
                          : "The sources in your current workspace."}
                      </p>
                      {sources.slice(0, 5).map((value, index) => (
                        <button
                          className="source-row"
                          key={value}
                          onClick={() => chooseSource(value)}
                        >
                          <span
                            className={`source-icon source-color-${index % 3}`}
                          >
                            <Database size={14} />
                          </span>
                          <span>
                            {shortSource(value)}
                            <small>
                              {
                                workspace.datasets.filter(
                                  (item) => item.source === value,
                                ).length
                              }{" "}
                              dataset
                              {workspace.datasets.filter(
                                (item) => item.source === value,
                              ).length === 1
                                ? ""
                                : "s"}
                            </small>
                          </span>
                          <ArrowUpRight size={13} />
                        </button>
                      ))}
                    </section>
                    <div className="example-note">
                      <FlaskConical size={17} />
                      <div>
                        <strong>
                          {workspace.mode === "demo"
                            ? "An example, ready to explore"
                            : workspace.mode === "live"
                              ? "Live metadata, human judgment"
                              : "Your research, in one place"}
                        </strong>
                        <p>
                          {workspace.mode === "demo"
                            ? "Illustrative literature and catalog summaries. Try a live search to follow your own question."
                            : "Inspect the original sources before making scientific or reuse decisions."}
                        </p>
                        {workspace.mode === "demo" && (
                          <button
                            className="text-button"
                            onClick={() => navigate("discover")}
                          >
                            Make it your own <ArrowRight size={13} />
                          </button>
                        )}
                      </div>
                    </div>
                  </aside>
                )}
              </div>
            </>
          )}
          {page === "datasets" && (
            <>
              <section className="page-intro">
                <div>
                  <div className="eyebrow">THE OTHER HALF OF THE QUESTION</div>
                  <h1>Data with possibilities.</h1>
                  <p>
                    Explore the dataset metadata behind your research
                    connections.
                  </p>
                </div>
                <button
                  className="button button-white"
                  onClick={() => navigate("discover")}
                >
                  <Plus size={16} />
                  Find datasets
                </button>
              </section>
              <div className="catalog-toolbar">
                <label>
                  <Search size={17} />
                  <input
                    aria-label="Search dataset catalog"
                    placeholder="Search datasets, sources, or variables…"
                    value={filter}
                    onChange={(e) => setFilter(e.target.value)}
                  />
                </label>
                <span>{workspace.datasets.length} candidate datasets</span>
              </div>
              <div className="dataset-grid">
                {workspace.datasets
                  .filter((item) =>
                    `${item.title} ${item.source} ${item.variables.join(" ")}`
                      .toLowerCase()
                      .includes(filter.toLowerCase()),
                  )
                  .map((item, index) => (
                    <article className="dataset-card" key={item.id}>
                      <div className="dataset-card-top">
                        <span
                          className={`stat-icon ${index % 2 ? "blue" : "green"}`}
                        >
                          <Database size={19} />
                        </span>
                        <span className="tag">{item.source}</span>
                      </div>
                      <h2>
                        <button onClick={() => setDatasetDetail(item)}>
                          {item.title}
                        </button>
                      </h2>
                      <p>
                        {item.description ||
                          "No description supplied by the source."}
                      </p>
                      <div className="term-list">
                        {item.variables.slice(0, 3).map((term) => (
                          <span key={term}>{formatTerm(term)}</span>
                        ))}
                      </div>
                      <div className="dataset-meta">
                        <span>
                          {item.sampleSize === null
                            ? "Sample size not reported"
                            : `${item.sampleSize.toLocaleString()} ${workspace.mode === "demo" ? "illustrative samples" : "reported samples"}`}
                        </span>
                        <button
                          className="text-button"
                          onClick={() => setDatasetDetail(item)}
                        >
                          View details <ArrowUpRight size={14} />
                        </button>
                      </div>
                    </article>
                  ))}
              </div>
              {!workspace.datasets.length && (
                <div className="empty-state">
                  <Database size={30} />
                  <h3>A new catalog starts with a question.</h3>
                  <p>Search for a topic to find publicly indexed datasets.</p>
                  <button
                    className="button button-dark"
                    onClick={() => navigate("discover")}
                  >
                    Discover datasets
                  </button>
                </div>
              )}
            </>
          )}
          {page === "methods" && (
            <>
              <section className="page-intro">
                <div>
                  <div className="eyebrow">
                    OPEN METHODS. TRACEABLE EVIDENCE.
                  </div>
                  <h1>A connection you can inspect.</h1>
                  <p>
                    Useful research tools should show their working. Here is
                    ours.
                  </p>
                </div>
                <a
                  className="button button-white"
                  href="https://github.com/ericrosenn1/LitDataMatcher"
                  target="_blank"
                  rel="noreferrer"
                >
                  <Github size={16} />
                  Explore the code
                </a>
              </section>
              <section className="methods-hero">
                <Asterisk size={36} />
                <h2>Built to support your judgment.</h2>
                <p>
                  LitDataMatcher connects language in research to available
                  dataset metadata. A match is an invitation to investigate: it
                  does not establish that a dataset can answer a question.
                </p>
              </section>
              <div className="method-steps">
                {[
                  {
                    icon: BookOpen,
                    title: "Start with the literature",
                    text: "Search abstracts through Europe PMC, paste your own research text, or import JSONL results from the Python pipeline.",
                  },
                  {
                    icon: Search,
                    title: "Identify a direction",
                    text: "The browser looks for explicit future-work and limitation phrases. When none exist, topic connections stay labeled as topic connections.",
                  },
                  {
                    icon: Database,
                    title: "Connect to public data",
                    text: "Live search retrieves dataset metadata from DataCite. The example workspace uses the Python pipeline’s illustrative catalog.",
                  },
                  {
                    icon: ShieldCheck,
                    title: "Review the evidence",
                    text: "Inspect source text, component scores, and missing information. Save notes and export your review with provenance intact.",
                  },
                ].map(({ icon: Icon, title, text }, i) => (
                  <section className="method-step" key={title}>
                    <span className="step-number">0{i + 1}</span>
                    <Icon size={23} />
                    <h3>{title}</h3>
                    <p>{text}</p>
                  </section>
                ))}
              </div>
              <div className="method-detail-grid">
                <section className="panel">
                  <h2>Two transparent scoring paths</h2>
                  <p>
                    <strong>Live search and pasted text</strong> use
                    deterministic lexical matching. Shared topic terms and
                    available metadata provide a ranking signal, not a
                    probability or an assessment of novelty.
                  </p>
                  <p>
                    <strong>Python pipeline results</strong> preserve the
                    original ranking and its components: variable coverage,
                    semantic or lexical similarity, population fit, data
                    quality, and feasibility. Exact weights and component values
                    appear in the match review.
                  </p>
                  <p>
                    The browser does not call an LLM, train a model, or claim to
                    understand causality. See the repository for the core
                    pipeline’s implementation and limitations.
                  </p>
                  <button
                    className="text-button"
                    onClick={() =>
                      workspace.opportunities[0] &&
                      setSelected(workspace.opportunities[0])
                    }
                  >
                    Inspect an example score <ArrowUpRight size={14} />
                  </button>
                </section>
                <section className="panel">
                  <h2>Your work, your browser</h2>
                  <p>
                    Workspace data, saved matches, and notes stay in local
                    browser storage. Export JSON to back them up or move them to
                    another device. Clearing browser storage removes local work.
                  </p>
                  <p>
                    A live search sends your search terms directly to Europe PMC
                    and DataCite. Pasted text and imported results are processed
                    locally. Provider availability and metadata completeness can
                    vary.
                  </p>
                  <div className="privacy-points">
                    <span>
                      <Check size={15} />
                      No account required
                    </span>
                    <span>
                      <Check size={15} />
                      No API key required
                    </span>
                    <span>
                      <Check size={15} />
                      No analytics or tracking
                    </span>
                  </div>
                </section>
              </div>
            </>
          )}
          {workspace.warnings.length > 0 && (
            <details className="provenance-note">
              <summary>
                <FileText size={14} />
                Source notes & limitations{" "}
                <span>{workspace.warnings.length}</span>
                <ChevronDown size={14} />
              </summary>
              <ul>
                {workspace.warnings.map((warning, index) => (
                  <li key={index}>{warning}</li>
                ))}
              </ul>
            </details>
          )}
          <footer className="footer">
            <span>
              <Asterisk size={16} />
              Made for questions worth asking.
            </span>
            <span>
              LitDataMatcher <span className="footer-dot">·</span> Open research
              software
              <a
                href="https://github.com/ericrosenn1/LitDataMatcher"
                target="_blank"
                rel="noreferrer"
                aria-label="LitDataMatcher on GitHub"
              >
                <Github size={15} />
              </a>
            </span>
          </footer>
        </main>
      </div>
      <input
        ref={fileRef}
        hidden
        type="file"
        accept=".json,.jsonl,application/json"
        aria-label="Import workspace or pipeline results"
        onChange={(e) => void importFile(e.target.files?.[0])}
      />
      {notice && (
        <div className="toast" role="status">
          <span>
            <Check size={15} />
          </span>
          {notice}
        </div>
      )}
      {selected &&
        (() => {
          const dataset = datasets.get(selected.datasetId);
          const paper = papers.get(selected.paperId);
          const review = reviews[selected.id] ?? emptyReview;
          return (
            <Modal
              title="Connection review"
              onClose={() => setSelected(null)}
              wide
            >
              <div className="detail-content">
                <div className="detail-heading">
                  <span className="tag tag-green">
                    {formatTerm(selected.extractionType)}
                  </span>
                  <Score value={selected.score} small />
                </div>
                <h2>{selected.question}</h2>
                <div className="detail-block">
                  <h3>
                    <BookOpen size={16} />
                    The source evidence
                  </h3>
                  <blockquote>
                    {selected.evidence ||
                      "No evidence text was supplied with this record."}
                  </blockquote>
                  <p className="source-citation">
                    {paper?.title}
                    {paper?.isIllustrative && (
                      <span className="tag tag-amber">Illustrative text</span>
                    )}
                  </p>
                  {safeUrl(paper?.url) && (
                    <a
                      className="text-button"
                      href={safeUrl(paper?.url)}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Read original source <ArrowUpRight size={14} />
                    </a>
                  )}
                </div>
                <div className="detail-block">
                  <h3>
                    <Database size={16} />
                    Candidate dataset
                  </h3>
                  <h4>{dataset?.title}</h4>
                  <p>{dataset?.description}</p>
                  <div className="term-list">
                    {dataset?.variables.map((term) => (
                      <span key={term}>{formatTerm(term)}</span>
                    ))}
                  </div>
                  {safeUrl(dataset?.url) && (
                    <a
                      className="text-button"
                      href={safeUrl(dataset?.url)}
                      target="_blank"
                      rel="noreferrer"
                    >
                      Visit {dataset?.source} <ArrowUpRight size={14} />
                    </a>
                  )}
                </div>
                <div className="detail-block">
                  <h3>
                    <SlidersHorizontal size={16} />
                    Why this connection?
                  </h3>
                  {selected.components.map((component, index) => (
                    <div
                      className="component"
                      key={`${component.label}-${index}`}
                    >
                      <span>
                        {component.label}
                        <small>
                          {Math.round(component.weight * 100)}% weight
                        </small>
                      </span>
                      <div>
                        <i
                          style={{
                            width: `${Math.max(0, Math.min(100, component.score))}%`,
                          }}
                        />
                      </div>
                      <strong>{Math.round(component.score)}</strong>
                    </div>
                  ))}
                  <ul className="rationale">
                    {selected.rationale.map((line, i) => (
                      <li key={i}>{line}</li>
                    ))}
                  </ul>
                  <p className="small-print">
                    Ranking score, not calibrated confidence. Inspect
                    population, variables, access, and study design before using
                    the data.
                  </p>
                </div>
                {selected.caveats.length > 0 && (
                  <div className="detail-caveats">
                    <h3>
                      <CircleHelp size={16} />
                      Keep in mind
                    </h3>
                    <ul>
                      {selected.caveats.map((line, i) => (
                        <li key={i}>{line}</li>
                      ))}
                    </ul>
                  </div>
                )}
                <div className="detail-block review-form">
                  <h3>Your review</h3>
                  <label>
                    Assessment
                    <select
                      value={review.status}
                      onChange={(e) =>
                        updateReview(selected.id, { status: e.target.value })
                      }
                    >
                      <option value="unreviewed">Not reviewed</option>
                      <option value="promising">Promising connection</option>
                      <option value="needs_review">Needs closer review</option>
                      <option value="not_suitable">Not suitable</option>
                    </select>
                  </label>
                  <label>
                    Research notes
                    <textarea
                      value={review.note}
                      maxLength={10000}
                      onChange={(e) =>
                        updateReview(selected.id, { note: e.target.value })
                      }
                      placeholder="What would you check next? Missing variables, design considerations, ideas…"
                      rows={4}
                    />
                  </label>
                  <button
                    className={`button ${review.saved ? "button-white" : "button-dark"}`}
                    onClick={() => toggleSave(selected.id)}
                  >
                    <Bookmark
                      size={15}
                      fill={review.saved ? "currentColor" : "none"}
                    />
                    {review.saved ? "Saved to library" : "Save to library"}
                  </button>
                  <span className="small-print">
                    Notes save automatically in this browser.
                  </span>
                </div>
              </div>
            </Modal>
          );
        })()}
      {datasetDetail && (
        <Modal title="Dataset details" onClose={() => setDatasetDetail(null)}>
          <div className="detail-content">
            <span className="tag tag-green">{datasetDetail.source}</span>
            <h2>{datasetDetail.title}</h2>
            <p>{datasetDetail.description || "No description provided."}</p>
            <dl className="metadata-list">
              <div>
                <dt>Sample size</dt>
                <dd>
                  {datasetDetail.sampleSize === null
                    ? "Not reported"
                    : `${datasetDetail.sampleSize.toLocaleString()}${workspace.mode === "demo" ? " (illustrative)" : ""}`}
                </dd>
              </div>
              <div>
                <dt>Population</dt>
                <dd>{datasetDetail.population.join(", ") || "Not reported"}</dd>
              </div>
              <div>
                <dt>Access</dt>
                <dd>{datasetDetail.access || "Not reported"}</dd>
              </div>
              <div>
                <dt>License</dt>
                <dd>{datasetDetail.license || "Not reported"}</dd>
              </div>
              <div>
                <dt>Identifier</dt>
                <dd>{datasetDetail.id}</dd>
              </div>
            </dl>
            <h3>Variables & topics</h3>
            <div className="term-list">
              {datasetDetail.variables.length ? (
                datasetDetail.variables.map((term) => (
                  <span key={term}>{formatTerm(term)}</span>
                ))
              ) : (
                <p>No variable-level metadata provided.</p>
              )}
            </div>
            <p className="small-print">
              Metadata labels are not verification that participant-level data
              are available. Review the original record and reuse terms.
            </p>
            {safeUrl(datasetDetail.url) && (
              <a
                className="button button-dark"
                href={safeUrl(datasetDetail.url)}
                target="_blank"
                rel="noreferrer"
              >
                Open source record <ArrowUpRight size={15} />
              </a>
            )}
          </div>
        </Modal>
      )}
      {commandOpen && (
        <Modal
          title="Explore your next question"
          onClose={() => setCommandOpen(false)}
        >
          <form className="command-form" onSubmit={runSearch}>
            <label htmlFor="command-query">
              Search scientific literature and public datasets
            </label>
            <div>
              <Search size={19} />
              <input
                id="command-query"
                autoFocus
                value={query}
                maxLength={300}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter a research topic…"
              />
            </div>
            <button
              type="submit"
              className="button button-dark"
              disabled={busy}
            >
              Discover connections <ArrowRight size={16} />
            </button>
            <p>
              Searches Europe PMC and DataCite. Your search terms are sent to
              these public providers.
            </p>
          </form>
          <div className="command-shortcuts">
            <button
              onClick={() => {
                setCommandOpen(false);
                fileRef.current?.click();
              }}
            >
              <Upload size={17} />
              Import pipeline results
              <ArrowRight size={15} />
            </button>
            <button
              onClick={() => {
                setCommandOpen(false);
                setAnalyzeOpen(true);
              }}
            >
              <FileText size={17} />
              Analyze your own text
              <ArrowRight size={15} />
            </button>
            <button
              onClick={() => {
                cancelPendingSearch();
                setWorkspace(demoWorkspace);
                setCommandOpen(false);
                navigate("overview");
                setNotice(
                  "Example workspace loaded. Saved notes are preserved.",
                );
              }}
            >
              <FlaskConical size={17} />
              Explore the example workspace
              <ArrowRight size={15} />
            </button>
          </div>
        </Modal>
      )}
      {exportOpen && (
        <Modal
          title="Take your research with you"
          onClose={() => setExportOpen(false)}
        >
          <div className="export-content">
            <p>
              Export all{" "}
              {page === "library"
                ? library.opportunities.filter(
                    (item) => reviews[item.id]?.saved,
                  ).length
                : workspace.opportunities.length}{" "}
              matches, source references, scores, and your review notes.
            </p>
            <button className="export-option" onClick={() => download("json")}>
              <span>
                <FileJson size={23} />
              </span>
              <div>
                <strong>Complete workspace</strong>
                <p>JSON · Reimport to restore data and reviews</p>
              </div>
              <ArrowDownToLine size={18} />
            </button>
            <button className="export-option" onClick={() => download("csv")}>
              <span>
                <FileText size={23} />
              </span>
              <div>
                <strong>Review spreadsheet</strong>
                <p>CSV · Open in Excel, Sheets, or your analysis tools</p>
              </div>
              <ArrowDownToLine size={18} />
            </button>
            <p className="small-print">
              Exports include current workspace records only. Store your JSON
              export somewhere safe to preserve your work.
            </p>
          </div>
        </Modal>
      )}
      {analyzeOpen && (
        <Modal
          title="Follow a question in your own text"
          onClose={() => {
            setAnalyzeOpen(false);
            setError("");
          }}
        >
          <form className="analyze-form" onSubmit={analyze}>
            <p>
              Paste an abstract or discussion. Explicit future directions and
              limitations are matched against the current{" "}
              {workspace.datasets.length} candidate datasets, locally in your
              browser.
            </p>
            <label>
              Paper title
              <input
                value={textTitle}
                onChange={(e) => setTextTitle(e.target.value)}
                maxLength={300}
                placeholder="Give this source a title"
              />
            </label>
            <label>
              Research text
              <textarea
                autoFocus
                value={paperText}
                maxLength={100000}
                onChange={(e) => setPaperText(e.target.value)}
                rows={8}
                placeholder="Future studies should investigate…"
                required
                minLength={40}
              />
            </label>
            {error && (
              <p role="alert" className="form-error">
                {error}
              </p>
            )}
            <button type="submit" className="button button-dark">
              <Microscope size={16} />
              Find connections
            </button>
            <p className="small-print">
              Your pasted text is never sent to an external service.
            </p>
          </form>
        </Modal>
      )}
    </div>
  );
}
