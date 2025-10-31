#!/usr/bin/env python3
import os, sys, json, argparse, pathlib, time
from typing import List, Dict, Any
import numpy as np

# Import all the models
from models_lit_analyzer import (
    Document,
    Evidence,
    ExtractionSignals,
    ResearchQuestion,
    FutureDirection,
    Limitation,
    NextStep,
    DocumentResults,
    AnalysisResults,
    ExtractionType
)

# Import extraction functions from extraction module
from extraction import (
    find_research_questions as _find_research_questions,
    find_limitations as _find_limitations,
    find_future_directions as _find_future_directions
)

try:
    from joblib import load as joblib_load
except Exception:
    joblib_load = None

# --------------- CLI ---------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="Directory of PDFs or a JSONL file with {title, abstract, text?}")
    ap.add_argument("--out", default="out_lit", help="Output directory")
    ap.add_argument("--use-grobid", action="store_true", help="Use a running GROBID service if available")
    ap.add_argument("--grobid-url", default="http://localhost:8070", help="GROBID server URL")
    ap.add_argument("--embed-model", default="allenai/specter", help="allenai/specter or allenai/scibert_scivocab_uncased")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--scispacy-model", default="en_core_sci_sm", help="scispaCy model name or empty to disable")
    ap.add_argument("--future-clf", default="", help="Path to joblib classifier for Option A (strong future-direction filter)")
    ap.add_argument("--future-th", type=float, default=0.70, help="Probability threshold for strong future directions (0..1)")
    ap.add_argument("--ctx-enabled", action="store_true", help="Attach local/semantic context to future-direction hits")
    ap.add_argument("--ctx-prev-next", type=int, default=1, help="Number of previous/next sentences to include as local context")
    ap.add_argument("--ctx-topk", type=int, default=2, help="Top-K semantically similar sentences from same text to include")
    ap.add_argument("--ctx-max-chars", type=int, default=600, help="Cap concatenated context length per hit")
    ap.add_argument("--device", default=os.getenv("LIT_DEVICE", "auto"),
                choices=["auto", "cpu", "cuda"], help="Where to run embedding models")
    ap.add_argument("--gpu-id", type=int, default=int(os.getenv("LIT_GPU_ID", "0")),
                help="GPU index (used if device=cuda)")
    ap.add_argument("--gpu-mem-frac", type=float, default=float(os.getenv("LIT_GPU_MEM_FRAC", "0.35")),
                help="Soft per-process GPU memory fraction when device=cuda")
    ap.add_argument("--threads", type=int, default=int(os.getenv("OMP_NUM_THREADS", "1")),
                help="Max CPU threads for PyTorch/BLAS")
    return ap.parse_args()

#--------- globals-----
FUTURE_CLF = None
FUTURE_TH  = 0.70
CTX_OPTS = {
    "enabled": False,
    "prev_next": 1,
    "topk": 2,
    "max_chars": 600,
}
RUNTIME_DEVICE = "cpu"
RUNTIME_GPU_ID = 0

FUTURE_HOOKS = [
    "future work", "future studies", "future experiments", "further work",
    "further study", "further studies", "further experiments",
    "remain to be determined", "remains to be determined", "remains unknown",
    "is unknown", "is unclear", "unclear whether", "not been examined",
    "not yet examined", "not yet understood", "not yet explored",
    "warrants investigation", "warrants further investigation",
    "needs to be investigated", "needs to be explored", "should be examined",
    "should be explored", "should be investigated", "should be addressed",
    "should be clarified", "should be validated", "should be replicated",
    "required to determine", "needed to determine", "will be important",
    "will be critical", "will be essential", "will clarify", "will elucidate",
    "next step", "next steps", "next direction", "next directions",
    "in the future", "in future work", "in future studies",
    "future research should", "further research should",
    "further investigation is warranted", "further investigation will",
    "remains an open question", "open question", "open questions",
    "outstanding question", "outstanding questions",
    "should be confirmed", "should be validated", "should be extended",
    "should be compared", "should be tested", "should be improved",
    "remains to be explored", "requires additional investigation",
    "calls for additional studies", "calls for future work",
    "merits further investigation", "needs confirmation",
    "yet to be elucidated", "yet to be determined"
]

#----- GPU helpers---
def set_threads(n: int):
    n = max(1, int(n))
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    try:
        import torch
        torch.set_num_threads(n)
    except Exception:
        pass

def pick_device(flag: str, gpu_id: int = 0) -> str:
    if flag == "cpu":
        return "cpu"
    if flag == "cuda":
        return "cuda" if _cuda_available() else "cpu"
    return "cuda" if _cuda_available() else "cpu"

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False

def set_gpu_mem_fraction(frac: float):
    try:
        import torch
        if torch.cuda.is_available():
            f = min(max(float(frac), 0.05), 0.95)
            torch.cuda.set_per_process_memory_fraction(f)
    except Exception:
        pass

# --------------- caches and helpers ---------------
EMB_CACHE: Dict[str, List[float]] = {}
OA_CACHE: Dict[str, Dict[str, Any]] = {}
LOG_DIR = "out_lit"

from typing import TypeVar, Callable, Dict, Any, List

T = TypeVar('T')

def cached(key: str, cache: Dict[str, T], fn: Callable[[], T]) -> T:
    """Generic caching with proper types"""
    if key in cache:
        return cache[key]
    val = fn()
    cache[key] = val
    return val

def trunc(s: str, n: int = 140) -> str:
    s = s or ""
    return s if len(s) <= n else (s[: n - 1] + "…")

# ---- Feature builder for classifier ----
from typing import Union

#Build feature vectors for future direction classification
def build_future_features(sentences: Union[str, List[str]]) -> np.ndarray:
    if isinstance(sentences, str):
        sentences = [sentences]
    emb = ensure_sentence_embedder()
    Xe = np.array(emb.encode(sentences), dtype=float)
    MODALS = [" will ", " should ", " could ", " might ", " may ", " need ", " requires "]
    rule_feats = []
    for s in sentences:
        low = s.lower()
        hook_cnt  = sum(h in low for h in FUTURE_HOOKS)
        modal_cnt = sum(m in f" {low} " for m in MODALS)
        will_inf  = 1 if has_will_infinitive(low) else 0
        rule_feats.append([hook_cnt, modal_cnt, will_inf])
    Xr = np.array(rule_feats, dtype=float)
    X_full = np.hstack([Xe, Xr])
    return X_full

def has_will_infinitive(low: str) -> bool:
    WILL_VERBS = ["reveal", "clarify", "determine", "validate", "replicate",
                  "elucidate", "quantify", "compare", "test", "assess",
                  "examine", "characterize", "investigate", "map", "measure"]
    if "will " not in low:
        return False
    for v in WILL_VERBS:
        if ("will " + v) in low:
            return True
    return False

# --------------- Models ---------------
class Embedder:
    def __init__(self, model_name: str, device: str = "cpu", gpu_id: int = 0):
        self.model_name = (model_name or "").strip()
        self.device = device
        self.gpu_id = gpu_id
        self.impl = None
        self.tok = None
        self.model = None
        if not self.model_name:
            return
        try:
            if self.model_name.startswith("allenai/specter"):
                from sentence_transformers import SentenceTransformer
                self.impl = SentenceTransformer(self.model_name, device=("cuda" if self.device=="cuda" else "cpu"))
            else:
                from transformers import AutoTokenizer, AutoModel
                import torch
                self.tok = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                if self.device == "cuda" and _cuda_available():
                    torch.cuda.set_device(self.gpu_id)
                    self.model.to("cuda")
                self.model.eval()
        except Exception:
            self.impl = None
            self.tok = None
            self.model = None

    def encode_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts or not self.model_name:
            return []
        try:
            if self.impl is not None:
                return [v.tolist() for v in self.impl.encode(texts, show_progress_bar=False,
                                                             batch_size=min(len(texts), 32))]
            import torch
            outs = []
            for t in texts:
                enc = self.tok(t, return_tensors="pt", truncation=True, max_length=512)
                if self.device == "cuda" and _cuda_available():
                    enc = {k: v.to("cuda") for k, v in enc.items()}
                with torch.no_grad():
                    h = self.model(**enc).last_hidden_state.mean(1).squeeze(0).detach().cpu().tolist()
                outs.append(h)
            return outs
        except Exception:
            return []

SENT_EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class SentenceEmbedder:
    def __init__(self, model_name=SENT_EMB_MODEL, device: str = "cpu", gpu_id: int = 0):
        self.model_name = model_name
        self.device = device
        self.gpu_id = gpu_id
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=("cuda" if self.device=="cuda" else "cpu"))
        except Exception:
            self.model = None

    def encode(self, texts):
        if not texts: return []
        if self.model is None: return [[] for _ in texts]
        try:
            return [v.tolist() for v in self.model.encode(texts, show_progress_bar=False,
                                                          batch_size=min(len(texts), 64))]
        except Exception:
            return [[] for _ in texts]

def ensure_sentence_embedder():
    global _SENT_EMB
    if _SENT_EMB is None:
        _SENT_EMB = SentenceEmbedder(SENT_EMB_MODEL, device=RUNTIME_DEVICE, gpu_id=RUNTIME_GPU_ID)
    return _SENT_EMB

# --------------- OpenAlex enrichment ---------------
import requests

def openalex_by_doi(doi: str) -> dict:
    if not doi:
        return {}
    try:
        r = requests.get(f"https://api.openalex.org/works/doi:{doi}", timeout=20)
        if r.status_code != 200:
            logger.warning(f"OpenAlex returned status {r.status_code} for DOI: {doi}")
            return {}
        w = r.json()
        authors = []
        for a in w.get("authorships", []):
            name = (a.get("author") or {}).get("display_name")
            if name:
                authors.append(name)
        return {
            "openalex_id": w.get("id", ""),
            "title_oa": w.get("title", ""),
            "year": w.get("publication_year", None),
            "cited_by_count": w.get("cited_by_count", 0),
            "concepts": [c.get("display_name", "") for c in w.get("concepts", [])][:10],
            "related_ids": [rel.get("id", "") for rel in w.get("related_works", [])][:20],
            "authors": authors,
            "authorships": w.get("authorships", [])
        }
    except requests.RequestException as e:
        logger.warning(f"OpenAlex API request failed for {doi}: {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error in OpenAlex lookup: {e}", exc_info=True)
        return {}

def openalex_by_title(title: str) -> dict:
    if not title:
        return {}
    try:
        q = {"search": title, "per_page": 1}
        r = requests.get("https://api.openalex.org/works", params=q, timeout=20)
        if r.status_code != 200:
            logger.warning(f"OpenAlex returned status {r.status_code} for title: {title}")
            return {}
        items = r.json().get("results", [])
        if not items:
            return {}
        w = items[0]
        authors = []
        for a in w.get("authorships", []):
            name = (a.get("author") or {}).get("display_name")
            if name:
                authors.append(name)
        return {
            "openalex_id": w.get("id", ""),
            "title_oa": w.get("title", ""),
            "year": w.get("publication_year", None),
            "cited_by_count": w.get("cited_by_count", 0),
            "concepts": [c.get("display_name", "") for c in w.get("concepts", [])][:10],
            "related_ids": [rel.get("id", "") for rel in w.get("related_works", [])][:20],
            "authors": authors,
            "authorships": w.get("authorships", [])
        }
    except requests.RequestException as e:
        logger.warning(f"OpenAlex API request failed for title '{title}': {e}")
        return {}
    except Exception as e:
        logger.error(f"Unexpected error in OpenAlex title lookup: {e}", exc_info=True)
        return {}
    
# --------------- GROBID ingest + fallbacks ---------------
def read_jsonl(path: pathlib.Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed JSON at {path}:{line_num} - {e}")
            except Exception as e:
                logger.error(f"Unexpected error reading {path}:{line_num} - {e}")
    return rows

def extract_with_grobid(pdf_path: pathlib.Path, url: str) -> Dict[str, Any]:
    try:
        with pdf_path.open("rb") as fh:
            files = {"input": (pdf_path.name, fh, "application/pdf")}
            resp = requests.post(url + "/api/processFulltextDocument", files=files, data={"consolidateCitations": 1}, timeout=60)
        if resp.status_code == 200 and "<TEI" in resp.text:
            return tei_to_doc(resp.text)
        else:
            logger.warning(f"GROBID returned status {resp.status_code} for {pdf_path.name}")

    except requests.RequestException as e:
        logger.warning(f"GROBID request failed for {pdf_path.name}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in GROBID extraction for {pdf_path.name}: {e}", exc_info=True)
    
    # Fallback to direct PDF extraction
    logger.info(f"Falling back to direct PDF extraction for {pdf_path.name}")
    return {"title": pdf_path.stem, "abstract": "", "text": pdf_to_text(pdf_path), "doi": ""}

def tei_to_doc(tei_xml: str) -> Dict[str, Any]:
    title, abstract, body, doi = "", "", "", ""
    try:
        s = tei_xml
        i0 = s.find("<title>")
        i1 = s.find("</title>", i0 + 7)
        if i0 != -1 and i1 != -1:
            title = s[i0 + 7:i1].strip()
        a0 = s.find("<abstract")
        if a0 != -1:
            a1 = s.find(">", a0)
            a2 = s.find("</abstract>", a1)
            if a1 != -1 and a2 != -1:
                abstract = strip_tags(s[a1 + 1:a2])
        b0 = s.find("<body")
        if b0 != -1:
            b1 = s.find(">", b0)
            b2 = s.find("</body>", b1)
            if b1 != -1 and b2 != -1:
                body = strip_tags(s[b1 + 1:b2])
        d0 = s.find('idno type="DOI"')
        if d0 != -1:
            d1 = s.find(">", d0)
            d2 = s.find("</idno>", d1)
            if d1 != -1 and d2 != -1:
                doi = s[d1 + 1:d2].strip()
    except Exception:
        pass
    return {"title": title, "abstract": abstract, "text": body, "doi": doi}

## log failed attempts - error handling
import logging
logger = logging.getLogger(__name__)

def pdf_to_text(pdf_path: pathlib.Path) -> str:
    try:
        from pdfminer.high_level import extract_text
        return extract_text(str(pdf_path)) or ""
    except ImportError:
        logger.warning("pdfminer not installed, cannot extract PDF text")
        return ""
    except Exception as e:
        logger.warning(f"Failed to extract text from {pdf_path}: {e}")
        return ""


def strip_tags(s: str) -> str:
    out, keep = [], True
    for ch in s:
        if ch == "<":
            keep = False
            continue
        if ch == ">":
            keep = True
            continue
        if keep:
            out.append(ch)
    return "".join(out)

# --------------- Sectioning ---------------
def split_sections(raw_text: str) -> Dict[str, str]:
    sections = {
        "introduction": "",
        "related": "",
        "methods": "",
        "results": "",
        "discussion": "",
        "conclusion": "",
        "limitations": "",
        "future": ""
    }
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    target = "introduction"
    for ln in lines:
        low = ln.lower()
        if "introduction" in low:
            target = "introduction"
        elif "related work" in low or "background" in low:
            target = "related"
        elif "methods" in low or "materials and methods" in low or "methodology" in low:
            target = "methods"
        elif "results" in low or "findings" in low or "outcomes" in low:
            target = "results"
        elif "discussion" in low or "interpretation" in low or "analysis" in low:
            target = "discussion"
        elif "conclusion" in low or "summary" in low or "closing remarks" in low or "overall" in low:
            target = "conclusion"
        elif "limitations" in low or "constraints" in low or "threats to validity" in low:
            target = "limitations"
        elif "future" in low and ("direction" in low or "work" in low or "prospect" in low or "plan" in low):
            target = "future"
        elif "outlook" in low or "perspective" in low or "perspectives" in low:
            target = "future"
        elif "next step" in low or "next steps" in low or "path forward" in low:
            target = "future"
        elif "outstanding question" in low or "open question" in low or "remaining question" in low:
            target = "future"
        elif "implication" in low or "significance" in low or "applications" in low:
            target = "future"
        sections[target] += ln + "\n"
    return sections

# --------------- Semantic matching ---------------
SEM_SIM_THRESHOLD = 0.62
FUTURE_PROTOS = [
    "future studies should examine",
    "further research is needed to",
    "future work will be important to",
    "it remains to be determined whether",
    "this question remains open",
    "further investigation is warranted",
    "it is unclear whether and requires study",
    "additional experiments are required to clarify",
    "the next step is to test",
    "this should be validated in future studies"
]
LIMIT_PROTOS = [
    "a limitation of this study is",
    "results should be interpreted with caution",
    "the small sample size limits",
    "we could not control for confounding",
    "generalizability is limited",
    "there is a risk of bias"
]
RQ_PROTOS = [
    "we ask whether",
    "we investigate whether",
    "the goal of this study is",
    "our central research question is",
    "we aim to determine whether"
]

_SENT_EMB = None
_PROTO_VECS = {}

# Compute cosine similarity between two vectors
def safe_cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    import math
    da = math.sqrt(sum(x*x for x in a))
    db = math.sqrt(sum(x*x for x in b))
    if da == 0 or db == 0:
        return 0.0
    return sum(x*y for x, y in zip(a, b)) / (da * db)

def get_proto_vecs(kind: str) -> List[List[float]]:
    """Returns list of embedding vectors"""
    global _PROTO_VECS
    if kind in _PROTO_VECS:
        return _PROTO_VECS[kind]
    emb = ensure_sentence_embedder()
    if kind == "future":
        _PROTO_VECS[kind] = emb.encode(FUTURE_PROTOS)
    elif kind == "limit":
        _PROTO_VECS[kind] = emb.encode(LIMIT_PROTOS)
    elif kind == "rq":
        _PROTO_VECS[kind] = emb.encode(RQ_PROTOS)
    else:
        _PROTO_VECS[kind] = []
    return _PROTO_VECS[kind]

def semantic_match(sentence: str, kind: str, threshold: float = SEM_SIM_THRESHOLD) -> bool:
    emb = ensure_sentence_embedder()
    sv = emb.encode([sentence])[0]
    if not sv:
        return False
    protos = get_proto_vecs(kind)
    best = 0.0
    for pv in protos:
        c = safe_cosine(sv, pv)
        if c > best:
            best = c
    return best >= threshold

from typing import Optional

def predict_discourse(sentence: str) -> Optional[str]:
    """
    Predict discourse type for a sentence.
    
    Returns:
        str: Predicted discourse type (e.g., "future", "limitation")
        None: If model unavailable or prediction fails
    """
    model_path = os.path.join(os.path.dirname(__file__), "models", "spec_classifier.joblib")
    if not os.path.exists(model_path):
        return None
    try:
        import joblib
        clf = joblib.load(model_path)
        pred = clf.predict([sentence])[0]
        return str(pred)
    except Exception:
        return None

# --------------- Context indexer ---------------
class ContextIndexer:
    def __init__(self, sentences, embedder=None):
        self.sents = sentences[:]
        self.emb = ensure_sentence_embedder() if embedder is None else embedder
        vecs = self.emb.encode(self.sents) if self.sents else []
        self.vecs = np.array(vecs, dtype=float)

    def neighbors_topk(self, idx, k=2):
        if self.vecs.size == 0: return []
        v = self.vecs[idx]
        da = np.sqrt((v * v).sum()) + 1e-12
        db = np.sqrt((self.vecs * self.vecs).sum(axis=1)) + 1e-12
        sims = (self.vecs @ v) / (db * da)
        sims[idx] = -1.0
        order = np.argsort(-sims)
        out = []
        for j in order[: max(0, k)]:
            out.append((int(j), float(sims[j])))
        return out

    def window(self, idx, n=1):
        n = max(0, int(n))
        lo = max(0, idx - n)
        hi = min(len(self.sents), idx + n + 1)
        return [(i, self.sents[i]) for i in range(lo, hi) if i != idx]

def build_context_index(text: str) -> "ContextIndexer":
    sents = split_sents(text)
    return ContextIndexer(sents)

def assemble_context(idx: int, index: "ContextIndexer", opts: dict) -> str:
    if index is None or not opts.get("enabled", False):
        return ""
    prev_next = int(opts.get("prev_next", 1))
    loc = [s for _, s in index.window(idx, prev_next)]
    k = int(opts.get("topk", 2))
    sem = []
    seen = set([idx])
    for j, _sim in index.neighbors_topk(idx, k=k):
        if j not in seen:
            seen.add(j); sem.append(index.sents[j])
    parts = []
    for seg in loc + sem:
        if not seg: continue
        parts.append(seg.strip())
    ctx = " ".join(parts)
    ctx = " ".join(ctx.split())
    max_chars = int(opts.get("max_chars", 600))
    return ctx[:max_chars]

# --------------- scispaCy entities ---------------
def scispacy_entities(text: str, model_name="en_core_sci_sm", max_chars=20000) -> list:
    if not text or not model_name:
        return []
    try:
        import spacy
        nlp = spacy.load(model_name)
        doc = nlp(text[:max_chars])
        ents = []
        for e in doc.ents:
            ents.append({"text": e.text, "label": e.label_})
        return ents[:500]
    except OSError:
        logger.warning(f"scispaCy model '{model_name}' not found. Install with: " f"pip install {model_name}")
        return []
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}", exc_info=True)
        return []

def split_sents(text: str) -> List[str]:
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_sci_sm")
        except Exception:
            nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        out = [s.text.strip() for s in doc.sents if s.text.strip()]
        if out:
            return out
    except Exception:
        pass
    buf, out = "", []
    for ch in text:
        buf += ch
        if ch in [".", "!", "?"]:
            if buf.strip():
                out.append(buf.strip())
            buf = ""
    if buf.strip():
        out.append(buf.strip())
    return out

# --------------- Wrapper functions for extraction ---------------
def find_research_questions(text: str) -> List[ResearchQuestion]:
    return _find_research_questions(text, split_sents, semantic_match, predict_discourse)

def find_limitations(text: str) -> List[Limitation]:
    return _find_limitations(text, split_sents, semantic_match, predict_discourse)

def find_future_directions(text: str, ctx_index=None) -> List[FutureDirection]:
    return _find_future_directions(
        text,
        split_sents,
        semantic_match,
        predict_discourse,
        build_future_features,
        assemble_context,
        ctx_index=ctx_index,
        future_clf=FUTURE_CLF,
        future_th=FUTURE_TH,
        ctx_opts=CTX_OPTS,
        log_dir=LOG_DIR
    )

# --------------- Next steps ---------------
def propose_next_steps(
    rqs: List[ResearchQuestion], 
    limits: List[Limitation], 
    title: str
) -> List[NextStep]:
    steps = []
    for rq in rqs[:5]:
        step = NextStep(
            question="Replicate and validate the main finding in an external cohort",
            rationale="External validity and robustness",
            suggested_approach="Define inclusion criteria, perform power analysis, preregister, report effect sizes",
            expected_impact=0.7,
            difficulty=0.4,
            links=[rq.evidence.text]
        )
        steps.append(step)
    for lm in limits[:5]:
        low = lm.text.lower()
        if "small sample" in low:
            step = NextStep(
                question="Increase sample size and rerun the analysis",
                rationale="Address low power risk",
                suggested_approach="Power analysis, recruit additional participants, adjust for multiple testing",
                expected_impact=0.6,
                difficulty=0.3,
                links=[lm.text]
            )
            steps.append(step)
        elif "lack of" in low:
            step = NextStep(
                question="Add the missing measurement or control",
                rationale="Reduce bias from missing variable",
                suggested_approach="Collect the missing covariate or add a control group",
                expected_impact=0.6,
                difficulty=0.5,
                links=[lm.text]
            )
            steps.append(step)
    return steps

# --------------- Load inputs ---------------
def load_inputs(inp: str, use_grobid: bool, grobid_url: str) -> List[Dict[str, Any]]:
    path = pathlib.Path(inp)
    docs = []
    if path.is_file() and path.suffix.lower() == ".jsonl":
        for row in read_jsonl(path):
            title = row.get("title", "")
            abstract = row.get("abstract", "")
            text = row.get("text", "")
            if not text:
                text = " ".join([abstract, title]).strip()
            rec = {"title": title, "abstract": abstract, "text": text, "doi": row.get("doi", "")}
            docs.append(rec)
        return docs
    if path.is_dir():
        for p in sorted(path.glob("*.pdf")):
            if use_grobid:
                docs.append(extract_with_grobid(p, grobid_url))
            else:
                docs.append({"title": p.stem, "abstract": "", "text": pdf_to_text(p), "doi": ""})
        return docs
    return []

def print_terminal_summary(args, docs, rq_map, st_map, fd_map, metrics):
    print("\n=== Literature Analyzer Summary (top 5) ===", flush=True)
    show_docs = docs[:5] if len(docs) > 5 else docs
    for d in show_docs:
        title = d.get("title", "")
        doi   = d.get("doi", "")
        header = title or doi or "(untitled)"
        print(f"\n{header}")
        rqs_for_title   = rq_map.get(title, [])
        fds_for_title   = fd_map.get(title, [])
        steps_for_title = st_map.get(title, [])
        print("  Explicit RQs:")
        if rqs_for_title:
            for idx, rq in enumerate(rqs_for_title[:3], 1):
                print(f"    {idx}. {trunc(rq.get('question',''))}")
                sig = rq.get("signals", {})
                print(f"       ↳ {sig}")
        else:
            print("    None detected.")
        print("  Future directions:")
        if fds_for_title:
            for idx, it in enumerate(fds_for_title[:3], 1):
                cat = it.get("category", "unspecified")
                print(f"    {idx}. [{cat}] {trunc(it.get('text',''))}")
                sig = it.get("signals", {})
                print(f"       ↳ {sig}")
                ctx = it.get("context","")
                if ctx:
                    print(f"       context: {trunc(ctx, 160)}")
        else:
            print("    None detected.")
        print("  Proposed next steps:")
        if steps_for_title:
            for idx, st in enumerate(steps_for_title[:3], 1):
                print(f"    {idx}. {trunc(st.get('question',''))}")
        else:
            print("    None detected.")
    if len(docs) > 5:
        print(f"\nShowing first 5 of {len(docs)} documents. See {args.out}/summary.md for full results.\n")
    else:
        print("\nAll documents shown above.\n")
    print(json.dumps({"type": "lit_done", "payload": metrics}), flush=True)

# --------------- Main ---------------
def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Configure logging at startup
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.out, 'lit_analyzer.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting analysis with inputs: {args.inputs}")

    
    set_threads(args.threads)
    global RUNTIME_DEVICE, RUNTIME_GPU_ID
    RUNTIME_DEVICE = pick_device(args.device, args.gpu_id)
    RUNTIME_GPU_ID = int(args.gpu_id)
    if RUNTIME_DEVICE == "cuda":
        set_gpu_mem_fraction(args.gpu_mem_frac)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_device(0 if os.environ.get("CUDA_VISIBLE_DEVICES") else RUNTIME_GPU_ID)
        except Exception:
            pass
        print(f"[INFO] Device: cuda (gpu_id={RUNTIME_GPU_ID}, mem_frac={args.gpu_mem_frac})")
    else:
        print("[INFO] Device: cpu")
    global _SENT_EMB
    _SENT_EMB = None
    embedder = Embedder(args.embed_model, device=RUNTIME_DEVICE, gpu_id=RUNTIME_GPU_ID)
    global CTX_OPTS, LOG_DIR, FUTURE_CLF, FUTURE_TH
    LOG_DIR = args.out
    CTX_OPTS = {
        "enabled": bool(args.ctx_enabled),
        "prev_next": int(args.ctx_prev_next),
        "topk": int(args.ctx_topk),
        "max_chars": int(args.ctx_max_chars),
    }
    FUTURE_TH = float(args.future_th)
    if args.future_clf and joblib_load is not None and os.path.exists(args.future_clf):
        try:
            FUTURE_CLF = joblib_load(args.future_clf)
            print(f"[INFO] Loaded future-direction classifier: {args.future_clf}")
        except Exception as e:
            print(f"[WARN] Could not load classifier: {e}")
            FUTURE_CLF = None
    else:
        FUTURE_CLF = None
    docs = load_inputs(args.inputs, args.use_grobid, args.grobid_url)
    if not docs:
        print("no inputs found")
        return
    f_rq = open(os.path.join(args.out, "explicit_rqs.jsonl"), "w", encoding="utf-8")
    f_ns = open(os.path.join(args.out, "proposed_next_steps.jsonl"), "w", encoding="utf-8")
    f_fd = open(os.path.join(args.out, "future_directions.jsonl"), "w", encoding="utf-8")
    f_mx = open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8")
    t0 = time.time()
    total_rqs, total_steps, total_future = 0, 0, 0
    for d in docs:
        title, abstract, text, doi = d.get("title",""), d.get("abstract",""), d.get("text",""), d.get("doi","")
        base = f"{abstract}\n{text}".strip()
        secs = split_sections(base)
        bucket = secs["introduction"] + "\n" + secs["related"] + "\n" + secs["discussion"] + "\n" + secs["conclusion"]
        rqs = find_research_questions(bucket)
        lims_txt = secs["limitations"] if secs["limitations"] else secs["discussion"]
        limits = find_limitations(lims_txt)
        fd_text = base
        ctx_index = build_context_index(fd_text) if CTX_OPTS["enabled"] else None
        future_items = find_future_directions(fd_text, ctx_index=ctx_index)
        def _oa():
            return openalex_by_doi(doi) if doi else openalex_by_title(title)
        oa = cached(doi or title, OA_CACHE, _oa)
        summary = f"Title: {title}. Abstract: {abstract}".strip()
        emb_key = (args.embed_model or "") + "|" + summary
        def _emb():
            if not args.embed_model:
                return []
            vecs = embedder.encode_texts([summary])
            return vecs[0] if vecs else []
        emb = cached(emb_key, EMB_CACHE, _emb)
        sections_text = (secs["introduction"] + "\n" + secs["discussion"] + "\n" + secs["conclusion"]).strip()
        entities = scispacy_entities(sections_text, model_name=args.scispacy_model) if args.scispacy_model else []
        for rq in rqs:
            rec = rq.to_dict()
            rec["title"] = title
            rec["embedding"] = emb
            rec["openalex"] = oa
            rec["entities"] = entities[:100]
            f_rq.write(json.dumps(rec) + "\n")
        total_rqs += len(rqs)
        steps = propose_next_steps(rqs, limits, title)
        for step in steps:
            rec = step.to_dict()
            rec["title"] = title
            rec["embedding"] = emb
            rec["openalex"] = oa
            rec["entities"] = entities[:100]
            f_ns.write(json.dumps(rec) + "\n")
        total_steps += len(steps)
        for fd in future_items:
            rec = fd.to_dict()
            rec["title"] = title
            rec["embedding"] = emb
            rec["openalex"] = oa
            rec["entities"] = entities[:100]
            f_fd.write(json.dumps(rec) + "\n")
        total_future += len(future_items)
    f_rq.close(); f_ns.close(); f_fd.close()
    elapsed = max(1e-6, time.time() - t0)
    metrics = {"docs": len(docs), "rqs": total_rqs, "steps": total_steps, "future": total_future, "secs": elapsed}
    json.dump(metrics, f_mx, indent=2)
    f_mx.close()
    summary_path = os.path.join(args.out, "summary.md")
    rq_path = os.path.join(args.out, "explicit_rqs.jsonl")
    steps_path = os.path.join(args.out, "proposed_next_steps.jsonl")
    fd_path = os.path.join(args.out, "future_directions.jsonl")
    def load_by_title(path):
        items = {}
        if not os.path.exists(path):
            return items
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    items.setdefault(rec.get("title",""), []).append(rec)
                except Exception:
                    pass
        return items
    rq_map = load_by_title(rq_path)
    st_map = load_by_title(steps_path)
    fd_map = load_by_title(fd_path)
    with open(summary_path, "w", encoding="utf-8") as fs:
        for d in docs:
            title, abstract, text, doi = d.get("title",""), d.get("abstract",""), d.get("text",""), d.get("doi","")
            fs.write(f"# {title}\n")
            if doi:
                fs.write(f"DOI: {doi}\n")
            oa = OA_CACHE.get(doi or title, {})
            if oa:
                fs.write("## OpenAlex Summary\n")
                fs.write(f"- Year: {oa.get('year','')}\n")
                fs.write(f"- Citations: {oa.get('cited_by_count',0)}\n")
                fs.write(f"- Concepts: {', '.join(oa.get('concepts', []))}\n\n")
            fs.write("## Explicit Research Questions\n")
            rqs_for_title = rq_map.get(title, [])
            if rqs_for_title:
                for i, rq in enumerate(rqs_for_title, 1):
                    fs.write(f"{i}. {rq.get('question','')}\n")
                    sig = rq.get("signals", {})
                    fs.write(f"   - Identified by: {sig}\n")
            else:
                fs.write("None detected.\n")
            fs.write("\n## Limitations\n")
            lims_txt = split_sections(text).get("limitations","")
            lims = find_limitations(lims_txt)
            if lims:
                for i, lm in enumerate(lims, 1):
                    fs.write(f"{i}. {lm.text}\n")
                    fs.write(f"   - Confidence: {lm.confidence:.2f}\n")
            else:
                fs.write("None detected.\n")
            fs.write("\n## Proposed Next Steps\n")
            steps_for_title = st_map.get(title, [])
            if steps_for_title:
                for i, st in enumerate(steps_for_title, 1):
                    fs.write(f"{i}. {st.get('question','')}\n")
                    fs.write(f"   - Rationale: {st.get('rationale','')}\n")
                    fs.write(f"   - Approach: {st.get('suggested_approach','')}\n")
                    fs.write(f"   - Expected Impact: {st.get('expected_impact','')}\n")
                    fs.write(f"   - Difficulty: {st.get('difficulty','')}\n")
            else:
                fs.write("None detected.\n")
            fs.write("\n## Future Directions Mined From the Paper\n")
            fds_for_title = fd_map.get(title, [])
            if fds_for_title:
                for i, it in enumerate(fds_for_title[:10], 1):
                    fs.write(f"{i}. [{it.get('category','unspecified')}] {it.get('text','')}\n")
                    sig = it.get("signals", {})
                    fs.write(f"   - Identified by: {sig}\n")
                    ctx = it.get("context","")
                    if ctx:
                        fs.write(f"   - Context: {ctx}\n")
            else:
                fs.write("None detected.\n")
            labels = []
            for rq in rqs_for_title:
                labels.extend([e.get("label","") for e in rq.get("entities", [])])
            labels = sorted(set([l for l in labels if l]))
            if labels:
                fs.write("\n## Entities Found\n")
                fs.write(", ".join(labels) + "\n")
            fs.write("\n---\n\n")
    print_terminal_summary(args, docs, rq_map, st_map, fd_map, metrics)
    print(json.dumps({"type": "lit_done", "payload": metrics}), flush=True)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass