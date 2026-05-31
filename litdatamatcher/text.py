"""Text normalization, sentence splitting, sectioning, and feature extraction."""

from __future__ import annotations

from collections import Counter
import math
import re


ABBREVIATIONS = {
    "al.",
    "dr.",
    "fig.",
    "i.e.",
    "e.g.",
    "et.",
    "mr.",
    "mrs.",
    "prof.",
    "vs.",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "can",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "may",
    "of",
    "on",
    "or",
    "our",
    "should",
    "study",
    "such",
    "than",
    "that",
    "the",
    "their",
    "these",
    "this",
    "to",
    "was",
    "we",
    "were",
    "whether",
    "which",
    "with",
}

SECTION_ALIASES = {
    "abstract": "abstract",
    "background": "introduction",
    "introduction": "introduction",
    "related work": "related",
    "methods": "methods",
    "materials and methods": "methods",
    "methodology": "methods",
    "results": "results",
    "findings": "results",
    "discussion": "discussion",
    "limitations": "limitations",
    "limitation": "limitations",
    "future directions": "future",
    "future work": "future",
    "conclusions": "conclusion",
    "conclusion": "conclusion",
}

VARIABLE_LEXICON = {
    "age": {"age", "adult", "infant", "elderly", "pediatric"},
    "sex": {"sex", "gender", "male", "female"},
    "antibiotic_exposure": {"antibiotic", "antibiotics", "antimicrobial"},
    "body_site": {"body site", "stool", "fecal", "gut", "intestinal", "mucosal"},
    "microbiome_composition": {"microbiome", "microbiota", "16s", "metagenomic", "metagenomics"},
    "metabolomics": {"metabolomic", "metabolomics", "metabolite", "metabolites"},
    "transcriptomics": {"transcriptomic", "transcriptomics", "rna-seq", "rnaseq"},
    "diet": {"diet", "dietary", "nutrition", "nutrient"},
    "timepoint": {"longitudinal", "time", "timepoint", "follow-up", "baseline"},
    "disease_activity": {"ibd", "crohn", "colitis", "inflammation", "remission"},
    "treatment": {"treatment", "therapy", "drug", "intervention"},
    "outcome": {"outcome", "response", "remission", "recovery", "relapse"},
}


def normalize_text(text: str) -> str:
    """Collapse whitespace and strip non-informative leading/trailing space."""

    return " ".join(str(text or "").replace("\u00a0", " ").split())


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric terms."""

    return re.findall(r"[a-zA-Z][a-zA-Z0-9_+-]{1,}", str(text or "").lower())


def split_sentences(text: str) -> list[str]:
    """Split scientific prose into sentences without requiring external NLP.

    The splitter protects common abbreviations and citation-like initials before
    applying punctuation boundaries. It is not a replacement for spaCy, but it is
    deterministic, dependency-free, and good enough for tests and offline runs.
    """

    cleaned = normalize_text(text)
    if not cleaned:
        return []

    protected = cleaned
    replacements: dict[str, str] = {}
    for idx, abbr in enumerate(sorted(ABBREVIATIONS, key=len, reverse=True)):
        pattern = re.compile(re.escape(abbr), flags=re.IGNORECASE)
        for hit_idx, match in enumerate(list(pattern.finditer(protected))):
            token = f"__ABBR{idx}_{hit_idx}__"
            replacements[token] = match.group(0)
            protected = protected.replace(match.group(0), token, 1)

    pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\(\[])", protected)
    out: list[str] = []
    for piece in pieces:
        sentence = piece
        for token, abbr in replacements.items():
            sentence = sentence.replace(token, abbr)
        sentence = normalize_text(sentence)
        if sentence:
            out.append(sentence)
    return out


def split_sections(raw_text: str) -> dict[str, str]:
    """Split article text into coarse IMRaD-style sections.

    Sectioning is intentionally conservative: a line must look like a short
    heading before it changes the active section, which reduces false positives
    from ordinary sentences containing words like "results" or "discussion".
    """

    sections = {
        "abstract": "",
        "introduction": "",
        "related": "",
        "methods": "",
        "results": "",
        "discussion": "",
        "limitations": "",
        "future": "",
        "conclusion": "",
    }
    active = "introduction"
    for raw_line in str(raw_text or "").splitlines():
        line = normalize_text(raw_line)
        if not line:
            continue
        heading = re.sub(r"^\d+(\.\d+)*\s+", "", line).strip(" :.-").lower()
        if len(heading) <= 40 and heading in SECTION_ALIASES:
            active = SECTION_ALIASES[heading]
            continue
        sections[active] += line + "\n"
    return sections


def extract_domain_terms(text: str, max_terms: int = 12) -> list[str]:
    """Extract high-signal lexical terms for matching and summaries."""

    counts = Counter(
        token
        for token in tokenize(text)
        if token not in STOPWORDS and len(token) >= 3 and not token.isdigit()
    )
    return [term for term, _ in counts.most_common(max_terms)]


def infer_required_variables(text: str) -> list[str]:
    """Infer likely data variables needed to answer a question."""

    from .ontology import infer_concepts_from_text

    ontology_hits = infer_concepts_from_text(text)
    lowered = str(text or "").lower()
    hits: list[str] = list(ontology_hits)
    for variable, cues in VARIABLE_LEXICON.items():
        if any(cue in lowered for cue in cues):
            hits.append(variable)
    return list(dict.fromkeys(hits))


def infer_population(text: str) -> str:
    """Infer a coarse study population from text."""

    lowered = str(text or "").lower()
    if any(term in lowered for term in ("infant", "neonate", "newborn")):
        return "infant"
    if any(term in lowered for term in ("child", "pediatric", "paediatric")):
        return "pediatric"
    if any(term in lowered for term in ("mouse", "murine", "mice")):
        return "mouse"
    if any(term in lowered for term in ("human", "patient", "adult", "cohort", "participant")):
        return "human"
    return ""


def lexical_similarity(left: str, right: str) -> float:
    """Return Jaccard similarity over normalized content tokens."""

    a = {token for token in tokenize(left) if token not in STOPWORDS}
    b = {token for token in tokenize(right) if token not in STOPWORDS}
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def sparse_cosine(left_terms: list[str], right_terms: list[str]) -> float:
    """Compute cosine similarity between two token lists."""

    left = Counter(left_terms)
    right = Counter(right_terms)
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    numerator = sum(left[key] * right[key] for key in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)
