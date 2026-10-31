"""Render the completed, audited large-pool development retrieval comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(source: Path, out: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(source.read_text(encoding="utf-8"))
    if data["status"] != "PASS" or data["independent_audit_provisional"]:
        raise ValueError("A completed, independently audited comparison is required")
    methods = [
        ("bm25", "BM25"),
        ("bm25_trained_rrf", "BM25 + trained fusion"),
        ("bm25_pretrained_rrf", "BM25 + pretrained fusion"),
        ("trained", "Fine-tuned MiniLM"),
        ("pretrained", "Pretrained MiniLM"),
    ]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "svg.hashsalt": "litdatamatcher-retrieval-scale-v1",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    fig.patch.set_facecolor("#fbfcfe")
    colors = ["#20334e", "#127f7b", "#526f8f", "#38a39a", "#a7b4c5"]
    for ax, view, title in zip(
        axes, ("full", "abstract_only"), ("Title + abstract", "Abstract only"), strict=True
    ):
        values, intervals = [], []
        for key, _ in methods:
            metric = data["conditions"][f"100000/{view}"]["metrics"][key]
            value, interval = metric["mrr_at_k"], metric["mrr_query_bootstrap_95_interval"]
            if not all(math.isfinite(x) and 0 <= x <= 1 for x in [value, *interval]):
                raise ValueError("Invalid MRR or confidence interval")
            values.append(value)
            intervals.append(interval)
        ax.set_facecolor("#fbfcfe")
        ax.barh(range(5), values, height=0.57, color=colors)
        ax.errorbar(
            values,
            range(5),
            xerr=[
                [x - ci[0] for x, ci in zip(values, intervals, strict=True)],
                [ci[1] - x for x, ci in zip(values, intervals, strict=True)],
            ],
            fmt="none",
            ecolor="#182c41",
            capsize=3,
            linewidth=1,
        )
        ax.set_yticks(range(5), [label for _, label in methods], fontsize=10)
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlabel("Known source-link MRR@10", fontsize=10, labelpad=10)
        ax.set_title(title, fontsize=13, color="#20334e", pad=15)
        ax.tick_params(length=0)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.15)
        for index, value in enumerate(values):
            ax.text(1.03, index, f"{value:.3f}", va="center", fontsize=10, color="#20334e")
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle("Retrieval among 100,000 source papers", fontsize=19, color="#20334e", y=0.98)
    fig.text(
        0.59,
        0.905,
        "Same 175 development queries · higher is better",
        ha="center",
        fontsize=11,
        color="#516176",
    )
    fig.text(
        0.025,
        0.06,
        "Bars: observed means. Whiskers: 95% query-bootstrap intervals.\n"
        "Source-link recovery does not establish dataset suitability; unjudged alternatives are not proven irrelevant.",
        fontsize=9,
        color="#516176",
    )
    fig.subplots_adjust(left=0.23, right=0.98, top=0.80, bottom=0.21, wspace=0.16)
    out.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "svg"):
        path = out / f"retrieval-scale.{suffix}"
        metadata = {"Creator": "LitDataMatcher"}
        if suffix == "svg":
            metadata["Date"] = None
        fig.savefig(path, dpi=180, bbox_inches="tight", metadata=metadata)
        if suffix == "svg":
            content = "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            ET.fromstring(content)
            path.write_text(content, encoding="utf-8", newline="\n")
        outputs.append({"path": path.name, "sha256": sha(path), "bytes": path.stat().st_size})
    plt.close(fig)
    receipt = {
        "source_sha256": sha(source),
        "script_sha256": sha(__file__),
        "matplotlib_version": matplotlib.__version__,
        "outputs": outputs,
    }
    (out / "retrieval-scale.manifest.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.source, args.out), indent=2))


if __name__ == "__main__":
    main()
