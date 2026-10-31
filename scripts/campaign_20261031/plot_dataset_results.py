"""Plot measured dataset-retrieval development results without scientific relabeling."""

from __future__ import annotations

import argparse
import csv
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

    report = json.loads(source.read_text(encoding="utf-8"))
    methods = [
        ("bm25", "BM25"),
        ("dense", "MiniLM dense"),
        ("hybrid_rrf", "BM25 + dense fusion"),
        ("requirement_aware_dense", "Availability-aware dense*"),
    ]
    rows = []
    for key, label in methods:
        metrics = report["metrics"][key]
        details = metrics["details"]
        if len(details) != report["queries"] or len({r["case_id"] for r in details}) != len(
            details
        ):
            raise ValueError("Incomplete or duplicate per-case measurements")
        value = metrics["mrr_at_k"]
        measured = sum(r["reciprocal_rank_at_k"] for r in details) / len(details)
        if not math.isfinite(value) or not 0 <= value <= 1 or abs(measured - value) > 1e-12:
            raise ValueError("Reported MRR does not match per-case measurements")
        rows.append({"method": key, "label": label, "mrr_at_5": value})
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "svg.hashsalt": "ldm-dataset-development-1",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.subplots_adjust(left=0.23, right=0.97, bottom=0.29, top=0.78, wspace=0.75)
    colors = ["#253c5b", "#147d80", "#8092a8", "#c5d5d9"]
    axes[0].barh(range(len(rows)), [r["mrr_at_5"] for r in rows], color=colors, height=0.6)
    axes[0].set_yticks(range(len(rows)), [r["label"] for r in rows], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 1.15)
    axes[0].set_xlabel("Known-target MRR@5")
    for i, row in enumerate(rows):
        axes[0].text(row["mrr_at_5"] + 0.025, i, f"{row['mrr_at_5']:.3f}", va="center", fontsize=9)
    targets = sorted({d["target"] for d in report["metrics"]["bm25"]["details"]})
    for i, (key, _) in enumerate(methods):
        details = report["metrics"][key]["details"]
        values = [
            sum(d["reciprocal_rank_at_k"] for d in details if d["target"] == target)
            / sum(d["target"] == target for d in details)
            for target in targets
        ]
        axes[1].bar(
            [j + (i - 1.5) * 0.18 for j in range(len(targets))], values, width=0.17, color=colors[i]
        )
    axes[1].set_xticks(range(len(targets)), targets)
    axes[1].set_ylim(0, 1.1)
    axes[1].set_ylabel("Known-target MRR@5")
    axes[1].set_title("Dependence by target dataset", fontsize=10, loc="left")
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="x" if ax == axes[0] else "y", alpha=0.15)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle(
        f"Dataset retrieval: development availability benchmark\n"
        f"{report['queries']} related queries · {report['candidate_count']} candidates · {len(targets)} target datasets",
        x=0.04,
        ha="left",
        fontsize=14,
    )
    fig.text(
        0.04,
        0.045,
        "*Availability-aware ranking uses the same measured-variable contract used to construct reference labels.\n"
        "This measures engineering consistency, not independently reviewed scientific feasibility or novelty.\n"
        "Queries reuse two datasets; no independence, confidence-interval or significance claim is made.",
        fontsize=8,
        color="#405168",
    )
    out.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "svg", "pdf"):
        target = out / f"dataset-retrieval-development.{suffix}"
        metadata = {"Creator": "LitDataMatcher"}
        if suffix == "svg":
            metadata["Date"] = None
        elif suffix == "pdf":
            metadata.update(CreationDate=None, ModDate=None)
        fig.savefig(target, dpi=180, bbox_inches="tight", metadata=metadata)
        if suffix == "svg":
            svg = "\n".join(target.read_text(encoding="utf-8").splitlines()) + "\n"
            target.write_text(svg, encoding="utf-8", newline="\n")
            ET.fromstring(svg)
        outputs.append({"path": target.name, "sha256": sha(target), "bytes": target.stat().st_size})
    plt.close(fig)
    csv_path = out / "dataset-retrieval-development.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    outputs.append(
        {"path": csv_path.name, "sha256": sha(csv_path), "bytes": csv_path.stat().st_size}
    )
    receipt = {
        "status": "PASS_WITH_LIMITATIONS",
        "source": str(source.resolve()),
        "source_sha256": sha(source),
        "script_sha256": sha(__file__),
        "matplotlib_version": matplotlib.__version__,
        "outputs": outputs,
        "limitations": report["limitations"],
    }
    (out / "FIGURE_MANIFEST.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.source, args.out), indent=2))


if __name__ == "__main__":
    main()
