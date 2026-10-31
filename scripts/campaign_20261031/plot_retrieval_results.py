"""Render the measured development comparison with source and output hashes."""

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

    report = json.loads(source.read_text(encoding="utf-8"))
    names = [("bm25", "BM25"), ("hybrid", "BM25 + pretrained fusion"),
             ("trained_epoch_02", "Fine-tuned MiniLM"), ("pretrained", "Pretrained MiniLM")]
    values, intervals = [], []
    for key, _ in names:
        metric = report["methods"][key]
        value = metric["mrr_at_k"]
        interval = metric["mrr_query_bootstrap_95_interval"]
        if not all(math.isfinite(x) and 0 <= x <= 1 for x in [value, *interval]):
            raise ValueError("Plot requires finite MRR values in [0, 1]")
        values.append(value)
        intervals.append(interval)
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "svg.hashsalt": "litdatamatcher-development-1"})
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    fig.set_facecolor("#fbfcfe")
    ax.set_facecolor("#fbfcfe")
    colors = ["#20334e", "#5b708d", "#127f7b", "#a7b4c5"]
    ax.barh(range(4), values, height=0.56, color=colors)
    ax.errorbar(values, range(4),
                xerr=[[value - interval[0] for value, interval in zip(values, intervals, strict=True)],
                      [interval[1] - value for value, interval in zip(values, intervals, strict=True)]],
                fmt="none", ecolor="#172536", capsize=4, linewidth=1.2)
    ax.set_yticks(range(4), [label for _, label in names], fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Known source-paper recovery · MRR@10", fontsize=10, labelpad=10)
    for index, value in enumerate(values):
        ax.text(1.025, index, f"{value:.4f}", va="center", fontsize=11, color="#20334e")
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=0.15)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    counts = report["counts"]
    ax.set_title(f"Development retrieval comparison\n{counts['development']:,} queries · "
                 f"{counts['documents']:,} candidate papers", loc="left", fontsize=16, pad=18,
                 color="#20334e")
    fig.text(0.02, -0.04, "Bars: observed means. Whiskers: 95% query-bootstrap intervals.\n"
             "Source-link recovery is not dataset-feasibility accuracy; source-title overlap limits interpretation.",
             fontsize=8.5, color="#516176")
    out.mkdir(parents=True, exist_ok=True)
    outputs = []
    for suffix in ("png", "svg"):
        target = out / f"retrieval-development.{suffix}"
        metadata = {"Creator": "LitDataMatcher"}
        if suffix == "svg":
            metadata["Date"] = None
        fig.savefig(target, dpi=180, bbox_inches="tight", metadata=metadata)
        if suffix == "svg":
            svg = "\n".join(line.rstrip() for line in target.read_text().splitlines()) + "\n"
            with target.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(svg)
            ET.fromstring(svg)
        outputs.append({"path": target.name, "sha256": sha(target), "bytes": target.stat().st_size})
    plt.close(fig)
    receipt = {"source_sha256": sha(source), "script_sha256": sha(__file__),
               "matplotlib_version": matplotlib.__version__, "outputs": outputs}
    with (out / "retrieval-development.manifest.json").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.source, args.out), indent=2))


if __name__ == "__main__":
    main()
