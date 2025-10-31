#!/usr/bin/env python3
import os, sys, json, time, argparse, threading
from collections import deque
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser("~/research_project/run/matches.jsonl"),
                    help="Output JSONL file for matched results")
    ap.add_argument("--report-interval", type=float, default=2.0, help="seconds between dashboard updates")
    return ap.parse_args()

# -----------------------------
# Utilities
# -----------------------------
def sparkline(values, width=20):
    blocks = " ▁▂▃▄▅▆▇█"
    if not values:
        return " " * width
    vals = list(values)[-width:]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return blocks[-1] * len(vals)
    out = []
    for v in vals:
        k = 1 + int((v - lo) / (hi - lo + 1e-9) * (len(blocks) - 2))
        out.append(blocks[k])
    return "".join(out).ljust(width)

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    data_cache, lit_cache = {}, {}
    done = {"data": False, "lit": False}
    matched_count = 0
    match_rate_hist = deque(maxlen=60)
    last_window_count = 0
    last_time = time.time()

    fout = open(args.out, "w", encoding="utf-8")

    def render_dashboard():
        now = time.time()
        tbl = Table(show_header=True, header_style="bold cyan")
        tbl.add_column("Metric"); tbl.add_column("Value")
        tbl.add_row("Unmatched Data", str(len(data_cache)))
        tbl.add_row("Unmatched Lit", str(len(lit_cache)))
        tbl.add_row("Total Matches", str(matched_count))
        tbl.add_row("Match Rate (per s)", f"{match_rate_hist[-1] if match_rate_hist else 0.0:.2f}")
        tbl.add_row("Throughput Graph", sparkline(match_rate_hist, width=24))
        return Panel(tbl, title="Matcher Live Dashboard", border_style="green")

    def dashboard_updater():
        with Live(render_dashboard(), refresh_per_second=4) as live:
            while not (done["data"] and done["lit"]):
                live.update(render_dashboard())
                time.sleep(args.report-interval)
            live.update(render_dashboard())

    threading.Thread(target=dashboard_updater, daemon=True).start()

    def try_match():
        nonlocal matched_count, last_window_count, last_time
        topics = set(data_cache).intersection(lit_cache)
        for topic in list(topics):
            d, l = data_cache.pop(topic), lit_cache.pop(topic)
            combined = round(0.6 * d["feasibility_score"] + 0.4 * l["significance_score"], 3)
            record = {
                "topic": topic,
                "scores": {
                    "feasibility": d["feasibility_score"],
                    "significance": l["significance_score"],
                    "combined": combined,
                },
                "why": {
                    "datasets": d["datasets"],
                    "variables": d["variables"],
                    "pub_trend": l["counts"],
                    "top_terms": l["top_terms"],
                },
            }
            fout.write(json.dumps(record) + "\n"); fout.flush()
            print(json.dumps({"type": "match", "payload": record}), flush=True)
            matched_count += 1

        # update throughput every few seconds
        now = time.time()
        dt = now - last_time
        if dt >= args.report-interval:
            rate = (matched_count - last_window_count) / max(dt, 1e-9)
            match_rate_hist.append(rate)
            last_window_count = matched_count
            last_time = now

    # main loop
    for line in sys.stdin:
        s = line.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        t = obj.get("type")
        if t == "data_result":
            p = obj["payload"]; data_cache[p["topic"]] = p
        elif t == "lit_result":
            p = obj["payload"]; lit_cache[p["topic"]] = p
        elif t == "data_done":
            done["data"] = True
        elif t == "lit_done":
            done["lit"] = True
        else:
            continue
        try_match()

    fout.close()
    print(json.dumps({"type": "matcher_done"}), flush=True)

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass