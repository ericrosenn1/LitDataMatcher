#!/usr/bin/env python3
import sys, csv, os

inp = sys.argv[1] if len(sys.argv) > 1 else "sentences.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "sentences_fixed.csv"

def normalize(line: str):
    # remove BOM/CR and trim
    return line.replace("\ufeff","").rstrip("\r\n")

with open(inp, "r", encoding="utf-8") as f_in, \
     open(out, "w", encoding="utf-8", newline="") as f_out:
    w = csv.writer(f_out, quoting=csv.QUOTE_MINIMAL)
    # write header
    w.writerow(["sentence","label"])
    bad = 0
    for n, raw in enumerate(f_in, start=1):
        line = normalize(raw)
        if not line or line.strip().lower() == "sentence,label":
            continue
        if line.lstrip().startswith("#"):
            # ignore commented provenance lines
            continue
        # split on the LAST comma so commas in the sentence are safe
        if "," not in line:
            bad += 1
            continue
        left, right = line.rsplit(",", 1)
        sent = left.strip()
        lab  = right.strip()
        # strip outer quotes if present; escape inner quotes
        if sent.startswith('"') and sent.endswith('"'):
            sent = sent[1:-1]
        sent = sent.replace('"', '""')
        # keep only 0/1/blank labels
        if lab not in {"0", "1", ""}:
            # try to sanitize '1 ' or '0 ' etc.
            lab = lab.strip()
            if lab not in {"0", "1", ""}:
                lab = ""  # leave unlabeled
        w.writerow([sent, lab])
    if bad:
        print(f"[WARN] Skipped {bad} malformed line(s) without a comma.")

print(f"[OK] Wrote clean CSV → {out}")