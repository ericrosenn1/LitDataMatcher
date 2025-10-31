#!/usr/bin/env python3
import os, sys, json, time, argparse, threading, subprocess, collections

TOPICS = [
    "human gut microbiome 16S",
    "IBD antibiotics recovery",
    "microbiome antibiotics longitudinal adults",
    "infant microbiome antibiotic exposure",
    "metagenomics IBD remission",
]

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="auto", choices=["auto","cpu","cuda"])
    ap.add_argument("--lit-batch", type=int, default=32)
    ap.add_argument("--init-rate-data", type=float, default=4.0, help="topics/sec to data worker")
    ap.add_argument("--init-rate-lit",  type=float, default=2.0, help="topics/sec to literature worker")
    ap.add_argument("--adjust-every", type=float, default=3.0, help="seconds between control updates")
    ap.add_argument("--out", default=os.path.expanduser("~/research_project/run/matches.jsonl"))
    return ap.parse_args()

def start_process(cmd, env=None):
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, env=env, bufsize=1)

class TokenBucket:
    def __init__(self, rate_per_s, burst=20.0):
        self.rate = float(rate_per_s); self.capacity = float(burst)
        self.tokens = float(burst); self.ts = time.perf_counter()
    def allow(self, n=1.0):
        now = time.perf_counter(); dt = now - self.ts; self.ts = now
        self.tokens = min(self.capacity, self.tokens + dt*self.rate)
        if self.tokens >= n: self.tokens -= n; return True
        return False
    def set_rate(self, r): self.rate = max(0.1, float(r))

def forward(src_proc, name, match_stdin, counters, last_ts_map):
    for line in src_proc.stdout:
        s = line.strip()
        if not s: continue
        print(f"[{name}] {s}")
        try:
            obj = json.loads(s); t = obj.get("type","")
            if t == "data_result":
                counters["data_out"] += 1; last_ts_map["data"] = time.perf_counter()
            elif t == "lit_result":
                counters["lit_out"]  += 1; last_ts_map["lit"]  = time.perf_counter()
            match_stdin.write(line); match_stdin.flush()
        except Exception:
            pass

def main():
    args = parse_args()
    base = os.path.dirname(__file__)
    run_dir = os.path.expanduser("~/research_project/run")
    os.makedirs(run_dir, exist_ok=True)

    env_lit = os.environ.copy()
    if args.gpu == "cpu":
        env_lit["CUDA_VISIBLE_DEVICES"] = ""; dev_flag = "cpu"
    elif args.gpu == "cuda":
        dev_flag = "cuda"
    else:
        dev_flag = "auto"

    p_data = start_process(["python", os.path.join(base, "data_worker.py")])
    p_lit  = start_process(["python", os.path.join(base, "lit_gpu_worker.py"),
                            "--device", dev_flag, "--batch-size", str(args.lit_batch)], env=env_lit)
    p_match= start_process(["python", os.path.join(base, "matcher.py"),
                            "--out", args.out])

    counters = collections.Counter(data_in=0, lit_in=0, data_out=0, lit_out=0)
    last_ts  = {"data": time.perf_counter(), "lit": time.perf_counter()}

    t_data = threading.Thread(target=forward, args=(p_data,"DATA", p_match.stdin, counters, last_ts), daemon=True)
    t_lit  = threading.Thread(target=forward, args=(p_lit, "LIT",  p_match.stdin, counters, last_ts), daemon=True)
    t_data.start(); t_lit.start()

    bucket_data = TokenBucket(rate_per_s=args.init_rate_data, burst=50.0)
    bucket_lit  = TokenBucket(rate_per_s=args.init_rate_lit,  burst=50.0)

    topics = list(TOPICS); i_data = 0; i_lit = 0
    last_adjust = time.perf_counter()
    print(f"[ORCH] starting feeder with data_rate={bucket_data.rate:.2f}/s, lit_rate={bucket_lit.rate:.2f}/s")

    while (i_data < len(topics)) or (i_lit < len(topics)):
        now = time.perf_counter()
        if i_data < len(topics) and bucket_data.allow(1.0):
            msg = json.dumps({"topic": topics[i_data]}) + "\n"
            try: p_data.stdin.write(msg); p_data.stdin.flush(); counters["data_in"] += 1; i_data += 1
            except BrokenPipeError: pass
        if i_lit < len(topics) and bucket_lit.allow(1.0):
            msg = json.dumps({"topic": topics[i_lit]}) + "\n"
            try: p_lit.stdin.write(msg); p_lit.stdin.flush(); counters["lit_in"] += 1; i_lit += 1
            except BrokenPipeError: pass

        if now - last_adjust >= args.adjust_every:
            last_adjust = now
            data_rate_obs = counters["data_out"] / max(1e-6, args.adjust_every)
            lit_rate_obs  = counters["lit_out"]  / max(1e-6, args.adjust_every)
            counters["data_out"] = 0; counters["lit_out"] = 0

            if data_rate_obs > 1.3 * lit_rate_obs:
                bucket_data.set_rate(bucket_data.rate * 0.85)
                bucket_lit.set_rate(min(bucket_lit.rate * 1.15, bucket_data.rate * 1.1 + 0.5))
            elif lit_rate_obs > 1.3 * data_rate_obs:
                bucket_lit.set_rate(bucket_lit.rate * 0.85)
                bucket_data.set_rate(min(bucket_data.rate * 1.15, bucket_lit.rate * 1.1 + 0.5))

            if data_rate_obs > 0.2 and lit_rate_obs > 0.2:
                bucket_data.set_rate(bucket_data.rate * 1.05)
                bucket_lit.set_rate(bucket_lit.rate * 1.05)

            print(f"[ORCH] adjust: data_rate≈{data_rate_obs:.2f}/s lit_rate≈{lit_rate_obs:.2f}/s "
                  f"→ feeders data={bucket_data.rate:.2f}/s lit={bucket_lit.rate:.2f}/s")
        time.sleep(0.01)

    for proc in (p_data, p_lit):
        try: proc.stdin.write("SHUTDOWN\n"); proc.stdin.flush()
        except Exception: pass

    p_data.wait(); p_lit.wait()
    for done_type in ("data_done","lit_done"):
        try: p_match.stdin.write(json.dumps({"type": done_type}) + "\n"); p_match.stdin.flush()
        except Exception: pass
    time.sleep(0.2)
    try: p_match.stdin.close()
    except Exception: pass
    p_match.wait()
    print(f"[ORCH] Done. Results at {args.out}")

if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt: pass
