#!/usr/bin/env python3
import os, sys, json, time, argparse, asyncio, threading
from collections import deque

# -----------------------------
# Argument parsing
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=int(os.getenv("OMP_NUM_THREADS", "1")))
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--default-rate", type=float, default=4.0)
    ap.add_argument("--report-interval", type=float, default=2.0)
    return ap.parse_args()

# -----------------------------
# Token bucket rate limiter
# -----------------------------
class TokenBucket:
    def __init__(self, rate_per_s: float, burst: int):
        self.rate = float(rate_per_s)
        self.capacity = float(burst)
        self.tokens = float(burst)
        self.ts = time.perf_counter()
        self.lock = threading.Lock()

    def set_rate(self, rate: float):
        with self.lock:
            self.rate = max(0.1, float(rate))

    def take(self, n=1) -> int:
        with self.lock:
            now = time.perf_counter()
            dt = now - self.ts
            self.ts = now
            self.tokens = min(self.capacity, self.tokens + dt * self.rate)
            got = int(min(self.tokens, n))
            self.tokens -= got
            return got

# -----------------------------
# Simulated data fetch
# -----------------------------
async def fetch_summary(topic: str) -> dict:
    # Simulate network / data lookup latency
    await asyncio.sleep(0.15)
    return {
        "topic": topic,
        "datasets": ["Qiita:123", "MGnify:ABC"],
        "variables": {"age": 1200, "antibiotics": 900, "body_site": 1500},
        "samples": 2000,
        "feasibility_score": 0.78,
    }

# -----------------------------
# Main worker logic
# -----------------------------
async def main():
    args = parse_args()

    # dynamic quotas
    state = {
        "max_batch": 32,
        "token_rate": args.default-rate if hasattr(args, "default-rate") else args.default_rate,
        "threads": args.threads,
        "cpu_cores": list(range(args.threads)),
    }

    # initialize rate limiter
    bucket = TokenBucket(rate_per_s=state["token_rate"], burst=state["max_batch"])

    pending = []
    completed_times = deque(maxlen=1000)
    last_batch_latency_ms = None

    # -----------------------------
    # Reporter thread (for metrics)
    # -----------------------------
    def reporter():
        while True:
            time.sleep(args.report_interval)
            now = time.time()
            # calculate recent throughput
            while completed_times and now - completed_times[0] > 10.0:
                completed_times.popleft()
            rps = len(completed_times) / 10.0
            metrics = {
                "rps": rps,
                "queue": len(pending),
                "latency_hint_ms": last_batch_latency_ms,
            }
            print(json.dumps({"type": "metrics", "payload": metrics}), flush=True)

    threading.Thread(target=reporter, daemon=True).start()

    # -----------------------------
    # Async stdin loop
    # -----------------------------
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    while True:
        line = await reader.readline()
        if not line:
            break
        msg = line.decode().strip()
        if not msg:
            continue

        if msg == "SHUTDOWN":
            # flush remaining topics
            if pending:
                results = await asyncio.gather(*(fetch_summary(t) for t in pending))
                for result in results:
                    print(json.dumps({"type": "data_result", "payload": result}), flush=True)
            print(json.dumps({"type": "data_done"}), flush=True)
            break

        try:
            obj = json.loads(msg)
        except Exception:
            continue

        typ = obj.get("type")

        # Handle quota updates from orchestrator
        if typ == "quota":
            q = obj.get("payload", {})
            if "max_batch" in q:
                state["max_batch"] = int(max(1, q["max_batch"]))
                bucket.capacity = state["max_batch"]
            if "token_rate" in q:
                state["token_rate"] = float(q["token_rate"])
                bucket.set_rate(state["token_rate"])
            continue

        # Handle incoming work item
        topic = obj.get("topic")
        if not topic:
            continue
        pending.append(topic)

        # Try to process items subject to rate limits
        while pending:
            n = min(len(pending), state["max_batch"])
            got = bucket.take(n)
            if got <= 0:
                await asyncio.sleep(0.01)
                break

            batch = [pending.pop(0) for _ in range(got)]
            t0 = time.perf_counter()
            results = await asyncio.gather(*(fetch_summary(t) for t in batch))
            for r in results:
                print(json.dumps({"type": "data_result", "payload": r}), flush=True)
                completed_times.append(time.time())
            last_batch_latency_ms = int((time.perf_counter() - t0) * 1000)

    # graceful cleanup
    try:
        if pending:
            results = await asyncio.gather(*(fetch_summary(t) for t in pending))
            for r in results:
                print(json.dumps({"type": "data_result", "payload": r}), flush=True)
                completed_times.append(time.time())
        print(json.dumps({"type": "data_done"}), flush=True)
    except Exception:
        pass

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass