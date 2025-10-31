#!/usr/bin/env python3
import os, sys, json, time, argparse, asyncio, threading
from collections import deque
from typing import List

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--threads", type=int, default=int(os.getenv("OMP_NUM_THREADS", "1")))
    ap.add_argument("--default-rate", type=float, default=2.0)
    ap.add_argument("--report-interval", type=float, default=2.0)
    return ap.parse_args()

# -----------------------------
# Token bucket limiter
# -----------------------------
class TokenBucket:
    def __init__(self, rate_per_s: float, burst: int):
        self.rate = float(rate_per_s)
        self.capacity = max(1, int(burst))
        self.tokens = float(self.capacity)
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
# Runtime helpers
# -----------------------------
def apply_cpu_affinity(cores: List[int]):
    try:
        if cores:
            os.sched_setaffinity(0, set(int(c) for c in cores))
    except Exception:
        pass

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

def pick_device(flag: str) -> str:
    if flag in ("cpu", "cuda"):
        return flag
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

# -----------------------------
# Model loader (optional GPU)
# -----------------------------
class Encoder:
    def __init__(self, model_name: str, device: str):
        self.device = device
        self.dim = 384
        self._impl = None
        # try to load sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self._impl = SentenceTransformer(model_name, device=device)
            # probe for embedding dimension
            emb = self._impl.encode(["probe"], batch_size=1, convert_to_tensor=True, show_progress_bar=False, device=device)
            try:
                self.dim = int(emb.shape[1])
            except Exception:
                pass
        except Exception:
            self._impl = None

        # best effort soft VRAM cap is set via quota handler below

    def encode(self, texts: List[str], batch_size: int) -> int:
        """
        Returns embedding dimension (int). Side effect is only to compute encodings.
        """
        if not texts:
            return self.dim
        if self._impl is None:
            # simulate compute if model not present
            time.sleep(0.01 * len(texts))
            return self.dim
        embs = self._impl.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
            device=self.device,
        )
        try:
            return int(embs.shape[1])
        except Exception:
            return self.dim

# -----------------------------
# Main worker
# -----------------------------
async def main():
    args = parse_args()

    device = pick_device(args.device)
    set_threads(args.threads)
    apply_cpu_affinity(list(range(args.threads)))

    # try to apply a default GPU memory fraction; will be overridden by quota
    def set_gpu_mem_frac(frac: float):
        try:
            import torch
            if device == "cuda":
                torch.cuda.set_per_process_memory_fraction(min(max(float(frac), 0.05), 0.95))
        except Exception:
            pass

    set_gpu_mem_frac(0.35)

    encoder = Encoder(args.model, device)

    # dynamic quota state
    state = {
        "max_batch": max(1, int(args.batch_size)),
        "token_rate": float(args.default_rate),
        "threads": int(args.threads),
        "cpu_cores": list(range(args.threads)),
        "gpu_mem_frac": 0.35,
    }
    bucket = TokenBucket(rate_per_s=state["token_rate"], burst=state["max_batch"])

    pending: List[str] = []
    completed_times = deque(maxlen=1000)
    last_batch_latency_ms = None

    # metrics reporter
    def reporter():
        while True:
            time.sleep(args.report_interval)
            now = time.time()
            while completed_times and now - completed_times[0] > 10.0:
                completed_times.popleft()
            rps = len(completed_times) / 10.0
            metrics = {
                "rps": rps,
                "queue": len(pending),
                "latency_hint_ms": last_batch_latency_ms,
                "device": device,
                "max_batch": state["max_batch"],
                "threads": state["threads"],
                "token_rate": state["token_rate"],
            }
            print(json.dumps({"type": "metrics", "payload": metrics}), flush=True)

    threading.Thread(target=reporter, daemon=True).start()

    # async stdin reader
    loop = asyncio.get_event_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)

    # processing loop
    while True:
        line = await reader.readline()
        if not line:
            break
        msg = line.decode().strip()
        if not msg:
            continue

        if msg == "SHUTDOWN":
            # flush remaining work quickly
            if pending:
                batch = pending[:]
                pending.clear()
                t0 = time.perf_counter()
                dim = encoder.encode([f"abstract for {t}" for t in batch], batch_size=min(state["max_batch"], len(batch)))
                for t in batch:
                    payload = {
                        "topic": t,
                        "counts": [120, 140, 155, 170, 182],
                        "significance_score": 0.80,
                        "top_terms": ["diversity", "antibiotics", "IBD"],
                        "emb_dim": int(dim),
                    }
                    print(json.dumps({"type": "lit_result", "payload": payload}), flush=True)
                    completed_times.append(time.time())
                last_batch_latency_ms = int((time.perf_counter() - t0) * 1000)
            print(json.dumps({"type": "lit_done"}), flush=True)
            break

        # parse JSON
        try:
            obj = json.loads(msg)
        except Exception:
            continue

        typ = obj.get("type")

        # quota update
        if typ == "quota":
            q = obj.get("payload", {})
            if "max_batch" in q:
                state["max_batch"] = int(max(1, q["max_batch"]))
                bucket.capacity = state["max_batch"]
            if "token_rate" in q:
                state["token_rate"] = float(q["token_rate"])
                bucket.set_rate(state["token_rate"])
            if "threads" in q:
                state["threads"] = int(max(1, q["threads"]))
                set_threads(state["threads"])
            if "cpu_cores" in q:
                state["cpu_cores"] = list(q["cpu_cores"])
                apply_cpu_affinity(state["cpu_cores"])
            if "gpu_mem_frac" in q:
                state["gpu_mem_frac"] = float(q["gpu_mem_frac"])
                set_gpu_mem_frac(state["gpu_mem_frac"])
            continue

        # normal work item
        topic = obj.get("topic", "")
        if not topic:
            continue
        pending.append(topic)

        # drain pending with rate limit and batching
        while pending:
            want = min(len(pending), state["max_batch"])
            got = bucket.take(want)
            if got <= 0:
                await asyncio.sleep(0.01)
                break

            batch = [pending.pop(0) for _ in range(got)]
            t0 = time.perf_counter()
            dim = encoder.encode([f"abstract for {t}" for t in batch], batch_size=min(state["max_batch"], got))
            for t in batch:
                payload = {
                    "topic": t,
                    "counts": [120, 140, 155, 170, 182],
                    "significance_score": 0.80,
                    "top_terms": ["diversity", "antibiotics", "IBD"],
                    "emb_dim": int(dim),
                }
                print(json.dumps({"type": "lit_result", "payload": payload}), flush=True)
                completed_times.append(time.time())
            last_batch_latency_ms = int((time.perf_counter() - t0) * 1000)

    # final guard
    try:
        if pending:
            batch = pending[:]
            pending.clear()
            t0 = time.perf_counter()
            dim = encoder.encode([f"abstract for {t}" for t in batch], batch_size=min(state["max_batch"], len(batch)))
            for t in batch:
                payload = {
                    "topic": t,
                    "counts": [120, 140, 155, 170, 182],
                    "significance_score": 0.80,
                    "top_terms": ["diversity", "antibiotics", "IBD"],
                    "emb_dim": int(dim),
                }
                print(json.dumps({"type": "lit_result", "payload": payload}), flush=True)
                completed_times.append(time.time())
            last_batch_latency_ms = int((time.perf_counter() - t0) * 1000)
        print(json.dumps({"type": "lit_done"}), flush=True)
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