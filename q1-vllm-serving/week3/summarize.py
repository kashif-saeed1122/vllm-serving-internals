"""
summarize.py -- read the Week 3 results folder and print two markdown tables.

    python q1-vllm-serving/week3/summarize.py results/week3

Reads
    results/week3/*.json            one per saved benchmark run
    results/week3/metrics.log       from metrics-sampler.sh          (optional)
    results/week3/week3-sweep.log   the sweep's tee output, for the RUN markers (optional)

Prints
    1. one row per benchmark JSON: TTFT / ITL / TPOT / throughput
    2. one row per run window from metrics.log: peak running, peak waiting, peak KV
       usage, KV usage at the FIRST moment waiting > 0, preemptions, prefix hit rate
"""
import glob
import json
import os
import sys

folder = sys.argv[1] if len(sys.argv) > 1 else "results/week3"


def num(value, digits):
    """Format a number, or '-' when the field is missing from the JSON."""
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


# ------------------------------------------------------------------ 1. JSONs ----
print("## Benchmark runs\n")
print("| run | completed | TTFT mean ms | TTFT p50 | TTFT p95 | TTFT p99 "
      "| ITL mean ms | TPOT mean ms | req/s | output tok/s |")
print("|---|---|---|---|---|---|---|---|---|---|")

for path in sorted(glob.glob(os.path.join(folder, "*.json"))):
    with open(path) as f:
        r = json.load(f)
    if "mean_ttft_ms" not in r:          # e.g. models.json -- not a benchmark result
        continue
    name = os.path.basename(path)[:-len(".json")]
    cells = [
        name,
        str(r.get("completed", "-")),
        num(r.get("mean_ttft_ms"), 2),
        num(r.get("p50_ttft_ms"), 2),
        num(r.get("p95_ttft_ms"), 2),
        num(r.get("p99_ttft_ms"), 2),
        num(r.get("mean_itl_ms"), 3),
        num(r.get("mean_tpot_ms"), 3),
        num(r.get("request_throughput"), 3),
        num(r.get("output_throughput"), 2),
    ]
    print("| " + " | ".join(cells) + " |")

print("\nWeek 2 concurrency-20 for comparison: TTFT mean 71.39 ms, ITL mean 12.033 ms, "
      "6.333 req/s, 1621.29 output tok/s. Week 2 concurrency-1 TTFT mean: 98.41 ms.")


# ------------------------------------------------------------ 2. metrics.log ----
log_path = os.path.join(folder, "metrics.log")
if not os.path.exists(log_path):
    print("\n(no metrics.log in this folder -- skipping the /metrics summary)")
    sys.exit(0)

# Each sample is a dict like {"ts": 1757340000, "running": 16.0, "waiting": 0.0, ...}
samples = []
current = None
with open(log_path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.isdigit():                     # a bare timestamp starts a new sample
            current = {"ts": int(line)}
            samples.append(current)
            continue
        if current is None or not line.startswith("vllm:"):
            continue
        name = line[len("vllm:"):].split("{")[0].split(" ")[0]
        if name.endswith("_created"):          # prometheus bookkeeping line, not data
            continue
        value = float(line.split()[-1])        # the value is always the last field
        if name in ("kv_cache_usage_perc", "gpu_cache_usage_perc"):
            current["kv_usage"] = value
        elif name.startswith("num_preemptions"):
            current["preemptions"] = value
        elif name.startswith("prefix_cache_queries"):
            current["prefix_queries"] = value
        elif name.startswith("prefix_cache_hits"):
            current["prefix_hits"] = value
        elif name == "num_requests_running":
            current["running"] = value
        elif name == "num_requests_waiting":
            current["waiting"] = value

# Run windows come from the "RUN <name>" + timestamp pairs the sweep printed.
runs = []                                       # list of (name, start_ts)
sweep_log = os.path.join(folder, "week3-sweep.log")
if os.path.exists(sweep_log):
    pending = None
    with open(sweep_log) as f:
        for line in f:
            line = line.strip()
            if line.startswith("RUN ") or line.startswith("SWEEP COMPLETE"):
                pending = line.replace("RUN ", "")
            elif pending is not None and line.isdigit():
                runs.append((pending, int(line)))
                pending = None
if not runs:
    runs = [("whole session", 0)]


def summarize(rows):
    """Peak values, counter deltas, and the KV usage at the first waiting > 0."""
    def col(key):
        return [s[key] for s in rows if key in s]

    out = {
        "n": len(rows),
        "max_running": max(col("running") or [0]),
        "max_waiting": max(col("waiting") or [0]),
        "max_kv": max(col("kv_usage") or [0]),
    }
    pre = col("preemptions")
    out["preemptions"] = pre[-1] - pre[0] if len(pre) > 1 else 0
    q, h = col("prefix_queries"), col("prefix_hits")
    if len(q) > 1 and len(h) > 1 and q[-1] - q[0] > 0:
        out["hit_rate"] = (h[-1] - h[0]) / (q[-1] - q[0]) * 100
    else:
        out["hit_rate"] = None
    first = next((s for s in rows if s.get("waiting", 0) > 0), None)
    out["first_wait_ts"] = first["ts"] if first else None
    out["kv_at_first_wait"] = first.get("kv_usage") if first else None
    return out


print("\n## Scheduler state per run (from metrics.log)\n")
print("| run | samples | peak running | peak waiting | peak KV usage % "
      "| KV usage % at first waiting>0 | preemptions in run | prefix hit % |")
print("|---|---|---|---|---|---|---|---|")

for i, (name, start) in enumerate(runs):
    end = runs[i + 1][1] if i + 1 < len(runs) else float("inf")
    if name.startswith("SWEEP COMPLETE"):
        continue
    rows = [s for s in samples if start <= s["ts"] < end]
    if not rows:
        continue
    s = summarize(rows)
    kv_first = "-" if s["kv_at_first_wait"] is None else f"{s['kv_at_first_wait'] * 100:.1f} (ts {s['first_wait_ts']})"
    cells = [
        name,
        str(s["n"]),
        num(s["max_running"], 0),
        num(s["max_waiting"], 0),
        num(s["max_kv"] * 100, 1),
        kv_first,
        num(s["preemptions"], 0),
        num(s["hit_rate"], 1),
    ]
    print("| " + " | ".join(cells) + " |")

print("""
How to read the "KV usage at first waiting>0" column:
  LOW  (under ~30%)                      -> T2: the per-step token budget (max_num_batched_tokens)
                                            bound first. NOT the memory ceiling.
  HIGH (near 100%) AND preemptions > 0   -> T3: KV blocks ran out. THIS is the memory ceiling.
  peak running flat at max_num_seqs, KV low -> T1: the sequence cap bound first.
Compare against the values in capacity-config.txt. Preemptions are the honest ceiling
signal: high KV usage alone can be prefix-cache retention, which is evicted on demand.
""")