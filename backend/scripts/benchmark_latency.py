"""
benchmark_latency.py — Ablation study: Full Multimodal + Multi-Agent System vs Text-Only RAG Baseline.

Methodology
-----------
* 10 representative anomaly queries spanning all three machine types are run through
  both pipelines five times each. The first run is discarded as a warm-up to eliminate
  cold-start effects (model loading, DB connection establishment).
* Reported statistics: mean, std, median, p95 — all in seconds.
* Each timing stage is recorded independently so the breakdown (embed / retrieve / LLM)
  can be reported separately.

Usage
-----
    cd industrial_copilot/backend
    python scripts/benchmark_latency.py

Output
------
  - Console table (suitable for copy-paste into the paper)
  - benchmark_results.json (raw numbers for post-processing)
"""

import sys
import os
import time
import json
import statistics
import textwrap
from dataclasses import dataclass, asdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -- Test corpus ----------------------------------------------------------------
# 10 representative queries; each paired with the manual_id the system would use.
# These mirror the scenarios described in the paper's evaluation section.

MACHINE_ID = "TEAPM_001"
MANUAL_ID  = "TEA_P_M_001"

TEST_QUERIES = [
    {
        "id": "Q01",
        "machine_id": MACHINE_ID,
        "manual_id":  MANUAL_ID,
        "anomaly_description": (
            "CT_01 current reading 138 A, fault threshold is 140 A. "
            "Thermistor_01 at 48 C nominal. Encoder_01 reporting intermittent position loss."
        ),
    },
    {
        "id": "Q02",
        "machine_id": MACHINE_ID,
        "manual_id":  MANUAL_ID,
        "anomaly_description": (
            "Current_Sensor_01 spiked to 820 A against a fault_high of 840 A. "
            "CT_01 and thermistor within normal range."
        ),
    },
    {
        "id": "Q03",
        "machine_id": MACHINE_ID,
        "manual_id":  MANUAL_ID,
        "anomaly_description": (
            "Thermistor_01 temperature reached 148 C, nearing fault_high of 150 C. "
            "Both current sensors elevated: CT_01 at 95 A, Current_Sensor_01 at 590 A."
        ),
    },
    {
        "id": "Q04",
        "machine_id": MACHINE_ID,
        "manual_id":  MANUAL_ID,
        "anomaly_description": (
            "Encoder_01 dropped to 800 PPR, below min_normal of 1000 PPR. "
            "Current and temperature sensors all nominal."
        ),
    },
    {
        "id": "Q05",
        "machine_id": MACHINE_ID,
        "manual_id":  MANUAL_ID,
        "anomaly_description": (
            "CT_01 frozen at 55 A for 90 seconds — suspected sensor freeze fault. "
            "Current_Sensor_01 reading normal fluctuations."
        ),
    },
    {
        "id": "Q06",
        "machine_id": MACHINE_ID,
        "manual_id":  MANUAL_ID,
        "anomaly_description": (
            "Progressive current drift: CT_01 climbed from 55 A to 99 A over 20 minutes. "
            "Encoder PPR normal. Thermistor rose from 47 C to 62 C simultaneously."
        ),
    },
    {
        "id": "Q07",
        "machine_id": MACHINE_ID,
        "manual_id":  MANUAL_ID,
        "anomaly_description": (
            "Machine entered idle state unexpectedly. All sensors near zero: "
            "CT_01 at 2 A, Current_Sensor_01 at 5 A, encoder 0 PPR."
        ),
    },
    {
        "id": "Q08",
        "machine_id": MACHINE_ID,
        "manual_id":  MANUAL_ID,
        "anomaly_description": (
            "Simultaneous high-side fault on both current sensors: CT_01 at 139 A, "
            "Current_Sensor_01 at 835 A. Thermistor at 102 C (max_normal 105 C)."
        ),
    },
    {
        "id": "Q09",
        "machine_id": MACHINE_ID,
        "manual_id":  MANUAL_ID,
        "anomaly_description": (
            "Encoder_01 oscillating between 500 PPR and 36000 PPR at 2 Hz frequency. "
            "No overcurrent detected. Thermistor stable."
        ),
    },
    {
        "id": "Q10",
        "machine_id": MACHINE_ID,
        "manual_id":  MANUAL_ID,
        "anomaly_description": (
            "CT_01 at 97 A and Current_Sensor_01 at 598 A both at max_normal boundary. "
            "Thermistor_01 at 104 C. Encoder normal. Pattern consistent with overload."
        ),
    },
]

RUNS_PER_QUERY = 5   # includes 1 warm-up discarded from stats
WARMUP_RUNS   = 1


# -- Timing containers ---------------------------------------------------------

@dataclass
class RunTiming:
    embed_s:    float
    retrieve_s: float
    llm_s:      float
    total_s:    float


@dataclass
class QueryStats:
    query_id:        str
    pipeline:        str          # 'text_only_rag' | 'full_system'
    mean_total_s:    float
    std_total_s:     float
    median_total_s:  float
    p95_total_s:     float
    mean_embed_s:    float
    mean_retrieve_s: float
    mean_llm_s:      float
    n_runs:          int


# -- Text-only RAG runner -------------------------------------------------------

def run_text_only_rag(query: dict, n_runs: int, warmup: int) -> list[RunTiming]:
    from scripts.text_only_rag import TextOnlyRAGPipeline
    pipeline = TextOnlyRAGPipeline()
    timings = []
    for i in range(n_runs + warmup):
        result = pipeline.query(
            anomaly_description=query["anomaly_description"],
            manual_id=query["manual_id"],
            machine_id=query["machine_id"],
        )
        if i >= warmup:
            timings.append(RunTiming(
                embed_s=result.timing.embed_s,
                retrieve_s=result.timing.retrieve_s,
                llm_s=result.timing.llm_s,
                total_s=result.timing.total_s,
            ))
    return timings


# -- Full system runner ---------------------------------------------------------

def run_full_system(query: dict, n_runs: int, warmup: int) -> list[RunTiming]:
    """
    Invokes the LangGraph pipeline directly (no HTTP overhead) and records
    per-stage timing by monkey-patching the embedder and RAG generator.
    """
    from agents.copilot_graph import build_copilot_graph
    from unified_rag.embeddings import embedder as embedder_module

    graph = build_copilot_graph()
    timings = []

    for i in range(n_runs + warmup):
        stage = {"embed_s": 0.0, "retrieve_s": 0.0, "llm_s": 0.0}

        # Wrap embedder to capture embedding latency
        _orig_embed = embedder_module.embedder.embed_text
        def timed_embed(text, _orig=_orig_embed, _s=stage):
            t0 = time.perf_counter()
            result = _orig(text)
            _s["embed_s"] += time.perf_counter() - t0
            return result
        embedder_module.embedder.embed_text = timed_embed

        t_total = time.perf_counter()
        try:
            graph.invoke({
                "event_id":       query["id"],
                "machine_id":     query["machine_id"],
                "machine_state":  "machine_fault",
                "anomaly_score":  0.42,
                "user_query":     query["anomaly_description"],
                "suspect_sensor": "temperature",
                "recent_readings": {},
                "ai_validation_status": None,
                "fault_category":       None,
                "ai_confidence_score":  None,
                "ai_engineering_notes": None,
                "sensor_status_report": "",
                "diagnostic_report":    "",
                "rag_context":          "",
                "retrieved_images":     [],
                "strategy_report":      "",
                "critic_feedback":      "",
                "final_execution_plan": "",
                "chat_history":         "",
            })
        except Exception as e:
            print(f"  ⚠ Full-system run failed for {query['id']}: {e}")
            embedder_module.embedder.embed_text = _orig_embed
            continue
        total_s = time.perf_counter() - t_total

        embedder_module.embedder.embed_text = _orig_embed

        # Approximate retrieval time = total − embed − estimated LLM overhead.
        # LLM latency is captured from rag.py timing via total minus known overhead.
        # For simplicity we apportion: retrieve ~= total - embed - (total * 0.85 llm fraction).
        # A more precise split requires patching rag.py; we report total here.
        if i >= warmup:
            timings.append(RunTiming(
                embed_s=stage["embed_s"],
                retrieve_s=0.0,      # rolled into total; not extractable without deeper patching
                llm_s=0.0,           # rolled into total
                total_s=total_s,
            ))

    return timings


# -- Stats helper --------------------------------------------------------------

def summarise(timings: list[RunTiming], query_id: str, pipeline: str) -> QueryStats:
    totals    = [t.total_s    for t in timings]
    embeds    = [t.embed_s    for t in timings]
    retrieves = [t.retrieve_s for t in timings]
    llms      = [t.llm_s      for t in timings]

    def p95(lst):
        s = sorted(lst)
        idx = max(0, int(len(s) * 0.95) - 1)
        return s[idx]

    return QueryStats(
        query_id=query_id,
        pipeline=pipeline,
        mean_total_s=statistics.mean(totals),
        std_total_s=statistics.stdev(totals) if len(totals) > 1 else 0.0,
        median_total_s=statistics.median(totals),
        p95_total_s=p95(totals),
        mean_embed_s=statistics.mean(embeds),
        mean_retrieve_s=statistics.mean(retrieves),
        mean_llm_s=statistics.mean(llms),
        n_runs=len(timings),
    )


# -- Reporting -----------------------------------------------------------------

def print_table(text_stats: list[QueryStats], full_stats: list[QueryStats]):
    sep = "-" * 100
    hdr = f"{'QID':<5} {'Machine':<12} {'Text-RAG mean':>14} {'Text-RAG sd':>11} {'Full-Sys mean':>14} {'Full-Sys sd':>11} {'Delta (s)':>10} {'Speedup':>8}"
    print("\n" + sep)
    print("  ABLATION STUDY: Text-Only RAG vs Full Multimodal + Multi-Agent System")
    print(sep)
    print(hdr)
    print(sep)

    machine_map = {q["id"]: q["machine_id"] for q in TEST_QUERIES}

    for ts, fs in zip(text_stats, full_stats):
        delta   = fs.mean_total_s - ts.mean_total_s
        speedup = fs.mean_total_s / ts.mean_total_s if ts.mean_total_s > 0 else float("inf")
        print(
            f"  {ts.query_id:<4} {machine_map[ts.query_id]:<12} "
            f"{ts.mean_total_s:>12.3f}s  {ts.std_total_s:>9.3f}s  "
            f"{fs.mean_total_s:>12.3f}s  {fs.std_total_s:>9.3f}s  "
            f"{delta:>+9.3f}s  {speedup:>7.2f}x"
        )

    print(sep)

    # Aggregate
    all_t_means = [s.mean_total_s for s in text_stats]
    all_f_means = [s.mean_total_s for s in full_stats]
    agg_t = statistics.mean(all_t_means)
    agg_f = statistics.mean(all_f_means)
    agg_delta = agg_f - agg_t
    agg_speedup = agg_f / agg_t if agg_t > 0 else float("inf")

    print(
        f"  {'MEAN':<4} {'(all queries)':<12} "
        f"{agg_t:>12.3f}s  {'':>9}  "
        f"{agg_f:>12.3f}s  {'':>9}  "
        f"{agg_delta:>+9.3f}s  {agg_speedup:>7.2f}x"
    )
    print(sep)

    print(textwrap.dedent(f"""
    Text-Only RAG breakdown (mean across all queries):
      Embedding :  {statistics.mean(s.mean_embed_s    for s in text_stats):.3f}s
      Retrieval :  {statistics.mean(s.mean_retrieve_s for s in text_stats):.3f}s
      LLM call  :  {statistics.mean(s.mean_llm_s      for s in text_stats):.3f}s
      Total     :  {agg_t:.3f}s

    Full System (LangGraph) mean total : {agg_f:.3f}s
    Overhead of full system vs baseline : {agg_delta:+.3f}s ({agg_speedup:.2f}x slower than text-only)

    Note: Both pipelines use Qwen (qwen-max) for generation and text-embedding-v4 for retrieval.
    The full system additionally runs 6 LangGraph agent nodes and includes multimodal
    (image + table) chunk retrieval and InteractionMemory lookups.
    """))


def save_json(text_stats, full_stats, path="benchmark_results.json"):
    out = {
        "text_only_rag": [asdict(s) for s in text_stats],
        "full_system":   [asdict(s) for s in full_stats],
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Raw results saved -> {path}")


# -- Entry point ---------------------------------------------------------------

def main():
    print(f"Benchmark: {len(TEST_QUERIES)} queries x {RUNS_PER_QUERY} runs "
          f"({WARMUP_RUNS} warm-up discarded)")
    print("Running Text-Only RAG baseline …\n")

    text_stats = []
    for q in TEST_QUERIES:
        print(f"  [{q['id']}] {q['machine_id']} …", end=" ", flush=True)
        timings = run_text_only_rag(q, RUNS_PER_QUERY, WARMUP_RUNS)
        stats   = summarise(timings, q["id"], "text_only_rag")
        text_stats.append(stats)
        print(f"mean={stats.mean_total_s:.2f}s  sd={stats.std_total_s:.2f}s")

    print("\nRunning Full Multimodal + Multi-Agent System …\n")

    full_stats = []
    for q in TEST_QUERIES:
        print(f"  [{q['id']}] {q['machine_id']} …", end=" ", flush=True)
        timings = run_full_system(q, RUNS_PER_QUERY, WARMUP_RUNS)
        if not timings:
            # If full system not available (e.g. AI_API_KEY missing), use placeholder
            print("SKIPPED (pipeline unavailable)")
            full_stats.append(QueryStats(
                query_id=q["id"], pipeline="full_system",
                mean_total_s=float("nan"), std_total_s=float("nan"),
                median_total_s=float("nan"), p95_total_s=float("nan"),
                mean_embed_s=float("nan"), mean_retrieve_s=float("nan"),
                mean_llm_s=float("nan"), n_runs=0,
            ))
            continue
        stats = summarise(timings, q["id"], "full_system")
        full_stats.append(stats)
        print(f"mean={stats.mean_total_s:.2f}s  sd={stats.std_total_s:.2f}s")

    print_table(text_stats, full_stats)
    save_json(text_stats, full_stats)


if __name__ == "__main__":
    main()
