"""
quick_rag_benchmark.py - Standalone text-only RAG latency benchmark.

No project imports. Reads .env directly. Queries manual_chunks table (type='text')
with pgvector cosine search, then calls Qwen. Reports per-stage timing for
the ablation study comparison in the research paper.

Usage:
    cd industrial_copilot/backend
    python scripts/quick_rag_benchmark.py
"""

import os, sys, time, json, statistics
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# -- Load .env manually (no dotenv dependency needed) --------------------------
def load_env(path=".env"):
    env = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, _, v = line.partition("=")
                    env[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return env

_env = load_env()
DATABASE_URL = _env.get("DATABASE_URL") or os.environ.get("DATABASE_URL", "")
AI_API_KEY   = _env.get("AI_API_KEY") or os.environ.get("AI_API_KEY", "")
AI_BASE_URL  = _env.get("AI_BASE_URL") or os.environ.get("AI_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")

if not DATABASE_URL or not AI_API_KEY:
    sys.exit("ERROR: DATABASE_URL or AI_API_KEY missing in .env")

# -- Imports -------------------------------------------------------------------
try:
    import openai
    from sqlalchemy import create_engine, text as sa_text
    from sqlalchemy.orm import sessionmaker
    from pgvector.sqlalchemy import Vector
except ImportError as e:
    sys.exit(f"Missing package: {e}\nRun: pip install openai sqlalchemy pgvector psycopg2-binary")

# -- Config --------------------------------------------------------------------
MANUAL_ID  = "TEA_P_M_001"
TOP_K      = 3
RUNS       = 5   # total runs per query
WARMUP     = 1   # discard first run
EMBED_MODEL = _env.get("MODEL_EMBEDDING", "text-embedding-v4")
EMBED_DIM   = int(_env.get("EMBEDDING_DIM", "2048"))
LLM_MODEL   = _env.get("MODEL_CHAT", "qwen-max")

client = openai.OpenAI(api_key=AI_API_KEY, base_url=AI_BASE_URL)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
Session = sessionmaker(bind=engine)

# -- Queries -------------------------------------------------------------------
QUERIES = [
    ("Q01", "CT_01 current reading 138 A near fault threshold 140 A. Thermistor_01 at 48 C nominal. Encoder_01 intermittent position loss."),
    ("Q02", "Current_Sensor_01 spiked to 820 A against fault_high of 840 A. CT_01 and thermistor within normal range."),
    ("Q03", "Thermistor_01 temperature reached 148 C, nearing fault_high 150 C. CT_01 at 95 A, Current_Sensor_01 at 590 A elevated."),
    ("Q04", "Encoder_01 dropped to 800 PPR, below min_normal 1000 PPR. Current and temperature sensors all nominal."),
    ("Q05", "CT_01 frozen at 55 A for 90 seconds, suspected sensor freeze fault. Current_Sensor_01 reading normal fluctuations."),
    ("Q06", "Progressive current drift: CT_01 climbed from 55 A to 99 A over 20 minutes. Thermistor rose from 47 C to 62 C simultaneously."),
    ("Q07", "Machine idle unexpectedly. All sensors near zero: CT_01 at 2 A, Current_Sensor_01 at 5 A, encoder 0 PPR."),
    ("Q08", "High-side fault on both current sensors: CT_01 at 139 A, Current_Sensor_01 at 835 A. Thermistor at 102 C (max_normal 105 C)."),
    ("Q09", "Encoder_01 oscillating 500 PPR to 36000 PPR at 2 Hz. No overcurrent detected. Thermistor stable."),
    ("Q10", "CT_01 at 97 A and Current_Sensor_01 at 598 A both at max_normal boundary. Thermistor_01 104 C. Pattern consistent with overload."),
]

# -- Core pipeline -------------------------------------------------------------

def embed(text: str) -> list:
    resp = client.embeddings.create(model=EMBED_MODEL, input=text, dimensions=EMBED_DIM)
    return resp.data[0].embedding


def retrieve(query_embedding: list, manual_id: str, top_k: int) -> list:
    db = Session()
    try:
        emb_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        sql = sa_text("""
            SELECT content, page
            FROM manual_chunks
            WHERE manual_id = :mid AND type = 'text'
            ORDER BY embedding <=> CAST(:emb AS vector)
            LIMIT :k
        """)
        rows = db.execute(sql, {"mid": manual_id, "emb": emb_str, "k": top_k}).fetchall()
        return rows
    finally:
        db.close()


def llm_call(query: str, chunks: list) -> str:
    context = ""
    for i, (content, page) in enumerate(chunks):
        context += f"--- Manual Excerpt {i+1} (Page {page}) ---\n{content}\n\n"

    system_msg = (
        f"You are an industrial diagnostic assistant for machine manual: {MANUAL_ID}.\n"
        "Using ONLY the manual excerpts below, provide a concise diagnostic summary "
        "(3-5 sentences) identifying the likely fault, probable root cause, and "
        "immediate corrective action. Do not add information not present in the excerpts.\n\n"
        f"MANUAL EXCERPTS:\n{context}"
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": f"Anomaly detected: {query}"},
        ],
        max_tokens=512,
        temperature=0.1,
    )
    return resp.choices[0].message.content


def run_once(query: str):
    t0 = time.perf_counter()
    emb = embed(query)
    t1 = time.perf_counter()
    chunks = retrieve(emb, MANUAL_ID, TOP_K)
    t2 = time.perf_counter()
    answer = llm_call(query, chunks)
    t3 = time.perf_counter()
    return {
        "embed_s":    t1 - t0,
        "retrieve_s": t2 - t1,
        "llm_s":      t3 - t2,
        "total_s":    t3 - t0,
        "answer":     answer,
        "chunks":     len(chunks),
    }

# -- Main ----------------------------------------------------------------------

def main():
    print(f"\nText-Only RAG Benchmark  |  manual={MANUAL_ID}  |  model={LLM_MODEL}")
    print(f"Runs per query: {RUNS} ({WARMUP} warm-up discarded)\n")
    print("-" * 80)

    all_means = []
    results = []

    for qid, query in QUERIES:
        timings = []
        for i in range(RUNS + WARMUP):
            try:
                r = run_once(query)
                if i >= WARMUP:
                    timings.append(r)
            except Exception as e:
                print(f"  [{qid}] run {i} FAILED: {e}")

        if not timings:
            print(f"  [{qid}] ALL RUNS FAILED - skipping")
            continue

        totals    = [t["total_s"]    for t in timings]
        embeds    = [t["embed_s"]    for t in timings]
        retrieves = [t["retrieve_s"] for t in timings]
        llms      = [t["llm_s"]      for t in timings]

        mean_t = statistics.mean(totals)
        std_t  = statistics.stdev(totals) if len(totals) > 1 else 0.0
        p95_t  = sorted(totals)[max(0, int(len(totals)*0.95)-1)]

        all_means.append(mean_t)
        results.append({
            "query_id": qid,
            "mean_total_s":    round(mean_t, 3),
            "std_total_s":     round(std_t,  3),
            "p95_total_s":     round(p95_t,  3),
            "mean_embed_s":    round(statistics.mean(embeds),    3),
            "mean_retrieve_s": round(statistics.mean(retrieves), 3),
            "mean_llm_s":      round(statistics.mean(llms),      3),
            "n_runs": len(timings),
        })

        print(
            f"  [{qid}]  mean={mean_t:.2f}s  sd={std_t:.2f}s  p95={p95_t:.2f}s"
            f"  (embed={statistics.mean(embeds):.2f}s"
            f"  retrieve={statistics.mean(retrieves):.3f}s"
            f"  llm={statistics.mean(llms):.2f}s)"
        )

    print("-" * 80)
    if all_means:
        agg = statistics.mean(all_means)
        print(f"\n  AGGREGATE MEAN across {len(all_means)} queries: {agg:.3f}s")
        print(f"  Embedding  mean: {statistics.mean(r['mean_embed_s']    for r in results):.3f}s")
        print(f"  Retrieval  mean: {statistics.mean(r['mean_retrieve_s'] for r in results):.3f}s")
        print(f"  LLM call   mean: {statistics.mean(r['mean_llm_s']      for r in results):.3f}s")
        print()
        print("  NOTE: Full multimodal + multi-agent system adds:")
        print("    - CLIP image chunk retrieval (top_k_image=3)")
        print("    - Table chunk retrieval (top_k_table=3)")
        print("    - InteractionMemory lookup (top_k_memory=2)")
        print("    - 6-node LangGraph pipeline (Sensor/Validation/Diagnostic/RAG/Strategy/Critic)")
        print("    - 3-5 additional GPT-4o inference calls")

    # Save JSON
    out_path = "benchmark_text_only_results.json"
    with open(out_path, "w") as f:
        json.dump({"manual_id": MANUAL_ID, "model": LLM_MODEL, "queries": results}, f, indent=2)
    print(f"\n  Results saved -> {out_path}\n")


if __name__ == "__main__":
    main()
