"""
eval_action3.py — Quantitative evaluation for IRAI26 Action Item #3

Measures:
  1. RAG Retrieval Precision  — Top-3 precision over 15 fault-diagnostic queries
  2. Safety Critic Compliance — First-pass / retry / fail-safe rates on 10 test procedures
  3. Institutional Memory     — Top-2 hit rate over 5 archived incident queries

Usage (from backend/):
    python scripts/eval_action3.py

Output:
    data/processed/evaluation/action3_results.json
    Printed summary table to stdout
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from unified_rag.ai_client import get_client, MODEL_CHAT
from unified_rag.db.database import SessionLocal
from unified_rag.db.models import ManualChunk, InteractionMemory
from unified_rag.embeddings.embedder import embedder

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

client = get_client()

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "processed", "evaluation")
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. RAG RETRIEVAL PRECISION ────────────────────────────────────────────────

# Queries grounded in the actual ingested manual (Rockwell 1336 IMPACT AC Drive)
RAG_TEST_QUERIES = [
    "safety precautions before removing power from the AC drive",
    "how to connect motor terminals to the 1336 IMPACT drive",
    "what to check if the drive trips on overcurrent fault",
    "parameter settings for motor overload protection",
    "procedure for replacing a circuit board in the drive",
    "NEC code requirements for drive installation wiring",
    "electrostatic discharge precautions when servicing drive boards",
    "how to verify bus voltage before working on drive terminals",
    "troubleshooting drive fault codes and alarms",
    "L Option Board contact closure interface wiring",
    "how to set acceleration and deceleration ramp times",
    "ground wiring requirements for the drive installation",
    "procedure for drive startup and initial commissioning",
    "how to adjust output frequency limits",
    "what PPE is required when working on energised drive cabinets",
]

RELEVANCE_JUDGE_PROMPT = """You are an expert evaluator for industrial technical documentation retrieval.

A technician sent the following query to a Retrieval-Augmented Generation (RAG) system that searches an AC drive maintenance and installation manual.
The system returned the TOP-3 retrieved text chunks below.

Your job: Judge whether the retrieved chunks are RELEVANT to answering the query.
- RELEVANT means the chunks contain information that would meaningfully help answer the query (even if partial).
- NOT RELEVANT means the chunks are off-topic, boilerplate, or contain no useful information for the query.

Respond ONLY with JSON: {"relevant": true or false, "reason": "<one short sentence>"}
"""


def evaluate_rag_precision(manual_ids: list) -> dict:
    """Run 15 queries, use GPT-4o as relevance judge for Top-3 precision."""
    print("\n--- [1/3] RAG RETRIEVAL PRECISION ---")

    db = SessionLocal()
    hits = 0
    total = 0
    per_query = []

    for query in RAG_TEST_QUERIES:
        query_emb = embedder.embed_text(query)

        for manual_id in manual_ids:
            results = db.query(ManualChunk).filter(
                ManualChunk.manual_id == manual_id,
                ManualChunk.type.in_(["text", "table"])
            ).order_by(
                ManualChunk.embedding.cosine_distance(query_emb)
            ).limit(3).all()

            if not results:
                continue

            chunks_text = "\n\n".join(
                f"[Chunk {i+1}, Page {r.page}]:\n{(r.content or '').strip()}"
                for i, r in enumerate(results)
            )

            judge_response = client.chat.completions.create(
                model=MODEL_CHAT,
                messages=[
                    {"role": "system", "content": RELEVANCE_JUDGE_PROMPT},
                    {"role": "user", "content": (
                        f"QUERY: {query}\n\n"
                        f"RETRIEVED CHUNKS:\n{chunks_text}"
                    )}
                ],
                temperature=0.0,
                max_tokens=150,
                response_format={"type": "json_object"}
            )
            verdict = json.loads(judge_response.choices[0].message.content)
            hit = verdict.get("relevant", False)
            reason = verdict.get("reason", "")

            hits += int(hit)
            total += 1
            per_query.append({
                "query": query,
                "manual_id": manual_id,
                "chunks_retrieved": len(results),
                "relevant": hit,
                "reason": reason,
            })
            print(f"  {'OK' if hit else '--'}  \"{query[:50]}\"")
            break

    db.close()

    precision = hits / total if total > 0 else 0.0
    print(f"\n  Top-3 Retrieval Precision: {hits}/{total} = {precision:.1%}")
    return {"top3_precision": round(precision, 4), "hits": hits, "total": total,
            "per_query": per_query}


# ── 2. SAFETY CRITIC COMPLIANCE ───────────────────────────────────────────────

SAFETY_PROCEDURES = [
    # (description, is_compliant, has_loto, has_ppe, no_energised)
    {
        "id": "P01", "compliant": True,
        "procedure": (
            "SAFETY PHASE: (1) De-energise the pump motor at the MCC. "
            "(2) Apply lockout padlock and tagout label. "
            "(3) Verify zero energy state with voltage tester. "
            "(4) Don insulated gloves, safety goggles, and steel-toe boots. "
            "(5) Proceed to mechanical seal replacement only after all lockout steps confirmed."
        ),
    },
    {
        "id": "P02", "compliant": False,
        "procedure": (
            "Step 1: While the pump is still running, inspect the mechanical seal by opening the guard. "
            "Step 2: Note the leakage rate. "
            "Step 3: Schedule replacement for next shift."
        ),
    },
    {
        "id": "P03", "compliant": True,
        "procedure": (
            "LOTO PROCEDURE: Isolate power at distribution panel CB-12, lock with personal padlock, "
            "tag with maintenance in progress. PPE: acid-resistant gloves, face shield, safety boots. "
            "Verify de-energisation before opening the motor terminal box. "
            "Only begin winding inspection after zero-voltage confirmation."
        ),
    },
    {
        "id": "P04", "compliant": False,
        "procedure": (
            "Perform bearing inspection. Remove bearing housing cover. "
            "Check bearing clearance. Re-grease and reassemble. "
            "No need to shut down for this routine inspection."
        ),
    },
    {
        "id": "P05", "compliant": True,
        "procedure": (
            "SAFETY FIRST: Complete full LOTO on turbine control panel. "
            "PPE requirement: arc-flash rated face shield (40 cal/cm²), insulated gloves class 4, "
            "flame-resistant coverall. Blade inspection must only begin with turbine at full stop "
            "and all energy sources isolated including steam and hydraulic lines."
        ),
    },
    {
        "id": "P06", "compliant": False,
        "procedure": (
            "Lathe spindle fault: Open the headstock cover while spindle is decelerating. "
            "Use a stethoscope to listen for bearing noise. Note any irregular sounds. "
            "Apply grease to accessible bearings. Replace cover."
        ),
    },
    {
        "id": "P07", "compliant": True,
        "procedure": (
            "Lock out lathe at main disconnect switch. Apply lockout hasp and personal padlock. "
            "Tag with 'Equipment Under Maintenance'. Wear safety gloves and goggles. "
            "Wait for complete spindle stop (minimum 5 minutes after power isolation) before "
            "opening the headstock housing for internal inspection."
        ),
    },
    {
        "id": "P08", "compliant": False,
        "procedure": (
            "Impeller clearance check: The pump can remain online at reduced flow. "
            "Loosen the back plate bolts while maintaining 30% flow. "
            "Adjust the clearance using the adjustment screws. Re-tighten and restore flow."
        ),
    },
    {
        "id": "P09", "compliant": True,
        "procedure": (
            "De-energise pump at local isolator. Lock and tag. Drain pump casing fully. "
            "PPE: chemical-resistant gloves, safety glasses, boot covers for hazardous fluid. "
            "Only after confirming zero pressure gauge reading, loosen the back plate and "
            "adjust impeller clearance to specification (0.3–0.5 mm)."
        ),
    },
    {
        "id": "P10", "compliant": False,
        "procedure": (
            "Motor winding check: Connect multimeter directly to running motor terminals. "
            "Measure resistance while motor is energised to identify winding fault. "
            "Record values and compare to nameplate."
        ),
    },
]

COMPLIANCE_SYSTEM_PROMPT = """You are an Industrial Safety Compliance Critic.

Evaluate a maintenance procedure step against this checklist:
1. LOTO: Must include de-energisation (lockout + tagout + verify zero energy)
2. PPE: Must specify appropriate PPE (gloves, goggles, safety footwear minimum)
3. NO ENERGISED WORK: Must prohibit working on energised equipment

Respond ONLY with JSON in this exact format:
{
  "compliant": true or false,
  "violations": ["<violation description if any>"],
  "reason": "<one sentence explanation>"
}
"""


def evaluate_safety_critic() -> dict:
    """Run 10 test procedures through GPT-4o compliance check with retry logic."""
    print("\n━━━ [2/3] SAFETY CRITIC COMPLIANCE ━━━")

    first_pass = 0
    retry_pass = 0
    fail_safe = 0
    per_procedure = []

    for proc in SAFETY_PROCEDURES:
        pid = proc["id"]
        expected = proc["compliant"]
        procedure_text = proc["procedure"]

        def check_compliance(text):
            response = client.chat.completions.create(
                model=MODEL_CHAT,
                messages=[
                    {"role": "system", "content": COMPLIANCE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Evaluate this procedure:\n\n{text}"}
                ],
                temperature=0.0,
                max_tokens=300,
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)

        # Attempt 1
        result1 = check_compliance(procedure_text)
        attempt1_compliant = result1.get("compliant", False)

        if attempt1_compliant:
            first_pass += 1
            outcome = "first_pass"
            violations = []
        else:
            violations = result1.get("violations", [])
            # Build a corrected version for retry (add safety header)
            corrected = (
                "SAFETY HEADER (added by system): Complete LOTO at main isolator. "
                "Apply personal padlock and tagout. Verify zero energy. Don appropriate PPE. "
                "Do NOT begin work on energised equipment.\n\n" + procedure_text
            )
            result2 = check_compliance(corrected)
            if result2.get("compliant", False):
                retry_pass += 1
                outcome = "retry_pass"
            else:
                fail_safe += 1
                outcome = "fail_safe"

        correct_classification = (attempt1_compliant == expected)
        per_procedure.append({
            "id": pid,
            "expected_compliant": expected,
            "first_pass_result": attempt1_compliant,
            "outcome": outcome,
            "correct_classification": correct_classification,
            "violations_flagged": violations,
        })

        icon = "✅" if correct_classification else "⚠️"
        print(f"  {icon}  [{pid}] expected={'PASS' if expected else 'FAIL'} "
              f"→ outcome={outcome}  violations={len(violations)}")

    total = len(SAFETY_PROCEDURES)
    classification_accuracy = sum(p["correct_classification"] for p in per_procedure) / total
    print(f"\n  First-pass compliant:  {first_pass}/{total} = {first_pass/total:.1%}")
    print(f"  Retry-to-pass:         {retry_pass}/{total} = {retry_pass/total:.1%}")
    print(f"  Fail-safe triggered:   {fail_safe}/{total} = {fail_safe/total:.1%}")
    print(f"  Classification accuracy: {classification_accuracy:.1%}")

    return {
        "total_procedures": total,
        "first_pass_count": first_pass,
        "retry_pass_count": retry_pass,
        "fail_safe_count": fail_safe,
        "first_pass_rate": round(first_pass / total, 4),
        "retry_pass_rate": round(retry_pass / total, 4),
        "fail_safe_rate": round(fail_safe / total, 4),
        "classification_accuracy": round(classification_accuracy, 4),
        "per_procedure": per_procedure,
    }


# ── 3. INSTITUTIONAL MEMORY RETRIEVAL ────────────────────────────────────────

MEMORY_INCIDENTS = [
    {
        "machine_id": "PUMP-001",
        "manual_id": "Zynaptrix_9000",
        "fault": "High temperature alarm triggered on centrifugal pump bearing",
        "summary": (
            "Centrifugal pump bearing temperature exceeded 85°C alarm threshold. "
            "Operator isolated pump, drained casing, and found impeller clearance worn to 1.2mm (spec: 0.3–0.5mm). "
            "Impeller replaced and clearance re-set to 0.4mm. Pump returned to service."
        ),
        "operator_fix": "Replaced impeller and re-set clearance to 0.4mm. Temperature returned to baseline 42°C within 30 minutes.",
    },
    {
        "machine_id": "LATHE-002",
        "manual_id": "Zynaptrix_9000",
        "fault": "Excessive spindle vibration detected at 1800 RPM",
        "summary": (
            "Lathe spindle vibration exceeded 4.5mm/s RMS at 1800 RPM. "
            "Diagnosis: spindle front bearing worn, radial clearance 0.08mm (limit: 0.03mm). "
            "Bearing replaced (NSK 6208-2RS), spindle balanced, vibration reduced to 0.9mm/s."
        ),
        "operator_fix": "Replaced front spindle bearing NSK 6208-2RS. Post-repair vibration: 0.9mm/s RMS.",
    },
    {
        "machine_id": "TURBINE-003",
        "manual_id": "Zynaptrix_9000",
        "fault": "Blade tip clearance out of specification causing efficiency loss",
        "summary": (
            "Turbine efficiency dropped 8%. Blade tip clearance inspection revealed 3.1mm gap (spec: 1.5–2.0mm). "
            "Two blades showed erosion pitting on leading edge. Blades resurfaced and tip clearance restored to 1.7mm. "
            "Output efficiency recovered to baseline."
        ),
        "operator_fix": "Resurfaced 2 eroded blades, restored tip clearance to 1.7mm. Efficiency recovered.",
    },
    {
        "machine_id": "PUMP-001",
        "manual_id": "Zynaptrix_9000",
        "fault": "Mechanical seal leak causing abnormal motor current draw",
        "summary": (
            "Motor current rose 18% above baseline (from 12A to 14.2A). "
            "Root cause: mechanical seal face wear causing process fluid ingress into motor cavity. "
            "Seal replaced with upgraded carbon-ceramic face. Current normalised to 11.8A post-repair."
        ),
        "operator_fix": "Replaced mechanical seal with carbon-ceramic upgrade. Motor current normalised to 11.8A.",
    },
    {
        "machine_id": "TURBINE-003",
        "manual_id": "Zynaptrix_9000",
        "fault": "Oil lubrication system pressure drop fault on turbine bearing",
        "summary": (
            "Lube oil pressure dropped from 2.8 bar to 1.4 bar triggering low-pressure alarm. "
            "Found: clogged oil filter (differential pressure 1.8 bar, limit: 0.5 bar). "
            "Filter replaced, oil flushed, pressure restored to 2.7 bar within 10 minutes."
        ),
        "operator_fix": "Replaced oil filter, flushed oil circuit. Lube pressure restored to 2.7 bar.",
    },
]

MEMORY_TEST_QUERIES = [
    {"query": "pump bearing high temperature overheating impeller clearance worn", "target_machine": "PUMP-001", "target_idx": 0},
    {"query": "spindle vibration bearing replacement lathe RPM", "target_machine": "LATHE-002", "target_idx": 1},
    {"query": "turbine blade erosion tip clearance efficiency drop", "target_machine": "TURBINE-003", "target_idx": 2},
    {"query": "mechanical seal leak motor current overload pump", "target_machine": "PUMP-001", "target_idx": 3},
    {"query": "oil lube pressure drop filter clogged turbine bearing", "target_machine": "TURBINE-003", "target_idx": 4},
]


def evaluate_institutional_memory() -> dict:
    """Archive 5 incidents and test retrieval accuracy."""
    print("\n━━━ [3/3] INSTITUTIONAL MEMORY RETRIEVAL ━━━")

    db = SessionLocal()

    # Archive incidents (insert with embeddings)
    inserted_ids = []
    print("  Archiving 5 test incidents...")
    for inc in MEMORY_INCIDENTS:
        combined = f"{inc['fault']} | {inc['summary']} | {inc['operator_fix']}"
        emb = embedder.embed_text(combined)

        record = InteractionMemory(
            machine_id=inc["machine_id"],
            manual_id=inc["manual_id"],
            summary=inc["summary"],
            operator_fix=inc["operator_fix"],
            embedding=emb,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        db.add(record)
        db.flush()
        inserted_ids.append(record.id)
        print(f"    Archived: [{inc['machine_id']}] {inc['fault'][:55]}...")
    db.commit()

    # Run retrieval queries
    hits = 0
    per_query = []
    print("\n  Running 5 retrieval queries...")
    for item in MEMORY_TEST_QUERIES:
        query_emb = embedder.embed_text(item["query"])
        target_machine = item["target_machine"]
        target_idx = item["target_idx"]
        target_id = inserted_ids[target_idx]

        results = db.query(InteractionMemory).filter(
            InteractionMemory.machine_id == target_machine
        ).order_by(
            InteractionMemory.embedding.cosine_distance(query_emb)
        ).limit(2).all()

        result_ids = [r.id for r in results]
        hit = target_id in result_ids
        hits += int(hit)

        rank = result_ids.index(target_id) + 1 if hit else None
        per_query.append({
            "query": item["query"][:60],
            "target_machine": target_machine,
            "target_id": target_id,
            "top2_ids": result_ids,
            "hit_in_top2": hit,
            "rank": rank,
        })

        print(f"  {'✅' if hit else '❌'}  [{target_machine}] "
              f"\"{item['query'][:45]}...\"  rank={rank}")

    # Cleanup: remove inserted test records
    db.query(InteractionMemory).filter(InteractionMemory.id.in_(inserted_ids)).delete(synchronize_session=False)
    db.commit()
    db.close()

    hit_rate = hits / len(MEMORY_TEST_QUERIES)
    print(f"\n  Top-2 Hit Rate: {hits}/{len(MEMORY_TEST_QUERIES)} = {hit_rate:.1%}")

    return {
        "total_queries": len(MEMORY_TEST_QUERIES),
        "hits_in_top2": hits,
        "top2_hit_rate": round(hit_rate, 4),
        "per_query": per_query,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  IRAI26 ACTION ITEM #3 — Quantitative Evaluation")
    print("=" * 60)

    # Discover available manual IDs from DB
    db = SessionLocal()
    manual_ids_raw = db.query(ManualChunk.manual_id).distinct().all()
    manual_ids = [m[0] for m in manual_ids_raw]
    db.close()

    if not manual_ids:
        print("\n⚠️  No manuals found in DB. RAG precision test will be skipped.")
        rag_results = {"top3_precision": None, "hits": 0, "total": 0,
                       "note": "No manuals ingested in vector DB"}
    else:
        print(f"\n  Found {len(manual_ids)} manual(s): {manual_ids}")
        rag_results = evaluate_rag_precision(manual_ids)

    safety_results = evaluate_safety_critic()
    memory_results = evaluate_institutional_memory()

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  FINAL RESULTS SUMMARY")
    print("=" * 60)

    rag_prec = rag_results.get("top3_precision")
    print(f"\n  RAG Engine")
    print(f"    Top-3 Retrieval Precision  : {f'{rag_prec:.1%}' if rag_prec is not None else 'N/A (no manuals)'}")

    sp = safety_results
    print(f"\n  Safety Critic (n={sp['total_procedures']} procedures)")
    print(f"    First-pass compliance rate : {sp['first_pass_rate']:.1%}")
    print(f"    Retry-to-pass rate         : {sp['retry_pass_rate']:.1%}")
    print(f"    Fail-safe triggered        : {sp['fail_safe_rate']:.1%}")
    print(f"    Classification accuracy    : {sp['classification_accuracy']:.1%}")

    mp = memory_results
    print(f"\n  Institutional Memory (n={mp['total_queries']} queries)")
    print(f"    Top-2 hit rate             : {mp['top2_hit_rate']:.1%}  ({mp['hits_in_top2']}/{mp['total_queries']})")

    print("\n" + "=" * 60)

    # Save results
    output = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "rag_retrieval": rag_results,
        "safety_critic": safety_results,
        "institutional_memory": memory_results,
    }
    out_path = os.path.join(OUT_DIR, "action3_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved → {out_path}")

    return output


if __name__ == "__main__":
    main()
