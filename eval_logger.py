"""
NOVA Evaluation Logger
-----------------------
Wraps the existing pipeline functions (unchanged) to produce structured,
comparable records for the three frozen hypotheses:

  H1 (RAG vs LLM-only)       -> compare condition "full" vs "no_rag"
  H2 (DVE vs no-DVE)         -> compare condition "full" vs "no_dve"
  H3 (memory reduces MTTR)   -> compare latency_sec, "full" vs "no_rag"

Every call appends one row to nova_eval_log.csv. Running all three
conditions on the SAME incident_id gives you paired data, which is what
makes a same-incident before/after comparison possible instead of just
three unrelated numbers.

This does not modify agents.py, nova_rca.py, or decision_verification.py —
it only calls their existing public functions in different combinations.
"""

import csv
import os
import time
from datetime import datetime

from nova_rca import retrieve_similar_incidents, generate_rca_analysis
from decision_verification import (
    verify_decision, extract_action, extract_confidence,
    calculate_evidence_score, POLICY
)

LOG_PATH = "nova_eval_log.csv"

FIELDNAMES = [
    "timestamp", "incident_id", "incident_text", "namespace",
    "condition",  # "full" | "no_rag" | "no_dve"
    "retrieved_ids", "retrieved_count", "top_match_score",
    "rca_confidence", "evidence_score",
    "extracted_action", "policy_risk",
    "dve_verdict", "dve_reason",
    "would_execute", "latency_sec",
]


def _ensure_log():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def _append_row(row: dict):
    _ensure_log()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)


def _base_row(incident_id, incident_text, namespace, condition):
    return {
        "timestamp": datetime.now().isoformat(),
        "incident_id": incident_id,
        "incident_text": incident_text,
        "namespace": namespace,
        "condition": condition,
    }


def run_full(incident_id, incident_text, namespace=None,
             retrieval_query=None, filter_severity=None):
    """Condition 'full': the actual production path — RAG retrieval +
    LLM RCA + Decision Verification Engine. This is what the live
    watcher/Coordinator already runs; calling it here separately just
    also captures it as a comparable, logged row."""
    start = time.time()
    similar = retrieve_similar_incidents(
        retrieval_query or incident_text, filter_severity=filter_severity
    )
    analysis = generate_rca_analysis(incident_text, similar)
    verification = verify_decision(
        incident_text, analysis, similar,
        namespace=namespace, auto_execute=False
    )
    latency = time.time() - start

    row = _base_row(incident_id, incident_text, namespace, "full")
    row.update({
        "retrieved_ids": ";".join(i["id"] for i in similar),
        "retrieved_count": len(similar),
        "top_match_score": similar[0]["score"] if similar else 0.0,
        "rca_confidence": verification["llm_confidence"],
        "evidence_score": verification["evidence_score"],
        "extracted_action": verification["action"],
        "policy_risk": verification["risk_level"],
        "dve_verdict": verification["final_verdict"],
        "dve_reason": verification["final_reason"],
        "would_execute": verification["final_verdict"] == "APPROVED",
        "latency_sec": round(latency, 3),
    })
    _append_row(row)
    return row


def run_no_rag(incident_id, incident_text, namespace=None):
    """Condition 'no_rag': H1 ablation. Forces the RCA call with an empty
    historical context (no retrieval at all), so the LLM reasons cold.
    Still runs the result through verify_decision so action/confidence
    are directly comparable to the 'full' row for the same incident."""
    start = time.time()
    analysis = generate_rca_analysis(incident_text, [])  # empty context, forces cold reasoning
    verification = verify_decision(
        incident_text, analysis, [],
        namespace=namespace, auto_execute=False
    )
    latency = time.time() - start

    row = _base_row(incident_id, incident_text, namespace, "no_rag")
    row.update({
        "retrieved_ids": "",
        "retrieved_count": 0,
        "top_match_score": 0.0,
        "rca_confidence": verification["llm_confidence"],
        "evidence_score": verification["evidence_score"],
        "extracted_action": verification["action"],
        "policy_risk": verification["risk_level"],
        "dve_verdict": verification["final_verdict"],
        "dve_reason": verification["final_reason"],
        "would_execute": verification["final_verdict"] == "APPROVED",
        "latency_sec": round(latency, 3),
    })
    _append_row(row)
    return row


def run_no_dve(incident_id, incident_text, namespace=None,
               retrieval_query=None, filter_severity=None):
    """Condition 'no_dve': H2 ablation. Same RAG + RCA as 'full', but skips
    verify_decision entirely — this is "what would have executed if the
    verification layer didn't exist." would_execute is always True by
    construction, since removing DVE means every LLM recommendation goes
    straight to execution. Label ground-truth correctness for these rows
    yourself (you know the injected fault) to turn this into the "actions
    DVE blocked that would have been wrong" count H2 needs."""
    start = time.time()
    similar = retrieve_similar_incidents(
        retrieval_query or incident_text, filter_severity=filter_severity
    )
    analysis = generate_rca_analysis(incident_text, similar)
    action = extract_action(analysis)
    confidence = extract_confidence(analysis)
    evidence = calculate_evidence_score(similar)
    latency = time.time() - start

    row = _base_row(incident_id, incident_text, namespace, "no_dve")
    row.update({
        "retrieved_ids": ";".join(i["id"] for i in similar),
        "retrieved_count": len(similar),
        "top_match_score": similar[0]["score"] if similar else 0.0,
        "rca_confidence": confidence,
        "evidence_score": evidence,
        "extracted_action": action,
        "policy_risk": POLICY.get(action, {}).get("risk", "UNKNOWN"),
        "dve_verdict": "SKIPPED_WOULD_AUTO_EXECUTE",
        "dve_reason": "Verification bypassed for H2 ablation.",
        "would_execute": True,
        "latency_sec": round(latency, 3),
    })
    _append_row(row)
    return row


def run_all_conditions(incident_id, incident_text, namespace=None,
                        retrieval_query=None, filter_severity=None):
    """Runs one incident through all three conditions back-to-back, so the
    CSV ends up with matched rows sharing the same incident_id — that
    pairing is what makes a same-incident before/after comparison valid
    instead of three unrelated numbers. This triples LLM calls per
    incident (roughly 3x the Groq usage of a single 'full' run)."""
    return {
        "full": run_full(incident_id, incident_text, namespace, retrieval_query, filter_severity),
        "no_rag": run_no_rag(incident_id, incident_text, namespace),
        "no_dve": run_no_dve(incident_id, incident_text, namespace, retrieval_query, filter_severity),
    }


if __name__ == "__main__":
    # Quick smoke test against one incident, all three conditions.
    test_incident = (
        "Pod cartservice in namespace online-boutique is stuck in "
        "CrashLoopBackOff. Container repeatedly failing to start."
    )
    results = run_all_conditions("SMOKE_TEST_001", test_incident, namespace="online-boutique")
    for condition, row in results.items():
        print(f"\n[{condition}] action={row['extracted_action']} "
              f"verdict={row['dve_verdict']} "
              f"confidence={row['rca_confidence']} "
              f"evidence={row['evidence_score']} "
              f"latency={row['latency_sec']}s")
    print(f"\nLogged to {LOG_PATH}")