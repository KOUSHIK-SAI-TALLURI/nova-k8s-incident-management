"""
NOVA Prometheus Auto-Watcher
-----------------------------
Polls Prometheus on a fixed interval, checks a set of PromQL rules that map
to NOVA's known incident types (OOMKilled, CrashLoopBackOff, high CPU,
PVCPending, NodeNotReady), and — when a real breach is detected — builds an
incident description string and feeds it straight into the existing
nova_analyze() pipeline (RAG retrieval -> LLM RCA -> Decision Verification
Engine -> execution), exactly the same as manually typed incidents.

Requires: `kubectl port-forward svc/prometheus-kube-prometheus-prometheus 9090:9090`
running in a separate terminal so this script can reach Prometheus's HTTP API.
"""

import time
import re
import requests
from datetime import datetime, timedelta

from agents import Coordinator
import eval_logger

coordinator = Coordinator()

PROM_URL = "http://localhost:9090"
POLL_INTERVAL_SECONDS = 30
COOLDOWN_MINUTES = 5   # don't refire the same pod/rule combo within this window


def stable_workload_name(name: str) -> str:
    """Strip the ReplicaSet-hash + pod-instance suffix Kubernetes appends to
    Deployment-managed pod names (e.g. 'sample-app-84844b99bf-2lx8d' ->
    'sample-app'). Without this, every pod restart gets a brand-new random
    name, which breaks two things: (1) cooldown never applies, because each
    restart looks like a new target, so the same recurring incident refires
    every poll cycle instead of respecting the cooldown window; and (2) RAG
    retrieval embeds the random hash as if it were meaningful content, which
    jitters the embedding and evidence score run to run for what is actually
    the identical incident."""
    match = re.match(r'^(.*)-[a-f0-9]{8,10}-[a-z0-9]{5}$', name)
    if match:
        return match.group(1)
    match = re.match(r'^(.*)-[a-z0-9]{5}$', name)
    if match:
        return match.group(1)
    return name

# ── Rule definitions ──────────────────────────────────────────────────────
# Each rule: a PromQL query returning one row per breaching target, plus a
# template for building the human-readable incident string NOVA expects,
# plus which labels to pull the pod/namespace from.
RULES = [
    {
        "name": "OOMKilled",
        "query": 'kube_pod_container_status_last_terminated_reason{reason="OOMKilled"} == 1',
        "describe": lambda labels: (
            f"Pod {labels.get('pod','unknown')} in namespace {labels.get('namespace','unknown')} "
            f"was killed. OOMKilled in logs. Memory usage exceeded container limit."
        ),
        "severity": "HIGH",
    },
    {
        "name": "CrashLoopBackOff",
        "query": 'kube_pod_container_status_waiting_reason{reason="CrashLoopBackOff"} == 1',
        "describe": lambda labels: (
            f"Pod {labels.get('pod','unknown')} in namespace {labels.get('namespace','unknown')} "
            f"is stuck in CrashLoopBackOff. Container repeatedly failing to start."
        ),
        "severity": "CRITICAL",
    },
    {
        "name": "HighRestartRate",
        "query": 'increase(kube_pod_container_status_restarts_total[10m]) > 3',
        "describe": lambda labels: (
            f"Pod {labels.get('pod','unknown')} in namespace {labels.get('namespace','unknown')} "
            f"has restarted more than 3 times in the last 10 minutes. Possible crash loop or memory issue."
        ),
        "severity": "HIGH",
    },
    {
        "name": "PVCPending",
        "query": 'kube_persistentvolumeclaim_status_phase{phase="Pending"} == 1',
        "describe": lambda labels: (
            f"Persistent volume claim {labels.get('persistentvolumeclaim','unknown')} "
            f"in namespace {labels.get('namespace','unknown')} is stuck in Pending state. "
            f"Pod likely stuck in init state."
        ),
        "severity": "MEDIUM",
    },
    {
        "name": "NodeNotReady",
        "query": 'kube_node_status_condition{condition="Ready", status="true"} == 0',
        "describe": lambda labels: (
            f"Node {labels.get('node','unknown')} is NotReady. Pods on this node may be evicted."
        ),
        "severity": "CRITICAL",
        # None of NOVA's POLICY actions (restart_pod, scale_deployment,
        # rollback_deployment, delete_pod, modify_config,
        # increase_memory_limit) actually address node health — draining,
        # cordoning, or replacing a node isn't something this pipeline can
        # do. Also, node-level incidents have no namespace, so letting this
        # go through normal verification would force the LLM to either
        # invent a namespace or force-fit the wrong action. Escalate to a
        # human instead of attempting auto-remediation.
        "auto_remediate": False,
    },
]

# key: (rule_name, target_signature) -> datetime of last fire
_last_fired = {}


def query_prometheus(promql: str):
    """Run an instant PromQL query and return the list of result rows."""
    try:
        resp = requests.get(
            f"{PROM_URL}/api/v1/query",
            params={"query": promql},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") != "success":
            print(f"   ⚠️  Prometheus query error: {data}")
            return []
        return data["data"]["result"]
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot reach Prometheus. Is `kubectl port-forward svc/prometheus-kube-prometheus-prometheus 9090:9090` running?")
        return []
    except Exception as e:
        print(f"   ❌ Prometheus query failed: {e}")
        return []


def target_signature(labels: dict) -> str:
    """Build a stable identifier for a specific pod/node/pvc so we can dedupe."""
    for key in ("pod", "node", "persistentvolumeclaim"):
        if key in labels:
            return f"{labels.get('namespace','')}/{labels[key]}"
    return str(sorted(labels.items()))


def is_in_cooldown(rule_name: str, signature: str) -> bool:
    last = _last_fired.get((rule_name, signature))
    if last is None:
        return False
    return datetime.now() - last < timedelta(minutes=COOLDOWN_MINUTES)


def mark_fired(rule_name: str, signature: str):
    _last_fired[(rule_name, signature)] = datetime.now()


def check_rule(rule: dict):
    results = query_prometheus(rule["query"])
    if not results:
        return

    for row in results:
        labels = row.get("metric", {})

        # Build a hash-stripped copy of the labels so cooldown tracking and
        # the RAG retrieval query treat every restart of the same workload
        # as the same logical incident, instead of a new one each time.
        stable_labels = dict(labels)
        for key in ("pod", "node", "persistentvolumeclaim"):
            if key in stable_labels:
                stable_labels[key] = stable_workload_name(stable_labels[key])

        signature = target_signature(stable_labels)

        if is_in_cooldown(rule["name"], signature):
            continue  # already handled recently, skip

        # Specific description (real pod name) — used for display and so
        # the Remediation Agent targets the exact current pod instance.
        incident_description = rule["describe"](labels)

        # Generic description (stable name, no random hash) — used only
        # for RAG retrieval + RCA, so embeddings and evidence scores are
        # consistent across restarts of the same underlying incident.
        retrieval_query = rule["describe"](stable_labels)

        print(f"\n🔔  Rule '{rule['name']}' triggered for {signature}")
        mark_fired(rule["name"], signature)

        # Note: we deliberately do NOT filter by namespace here. The live
        # cluster's namespaces (e.g. "default" for sample-app) don't match
        # the synthetic incident corpus's namespaces (all "production" or
        # "kube-system"), so a namespace filter zeroes out vector retrieval
        # and collapses evidence scores. Once Online Boutique is deployed
        # under "production" in Week 13-14, filter_namespace can be
        # reintroduced safely.
        # Real namespace, straight from the Prometheus label — this is
        # ground truth for the incident and must not be re-guessed
        # downstream. labels (not stable_labels) is used here because we
        # want the actual namespace value, not the hash-stripped workload
        # name transform (which only applies to pod/node/pvc names).
        real_namespace = labels.get("namespace")

        coordinator.handle_incident(
            incident_description,
            filter_severity=rule["severity"],
            retrieval_query=retrieval_query,
            namespace=real_namespace,
            auto_remediate=rule.get("auto_remediate", True)
        )

        # Evaluation logging for H1/H2/H3 — separate from the live
        # remediation path above, so a logging failure or extra LLM call
        # here never blocks or delays real incident handling. Skipped for
        # incident types with no valid auto-remediation action (e.g.
        # NodeNotReady), since there's no meaningful action/DVE comparison
        # to log for those. incident_id combines rule name + signature +
        # timestamp so repeated fires of the same pod/rule stay distinguishable
        # rows in the CSV rather than overwriting each other.
        if rule.get("auto_remediate", True):
            incident_id = f"{rule['name']}__{signature}__{datetime.now().strftime('%Y%m%dT%H%M%S')}"
            try:
                eval_logger.run_all_conditions(
                    incident_id,
                    retrieval_query,
                    namespace=real_namespace,
                    retrieval_query=retrieval_query,
                    filter_severity=rule["severity"]
                )
                print(f"   📝 Logged evaluation rows (full/no_rag/no_dve) for {incident_id}")
            except Exception as e:
                print(f"   ⚠️  Evaluation logging failed (live remediation was unaffected): {e}")


def watch_loop():
    print("=" * 60)
    print("👁️   NOVA PROMETHEUS AUTO-WATCHER STARTED")
    print(f"   Polling every {POLL_INTERVAL_SECONDS}s | Cooldown {COOLDOWN_MINUTES}min per target")
    print(f"   Prometheus: {PROM_URL}")
    print("=" * 60)

    while True:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{timestamp}] Polling Prometheus for {len(RULES)} rule(s)...")

        for rule in RULES:
            check_rule(rule)

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    try:
        watch_loop()
    except KeyboardInterrupt:
        print("\n\n👋 Watcher stopped.")