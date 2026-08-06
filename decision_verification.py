import os
import re
import subprocess
from groq import Groq
from dotenv import load_dotenv
from datetime import datetime
import shlex

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

POLICY = {
    "restart_pod": {
        "risk": "LOW",
        "min_confidence": 75,
        "min_evidence_score": 65,
        "auto_execute": True,
        "description": "Restart a crashed or unhealthy pod (underlying command deletes the pod; Deployment recreates it)"
    },
    "scale_deployment": {
        "risk": "LOW",
        "min_confidence": 65,
        "min_evidence_score": 55,
        "auto_execute": True,
        "description": "Scale deployment replicas up or down"
    },
    "rollback_deployment": {
        "risk": "MEDIUM",
        "min_confidence": 75,
        "min_evidence_score": 65,
        "auto_execute": True,
        "description": "Rollback to previous deployment version"
    },
    "delete_pod": {
        "risk": "MEDIUM",
        "min_confidence": 80,
        "min_evidence_score": 70,
        "auto_execute": False,
        "description": "Force delete a pod"
    },
    "modify_config": {
        "risk": "MEDIUM",
        "min_confidence": 80,
        "min_evidence_score": 70,
        "auto_execute": False,
        "description": "Modify configmap or environment variables"
    },
    "increase_memory_limit": {
        "risk": "LOW",
        "min_confidence": 70,
        "min_evidence_score": 60,
        "auto_execute": True,
        "description": "Increase container memory limits"
    },
    "rollback_production_db": {
        "risk": "HIGH",
        "min_confidence": 95,
        "min_evidence_score": 90,
        "auto_execute": False,
        "description": "Rollback production database"
    },
    "delete_namespace": {
        "risk": "HIGH",
        "min_confidence": 100,
        "min_evidence_score": 100,
        "auto_execute": False,
        "description": "BLOCKED — too destructive"
    }
}

ACTION_SYNTAX_HINTS = {
    "increase_memory_limit": (
        "Use exactly this form: "
        "kubectl set resources deployment <name> --namespace <namespace> "
        "--limits=memory=<value> --requests=memory=<value>. "
        "Do NOT use 'kubectl scale' for this action, and do NOT invent flags "
        "like --resource-requests or --resource-limits — they do not exist."
    ),
    "restart_pod": (
        "Use exactly this form: kubectl delete pod <pod_name> --namespace <namespace> "
        "(the Deployment will recreate it)."
    ),
    "scale_deployment": (
        "Use exactly this form: kubectl scale deployment <name> --namespace <namespace> "
        "--replicas=<n>."
    ),
    "rollback_deployment": (
        "Use exactly this form: kubectl rollout undo deployment/<name> --namespace <namespace>."
    ),
}


ACTION_CONSEQUENCES = {
    "restart_pod": (
        "Deletes the running pod immediately. The Deployment will recreate it "
        "automatically, but any requests in-flight to this pod are dropped, and "
        "there is a brief gap (typically a few seconds) before the replacement "
        "pod is Ready. If this is the only replica, that gap is real downtime."
    ),
    "delete_pod": (
        "Force-deletes a pod that is not terminating gracefully. Same "
        "availability impact as restart_pod. Any data written only to "
        "non-persistent (emptyDir) volumes on this specific pod is lost."
    ),
    "scale_deployment": (
        "Changes the replica count. Scaling down terminates excess pods "
        "immediately — in-flight requests to those pods are dropped. Scaling "
        "up starts new pods, which may take time to pass readiness checks."
    ),
    "rollback_deployment": (
        "Reverts the deployment to its previous ReplicaSet/image. Any "
        "behavior only present in the current version is lost until "
        "redeployed, and pods are recreated to match the prior version — "
        "expect a brief rollout disruption."
    ),
    "modify_config": (
        "Changes environment variables or a ConfigMap. Pods typically must "
        "restart to pick up the change. If this drifts from what's tracked "
        "in source control, it can cause config inconsistency later."
    ),
    "increase_memory_limit": (
        "Updates the resource spec on the deployment. Kubernetes performs a "
        "rolling restart to apply it — each replica is recreated in turn, "
        "with a brief gap in availability per pod as it restarts."
    ),
    "rollback_production_db": (
        "Reverts a production database to a prior state. This can be "
        "IRREVERSIBLE and cause permanent loss of any writes made since "
        "that state. Requires human approval — NOVA will not auto-execute this."
    ),
    "delete_namespace": (
        "Deletes an entire namespace and everything in it — deployments, "
        "pods, services, secrets, ConfigMaps, PVCs. Destructive and "
        "irreversible. NOVA will never auto-execute this action."
    ),
}


def print_consequence_warning(action: str):
    consequence = ACTION_CONSEQUENCES.get(action)
    if consequence:
        print(f"\n   ⚠️  CONSEQUENCE: {consequence}")


def generate_kubectl_command(action: str, analysis: str, incident: str = "",
                              namespace: str = None) -> str:
    """Generate the kubectl command for an action. Split out from
    execute_recovery() so a command can be generated for human review
    (preview, not executed) as well as for actual auto-execution.

    `namespace` should be the real, detected namespace threaded down from
    the watcher (or wherever the incident originated) — never a hardcoded
    guess. If it's genuinely unresolved (e.g. a node-level incident with
    no namespace concept), that's surfaced to the LLM explicitly instead
    of silently substituting "production"."""
    syntax_hint = ACTION_SYNTAX_HINTS.get(action, "")
    namespace_line = (
        f"Namespace (confirmed from the incident source — use this exactly): {namespace}"
        if namespace else
        "Namespace: NOT PROVIDED — extract it from CURRENT INCIDENT text below. "
        "If it truly is not present there either, do not guess a value; use 'default'."
    )

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a Kubernetes command generator. Reply with ONLY the raw kubectl command. No explanation, no markdown, no backticks."
            },
            {
                "role": "user",
                "content": f"""Generate the exact kubectl command for this action: {action}

CURRENT INCIDENT (this is the pod/namespace to target — use ONLY names found here):
{incident}

{namespace_line}

Supporting analysis (for context only — this may reference OTHER, HISTORICAL
incidents and their service names for comparison purposes. Do NOT use any
pod, deployment, or service name from this analysis unless it is also
explicitly present in the CURRENT INCIDENT text above):
{analysis}

Command syntax to use for this action: {syntax_hint}

Rules:
- The target pod/deployment name and namespace MUST come from CURRENT INCIDENT, not from historical incidents mentioned in the analysis.
- If no real deployment name is found in CURRENT INCIDENT, use 'sample-app'.
- Never use placeholder values like <deployment_name> or <new_memory_limit>.
- Never use single quotes around arguments. Use no quotes, or double quotes only.
- Keep the command simple, complete, and directly executable — no explanations."""
            }
        ],
        temperature=0,
        max_tokens=150,
        timeout=30
    )

    kubectl_command = response.choices[0].message.content.strip()
    return kubectl_command.replace("`", "").replace("bash", "").strip()


def print_command_box(kubectl_command: str, label: str = "GENERATED COMMAND"):
    box_width = 70
    print(f"\n   ┏{'━'*box_width}┓")
    print(f"   ┃  🔧 {label}")
    print(f"   ┃")
    for line in [kubectl_command[i:i+box_width-4] for i in range(0, len(kubectl_command), box_width-4)] or [""]:
        print(f"   ┃  {line}")
    print(f"   ┗{'━'*box_width}┛")


def execute_recovery(action: str, analysis: str, incident: str = "",
                      namespace: str = None) -> dict:
    print("\n" + "─"*60)
    print("⚙️   RECOVERY EXECUTOR")
    print("─"*60)

    kubectl_command = generate_kubectl_command(action, analysis, incident, namespace)

    print_consequence_warning(action)
    print_command_box(kubectl_command, "GENERATED COMMAND — AUTO-EXECUTING")
    print(f"\n   📅 Execution Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        result = subprocess.run(
            shlex.split(kubectl_command),
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print(f"\n   ✅ EXECUTION SUCCESS")
            print(f"   Output: {result.stdout.strip()}")
            return {"status": "SUCCESS", "command": kubectl_command, "output": result.stdout.strip(), "error": None}
        else:
            print(f"\n   ❌ EXECUTION FAILED")
            print(f"   Error: {result.stderr.strip()}")
            return {"status": "FAILED", "command": kubectl_command, "output": None, "error": result.stderr.strip()}
    except Exception as e:
        print(f"\n   ❌ EXECUTION ERROR: {str(e)}")
        return {"status": "ERROR", "command": kubectl_command, "output": None, "error": str(e)}


def extract_confidence(analysis: str) -> float:
    patterns = [
        r"(\d+)%\s*confident",
        r"confidence[:\s]+(\d+)%",
        r"(\d+)%\s*confidence",
        r"confidence\s*of\s*(\d+)%"
    ]
    for pattern in patterns:
        match = re.search(pattern, analysis.lower())
        if match:
            return float(match.group(1))
    return 50.0


def calculate_evidence_score(retrieved_incidents: list) -> float:
    """Weighted-average match score across however many incidents were
    retrieved (retrieval is now floor+ceiling, so this can be 1 to top_k,
    not always exactly 3). Weight decays by rank (1/(i+1)), normalized so
    weights always sum to 1 regardless of list length — the top match
    still dominates, but a longer list doesn't silently reweight everything
    the way a fixed 3-slot table would once retrieval returns 4-6 results."""
    if not retrieved_incidents:
        return 0.0
    n = len(retrieved_incidents)
    raw_weights = [1 / (i + 1) for i in range(n)]
    weight_total = sum(raw_weights)
    total = 0.0
    for i, inc in enumerate(retrieved_incidents):
        weight = raw_weights[i] / weight_total
        total += inc['score'] * weight
    return round(total, 2)


def extract_action(analysis: str) -> str:
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": """You are an action classifier.
Reply with ONLY one of these exact strings:
restart_pod, scale_deployment, rollback_deployment, delete_pod,
modify_config, increase_memory_limit, rollback_production_db,
delete_namespace, unknown"""
            },
            {
                "role": "user",
                "content": f"What is the primary recommended action?\n\n{analysis}"
            }
        ],
        temperature=0,
        max_tokens=20,
        timeout=30
    )
    action = response.choices[0].message.content.strip().lower()
    for key in POLICY.keys():
        if key in action:
            return key
    return "unknown"


def independent_verification(incident: str, original_analysis: str,
                              retrieved_incidents: list, action: str) -> dict:
    context = ""
    for i, inc in enumerate(retrieved_incidents):
        context += f"""
Historical Incident {i+1} (Match: {inc['score']}%):
ID: {inc['id']}
Service: {inc['metadata']['service']}
Type: {inc['metadata']['incident_type']}
Details: {inc['clean_document']}
---"""

    verification_prompt = f"""You are a SENIOR DevOps engineer reviewing an AI diagnosis.

CURRENT INCIDENT: {incident}

HISTORICAL MEMORY:
{context}

AI DIAGNOSIS:
{original_analysis}

PROPOSED ACTION: {action}

Respond in this EXACT format. One word only after each label line:

VERIFICATION_VERDICT: AGREE
REASONING: your reasoning
EVIDENCE_QUALITY: STRONG
EVIDENCE_EXPLANATION: your explanation
ALTERNATIVE_CAUSE: NONE
SAFE_TO_EXECUTE: YES
SAFE_EXPLANATION: your explanation
VERIFIER_CONFIDENCE: 85

Rules:
- VERIFICATION_VERDICT must be: AGREE, DISAGREE, or NEEDS_MORE_INFO
- EVIDENCE_QUALITY must be: STRONG, MODERATE, or WEAK
- SAFE_TO_EXECUTE must be: YES, NO, or CONDITIONAL
- VERIFIER_CONFIDENCE must be a number only"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a senior DevOps engineer. Follow the exact output format. One word after each label."
            },
            {
                "role": "user",
                "content": verification_prompt
            }
        ],
        temperature=0.1,
        max_tokens=600,
        timeout=30
    )

    verification_text = response.choices[0].message.content

    def extract_field_clean(text, field):
        pattern = rf"^{field}:\s*(\S+)"
        match = re.search(pattern, text, re.MULTILINE)
        return match.group(1).strip() if match else "UNKNOWN"

    def extract_field_full(text, field, next_field):
        pattern = rf"{field}:\s*(.*?)(?={next_field}:|$)"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else "UNKNOWN"

    verdict = extract_field_clean(verification_text, "VERIFICATION_VERDICT")
    evidence_quality = extract_field_clean(verification_text, "EVIDENCE_QUALITY")
    safe_to_execute = extract_field_clean(verification_text, "SAFE_TO_EXECUTE")
    alternative_cause = extract_field_full(verification_text, "ALTERNATIVE_CAUSE", "SAFE_TO_EXECUTE")

    conf_raw = extract_field_clean(verification_text, "VERIFIER_CONFIDENCE")
    conf_match = re.search(r"(\d+)", conf_raw)
    verifier_confidence = float(conf_match.group(1)) if conf_match else 50.0

    return {
        "verdict": verdict,
        "evidence_quality": evidence_quality,
        "safe_to_execute": safe_to_execute,
        "alternative_cause": alternative_cause,
        "verifier_confidence": verifier_confidence,
        "full_verification": verification_text
    }


def verify_decision(incident: str, analysis: str,
                    retrieved_incidents: list, namespace: str = None,
                    auto_execute: bool = True) -> dict:

    print("\n" + "─"*60)
    print("🔍  DECISION VERIFICATION ENGINE")
    print("─"*60)

    if not namespace:
        print("   ⚠️  No namespace was passed in — the caller didn't resolve "
              "one (or this is a node-level incident with no namespace). "
              "The kubectl generator will fall back to extracting it from "
              "the incident text, or 'default' if it truly isn't present "
              "anywhere. It will NOT silently assume 'production'.")

    llm_confidence = extract_confidence(analysis)
    evidence_score = calculate_evidence_score(retrieved_incidents)

    print(f"\n   📊 Initial Metrics:")
    print(f"   ├── LLM Confidence       : {llm_confidence}%")
    print(f"   └── Evidence Score       : {evidence_score}%")

    print(f"\n   🎯 Identifying recommended action...")
    action = extract_action(analysis)
    policy = POLICY.get(action, {
        "risk": "HIGH", "min_confidence": 90,
        "min_evidence_score": 80, "auto_execute": False
    })

    print(f"   ├── Action               : {action}")
    print(f"   ├── Risk Level           : {policy['risk']}")
    print(f"   └── Auto-Execute Allowed : {policy['auto_execute']}")

    print(f"\n   🤖 Running independent verification...")
    verification = independent_verification(
        incident=incident,
        original_analysis=analysis,
        retrieved_incidents=retrieved_incidents,
        action=action
    )

    print(f"\n   📋 Verifier Results:")
    print(f"   ├── Verdict              : {verification['verdict']}")
    print(f"   ├── Evidence Quality     : {verification['evidence_quality']}")
    print(f"   ├── Safe to Execute      : {verification['safe_to_execute']}")
    print(f"   ├── Verifier Confidence  : {verification['verifier_confidence']}%")
    print(f"   └── Alternative Cause    : {verification['alternative_cause'][:80]}")

    print(f"\n   📝 Full Verifier Reasoning:")
    print("   " + verification['full_verification'].replace("\n", "\n   "))

    checks = {
        "llm_confidence_ok": llm_confidence >= policy['min_confidence'],
        "evidence_score_ok": evidence_score >= policy['min_evidence_score'],
        "verifier_agrees": verification['verdict'] == "AGREE",
        "evidence_strong": verification['evidence_quality'] in ["STRONG", "MODERATE"],
        "safe_to_execute": verification['safe_to_execute'] in ["YES", "CONDITIONAL"],
        "auto_execute_allowed": policy['auto_execute']
    }

    print(f"\n   ✅ Policy Checks:")
    print(f"   ├── LLM confidence OK    : {checks['llm_confidence_ok']} ({llm_confidence}% >= {policy['min_confidence']}%)")
    print(f"   ├── Evidence score OK    : {checks['evidence_score_ok']} ({evidence_score}% >= {policy['min_evidence_score']}%)")
    print(f"   ├── Verifier agrees      : {checks['verifier_agrees']} ({verification['verdict']})")
    print(f"   ├── Evidence strong      : {checks['evidence_strong']} ({verification['evidence_quality']})")
    print(f"   ├── Safe to execute      : {checks['safe_to_execute']} ({verification['safe_to_execute']})")
    print(f"   └── Policy allows auto   : {checks['auto_execute_allowed']}")

    if (checks['llm_confidence_ok'] and checks['evidence_score_ok'] and
            checks['verifier_agrees'] and checks['evidence_strong'] and
            checks['safe_to_execute'] and checks['auto_execute_allowed']):
        final_verdict = "APPROVED"
        final_reason = "All checks passed. Executing recovery action now."
        emoji = "✅"
    elif (checks['llm_confidence_ok'] and checks['evidence_score_ok'] and
          checks['verifier_agrees'] and not checks['auto_execute_allowed']):
        final_verdict = "ESCALATE"
        final_reason = "Diagnosis verified but policy requires human approval."
        emoji = "⚠️"
    elif (checks['llm_confidence_ok'] and checks['evidence_score_ok'] and
          not checks['verifier_agrees']):
        final_verdict = "HOLD"
        final_reason = "LLM confident but verifier disagrees. Human review required."
        emoji = "🔴"
    elif verification['verdict'] == "NEEDS_MORE_INFO":
        final_verdict = "NEEDS_MORE_INFO"
        final_reason = "Insufficient information. Collect more data before acting."
        emoji = "🟡"
    else:
        final_verdict = "REJECTED"
        final_reason = "Failed critical checks. Escalate to senior engineer."
        emoji = "❌"

    print(f"\n{'═'*60}")
    print(f"   {emoji}  FINAL VERDICT  : {final_verdict}")
    print(f"   📌  REASON        : {final_reason}")
    print(f"   🔧  ACTION        : {action}")
    print(f"   ⚠️   RISK          : {policy['risk']}")
    print(f"   🕐  TIMESTAMP     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'═'*60}")

    execution_result = None
    if final_verdict == "APPROVED":
        if auto_execute:
            print(f"\n   🚀 EXECUTING RECOVERY...")
            execution_result = execute_recovery(action, analysis, incident, namespace)
        else:
            print(f"\n   ⏸  Execution deferred to Remediation Agent.")
    elif final_verdict == "ESCALATE":
        print(f"\n   📢 Human approval required for action: {action}")
    elif final_verdict == "HOLD":
        print(f"\n   🛑 BLOCKED. Alternative cause: {verification['alternative_cause'][:120]}")
    else:
        print(f"\n   🛑 REJECTED. Do not execute. Investigate further.")

    return {
        "final_verdict": final_verdict,
        "final_reason": final_reason,
        "action": action,
        "llm_confidence": llm_confidence,
        "evidence_score": evidence_score,
        "risk_level": policy['risk'],
        "verifier_verdict": verification['verdict'],
        "verifier_confidence": verification['verifier_confidence'],
        "evidence_quality": verification['evidence_quality'],
        "alternative_cause": verification['alternative_cause'],
        "checks": checks,
        "execution_result": execution_result,
        "timestamp": datetime.now().isoformat()
    }