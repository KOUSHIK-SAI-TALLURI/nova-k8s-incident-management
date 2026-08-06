"""
NOVA Agent Layer
-----------------
Thin agent wrappers around NOVA's existing, already-working functions.
This gives the pipeline real agent boundaries and a coordinator that
dispatches between them — instead of one script calling functions in a
fixed sequence — without changing any of the underlying retrieval, RCA,
verification, or execution logic. Each agent owns one responsibility;
the Coordinator owns the routing decision between them.
"""

from nova_rca import (
    retrieve_similar_incidents,
    print_retrieved_incidents,
    generate_rca_analysis
)
from decision_verification import verify_decision, execute_recovery, generate_kubectl_command, print_command_box, print_consequence_warning


class RCAAgent:
    """Owns retrieval from memory and root cause analysis."""

    def analyze(self, incident: str, filter_severity: str = None,
                filter_namespace: str = None, retrieval_query: str = None) -> dict:
        similar = retrieve_similar_incidents(
            retrieval_query or incident,
            filter_severity=filter_severity,
            filter_namespace=filter_namespace
        )
        print_retrieved_incidents(similar)

        print("\n" + "─"*60)
        print("🤖  NOVA ROOT CAUSE ANALYSIS")
        print("─"*60)
        analysis = generate_rca_analysis(incident, similar)
        print(analysis)
        print("\n" + "="*60)

        return {
            "incident": incident,
            "retrieved_incidents": similar,
            "analysis": analysis
        }


class VerificationAgent:
    """Owns the Decision Verification Engine's judgment.
    Deliberately does NOT execute anything itself — it only decides
    whether an action is safe and policy-approved. The Coordinator
    is responsible for deciding whether to act on that verdict."""

    def verify(self, incident: str, analysis: str,
               retrieved_incidents: list, namespace: str = None) -> dict:
        return verify_decision(
            incident=incident,
            analysis=analysis,
            retrieved_incidents=retrieved_incidents,
            namespace=namespace,
            auto_execute=False
        )


class RemediationAgent:
    """Owns real-world execution of an already-approved action.
    Only ever called by the Coordinator after VerificationAgent
    has returned an APPROVED verdict."""

    def remediate(self, action: str, analysis: str, incident: str = "",
                  namespace: str = None) -> dict:
        return execute_recovery(action, analysis, incident, namespace)

    def preview(self, action: str, analysis: str, incident: str = "",
                namespace: str = None) -> str:
        """Generate a suggested command WITHOUT executing it. Used when the
        verifier agrees an action is safe but policy blocked auto-execution
        (ESCALATE) or a threshold check narrowly failed (REJECTED) — the
        human reviewer gets something actionable instead of just 'REJECTED,
        investigate further' with no next step."""
        return generate_kubectl_command(action, analysis, incident, namespace)


class Coordinator:
    """Routes an incident through RCA -> Verification -> Remediation.
    This is the single place that decides what happens next based on
    each agent's output — the seam that makes this a dispatched
    multi-agent system rather than a fixed linear script."""

    def __init__(self):
        self.rca_agent = RCAAgent()
        self.verification_agent = VerificationAgent()
        self.remediation_agent = RemediationAgent()

    def handle_incident(self, incident: str, filter_severity: str = None,
                         filter_namespace: str = None,
                         retrieval_query: str = None,
                         namespace: str = None,
                         auto_remediate: bool = True) -> dict:
        """
        filter_namespace: used ONLY to restrict RAG retrieval to incidents
            from a specific namespace. Left as None by the watcher on
            purpose (see watcher.py) so retrieval isn't zeroed out by a
            corpus/cluster namespace mismatch.
        namespace: the REAL namespace the incident is actually happening
            in (e.g. from the Prometheus label). This is what gets used
            for verification and execution — it must never fall back to
            a hardcoded value like "production". If it's None, that means
            no real namespace was resolved upstream, and decision_verification
            will surface that loudly instead of guessing.
        auto_remediate: False for incident types that have no valid action
            in POLICY (e.g. NodeNotReady — no namespace, no matching action).
            When False, RCA still runs for context, but verification and
            remediation are skipped entirely rather than forcing the LLM
            to invent a namespace or force-fit the wrong action.
        """

        print(f"\n{'='*60}")
        print(f"🚨  NEW INCIDENT DETECTED")
        print(f"{'='*60}")
        print(f"   {incident}")
        if filter_severity:
            print(f"   Filter → Severity : {filter_severity}")
        if filter_namespace:
            print(f"   Filter → Namespace: {filter_namespace}")
        if namespace:
            print(f"   Target  → Namespace: {namespace}")
        print(f"{'='*60}")

        # 1. Delegate to RCAAgent (always — useful context even when this
        #    incident type can't be auto-remediated)
        rca_result = self.rca_agent.analyze(
            incident, filter_severity, filter_namespace, retrieval_query
        )

        if not auto_remediate:
            print(f"\n   📢 ESCALATED — no valid auto-remediation action exists "
                  f"for this incident type. Skipping verification and "
                  f"remediation; a human needs to handle this directly.")
            return {
                "incident": incident,
                "retrieved_incidents": rca_result["retrieved_incidents"],
                "analysis": rca_result["analysis"],
                "verification": {
                    "final_verdict": "ESCALATE_NO_ACTION",
                    "final_reason": "No POLICY action addresses this incident type "
                                     "(and/or no namespace applies). Requires human response.",
                    "action": None,
                    "execution_result": None
                }
            }

        # 2. Delegate to VerificationAgent
        verification = self.verification_agent.verify(
            incident=incident,
            analysis=rca_result["analysis"],
            retrieved_incidents=rca_result["retrieved_incidents"],
            namespace=namespace
        )

        # 3. Coordinator decides: only dispatch to RemediationAgent
        #    if VerificationAgent approved the action.
        execution_result = None
        if verification["final_verdict"] == "APPROVED":
            print(f"\n   🚀 COORDINATOR DISPATCHING TO REMEDIATION AGENT...")
            execution_result = self.remediation_agent.remediate(
                verification["action"],
                rca_result["analysis"],
                incident,
                namespace
            )
            verification["execution_result"] = execution_result

        elif (verification["checks"].get("verifier_agrees") and
              verification["checks"].get("safe_to_execute")):
            # Policy blocked auto-execution (ESCALATE) or a threshold check
            # narrowly failed (REJECTED), but the independent verifier still
            # thinks the action is safe. Generate the command for a human to
            # review rather than leaving them with nothing to act on.
            suggested_command = self.remediation_agent.preview(
                verification["action"],
                rca_result["analysis"],
                incident,
                namespace
            )
            verification["suggested_command"] = suggested_command
            print_consequence_warning(verification["action"])
            print_command_box(suggested_command, "SUGGESTED COMMAND — NOT EXECUTED, REVIEW BEFORE RUNNING")

        return {
            "incident": incident,
            "retrieved_incidents": rca_result["retrieved_incidents"],
            "analysis": rca_result["analysis"],
            "verification": verification
        }