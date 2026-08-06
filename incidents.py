import chromadb
from sentence_transformers import SentenceTransformer
import json
import os
import shutil

# Clear old ChromaDB to re-embed with new model
if os.path.exists("./nova_memory"):
    shutil.rmtree("./nova_memory")
    print("🗑️  Cleared old ChromaDB")

# Initialize ChromaDB
client = chromadb.PersistentClient(path="./nova_memory")
collection = client.get_or_create_collection(name="incidents")

# Better embedding model

model = SentenceTransformer('BAAI/bge-base-en-v1.5')
print("✅ Model loaded")

incidents = [
    {
        "id": "INC001",
        "description": "Pod crashed due to memory limit exceeded. OOMKilled error observed.",
        "root_cause": "Memory leak in application code. Container limit set too low.",
        "resolution": "Increased memory limit to 512Mi. Restarted pod. Fixed memory leak.",
        "severity": "HIGH",
        "service": "payment-service",
        "namespace": "production",
        "incident_type": "OOMKilled"
    },
    {
        "id": "INC002",
        "description": "CPU usage spiked to 95% causing slow response times.",
        "root_cause": "Infinite loop in request handler triggered by malformed input.",
        "resolution": "Identified and fixed infinite loop. Added input validation.",
        "severity": "HIGH",
        "service": "api-gateway",
        "namespace": "production",
        "incident_type": "CPUSpike"
    },
    {
        "id": "INC003",
        "description": "Service unavailable. All pods showing CrashLoopBackOff.",
        "root_cause": "Wrong environment variable in deployment config.",
        "resolution": "Corrected environment variable. Redeployed application.",
        "severity": "CRITICAL",
        "service": "order-service",
        "namespace": "production",
        "incident_type": "CrashLoopBackOff"
    },
    {
        "id": "INC004",
        "description": "Database connection timeout. Application returning 500 errors.",
        "root_cause": "Database pod ran out of storage. PVC full.",
        "resolution": "Expanded PVC size. Cleared old logs from database.",
        "severity": "HIGH",
        "service": "database",
        "namespace": "production",
        "incident_type": "ConnectionTimeout"
    },
    {
        "id": "INC005",
        "description": "Network latency increased to 2000ms between services.",
        "root_cause": "DNS resolution failure due to CoreDNS pod crash.",
        "resolution": "Restarted CoreDNS pods. Network latency returned to normal.",
        "severity": "MEDIUM",
        "service": "core-dns",
        "namespace": "kube-system",
        "incident_type": "NetworkLatency"
    },
    {
        "id": "INC006",
        "description": "Deployment rollout stuck. New pods not becoming ready.",
        "root_cause": "New image failed health check. Readiness probe failing.",
        "resolution": "Rolled back to previous image version. Fixed health check endpoint.",
        "severity": "HIGH",
        "service": "frontend",
        "namespace": "production",
        "incident_type": "DeploymentStuck"
    },
    {
        "id": "INC007",
        "description": "Node NotReady. All pods on node evicted.",
        "root_cause": "Node disk pressure. Disk usage reached 95%.",
        "resolution": "Cleared unused images and logs. Node recovered.",
        "severity": "CRITICAL",
        "service": "node",
        "namespace": "kube-system",
        "incident_type": "NodeFailure"
    },
    {
        "id": "INC008",
        "description": "Horizontal Pod Autoscaler not scaling. Traffic increasing.",
        "root_cause": "Metrics server not running. HPA could not read CPU metrics.",
        "resolution": "Reinstalled metrics server. HPA resumed normal scaling.",
        "severity": "MEDIUM",
        "service": "hpa",
        "namespace": "kube-system",
        "incident_type": "ScalingFailure"
    },
    {
        "id": "INC009",
        "description": "Secret not found error. Application failing to start.",
        "root_cause": "Kubernetes secret deleted accidentally during cleanup.",
        "resolution": "Recreated secret from backup. Application started successfully.",
        "severity": "HIGH",
        "service": "auth-service",
        "namespace": "production",
        "incident_type": "SecretMissing"
    },
    {
        "id": "INC010",
        "description": "Persistent volume claim pending. Pod stuck in init state.",
        "root_cause": "Storage class not available in cluster.",
        "resolution": "Created correct storage class. PVC bound successfully.",
        "severity": "MEDIUM",
        "service": "database",
        "namespace": "production",
        "incident_type": "PVCPending"
    },
    {
        "id": "INC011",
        "description": "Memory usage gradually increasing over 6 hours. OOMKill imminent.",
        "root_cause": "Memory leak in third party library. Objects not garbage collected.",
        "resolution": "Scheduled rolling restart every 24 hours. Filed bug report upstream.",
        "severity": "MEDIUM",
        "service": "recommendation-service",
        "namespace": "production",
        "incident_type": "MemoryLeak"
    },
    {
        "id": "INC012",
        "description": "API gateway returning 503. Downstream service unreachable.",
        "root_cause": "Service selector label mismatch after deployment update.",
        "resolution": "Fixed label selector in service definition. Traffic restored.",
        "severity": "HIGH",
        "service": "api-gateway",
        "namespace": "production",
        "incident_type": "ServiceUnavailable"
    },
    {
        "id": "INC013",
        "description": "Liveness probe failing repeatedly. Pod restarting every 2 minutes.",
        "root_cause": "Application startup time exceeded probe initial delay.",
        "resolution": "Increased initialDelaySeconds from 10 to 60. Pod stabilized.",
        "severity": "MEDIUM",
        "service": "cart-service",
        "namespace": "production",
        "incident_type": "ProbeFailure"
    },
    {
        "id": "INC014",
        "description": "Cluster autoscaler not adding nodes. Pods in Pending state.",
        "root_cause": "Cloud provider quota exceeded for VM instances.",
        "resolution": "Requested quota increase. Pending pods scheduled after approval.",
        "severity": "HIGH",
        "service": "cluster-autoscaler",
        "namespace": "kube-system",
        "incident_type": "ScalingFailure"
    },
    {
        "id": "INC015",
        "description": "Config map update not reflected in running pods.",
        "root_cause": "Application caches config at startup. No hot reload implemented.",
        "resolution": "Performed rolling restart after config map update.",
        "severity": "LOW",
        "service": "config-service",
        "namespace": "production",
        "incident_type": "ConfigIssue"
    },
    {
        "id": "INC016",
        "description": "Ingress returning 404 for all routes after update.",
        "root_cause": "Ingress class annotation removed during helm upgrade.",
        "resolution": "Added back ingress class annotation. Routes restored.",
        "severity": "HIGH",
        "service": "ingress",
        "namespace": "production",
        "incident_type": "NetworkIssue"
    },
    {
        "id": "INC017",
        "description": "Pod evicted due to node memory pressure.",
        "root_cause": "No resource limits set. Pod consumed entire node memory.",
        "resolution": "Set memory limits on all containers. Added resource quotas.",
        "severity": "HIGH",
        "service": "shipping-service",
        "namespace": "production",
        "incident_type": "OOMKilled"
    },
    {
        "id": "INC018",
        "description": "RBAC error. Service account cannot access secrets.",
        "root_cause": "ClusterRole missing secrets read permission after policy update.",
        "resolution": "Updated ClusterRole to include secrets read permission.",
        "severity": "MEDIUM",
        "service": "auth-service",
        "namespace": "production",
        "incident_type": "RBACError"
    },
    {
        "id": "INC019",
        "description": "Cronjob not executing at scheduled time.",
        "root_cause": "Cluster timezone mismatch. Cron schedule in UTC but cluster in IST.",
        "resolution": "Adjusted cron schedule to UTC equivalent time.",
        "severity": "LOW",
        "service": "cronjob-service",
        "namespace": "production",
        "incident_type": "CronJobFailure"
    },
    {
        "id": "INC020",
        "description": "Service mesh sidecar injecting incorrectly. mTLS handshake failing.",
        "root_cause": "Namespace not labeled for automatic sidecar injection.",
        "resolution": "Added istio-injection=enabled label to namespace.",
        "severity": "HIGH",
        "service": "istio",
        "namespace": "istio-system",
        "incident_type": "NetworkIssue"
    },
    {
        "id": "INC021",
        "description": "Pod is stuck in CrashLoopBackOff. Container repeatedly failing to start after being killed by an external termination signal. No OOMKilled event, no memory pressure, no misconfigured environment variable — the container process itself was terminated (e.g. by a node-level or infrastructure kill event) and Kubernetes is repeatedly trying and failing to bring it back up cleanly.",
        "root_cause": "Container process terminated externally (SIGKILL from node/runtime-level disruption), not an application bug or config error. Kubernetes' restart backoff interprets the repeated terminations as CrashLoopBackOff.",
        "resolution": "Restarted the affected pod (deleted it so the Deployment would recreate it cleanly). Monitored restart count post-recreation; no config or code change was required since the root cause was external, not application-level.",
        "severity": "CRITICAL",
        "service": "cartservice",
        "namespace": "online-boutique",
        "incident_type": "CrashLoopBackOff"
    },
    {
        "id": "INC022",
        "description": "Pod has restarted more than 3 times in the last 10 minutes. Restart count climbing on the same pod instance rather than a new pod being created each time. No OOMKilled event and no memory limit breach — restarts are being caused by the container being killed repeatedly by an external signal rather than a resource or config issue.",
        "root_cause": "Repeated external container termination (node instability or infrastructure-level kill) causing the restart counter to climb without any underlying memory leak, OOM, or misconfiguration.",
        "resolution": "Restarted the pod to clear transient state and confirmed the restart counter stabilized afterward. No memory limit increase or config change was needed since the trigger was external, not resource exhaustion.",
        "severity": "HIGH",
        "service": "cartservice",
        "namespace": "online-boutique",
        "incident_type": "HighRestartRate"
    }
]

print("Seeding ChromaDB with incident history...")

documents = []
embeddings = []
ids = []
metadatas = []

for incident in incidents:
    # BGE models work better with this prefix
    full_text = f"Represent this DevOps incident for retrieval: " \
                f"Incident: {incident['description']} " \
                f"Root Cause: {incident['root_cause']} " \
                f"Resolution: {incident['resolution']} " \
                f"Severity: {incident['severity']}"

    documents.append(full_text)
    ids.append(incident['id'])
    metadatas.append({
        "severity": incident['severity'],
        "service": incident['service'],
        "namespace": incident['namespace'],
        "incident_type": incident['incident_type'],
        "id": incident['id']
    })

print("Generating embeddings with bge-large...")
embeddings = model.encode(documents, normalize_embeddings=True).tolist()

collection.add(
    documents=documents,
    embeddings=embeddings,
    ids=ids,
    metadatas=metadatas
)

print(f"✅ Successfully stored {len(incidents)} incidents in ChromaDB")
print(f"Collection size: {collection.count()} incidents")