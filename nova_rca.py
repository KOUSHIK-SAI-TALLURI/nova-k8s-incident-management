import os
import chromadb
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
from decision_verification import verify_decision

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./nova_memory")
collection = chroma_client.get_or_create_collection(name="incidents")
model = SentenceTransformer('BAAI/bge-base-en-v1.5')

all_data = collection.get()
all_documents = all_data['documents']
all_ids = all_data['ids']
all_metadatas = all_data['metadatas']

tokenized_corpus = [doc.lower().split() for doc in all_documents]
bm25 = BM25Okapi(tokenized_corpus)


def retrieve_similar_incidents(query: str, top_k: int = 6,
                                min_score: float = 35.0,
                                candidate_pool: int = 15,
                                filter_severity: str = None,
                                filter_namespace: str = None):
    """Floor + ceiling retrieval.

    Pulls a larger candidate pool (candidate_pool) from vector + BM25 so a
    correct match sitting outside the old fixed top-3 window still has a
    chance to surface. From that pool:
      - drop anything below min_score (raw noise shouldn't reach the RCA)
      - keep at most top_k (bounds prompt size/cost)
      - always return at least 1 result, even below min_score, so the RCA
        is never left with zero grounding to reason over.
    """
    conditions = []
    if filter_severity:
        conditions.append({"severity": {"$eq": filter_severity}})
    if filter_namespace:
        conditions.append({"namespace": {"$eq": filter_namespace}})

    if len(conditions) == 0:
        where_filter = None
    elif len(conditions) == 1:
        where_filter = conditions[0]
    else:
        where_filter = {"$and": conditions}

    prefixed_query = f"Represent this DevOps incident for retrieval: {query}"
    query_embedding = model.encode(
        [prefixed_query],
        normalize_embeddings=True
    ).tolist()

    # Pull a wider candidate pool than top_k so a correct match ranked
    # below the old fixed window can still be considered.
    pool_size = max(candidate_pool, top_k)

    vector_results = collection.query(
        query_embeddings=query_embedding,
        n_results=pool_size,
        where=where_filter
    )

    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:pool_size]

    combined_scores = {}
    for id_, dist in zip(vector_results['ids'][0], vector_results['distances'][0]):
        combined_scores[id_] = (1 - dist) * 0.7

    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    for i in bm25_top_indices:
        id_ = all_ids[i]
        combined_scores[id_] = combined_scores.get(id_, 0) + (bm25_scores[i] / max_bm25) * 0.3

    # Rank the full candidate pool first, then apply floor + ceiling.
    ranked_ids = sorted(combined_scores, key=combined_scores.get, reverse=True)

    above_floor = [id_ for id_ in ranked_ids if combined_scores[id_] * 100 >= min_score]

    if above_floor:
        sorted_ids = above_floor[:top_k]
    else:
        # Nothing cleared the floor — guarantee at least 1 so the RCA
        # never gets an empty context and has to hallucinate ungrounded.
        sorted_ids = ranked_ids[:1]

    retrieved = []
    for id_ in sorted_ids:
        idx = all_ids.index(id_)
        raw_doc = all_documents[idx]
        clean_doc = raw_doc.replace(
            "Represent this DevOps incident for retrieval: ", ""
        ).strip()
        retrieved.append({
            "id": id_,
            "score": round(combined_scores[id_] * 100, 2),
            "raw_document": raw_doc,
            "clean_document": clean_doc,
            "metadata": all_metadatas[idx]
        })

    return retrieved


def print_retrieved_incidents(similar: list):
    print("\n" + "─"*60)
    print("📚  HISTORICAL MEMORY — SIMILAR PAST INCIDENTS RETRIEVED")
    print("─"*60)

    for i, inc in enumerate(similar):
        meta = inc['metadata']
        print(f"""
┌─────────────────────────────────────────────────────────┐
│  Historical Incident {i+1}  │  Match Score: {inc['score']}%
├─────────────────────────────────────────────────────────┤
│  Incident ID   : {inc['id']}
│  Service       : {meta['service']}
│  Namespace     : {meta['namespace']}
│  Severity      : {meta['severity']}
│  Type          : {meta['incident_type']}
├─────────────────────────────────────────────────────────┤
│  {inc['clean_document'].replace('. ', '.'+chr(10)+'│  ')}
└─────────────────────────────────────────────────────────┘""")


def generate_rca_analysis(incident: str, similar: list) -> str:
    """Build the RCA prompt from retrieved incidents and call the LLM.
    Pulled out of nova_analyze() so the RCAAgent can reuse it directly
    without duplicating the prompt."""

    context = ""
    for i, inc in enumerate(similar):
        meta = inc['metadata']
        context += f"""
Historical Incident {i+1}:
- Incident ID   : {inc['id']}
- Service       : {meta['service']}
- Namespace     : {meta['namespace']}
- Severity      : {meta['severity']}
- Type          : {meta['incident_type']}
- Match Score   : {inc['score']}%
- Details       : {inc['clean_document']}
---
"""

    prompt = f"""You are NOVA, an intelligent DevOps incident management AI.

You have retrieved the following similar historical incidents from memory:

{context}

CURRENT INCIDENT:
{incident}

Analyze this incident and respond in this exact format:

ROOT CAUSE
What is the most likely root cause? Be specific. Reference the historical incident ID.

CONFIDENCE
Give a percentage (0-100%) and explain why.

EVIDENCE
List the specific evidence from current incident and historical memory.

SIMILAR INCIDENT
Which historical incident matches best? State the Incident ID, service name, and why.

RECOMMENDED ACTION
Give step-by-step actions a new engineer can follow.

RISK LEVEL
State LOW, MEDIUM, or HIGH and explain."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are NOVA, a trustworthy DevOps incident management AI. Always reference specific historical incident IDs. Write clearly enough for a new engineer to understand."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.1,
        max_tokens=1000,
        timeout=60
    )

    return response.choices[0].message.content


def nova_analyze(incident: str,
                 filter_severity: str = None,
                 filter_namespace: str = None):

    print(f"\n{'='*60}")
    print(f"🚨  NEW INCIDENT DETECTED")
    print(f"{'='*60}")
    print(f"   {incident}")
    if filter_severity:
        print(f"   Filter → Severity : {filter_severity}")
    if filter_namespace:
        print(f"   Filter → Namespace: {filter_namespace}")
    print(f"{'='*60}")

    # Step 1: Retrieve
    similar = retrieve_similar_incidents(
        incident,
        filter_severity=filter_severity,
        filter_namespace=filter_namespace
    )

    # Step 2: Print retrieved incidents
    print_retrieved_incidents(similar)

    # Step 3: LLM RCA
    print("\n" + "─"*60)
    print("🤖  NOVA ROOT CAUSE ANALYSIS")
    print("─"*60)

    analysis = generate_rca_analysis(incident, similar)
    print(analysis)
    print("\n" + "="*60)

    # Step 4: Decision Verification
    # No hardcoded "production" fallback — if filter_namespace wasn't
    # passed, namespace stays None and verify_decision/generate_kubectl_command
    # will surface that loudly instead of silently guessing, same as the
    # Coordinator path in agents.py.
    verification = verify_decision(
        incident=incident,
        analysis=analysis,
        retrieved_incidents=similar,
        namespace=filter_namespace
    )

    return {
        "incident": incident,
        "retrieved_incidents": similar,
        "analysis": analysis,
        "verification": verification
    }


if __name__ == "__main__":

    # Test 1: Memory incident
    nova_analyze(
        "Pod is being killed. Memory usage at 98%. OOMKilled in logs.",
        filter_namespace="production"
    )

    # Test 2: Deployment incident
    nova_analyze(
        "New deployment pods stuck. Not becoming ready. Health check failing."
    )

    # Test 3: Network incident
    nova_analyze(
        "Services cannot communicate. DNS lookup failing. High latency observed."
    )