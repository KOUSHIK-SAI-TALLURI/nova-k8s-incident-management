import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import numpy as np

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./nova_memory")
collection = client.get_or_create_collection(name="incidents")
model = SentenceTransformer('BAAI/bge-base-en-v1.5')

# Load all documents for BM25
all_data = collection.get()
all_documents = all_data['documents']
all_ids = all_data['ids']
all_metadatas = all_data['metadatas']

# Build BM25 index
tokenized_corpus = [doc.lower().split() for doc in all_documents]
bm25 = BM25Okapi(tokenized_corpus)

def hybrid_search(query: str, top_k: int = 3,
                  filter_severity: str = None,
                  filter_service: str = None,
                  filter_namespace: str = None,
                  filter_incident_type: str = None):

    print(f"\n{'='*60}")
    print(f"NEW INCIDENT: {query}")

    # Build metadata filter
    conditions = []
    if filter_severity:
        conditions.append({"severity": {"$eq": filter_severity}})
    if filter_service:
        conditions.append({"service": {"$eq": filter_service}})
    if filter_namespace:
        conditions.append({"namespace": {"$eq": filter_namespace}})
    if filter_incident_type:
        conditions.append({"incident_type": {"$eq": filter_incident_type}})

    if len(conditions) == 0:
        where_filter = None
    elif len(conditions) == 1:
        where_filter = conditions[0]
    else:
        where_filter = {"$and": conditions}

    if where_filter:
        print(f"FILTERS: {where_filter}")
    print(f"{'='*60}")

    # BGE query prefix
    prefixed_query = f"Represent this DevOps incident for retrieval: {query}"

    # Vector search
    query_embedding = model.encode(
        [prefixed_query],
        normalize_embeddings=True
    ).tolist()

    vector_results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_filter
    )

    # BM25 keyword search
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k]

    # Combine scores (hybrid)
    vector_ids = vector_results['ids'][0]
    vector_distances = vector_results['distances'][0]

    combined_scores = {}

    # Add vector scores (weight: 0.7)
    for idx, (id_, dist) in enumerate(zip(vector_ids, vector_distances)):
        vector_score = (1 - dist) * 0.7
        combined_scores[id_] = combined_scores.get(id_, 0) + vector_score

    # Add BM25 scores (weight: 0.3)
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    for i in bm25_top_indices:
        id_ = all_ids[i]
        bm25_normalized = (bm25_scores[i] / max_bm25) * 0.3
        combined_scores[id_] = combined_scores.get(id_, 0) + bm25_normalized

    # Sort by combined score
    sorted_ids = sorted(combined_scores,
                        key=combined_scores.get,
                        reverse=True)[:top_k]

    print(f"\nTop {top_k} similar past incidents:\n")

    retrieved = []
    for rank, id_ in enumerate(sorted_ids):
        score = round(combined_scores[id_] * 100, 2)
        idx = all_ids.index(id_)
        doc = all_documents[idx]
        meta = all_metadatas[idx]

        print(f"--- Match {rank+1} | Score: {score}% ---")
        print(f"Service: {meta['service']} | "
              f"Severity: {meta['severity']} | "
              f"Type: {meta['incident_type']}")
        print(doc.replace("Represent this DevOps incident for retrieval: ", "").strip())
        print()

        retrieved.append({
            "id": id_,
            "score": score,
            "document": doc,
            "metadata": meta
        })

    return retrieved


if __name__ == "__main__":

    # Test 1: Memory issue - no filter
    hybrid_search(
        "Pod is being killed. Memory usage at 98%. OOMKilled in logs."
    )

    # Test 2: Deployment issue - no filter
    hybrid_search(
        "New deployment pods stuck. Not becoming ready. Health check failing."
    )

    # Test 3: Network issue - no filter
    hybrid_search(
        "Services cannot communicate. DNS lookup failing. High latency observed."
    )

    # Test 4: Metadata filtering demo
    hybrid_search(
        "Pod crashing with memory errors",
        filter_severity="HIGH",
        filter_namespace="production"
    )