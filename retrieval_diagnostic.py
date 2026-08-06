"""
Retrieval Diagnostic
---------------------
Breaks down the combined hybrid score into its raw vector-similarity and
raw BM25 components, per candidate incident, so we can see exactly which
signal is pulling a given incident up or down in the ranking. Run this
after nova_rca.py's setup is importable (same folder, same venv).
"""

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

chroma_client = chromadb.PersistentClient(path="./nova_memory")
collection = chroma_client.get_or_create_collection(name="incidents")
model = SentenceTransformer('BAAI/bge-base-en-v1.5')

all_data = collection.get()
all_documents = all_data['documents']
all_ids = all_data['ids']
all_metadatas = all_data['metadatas']

tokenized_corpus = [doc.lower().split() for doc in all_documents]
bm25 = BM25Okapi(tokenized_corpus)

QUERY = (
    "Pod memory usage has been climbing steadily over the past 6 hours, now at "
    "85% of a 2Gi memory limit — a limit that is already generous compared to "
    "what this service normally needs. No OOMKilled event or restart has "
    "occurred yet, but the trend suggests the limit will be hit within 24 hours "
    "if unaddressed. This same gradual climb pattern has been observed after "
    "each of the last three redeploys."
)

CHECK_IDS = ["INC001", "INC011", "INC017"]

prefixed_query = f"Represent this DevOps incident for retrieval: {QUERY}"
query_embedding = model.encode([prefixed_query], normalize_embeddings=True)[0]

tokenized_query = QUERY.lower().split()
bm25_scores = bm25.get_scores(tokenized_query)
max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1

print(f"{'ID':<8}{'Vector cos-sim':<18}{'Vector*0.7':<14}{'BM25 raw':<12}{'BM25 norm*0.3':<16}{'Combined %':<12}")
print("-" * 80)

for id_ in CHECK_IDS:
    idx = all_ids.index(id_)
    doc_embedding = model.encode([all_documents[idx]], normalize_embeddings=True)[0]
    cos_sim = float(np.dot(query_embedding, doc_embedding))  # normalized, so dot = cosine
    vector_weighted = cos_sim * 0.7

    raw_bm25 = bm25_scores[idx]
    bm25_weighted = (raw_bm25 / max_bm25) * 0.3

    combined = round((vector_weighted + bm25_weighted) * 100, 2)

    print(f"{id_:<8}{cos_sim:<18.4f}{vector_weighted:<14.4f}{raw_bm25:<12.4f}{bm25_weighted:<16.4f}{combined:<12}")

print("\nFor reference, raw document text used for embedding/BM25:\n")
for id_ in CHECK_IDS:
    idx = all_ids.index(id_)
    clean = all_documents[idx].replace("Represent this DevOps incident for retrieval: ", "")
    print(f"[{id_}] {clean}\n")
