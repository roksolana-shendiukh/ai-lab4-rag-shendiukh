import json, time
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMB_MODEL, CHUNKS, QUERY_PREFIX, PASSAGE_PREFIX


def get_embeddings():
    """Embedding model with normalized vectors (cosine similarity = dot product)."""
    enc = {"normalize_embeddings": True}
    return HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        encode_kwargs=dict(enc, prompt=PASSAGE_PREFIX) if PASSAGE_PREFIX else enc,
        query_encode_kwargs=dict(enc, prompt=QUERY_PREFIX) if QUERY_PREFIX else enc)


def load_chunks(name):
    return json.load(open(f"data/chunks_{name}.json", encoding="utf-8"))


def load_store(name, emb):
    return InMemoryVectorStore.load(f"index/{name}.json", emb)


if __name__ == "__main__":
    emb = get_embeddings()
    for name in CHUNKS:
        # Each chunk keeps its id and page number as metadata
        docs = [Document(page_content=c["text"],
                         metadata={"chunk_id": c["chunk_id"], "page": c["page"]})
                for c in load_chunks(name)]
        t0 = time.perf_counter()
        store = InMemoryVectorStore.from_documents(docs, emb)
        store.dump(f"index/{name}.json")
        dt = time.perf_counter() - t0
        print(f"{name}: {len(docs)} chunks, indexing time {dt:.1f} s")