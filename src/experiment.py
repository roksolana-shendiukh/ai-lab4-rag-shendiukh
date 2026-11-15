# src/experiment.py
"""E5: effect of chunk overlap (0 vs 25 %) at the same chunk size."""
import csv, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # save plots to files without a display
import matplotlib.pyplot as plt
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from chunking import make_chunks
from index import get_embeddings
from eval_retrieval import (load_questions, tokenize, bm25_ranking, rrf,
                            hit_at_k, reciprocal_rank, K_MAX)
from config import EMB_MODEL, TOP_K, THRESHOLD

# Same chunk size, overlap 0 and 25 %
CONFIGS = {"128/0": (128, 0), "128/32": (128, 32),
           "192/0": (192, 0), "192/48": (192, 48)}
ANSWER_CONFIG = "128/32"  # overlap config used to generate RAG answers


def build(pages, size, overlap, tok, emb):
    """Chunk the document, build dense index and BM25 for one configuration."""
    chunks = make_chunks(pages, size, overlap, tok)
    docs = [Document(page_content=c["text"],
                     metadata={"chunk_id": c["chunk_id"], "page": c["page"]})
            for c in chunks]
    t0 = time.perf_counter()
    store = InMemoryVectorStore.from_documents(docs, emb)
    index_time = time.perf_counter() - t0
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
    return chunks, store, bm25, index_time


def retrieve(q, chunks, store, bm25, ids):
    """Return dense ranking, hybrid ranking (chunk ids) and dense top-1 score."""
    hits = store.similarity_search_with_score(q, k=K_MAX)
    dense = [d.metadata["chunk_id"] for d, _ in hits]
    hybrid = rrf(dense, bm25_ranking(bm25, ids, q))[:K_MAX]
    return dense, hybrid, hits[0][1]


if __name__ == "__main__":
    import json
    pages = json.load(open("data/pages.json", encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    emb = get_embeddings()
    qs = load_questions()
    answerable = [q for q in qs if q["gold"]]
    ks = range(1, K_MAX + 1)

    rows, curves, built = [], {}, {}
    for cfg, (size, overlap) in CONFIGS.items():
        chunks, store, bm25, index_time = build(pages, size, overlap, tok, emb)
        built[cfg] = (chunks, store, bm25)
        page_of = {c["chunk_id"]: c["page"] for c in chunks}
        ids = [c["chunk_id"] for c in chunks]
        ranked = {"dense": {}, "hybrid": {}}
        top1 = {}
        for q in qs:
            dense, hybrid, s = retrieve(q["question"], chunks, store, bm25, ids)
            ranked["dense"][q["qid"]] = [page_of[c] for c in dense]
            ranked["hybrid"][q["qid"]] = [page_of[c] for c in hybrid]
            top1[q["qid"]] = s
        # Refusal threshold effect: answerable questions below tau are falsely refused
        false_ref = sum(top1[q["qid"]] < THRESHOLD for q in answerable)
        for method, pg in ranked.items():
            curve = [np.mean([hit_at_k(pg[q["qid"]], q["gold"], k) for q in answerable])
                     for k in ks]
            mrr = np.mean([reciprocal_rank(pg[q["qid"]], q["gold"]) for q in answerable])
            curves[(cfg, method)] = curve
            rows.append({"config": cfg, "overlap_%": round(100 * overlap / size),
                         "method": method, "chunks": len(chunks),
                         "index_time_s": round(index_time, 1),
                         "Hit@1": round(curve[0], 2), "Hit@3": round(curve[2], 2),
                         "Hit@5": round(curve[4], 2), "MRR": round(mrr, 2),
                         "false_refusals_at_tau": false_ref})
        print(f"{cfg}: {len(chunks)} chunks, indexing {index_time:.1f} s")

    pd.DataFrame(rows).to_csv("eval/experiment_e5_retrieval.csv", index=False)

    # Hit@k curves: solid = overlap 0, dashed = overlap 25 %
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, method in zip(axes, ["dense", "hybrid"]):
        for cfg, (size, overlap) in CONFIGS.items():
            ax.plot(ks, curves[(cfg, method)], marker="o",
                    linestyle="-" if overlap == 0 else "--", label=cfg)
        ax.set_title(f"E5: Hit@k, {method}"); ax.set_xlabel("k")
        ax.grid(alpha=0.3); ax.legend()
    axes[0].set_ylabel("Hit@k"); axes[0].set_ylim(0, 1.05)
    plt.tight_layout(); plt.savefig("reports/figures/e5_overlap_hit_at_k.png", dpi=150)

    # RAG answers for the overlap configuration (label them manually afterwards)
    from rag import rag_chain, format_docs, invoke_with_retry, REFUSAL
    chunks, store, bm25 = built[ANSWER_CONFIG]
    by_id = {c["chunk_id"]: c for c in chunks}
    ids = list(by_id)
    with open("eval/experiment_e5_answers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "config", "answer", "pages", "t_retr", "t_gen",
                    "label", "citation_ok"])
        for q in qs:
            t0 = time.perf_counter()
            _, hybrid, s = retrieve(q["question"], chunks, store, bm25, ids)
            t1 = time.perf_counter()
            if s < THRESHOLD:
                text, pg, t_gen = REFUSAL, [], 0.0
            else:
                ctx = [by_id[c] for c in hybrid[:TOP_K]]
                text = invoke_with_retry(rag_chain, {"context": format_docs(ctx),
                                                     "question": q["question"]})
                pg, t_gen = [c["page"] for c in ctx], time.perf_counter() - t1
            w.writerow([q["qid"], ANSWER_CONFIG, text, ";".join(map(str, pg)),
                        f"{t1 - t0:.3f}", f"{t_gen:.3f}", "", ""])
            print(f"{q['qid']} answered")