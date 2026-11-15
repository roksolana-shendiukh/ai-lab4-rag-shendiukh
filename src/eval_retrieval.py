import csv, re
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from rank_bm25 import BM25Okapi
from index import get_embeddings, load_chunks, load_store
from config import CHUNKS

K_MAX = 10


def load_questions(path="eval/questions.csv"):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        # Gold pages; empty set for unanswerable questions
        r["gold"] = {int(p) for p in r["pages"].split(";") if p.strip()}
    return rows


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def bm25_ranking(bm25, ids, q, k=K_MAX):
    scores = bm25.get_scores(tokenize(q))
    return [ids[i] for i in np.argsort(scores)[::-1][:k]]


def rrf(*rankings, k=60):
    """Reciprocal Rank Fusion: uses only positions in the rankings."""
    score = {}
    for ranking in rankings:
        for r, cid in enumerate(ranking, start=1):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + r)
    return sorted(score, key=score.get, reverse=True)


def hit_at_k(pages, gold, k):
    return float(any(p in gold for p in pages[:k]))


def reciprocal_rank(pages, gold):
    return next((1.0 / r for r, p in enumerate(pages, 1) if p in gold), 0.0)


def first_hit_rank(pages, gold):
    return next((r for r, p in enumerate(pages, 1) if p in gold), "")


if __name__ == "__main__":
    qs = load_questions()
    answerable = [q for q in qs if q["gold"]]
    emb = get_embeddings()
    rankings = {}
    top1 = {"answerable": [], "unanswerable": []}
    top1_by_q = {}

    for name in CHUNKS:
        chunks = load_chunks(name)
        page_of = {c["chunk_id"]: c["page"] for c in chunks}
        ids = [c["chunk_id"] for c in chunks]
        bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])
        store = load_store(name, emb)
        for q in qs:
            hits = store.similarity_search_with_score(q["question"], k=K_MAX)
            dense = [d.metadata["chunk_id"] for d, _ in hits]
            lex = bm25_ranking(bm25, ids, q["question"])
            if name == "A":
                # Top-1 cosine similarity is used to choose the refusal threshold
                group = "answerable" if q["gold"] else "unanswerable"
                top1[group].append(hits[0][1])
                top1_by_q[q["qid"]] = hits[0][1]
            for method, ranking in [("dense", dense), ("bm25", lex),
                                    ("hybrid", rrf(dense, lex)[:K_MAX])]:
                pages = [page_of[c] for c in ranking]
                rankings[(f"{name}-{method}", q["qid"])] = pages

    systems = sorted({s for s, _ in rankings})
    ks = range(1, K_MAX + 1)

    # Aggregated metrics for every system + Hit@k curves
    with open("eval/retrieval_metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "Hit@1", "Hit@3", "Hit@5", "MRR"])
        for s in systems:
            curve = [np.mean([hit_at_k(rankings[(s, q["qid"])], q["gold"], k)
                              for q in answerable]) for k in ks]
            mrr = np.mean([reciprocal_rank(rankings[(s, q["qid"])], q["gold"])
                           for q in answerable])
            values = (curve[0], curve[2], curve[4], mrr)
            w.writerow([s] + [f"{v:.2f}" for v in values])
            plt.plot(ks, curve, marker="o", label=s)
    plt.xlabel("k"); plt.ylabel("Hit@k"); plt.ylim(0, 1.05)
    plt.title("Hit@k for 6 retrieval systems")
    plt.grid(alpha=0.3); plt.legend()
    plt.savefig("reports/figures/03_hit_at_k.png", dpi=150); plt.close()

    # Per-question rank of the first relevant page (empty = not found in top-10)
    with open("eval/retrieval_per_question.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "type", "top1_score_A"] + systems)
        for q in qs:
            w.writerow([q["qid"], q["type"], f"{top1_by_q[q['qid']]:.3f}"] +
                       [first_hit_rank(rankings[(s, q["qid"])], q["gold"]) for s in systems])

    # Distribution of top-1 similarity: answerable vs unanswerable questions
    plt.hist(top1["answerable"], bins=15, alpha=0.6, label="answerable")
    plt.hist(top1["unanswerable"], bins=15, alpha=0.6, label="unanswerable")
    plt.xlabel("Cosine similarity of top-1 chunk"); plt.ylabel("Number of questions")
    plt.title("Top-1 similarity distribution (configuration A, dense)")
    plt.legend()
    plt.savefig("reports/figures/04_top1_scores.png", dpi=150)
    print("Done: eval/retrieval_metrics.csv, eval/retrieval_per_question.csv, figures 03-04")