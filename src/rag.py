import csv, time
from rank_bm25 import BM25Okapi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from databricks_langchain import ChatDatabricks
from index import get_embeddings, load_chunks, load_store
from eval_retrieval import load_questions, tokenize, bm25_ranking, rrf, K_MAX
from config import LLM_MODEL, TOP_K, THRESHOLD, BEST_CONFIG

REFUSAL = "NOT IN THE DOCUMENT"
SYSTEM = ("You are an assistant that answers ONLY on the basis of the provided context. "
          "After every statement cite the source in the format [p. N]. "
          f"If the context does not contain the answer, reply exactly: {REFUSAL}.")
rag_prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM), ("human", "Context:\n{context}\n\nQuestion: {question}")])
plain_prompt = ChatPromptTemplate.from_messages([("human", "{question}")])

llm = ChatDatabricks(endpoint=LLM_MODEL, temperature=0)
rag_chain = rag_prompt | llm | StrOutputParser()      # LCEL: prompt -> LLM -> text
plain_chain = plain_prompt | llm | StrOutputParser()  # baseline without RAG


class HybridRetriever:
    """Dense + BM25 retrieval fused with RRF (best system at step 6)."""

    def __init__(self, name, emb):
        self.chunks = {c["chunk_id"]: c for c in load_chunks(name)}
        self.ids = list(self.chunks)
        self.bm25 = BM25Okapi([tokenize(self.chunks[i]["text"]) for i in self.ids])
        self.store = load_store(name, emb)

    def search(self, question, k=TOP_K):
        hits = self.store.similarity_search_with_score(question, k=K_MAX)
        top1 = hits[0][1] if hits else 0.0  # dense top-1 score for the threshold
        dense = [d.metadata["chunk_id"] for d, _ in hits]
        lex = bm25_ranking(self.bm25, self.ids, question)
        fused = rrf(dense, lex)[:k]          # most relevant chunk goes first
        return [self.chunks[i] for i in fused], top1


def format_docs(chunks):
    return "\n\n".join(f"[p. {c['page']}] {c['text']}" for c in chunks)


def invoke_with_retry(chain, inputs, attempts=3):
    """Retry on temporary API errors (rate limits of the free tier)."""
    for i in range(attempts):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            if i == attempts - 1:
                raise
            print(f"API error, retrying in 10 s: {e}")
            time.sleep(10)


def ask(question, retriever, k=TOP_K, threshold=THRESHOLD):
    t0 = time.perf_counter()
    chunks, top1 = retriever.search(question, k)
    t1 = time.perf_counter()
    if top1 < threshold:                     # nothing similar found -> refuse without LLM
        return REFUSAL, [], t1 - t0, 0.0
    text = invoke_with_retry(rag_chain, {"context": format_docs(chunks),
                                         "question": question})
    pages = [c["page"] for c in chunks]
    return text, pages, t1 - t0, time.perf_counter() - t1


if __name__ == "__main__":
    retriever = HybridRetriever(BEST_CONFIG, get_embeddings())
    with open("eval/answers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "mode", "answer", "pages", "t_retr", "t_gen",
                    "label", "citation_ok"])
        for q in load_questions():
            text, pages, t_r, t_g = ask(q["question"], retriever)
            w.writerow([q["qid"], "rag", text, ";".join(map(str, pages)),
                        f"{t_r:.3f}", f"{t_g:.3f}", "", ""])
            t0 = time.perf_counter()
            text = invoke_with_retry(plain_chain, {"question": q["question"]})
            t_gen = time.perf_counter() - t0
            w.writerow([q["qid"], "llm", text, "", "0", f"{t_gen:.3f}", "", ""])
            print(f"{q['qid']} done")