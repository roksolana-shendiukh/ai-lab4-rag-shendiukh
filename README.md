# Lab 4. Building a Retrieval-Augmented Generation (RAG) Solution with an LLM

- **Student:** Shendiukh Roksolana Yaroslavivna
- **Group:** PZ-46
- **Variant:** 4
- **Document:** Introductory Statistics 2e (OpenStax) -
  https://openstax.org/details/books/introductory-statistics-2e
- **Fragment:** the book has 847 pages, so Chapters 2-3 (Descriptive Statistics,
  Probability Topics) are used: book pages 77-238, 162 pages in total.
  All page numbers in this project refer to data/document.pdf.

## Environment
The project runs in Databricks (serverless, Python). The LLM is accessed through
the Databricks Foundation Model API (`databricks-langchain`, `ChatDatabricks`)
with temperature 0. Embedding models are downloaded from Hugging Face.

## Variant parameters
| Parameter | Value |
|---|---|
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 (max_seq_length = 256) |
| Chunks A | 128 tokens, overlap 0 |
| Chunks B | 128 tokens, overlap 32 |
| LLM | databricks-meta-llama-3-3-70b-instruct |
| Additional experiment | E5: overlap 0 and 25 % |

## How to run

    pip install -r requirements.txt
    python src/ingest.py
    python src/chunking.py
    python src/index.py
    python src/eval_retrieval.py
    python src/rag.py
    python src/eval_answers.py

## Prediction
- **Best configuration and method for Hit@5:** configuration **B** with **hybrid search (RRF)**.
  Overlap of 32 tokens reduces the risk that an answer is split by a chunk boundary.
  Hybrid search combines dense retrieval (paraphrased questions, definitions) with
  BM25 (exact terms and numbers such as "IQR", "1.5", "P(A AND B)").
- **BM25 vs dense:** BM25 is expected to be competitive on factual questions with
  numbers, dense retrieval is expected to win on paraphrased questions.
- **Expected Hit@5:** about **0.80** for the best system (10 of 12 answerable questions).
- **Expected refusal threshold:** tau ≈ **0.45**. In trial queries top-1 cosine
  similarity of relevant chunks was 0.68-0.77; unanswerable questions are close to
  the topic (statistics), so their top-1 scores are expected around 0.40-0.55.

## Error analysis and conclusions

### Error 1 - q09 (partial)
- **Question:** What is the difference between independent and mutually exclusive events?
- **Gold pages:** 109; 110. **Retrieved pages (A-hybrid, top-6):** 136, 108, 148, 146, 110, 60.
- **Quote (p. 108, chunk 454):** "Independent Events Two events are independent if the following are true: • P(A|B) = P"
- **Cause:** the answer is split by a chunk boundary. Every page is chunked separately, so the
  list of independence conditions starts on p. 108 and ends on p. 109 (chunk 455,
  "P(A AND B) = P(A)P(B)"), which was not retrieved. The LLM explained both concepts
  but gave no formulas.
- **Fix:** chunk the text across page boundaries (join consecutive pages before splitting and
  keep the start page in metadata), or ask for formulas explicitly in the system prompt.

### Error 2 - q02 (retrieval ranking)
- **Question:** In the left-skewed example, what are the mean and the median?
- **Gold page:** 40. **Retrieved pages (A-hybrid, top-6):** 96, 41, 57, 67, 41, 40.
- **Quote (p. 40, chunk 184, rank 6):** "Figure 2.17 The mean is 6.3, the median is 6.5, and the mode is seven."
- **Cause:** the search did not rank the relevant chunk high. Exercise and answer chunks
  (p. 96: "...the data are skewed right. When the data are symmetrical, the mean and median
  are close...") share more words with the question than the example itself. The answer was
  correct only because TOP_K = 6; with k <= 5 the context would miss it.
- **Fix:** re-rank top-20 candidates with a cross-encoder or down-weight exercise sections
  (Practice, Homework, Solutions) using page metadata.

### Error 3 - q03 (error in the gold set)
- **Question:** At supermarket A, what are the mean waiting time and the standard deviation?
- **Gold page:** 44. **Retrieved pages (A-hybrid, top-6):** 44, 43, 45, 44, 53, 36.
- **Quote (p. 43, chunk 193):** "The average wait time at both supermarkets is five minutes.
  At supermarket A, the standard deviation for the wait time is two minutes"
- **Cause:** error in the reference set. The answer is on both p. 43 and p. 44, but only p. 44
  was listed, so systems that rank p. 43 first are counted as misses and Hit@1 / MRR are
  underestimated. The RAG answer of configuration 128/32 cited p. 43 and was correct.
- **Fix:** set the gold pages of q03 to "43;44" and re-run eval_retrieval.py.

### Conclusions
The best retrieval system is A-hybrid (chunks 128/0, dense + BM25 with RRF): Hit@5 = 0.92,
MRR = 0.63, and Hit@k reaches 1.0 at k = 6, so TOP_K = 6 was chosen. Top-1 cosine similarity
separates answerable (0.570-0.825) and unanswerable (0.302-0.517) questions, and the threshold
tau = 0.55 gave 3 of 3 correct refusals with no false refusals. RAG raised answer correctness
from 0.79 to 0.96 and reduced the hallucination rate from 0.20 to 0, with citation accuracy
1.00. Generation takes about 1.5 s against 0.05 s for retrieval, so the LLM dominates latency.
Experiment E5 showed that 25 % overlap does not help on this document: for 128-token chunks
hybrid MRR dropped from 0.63 to 0.54 while the number of chunks grew from 723 to 836.
The remaining errors come from page-level chunking and exercise sections that compete with
explanatory text in retrieval.

## Use of AI assistants
| # | Tool | Step | Purpose | Verified independently |
|---|---|---|---|---|
| 1 | Claude (Opus 5.5) | 1-2 | Databricks Git folder setup, config.py and README templates | Parameters checked against table 6.1 of the guidelines |
