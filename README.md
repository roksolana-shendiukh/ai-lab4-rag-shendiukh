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
_(filled in at step 5)_

## Error analysis and conclusions
_(filled in at step 9)_

## Use of AI assistants
| # | Tool | Step | Purpose | Verified independently |
|---|---|---|---|---|
| 1 | Claude (Opus 5.5) | 1-2 | Databricks Git folder setup, config.py and README templates | Parameters checked against table 6.1 of the guidelines |
