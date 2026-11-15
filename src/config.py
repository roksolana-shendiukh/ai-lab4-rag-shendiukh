# src/config.py
# Variant 4 parameters (table 6.1)
VARIANT = 4
DOC_PATH = "data/document.pdf"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # max_seq_length = 256
QUERY_PREFIX, PASSAGE_PREFIX = "", ""                  # prefixes are needed only for e5 models
CHUNKS = {"A": (128, 0), "B": (128, 32)}              # (chunk_size, chunk_overlap) in tokens
LLM_MODEL = "databricks-meta-llama-3-3-70b-instruct"  # Databricks Foundation Model API

# Chosen at step 6 (eval/retrieval_metrics.csv, figures 03-04)
BEST_CONFIG = "A"        # A-hybrid: highest MRR = 0.63, Hit@5 = 0.92
SEARCH_METHOD = "hybrid" # dense + BM25 fused with RRF
TOP_K = 6                # Hit@k of A-hybrid reaches 1.0 at k = 6 (plateau)
THRESHOLD = 0.55         # gap between unanswerable (max 0.517) and answerable (min 0.570)
SEED = VARIANT