# src/config.py
# Variant 4 parameters (table 6.1)
VARIANT = 4
DOC_PATH = "data/document.pdf"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # max_seq_length = 256
QUERY_PREFIX, PASSAGE_PREFIX = "", ""                  # prefixes are needed only for e5 models
CHUNKS = {"A": (128, 0), "B": (128, 32)}              # (chunk_size, chunk_overlap) in tokens
LLM_MODEL = "databricks-meta-llama-3-3-70b-instruct"  # Databricks Foundation Model API
TOP_K = 4                                              # number of chunks in context; tuned at step 6
THRESHOLD = 0.0                                        # refusal threshold; tuned at step 6
SEED = VARIANT
