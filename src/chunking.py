import json
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import EMB_MODEL, CHUNKS


def make_chunks(pages, size, overlap, tok):
    """Split pages into chunks measured in tokens of the embedding model."""
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tok, chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""])
    # Each page is a separate document, so every chunk keeps its page number
    docs = [Document(page_content=p["text"], metadata={"page": p["page"]})
            for p in pages]
    chunks = splitter.split_documents(docs)
    return [{"chunk_id": i, "page": c.metadata["page"], "text": c.page_content}
            for i, c in enumerate(chunks)]


if __name__ == "__main__":
    pages = json.load(open("data/pages.json", encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)

    for name, (size, overlap) in CHUNKS.items():
        chunks = make_chunks(pages, size, overlap, tok)
        lens = [len(tok.encode(c["text"], add_special_tokens=False))
                for c in chunks]
        print(f"{name}: size={size}, overlap={overlap}, chunks={len(chunks)}, "
              f"mean length={sum(lens) / len(lens):.0f} tokens, "
              f"max length={max(lens)} tokens")
        with open(f"data/chunks_{name}.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)
        plt.hist(lens, bins=40, alpha=0.6, label=f"{name}: {size}/{overlap}")

    plt.xlabel("Chunk length, tokens"); plt.ylabel("Number of chunks")
    plt.title("Chunk length distribution (all-MiniLM-L6-v2 tokenizer)")
    plt.legend()
    plt.savefig("reports/figures/02_chunk_lengths.png", dpi=150)