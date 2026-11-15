import json, re
import matplotlib
matplotlib.use("Agg")  # save plots to files without a display
import matplotlib.pyplot as plt
from pypdf import PdfReader
from transformers import AutoTokenizer
from config import DOC_PATH, EMB_MODEL


def load_pages(path):
    """Extract text page by page and normalize whitespace."""
    pages = []
    for i, page in enumerate(PdfReader(path).pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text)          # collapse spaces and tabs
        text = re.sub(r"\n{3,}", "\n\n", text).strip()  # collapse empty lines
        if text:
            pages.append({"page": i, "text": text})
    return pages


if __name__ == "__main__":
    pages = load_pages(DOC_PATH)

    # Count tokens with the tokenizer of the embedding model
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    n_tok = [len(tok.encode(p["text"], add_special_tokens=False)) for p in pages]
    print(f"Pages with text: {len(pages)}, total tokens: {sum(n_tok)}, "
          f"median: {sorted(n_tok)[len(n_tok) // 2]}, max: {max(n_tok)}")

    with open("data/pages.json", "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False)

    plt.hist(n_tok, bins=40)
    plt.xlabel("Tokens per page"); plt.ylabel("Number of pages")
    plt.title("Tokens per page (all-MiniLM-L6-v2 tokenizer)")
    plt.savefig("reports/figures/01_tokens_per_page.png", dpi=150)