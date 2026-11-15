from pypdf import PdfReader, PdfWriter

# 1) вирізаємо розділи 2–3 (сторінки 77–238 книжки) у data/document.pdf
reader = PdfReader("data/full_book.pdf")
writer = PdfWriter()
for i in range(77 - 1, 238):
    writer.add_page(reader.pages[i])
with open("data/document.pdf", "wb") as f:
    writer.write(f)
print("Сторінок у document.pdf:", len(PdfReader("data/document.pdf").pages))

# 2) примітка про фрагмент у README
readme = open("README.md", encoding="utf-8").read()
readme = readme.replace(
    "  https://openstax.org/details/books/introductory-statistics-2e\n",
    "  https://openstax.org/details/books/introductory-statistics-2e\n"
    "- **Фрагмент:** документ має 847 сторінок, тому використано розділи 2–3\n"
    "  (Descriptive Statistics, Probability Topics), сторінки 77–238 книжки,\n"
    "  162 сторінки. Номери сторінок у роботі відповідають файлу data/document.pdf.\n")
open("README.md", "w", encoding="utf-8").write(readme)

# 3) src/ingest.py
code = '''# src/ingest.py
import json, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pypdf import PdfReader
from transformers import AutoTokenizer
from config import DOC_PATH, EMB_MODEL


def load_pages(path):
    pages = []
    for i, page in enumerate(PdfReader(path).pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"[ \\t]+", " ", text)
        text = re.sub(r"\\n{3,}", "\\n\\n", text).strip()
        if text:
            pages.append({"page": i, "text": text})
    return pages


if __name__ == "__main__":
    pages = load_pages(DOC_PATH)
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    n_tok = [len(tok.encode(p["text"], add_special_tokens=False)) for p in pages]
    print(f"Сторінок з текстом: {len(pages)}, токенів усього: {sum(n_tok)}, "
          f"медіана: {sorted(n_tok)[len(n_tok) // 2]}, максимум: {max(n_tok)}")
    with open("data/pages.json", "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False)
    plt.hist(n_tok, bins=40)
    plt.xlabel("Токенів на сторінці"); plt.ylabel("Кількість сторінок")
    plt.title("Розподіл кількості токенів на сторінці (all-MiniLM-L6-v2)")
    plt.savefig("reports/figures/01_tokens_per_page.png", dpi=150)
'''
open("src/ingest.py", "w", encoding="utf-8").write(code)
print("src/ingest.py створено")