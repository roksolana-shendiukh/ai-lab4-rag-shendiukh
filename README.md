# Лабораторна робота №4. Побудова RAG-рішення на базі LLM

- **Студентка:** Shendiukh Roksolana Yaroslavivna
- **Група:** ПЗ-46
- **Варіант:** 4
- **Документ:** Introductory Statistics 2e (OpenStax) —
  https://openstax.org/details/books/introductory-statistics-2e

## Середовище
Робота виконується в Databricks (serverless, Python). LLM викликається через
Databricks Foundation Model API (`databricks-langchain`, `ChatDatabricks`),
температура 0. Моделі ембедингів завантажуються з Hugging Face.

## Параметри варіанта
| Параметр | Значення |
|---|---|
| Модель ембедингів | sentence-transformers/all-MiniLM-L6-v2 (max_seq_length = 256) |
| Чанки A | 128 токенів, перекриття 0 |
| Чанки B | 128 токенів, перекриття 32 |
| LLM | databricks-meta-llama-3-3-70b-instruct |
| Додатковий експеримент | Е5: перекриття 0 і 25 % |

## Запуск

    pip install -r requirements.txt
    python src/ingest.py
    python src/chunking.py
    python src/index.py
    python src/eval_retrieval.py
    python src/rag.py
    python src/eval_answers.py

## Прогноз
_(заповнюється на кроці 5)_

## Аналіз помилок і висновки
_(заповнюється на кроці 9)_

## Використання ШІ-асистентів
| № | Інструмент | Крок | Для чого використано | Що перевірено самостійно |
|---|---|---|---|---|
| 1 | Claude (Opus 5.5) | 1 | Налаштування Databricks Git folder, шаблон config.py і README | Параметри звірено з таблицею 6.1 методички |
