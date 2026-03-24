import io
import sys

from config import CHILD_JSON, DATA_PATH, DB_PATH, PARENT_DB_PATH, PARENT_JSON
from create_json import create_parent_child_json
from db import create_parent_child_store, load_vectorstores
from file_reader import load_documents
from search_in_db import interactive_search

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")


def main():
    select_category = 0
    select_category = int(
        input("""
1. Создать БД
2. Поиск по БД
3. Сохранить в JSON
Выбор: """)
    )

    print("=" * 60)
    print(" RAG Database Generator (Parent-Child + Batch)")
    print("=" * 60)
    # print(f" Скрипт: {SCRIPT_DIR.absolute()}")
    print(f" Документы: {DATA_PATH.absolute()}")
    print("=" * 60 + "\n")

    if select_category == 1:
        try:
            docs = load_documents(DATA_PATH)
            if docs:
                child_db, parent_db, embeddings = create_parent_child_store(
                    docs, DB_PATH, PARENT_DB_PATH
                )

                # if child_db:
                #     print("\n--- Тестовый поиск ---")
                #     query = "содержание"
                #     results = child_db.similarity_search(query, k=2)
                #     for i, res in enumerate(results):
                #         print(
                #             f"[{i + 1}] [{res.metadata.get('doc_type')}] {res.metadata.get('source_file')}"
                #         )
                #         print(f"    {res.page_content[:100]}...\n")
            else:
                print("\n Документы не загружены.")
        except Exception as e:
            print(f"\n Критическая ошибка: {e}")

    elif select_category == 2:
        try:
            child_db, parent_db, reranker, types = load_vectorstores(
                DB_PATH, PARENT_DB_PATH
            )
            interactive_search(child_db, parent_db, reranker, types)
        except Exception as e:
            print(f" Критическая ошибка: {e}")

    elif select_category == 3:
        try:
            docs = load_documents(DATA_PATH)
            if docs:
                total_parent, total_child = create_parent_child_json(docs)

                # if total_parent and total_child:
                #     print("\n--- Структура папок ---")
                #     print(f"  {PARENT_FOLDER.absolute()}")
                #     print(f"  └── *.json (по одному на документ)")
                #     print(f"  {CHILD_FOLDER.absolute()}")
                #     print(f"  └── *.json (по одному на документ)")
                #     print(f"\nВсего: {total_parent} родителей, {total_child} детей")
            else:
                print("\nДокументы не загружены.")
        except Exception as e:
            print(f"\nКритическая ошибка: {e}")


if __name__ == "__main__":
    main()
