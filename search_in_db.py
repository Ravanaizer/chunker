def rerank_results(query, docs, reranker, top_k=5):
    """Пересортировка результатов через Cross-Encoder."""
    if not docs:
        return []

    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, score in scored_docs[:top_k]]


def diversify_results(results, max_per_doc=2):
    """Оставляет не более N чанков из одного документа."""
    doc_counts = {}
    diversified = []

    for doc in results:
        source = doc.metadata.get("source_file", "unknown")
        doc_counts[source] = doc_counts.get(source, 0) + 1
        if doc_counts[source] <= max_per_doc:
            diversified.append(doc)

    return diversified


def interactive_search(child_db, parent_db, reranker, available_types):
    """Поиск с фильтрацией и reranking."""
    print(" Поиск запущен.")
    print("Пример: 'параметры в ТЗ' (авто-фильтр)")
    print("Для выхода: 'exit'\n")

    while True:
        try:
            query = input("Запрос: ").strip()
            if query.lower() in ["exit", "quit", "выход"]:
                break
            if not query:
                continue

            doc_filter = None
            query_lower = query.lower()

            for dtype in available_types:
                if dtype.lower() in query_lower:
                    doc_filter = {"doc_type": dtype}
                    print(f" Фильтр: {dtype}")
                    break

            if not doc_filter:
                if any(x in query_lower for x in ["тз", "задание"]):
                    doc_filter = {"doc_type": "ТЗ"}
                elif any(x in query_lower for x in ["договор", "соглашение"]):
                    doc_filter = {"doc_type": "Соглашение"}
                elif any(x in query_lower for x in ["расчет", "смета"]):
                    doc_filter = {"doc_type": "Расчет"}
                elif any(x in query_lower for x in ["приказ"]):
                    doc_filter = {"doc_type": "Приказ"}

            raw_results = child_db.similarity_search(query, k=20, filter=doc_filter)

            if not raw_results:
                print(" Ничего не найдено.\n")
                continue

            print(" Reranking...")
            final_results = rerank_results(query, raw_results, reranker, top_k=10)
            final_results = diversify_results(final_results, max_per_doc=2)

            print(f"\n Топ результатов: {len(final_results)}")
            print("=" * 60)

            for i, doc in enumerate(final_results):
                doc_type = doc.metadata.get("doc_type", "?")
                source = doc.metadata.get("source_file", "?")

                print(f"\n#{i + 1} [{doc_type}] {source}")
                print(f"Текст:\n{doc.page_content[:300]}...")

            print()

        except KeyboardInterrupt:
            print("\nПрервано.")
            break
        except Exception as e:
            print(f"Ошибка: {e}")
            import traceback

            traceback.print_exc()
