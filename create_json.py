import json
import uuid
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHILD_CHUNK_SIZE,
    CHILD_JSON,
    CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    PARENT_JSON,
)
from text_utils import extract_header, is_structural_chunk


def create_parent_child_json(documents: List[Document]) -> tuple:
    if not documents:
        return None, None

    print("\nПодготовка чанков (Parent-Child)...")

    # Создаём папки для JSON
    PARENT_JSON.mkdir(parents=True, exist_ok=True)
    CHILD_JSON.mkdir(parents=True, exist_ok=True)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    total_parent = 0
    total_child = 0
    skipped_count = 0
    files_created = 0

    for doc in documents:
        doc_id = doc.metadata.get("doc_id", str(uuid.uuid4()))
        source_file = doc.metadata.get("source_file", "unknown")
        full_text = doc.page_content

        # Безопасное имя файла
        safe_name = (
            source_file.replace(".txt", "").replace(".docx", "").replace(".pdf", "")
        )
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in " -_")[:50]

        parent_chunks = []
        child_chunks = []

        # Child chunks
        child_split_docs = child_splitter.split_documents([doc])
        for i, c_chunk in enumerate(child_split_docs):
            if is_structural_chunk(c_chunk.page_content):
                skipped_count += 1
                continue

            header = extract_header(full_text, c_chunk.page_content)
            c_chunk.page_content = f"Раздел: {header}\n\n{c_chunk.page_content}"

            chunk_data = {
                "chunk_id": f"{doc_id}_{i}",
                "parent_id": doc_id,
                "is_parent": False,
                "content": c_chunk.page_content,
                "metadata": {
                    "doc_type": doc.metadata.get("doc_type", "Прочее"),
                    "source_file": source_file,
                    "folder_path": doc.metadata.get("folder_path", ""),
                    "header": header,
                },
            }
            child_chunks.append(chunk_data)

        # Parent chunks
        parent_split_docs = parent_splitter.split_documents([doc])
        for i, p_chunk in enumerate(parent_split_docs):
            chunk_data = {
                "chunk_id": f"{doc_id}_p{i}",
                "parent_id": doc_id,
                "is_parent": True,
                "content": p_chunk.page_content,
                "metadata": {
                    "doc_type": doc.metadata.get("doc_type", "Прочее"),
                    "source_file": source_file,
                    "folder_path": doc.metadata.get("folder_path", ""),
                },
            }
            parent_chunks.append(chunk_data)

        # Сохраняем JSON для этого документа
        if parent_chunks or child_chunks:
            files_created += 1

            if parent_chunks:
                parent_file = PARENT_JSON / f"{safe_name}_parents.json"
                with open(parent_file, "w", encoding="utf-8") as f:
                    json.dump(parent_chunks, f, ensure_ascii=False, indent=2)
                total_parent += len(parent_chunks)

            if child_chunks:
                child_file = CHILD_JSON / f"{safe_name}_children.json"
                with open(child_file, "w", encoding="utf-8") as f:
                    json.dump(child_chunks, f, ensure_ascii=False, indent=2)
                total_child += len(child_chunks)

        if files_created % 20 == 0:
            print(f"  Обработано файлов: {files_created}...")

    print(
        f"\nРодителей: {total_parent}, Детей: {total_child}, Пропущено: {skipped_count}"
    )
    print(f"Создано JSON-файлов: {files_created * 2}")

    return total_parent, total_child
