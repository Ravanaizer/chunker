import os
import shutil
import uuid
import warnings
from pathlib import Path
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    BATCH_SIZE,
    CHILD_CHUNK_SIZE,
    CHUNK_OVERLAP,
    MODEL_NAME,
    PARENT_CHUNK_SIZE,
    RERANK_MODEL,
)
from text_utils import extract_header, is_structural_chunk

warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder


def create_parent_child_store(
    documents: List[Document], child_db_path: Path, parent_db_path: Path
):
    # generate 2 db: parent and child with package loader
    if not documents:
        return None, None, None

    # embedding config
    print("\n Инициализация эмбеддингов...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # delete db
    for path in [child_db_path, parent_db_path]:
        if path.exists():
            shutil.rmtree(path)
            print(f"Удалено: {path}")

    # configure db
    parent_vectorstore = Chroma(
        persist_directory=str(parent_db_path), embedding_function=embeddings
    )
    child_vectorstore = Chroma(
        persist_directory=str(child_db_path), embedding_function=embeddings
    )
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    print("Индексация (Parent-Child)...")

    parent_docs = []
    child_docs = []
    skipped_count = 0

    # generate vectors for db
    for doc in documents:
        doc_id = doc.metadata.get("doc_id", str(uuid.uuid4()))
        full_text = doc.page_content

        child_chunks = child_splitter.split_documents([doc])
        valid_child_chunks = []

        for i, c_chunk in enumerate(child_chunks):
            if is_structural_chunk(c_chunk.page_content):
                skipped_count += 1
                continue

            header = extract_header(full_text, c_chunk.page_content)
            c_chunk.page_content = f"Раздел: {header}\n\n{c_chunk.page_content}"

            c_chunk.metadata["parent_id"] = doc_id
            c_chunk.metadata["is_parent"] = False
            c_chunk.metadata["chunk_id"] = f"{doc_id}_{i}"
            valid_child_chunks.append(c_chunk)

        parent_chunks = parent_splitter.split_documents([doc])
        for p_chunk in parent_chunks:
            p_chunk.metadata["parent_id"] = doc_id
            p_chunk.metadata["is_parent"] = True
            parent_docs.append(p_chunk)

        child_docs.extend(valid_child_chunks)

    print(
        f"Родителей: {len(parent_docs)}, Детей: {len(child_docs)}, Пропущено мусора: {skipped_count}"
    )

    # parents package loader with static int
    if parent_docs:
        print(
            f"Добавляем родителей ({len(parent_docs)} шт., батчами по {BATCH_SIZE})..."
        )
        total = int(len(parent_docs))
        for start in range(0, total, BATCH_SIZE):
            end = min(int(start + BATCH_SIZE), total)
            batch = parent_docs[int(start) : int(end)]
            parent_vectorstore.add_documents(batch)
            print(f"Пакет {int(start // BATCH_SIZE) + 1}: {len(batch)} документов")

    # child package loader with static int
    if child_docs:
        print(f"Добавляем детей ({len(child_docs)} шт., батчами по {BATCH_SIZE})...")
        total = int(len(child_docs))
        for start in range(0, total, BATCH_SIZE):
            end = min(int(start + BATCH_SIZE), total)
            batch = child_docs[int(start) : int(end)]
            child_vectorstore.add_documents(batch)
            print(f"Пакет {int(start // BATCH_SIZE) + 1}: {len(batch)} чанков")

    print(f"\n База готова:")
    print(f"  - Дети: {child_db_path.absolute()}")
    print(f"  - Родители: {parent_db_path.absolute()}")

    return child_vectorstore, parent_vectorstore, embeddings


def load_vectorstores(child_db_path: str, parent_db_path: str):
    # load db and reranking model
    if not os.path.exists(child_db_path):
        raise FileNotFoundError(f"База детей не найдена: {child_db_path}")
    if not os.path.exists(parent_db_path):
        raise FileNotFoundError(f"База родителей не найдена: {parent_db_path}")

    print(f" Подключение к базам...")

    # embedding on cpu
    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    parent_vectorstore = Chroma(
        persist_directory=parent_db_path, embedding_function=embeddings
    )
    child_vectorstore = Chroma(
        persist_directory=child_db_path, embedding_function=embeddings
    )

    print(" Загрузка модели reranking...")
    reranker = CrossEncoder(RERANK_MODEL)

    child_count = child_vectorstore._collection.count()
    print(f" Подключено. Чанков: {child_count}\n")

    try:
        sample = child_vectorstore._collection.get(limit=1000, include=["metadatas"])
        types = set()
        if sample and "metadatas" in sample:
            for meta in sample["metadatas"]:
                if meta and "doc_type" in meta:
                    types.add(meta["doc_type"])
        print(f" Типы документов: {', '.join(sorted(types))}\n")
        return child_vectorstore, parent_vectorstore, reranker, sorted(types)
    except:
        return child_vectorstore, parent_vectorstore, reranker, []
