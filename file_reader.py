import uuid
from collections import Counter
from pathlib import Path
from typing import List

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document

from text_utils import classify_document_type, clean_text


def load_documents(folder_path: Path) -> List[Document]:
    if not folder_path.exists():
        raise FileNotFoundError(f"Папка {folder_path} не найдена.")

    documents = []
    type_stats = Counter()

    print(f"Сканирование: {folder_path}")

    for file_path in folder_path.rglob("*"):
        if file_path.is_file():
            suffix = file_path.suffix.lower()
            try:
                # select loader by file format
                loader = None
                if suffix == ".pdf":
                    loader = PyPDFLoader(str(file_path))
                elif suffix in [".txt", ".md", ".csv", ".log"]:
                    loader = TextLoader(str(file_path), encoding="utf-8")
                elif suffix == ".docx":
                    loader = Docx2txtLoader(str(file_path))

                if loader:
                    # load file and generate type
                    docs = loader.load()
                    preview = docs[0].page_content if docs else ""
                    doc_type = classify_document_type(str(file_path), preview)
                    type_stats[doc_type] += 1

                    # generate metadata and clean text
                    for doc in docs:
                        doc.page_content = clean_text(doc.page_content)
                        doc.metadata["doc_type"] = doc_type
                        doc.metadata["source_file"] = file_path.name
                        doc.metadata["folder_path"] = str(file_path.parent)
                        doc.metadata["doc_id"] = str(uuid.uuid4())

                    documents.extend(docs)
                    # print(f"[{doc_type}] {file_path.name}")
            except Exception as e:
                print(f"[{file_path.name}]: {e}")

    print(f"\n\nСтатистика типов:")
    for dtype, count in type_stats.most_common():
        print(f"  {dtype}: {count} шт.")

    return documents
