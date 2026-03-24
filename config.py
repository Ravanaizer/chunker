from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "txt"  # folder with document path
DB_PATH = SCRIPT_DIR / "vector_db"  # child db path
PARENT_DB_PATH = SCRIPT_DIR / "parent_db"  # parent db path

# chunks size
PARENT_CHUNK_SIZE = 1500
CHILD_CHUNK_SIZE = 500
CHUNK_OVERLAP = 150
BATCH_SIZE = 1000  # Size for ChromaDB

# test config
DB_PATH = "./vector_db"
PARENT_DB_PATH = "./parent_db"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
