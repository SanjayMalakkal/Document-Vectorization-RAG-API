import chromadb
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

# Initialize ChromaDB client
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

collection = client.get_or_create_collection(name="documents")

# Add vectors to ChromaDB
def add_vectors(vectors, metadatas, ids):
    collection.add(
        embeddings=vectors,
        metadatas=metadatas,
        ids=ids
    )

# LangChain vector store wrapper
text_embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
chroma_vectorstore = Chroma(
    collection_name="documents",
    persist_directory="./chroma_db",
    embedding_function=text_embeddings
)
