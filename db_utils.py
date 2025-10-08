import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings

# Initialize ChromaDB PersistentClient
client = chromadb.PersistentClient(path="./chroma_db")

# Get or create separate collections for text and images
text_collection = client.get_or_create_collection(name="text_documents")
image_collection = client.get_or_create_collection(name="image_documents")

# Add vectors to ChromaDB
def add_text_vectors(vectors, metadatas, ids, documents):
    text_collection.add(
        embeddings=vectors,
        metadatas=metadatas,
        ids=ids,
        documents=documents  # Store text content for LangChain
    )

def add_image_vectors(vectors, metadatas, ids):
    image_collection.add(
        embeddings=vectors,
        metadatas=metadatas,
        ids=ids
    )

# LangChain vector store wrapper (for text only, since LangChain uses text embeddings)
text_embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
chroma_vectorstore = Chroma(
    collection_name="text_documents",
    persist_directory="./chroma_db",
    embedding_function=text_embeddings
)