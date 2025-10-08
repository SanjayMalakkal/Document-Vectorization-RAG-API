from dotenv import load_dotenv
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from vector_utils import extract_pdf, extract_ppt, vectorize_texts, vectorize_images
from db_utils import add_text_vectors, add_image_vectors, chroma_vectorstore
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Load environment variables from .env
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment")

app = FastAPI(title="Document Vectorization & RAG API")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1].lower()
    file_location = f"temp_{uuid.uuid4()}.{ext}"
    with open(file_location, "wb") as f:
        f.write(await file.read())

    try:
        if ext == "pdf":
            texts, images, tables = extract_pdf(file_location)
        elif ext in ["ppt", "pptx"]:
            texts, images, tables = extract_ppt(file_location)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        # Filter out empty or None text and table content
        texts = [text for text in texts if text and text.strip()]
        tables = [table for table in tables if table and table.strip()]
        logger.info(f"Filtered text blocks: {len(texts)}, tables: {len(tables)}, images: {len(images)}")

        # Vectorize content
        text_vectors = vectorize_texts(texts + tables)
        image_vectors = vectorize_images(images)

        # Store in ChromaDB
        if text_vectors:
            text_ids = [str(uuid.uuid4()) for _ in text_vectors]
            text_metadatas = (
                [{"source": file.filename, "type": "text", "content": text} for text in texts] +
                [{"source": file.filename, "type": "table", "content": table} for table in tables]
            )
            # Pass the combined texts and tables as documents
            add_text_vectors(text_vectors, text_metadatas, text_ids, documents=texts + tables)
            logger.info(f"Stored {len(text_vectors)} text vectors in ChromaDB")

        if image_vectors:
            image_ids = [str(uuid.uuid4()) for _ in image_vectors]
            image_metadatas = [{"source": file.filename, "type": "image"} for _ in images]
            add_image_vectors(image_vectors, image_metadatas, image_ids)
            logger.info(f"Stored {len(image_vectors)} image vectors in ChromaDB")

        return {
            "text_blocks": len(texts),
            "tables": len(tables),
            "images": len(images)
        }
    finally:
        if os.path.exists(file_location):
            os.remove(file_location)

@app.post("/query")
async def query_knowledge(query: str):
    logger.info(f"Processing query: {query}")
    retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, model_name="gpt-4o-mini"),
        retriever=retriever,
        return_source_documents=True
    )

    # Run query
    result = qa_chain.invoke({"query": query})
    answer = result['result']
    source_docs = result['source_documents']

    sources = []
    for doc in source_docs:
        sources.append({
            "source_file": doc.metadata.get("source"),
            "type": doc.metadata.get("type"),
            "content": doc.metadata.get("content") or doc.page_content
        })
    logger.info(f"Query returned {len(sources)} sources")

    return {
        "query": query,
        "answer": answer,
        "sources": sources
    }