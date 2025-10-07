from dotenv import load_dotenv
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from vector_utils import extract_pdf, extract_ppt, vectorize_texts, vectorize_images
from db_utils import add_vectors, chroma_vectorstore
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import uuid

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

        # Vectorize content
        text_vectors = vectorize_texts(texts + tables)
        image_vectors = vectorize_images(images)

        # Store in ChromaDB
        total_vectors = text_vectors + image_vectors
        if total_vectors:
            ids = [str(uuid.uuid4()) for _ in total_vectors]
            metadatas = (
                [{"source": file.filename, "type": "text"} for _ in texts] +
                [{"source": file.filename, "type": "table"} for _ in tables] +
                [{"source": file.filename, "type": "image"} for _ in images]
            )
            add_vectors(total_vectors, metadatas, ids)

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
    
    retriever = chroma_vectorstore.as_retriever(search_kwargs={"k": 5})
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0, model_name="gpt-4o-mini"),
        retriever=retriever,
        return_source_documents=True
    )

    # Run query
    result = qa_chain({"query": query})
    answer = result['result']
    source_docs = result['source_documents']

    sources = []
    for doc in source_docs:
        sources.append({
            "source_file": doc.metadata.get("source"),
            "type": doc.metadata.get("type")
        })

    return {
        "query": query,
        "answer": answer,
        "sources": sources
    }
