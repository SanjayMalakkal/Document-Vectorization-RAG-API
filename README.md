# Document Vectorization & RAG API

A powerful FastAPI-based application that extracts, vectorizes, and indexes content (text, tables, and images) from PDF and PowerPoint documents. It enables Retrieval-Augmented Generation (RAG) querying using OpenAI's GPT-4o-mini to answer questions based on the uploaded documents.

## 🚀 Features

- **Multi-Format Support**: Extraction from **PDF** (`.pdf`) and **PowerPoint** (`.ppt`, `.pptx`) files.
- **Advanced Extraction**:
  - **Text**: Extracted using `PyMuPDF` (PDF) and `python-pptx` (PPT).
  - **Tables**: Extracted using `Camelot` (PDF) and structured parsing (PPT).
  - **Images**: extracted and embedded for multimodal capabilities.
- **Vectorization**:
  - **Text Embeddings**: Uses `all-MiniLM-L6-v2` via `SentenceTransformer`.
  - **Image Embeddings**: Uses `clip-vit-base-patch32` via `transformers` (CLIP).
- **Vector Database**: Stores embeddings in **ChromaDB** for efficient similarity search.
- **RAG Query Engine**: Uses **LangChain** and **OpenAI GPT-4o-mini** to provide accurate answers with source citations.
- **Containerized**: Fully Dockerized for easy deployment.

## 🛠️ Tech Stack

- **Framework**: FastAPI, Uvicorn
- **LLM & AI**: LangChain, OpenAI GPT-4o-mini, SentenceTransformers, CLIP
- **Database**: ChromaDB (Vector Store)
- **Document Processing**: PyMuPDF, python-pptx, Camelot, Pillow, OpenCV
- **Deployment**: Docker, Docker Compose

## 📋 Prerequisites

- **Docker** and **Docker Compose** installed.
- **OpenAI API Key** (for RAG querying).

## 🚀 Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Document-Vectorization-RAG-API
   ```

2. **Configure Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Build and Run with Docker**:
   ```bash
   docker-compose up --build
   ```
   The API will be available at `http://localhost:5000`.

## 🔌 API Endpoints

### 1. Upload Document
Uploads a file, extracts content, vectorizes it, and stores it in the knowledge base.

- **Endpoint**: `POST /upload`
- **Form Data**: `file` (Binary file: `.pdf`, `.ppt`, `.pptx`)
- **Response**:
  ```json
  {
    "text_blocks": 15,
    "tables": 2,
    "images": 5
  }
  ```

### 2. Query Knowledge Base
Ask questions based on the uploaded documents.

- **Endpoint**: `POST /query`
- **Query Parameters**: `query` (string)
- **Response**:
  ```json
  {
    "query": "What are the Q3 financial results?",
    "answer": "The Q3 financial results show a 15% increase in revenue...",
    "sources": [
      {
        "source_file": "report.pdf",
        "type": "table",
        "content": "..."
      }
    ]
  }
  ```

## 📂 Project Structure

```
├── .env                # Environment variables
├── main.py             # FastAPI entry point
├── vector_utils.py     # Extraction & vectorization logic
├── db_utils.py         # ChromaDB interactions
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker build instructions
└── docker-compose.yaml # Docker services config
```

## ⚠️ Notes

- Ensure your files are readable (not encrypted/password protected).
- Supported table extraction works best on native PDFs (not scanned images).
