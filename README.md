# LangChain RAG System with PGVector

A complete Retrieval-Augmented Generation (RAG) system built with:
- **LangChain** for document processing and retrieval
- **PGVector** for vector storage with PostgreSQL
- **Ollama** for local LLM inference
- **SentenceTransformersTokenTextSplitter** for intelligent text chunking
- **MMR (Max Marginal Relevance)** for diverse retrieval
- **Rich metadata** support for document tracking

## Features

- 📄 PDF document processing with page-level metadata
- 🔍 MMR retrieval for diverse, relevant results
- 🗄️ PostgreSQL with PGVector for scalable vector storage
- 🤖 Local Ollama LLM integration
- 📊 Rich metadata tracking (page numbers, timestamps, chunk IDs)
- 🔧 Configurable retrieval parameters
- ✅ Health check endpoints

## Prerequisites

### 1. PostgreSQL with PGVector Extension

Install PostgreSQL and the PGVector extension:

```bash
# On Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib

# Install PGVector extension
sudo apt install postgresql-14-pgvector

# Or compile from source (if package not available)
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

### 2. Setup Database

```sql
-- Connect to PostgreSQL as superuser
sudo -u postgres psql

-- Create database and user
CREATE DATABASE rag_db;
CREATE USER rag_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE rag_db TO rag_user;

-- Connect to the new database
\c rag_db

-- Enable the vector extension
CREATE EXTENSION vector;
```

### 3. Ollama Installation

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (adjust model name as needed)
ollama pull llama2
# or
ollama pull mistral
```

## Installation

1. **Clone and setup environment:**
```bash
cd /home/uday/IdeaProjects/Rag
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure environment:**
```bash
# Set database URL (adjust credentials)
export DATABASE_URL="postgresql://rag_user:your_password@localhost:5432/rag_db"

# Or create a .env file
echo "DATABASE_URL=postgresql://rag_user:your_password@localhost:5432/rag_db" > .env
```

3. **Start the services:**
```bash
# Ensure PostgreSQL is running
sudo systemctl start postgresql

# Ensure Ollama is running
ollama serve

# Start the FastAPI application
uvicorn main:app --reload --port 8000
```

## Usage

### 1. Health Check
```bash
curl http://localhost:8000/health
```

### 2. Upload a PDF
```bash
curl -X POST "http://localhost:8000/upload_pdf" \
     -F "file=@/path/to/your/document.pdf"
```

### 3. Query with MMR Retrieval
```bash
# Basic query
curl "http://localhost:8000/query?q=What%20is%20the%20main%20topic?"

# Advanced query with custom parameters
curl "http://localhost:8000/query?q=Explain%20the%20methodology&search_type=mmr&k=6&lambda_mult=0.3"
```

### 4. Get Collection Info
```bash
curl http://localhost:8000/collections/info
```

## API Endpoints

- `POST /upload_pdf` - Upload and process PDF documents
- `GET /query` - Query documents with MMR retrieval
- `GET /health` - System health check
- `GET /collections/info` - Vector store information

## Configuration

Key parameters you can adjust:

- `OLLAMA_MODEL`: LLM model name (default: 'llama2')
- `TOP_K_RESULTS`: Number of chunks to retrieve (default: 4)
- `MMR_DIVERSITY_SCORE`: Balance between relevance and diversity (default: 0.5)
- `DATABASE_URL`: PostgreSQL connection string

## Metadata Schema

Each document chunk includes rich metadata:

```json
{
  "source": "document.pdf",
  "page": 1,
  "total_pages": 10,
  "upload_timestamp": "2025-09-05T19:30:00",
  "file_type": "pdf",
  "chunk_type": "sentence_transformer_split",
  "chunk_id": "document.pdf_page_1_chunk_0",
  "chunk_index": 0,
  "total_chunks_in_page": 3,
  "original_page": 1
}
```

## Troubleshooting

1. **PGVector connection issues:**
   - Ensure PostgreSQL is running: `sudo systemctl status postgresql`
   - Check database credentials and URL
   - Verify PGVector extension: `SELECT * FROM pg_extension WHERE extname = 'vector';`

2. **Ollama connection issues:**
   - Ensure Ollama is running: `ollama serve`
   - Check available models: `ollama list`
   - Pull required model: `ollama pull llama2`

3. **Memory issues:**
   - Reduce chunk size in text splitter
   - Lower the `k` parameter in queries
   - Use smaller embedding models

## Development

To modify the system:

1. **Change embedding model:**
   ```python
   embeddings_model = SentenceTransformerEmbeddings(model_name='your-model-name')
   ```

2. **Adjust text splitting:**
   ```python
   text_splitter = SentenceTransformersTokenTextSplitter(
       model_name='all-MiniLM-L6-v2',
       chunk_size=600,  # Increase chunk size
       chunk_overlap=100  # Increase overlap
   )
   ```

3. **Customize MMR parameters:**
   ```python
   retriever = vectorstore.as_retriever(
       search_type="mmr",
       search_kwargs={
           "k": 8,  # More results
           "lambda_mult": 0.3,  # More diversity
           "fetch_k": 20  # Larger candidate pool
       }
   )
   ```

## License

This project is provided as-is for educational and development purposes.
