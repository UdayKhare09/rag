# main.py
# A complete RAG system using FastAPI, LangChain, Ollama, and PGVector.
import fitz
import os
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

# LangChain imports
from langchain_community.llms import Ollama
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# FastAPI imports
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
import logging
import tempfile

# --- Configuration ---
# Set up logging to monitor the application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# PostgreSQL connection settings for PGVector
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://uday:Uday88717@localhost:5432/rag_db")
COLLECTION_NAME = "documents"

# --- Model and Global State Initialization ---
# Initialize the FastAPI app
app = FastAPI(
    title="LangChain RAG API",
    description="A RAG system using LangChain, Ollama, and PGVector with MMR retrieval.",
    version="2.0.0"
)

# Global variables to hold the LangChain components
embeddings_model = None
text_splitter = None
vectorstore = None
llm = None
retriever = None
qa_chain = None

OLLAMA_MODEL = 'gemma3:1b'  # Default model, can be changed
TOP_K_RESULTS = 4
MMR_DIVERSITY_SCORE = 0.5

# --- Helper Functions ---

def initialize_components():
    """
    Initialize LangChain components at startup.
    """
    global embeddings_model, text_splitter, vectorstore, llm, retriever, qa_chain
    
    try:
        # Initialize embeddings model
        embeddings_model = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
        logging.info("SentenceTransformer embeddings model loaded successfully.")
        
        # Initialize text splitter with token-aware splitting
        text_splitter = SentenceTransformersTokenTextSplitter(
            model_name='all-MiniLM-L6-v2',
            chunk_size=400,
            chunk_overlap=50
        )
        logging.info("SentenceTransformersTokenTextSplitter initialized.")
        
        # Initialize Ollama LLM
        llm = Ollama(model=OLLAMA_MODEL, temperature=0.1)
        logging.info(f"Ollama LLM '{OLLAMA_MODEL}' initialized.")
        
        # Initialize PGVector store
        vectorstore = PGVector(
            connection_string=DATABASE_URL,
            embedding_function=embeddings_model,
            collection_name=COLLECTION_NAME,
        )
        logging.info("PGVector store initialized.")
        
        # Initialize MMR retriever
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": TOP_K_RESULTS,
                "lambda_mult": MMR_DIVERSITY_SCORE,
                "fetch_k": TOP_K_RESULTS * 2
            }
        )
        logging.info("MMR retriever initialized.")
        
        # Initialize QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            verbose=True
        )
        logging.info("QA chain initialized.")
        
    except Exception as e:
        logging.error(f"Failed to initialize components: {e}")
        raise RuntimeError(f"Could not initialize LangChain components: {e}")


def process_pdf_with_metadata(file_content: bytes, filename: str) -> List[Document]:
    """
    Process PDF and create Document objects with metadata.
    
    Args:
        file_content: PDF file content as bytes
        filename: Original filename for metadata
        
    Returns:
        List of Document objects with content and metadata
    """
    try:
        # Save content to temporary file for PyMuPDF processing
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_file_path = tmp_file.name
        
        # Extract text using PyMuPDF with page-level metadata
        documents = []
        with fitz.open(tmp_file_path) as doc:
            for page_num, page in enumerate(doc):
                text = page.get_text()
                if text.strip():  # Only add non-empty pages
                    # Create document with rich metadata
                    document = Document(
                        page_content=text,
                        metadata={
                            "source": filename,
                            "page": page_num + 1,
                            "total_pages": len(doc),
                            "upload_timestamp": datetime.now().isoformat(),
                            "file_type": "pdf",
                            "chunk_type": "page"
                        }
                    )
                    documents.append(document)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        
        logging.info(f"Extracted {len(documents)} pages from {filename}")
        return documents
        
    except Exception as e:
        logging.error(f"Error processing PDF {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")


def split_documents_with_metadata(documents: List[Document]) -> List[Document]:
    """
    Split documents into chunks while preserving and enhancing metadata.
    
    Args:
        documents: List of Document objects to split
        
    Returns:
        List of split Document objects with enhanced metadata
    """
    try:
        all_chunks = []
        
        for doc in documents:
            # Split the document
            chunks = text_splitter.split_documents([doc])
            
            # Add chunk-specific metadata
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "chunk_id": f"{doc.metadata['source']}_page_{doc.metadata['page']}_chunk_{i}",
                    "chunk_index": i,
                    "total_chunks_in_page": len(chunks),
                    "chunk_type": "sentence_transformer_split",
                    "original_page": doc.metadata['page']
                })
                all_chunks.append(chunk)
        
        logging.info(f"Split {len(documents)} documents into {len(all_chunks)} chunks")
        return all_chunks
        
    except Exception as e:
        logging.error(f"Error splitting documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to split documents: {e}")

# --- FastAPI Lifespan Events ---

@app.on_event("startup")
async def startup_event():
    """
    Event handler for application startup. Initialize LangChain components.
    """
    logging.info("Application startup...")
    initialize_components()

# --- FastAPI Endpoints ---

@app.post("/upload_pdf", summary="Upload and index a PDF")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Endpoint to upload a PDF file. The file is processed using LangChain,
    chunked with SentenceTransformersTokenTextSplitter, and stored in PGVector
    with rich metadata.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF.")

    logging.info(f"Processing uploaded file: {file.filename}")

    try:
        # Read file content into memory
        file_content = await file.read()

        # Process PDF with metadata
        documents = process_pdf_with_metadata(file_content, file.filename)
        
        if not documents:
            raise HTTPException(status_code=400, detail="Could not extract any content from the PDF.")

        # Split documents into chunks with enhanced metadata
        chunks = split_documents_with_metadata(documents)
        
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not create any text chunks from the PDF.")

        # Add chunks to PGVector store
        global vectorstore
        if vectorstore is None:
            raise HTTPException(status_code=500, detail="Vector store not initialized.")
        
        # Store documents in PGVector
        doc_ids = vectorstore.add_documents(chunks)
        logging.info(f"Added {len(chunks)} chunks to PGVector store with IDs: {doc_ids[:5]}...")

        return {
            "status": "success",
            "filename": file.filename,
            "pages_processed": len(documents),
            "chunks_indexed": len(chunks),
            "document_ids": doc_ids[:10],  # Return first 10 IDs as sample
            "metadata_sample": chunks[0].metadata if chunks else {}
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process uploaded file: {e}")


@app.get("/query", summary="Query the indexed documents using MMR")
async def query(
    q: str = Query(..., min_length=3, description="The question to ask the documents."),
    search_type: str = Query("mmr", description="Search type: 'mmr' or 'similarity'"),
    k: int = Query(4, description="Number of documents to retrieve"),
    lambda_mult: float = Query(0.5, description="MMR diversity parameter (0=diverse, 1=relevant)")
):
    """
    Endpoint to query the indexed documents using MMR retrieval.
    Returns an answer generated by Ollama based on the most relevant chunks.
    """
    if vectorstore is None or qa_chain is None:
        raise HTTPException(
            status_code=404, 
            detail="No documents have been indexed yet. Please upload a PDF first."
        )

    logging.info(f"Received query: {q}")

    try:
        # Update retriever settings dynamically
        global retriever
        if search_type == "mmr":
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "lambda_mult": lambda_mult,
                    "fetch_k": k * 2
                }
            )
        else:
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
        
        # Update QA chain with new retriever
        qa_chain.retriever = retriever

        # Execute the query
        logging.info(f"Executing query with {search_type} retrieval...")
        result = qa_chain({"query": q})
        
        # Extract source documents and their metadata
        source_docs = result.get("source_documents", [])
        sources_info = []
        
        for doc in source_docs:
            sources_info.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata,
                "page": doc.metadata.get("page", "unknown"),
                "source": doc.metadata.get("source", "unknown"),
                "chunk_id": doc.metadata.get("chunk_id", "unknown")
            })

        logging.info(f"Query completed. Found {len(source_docs)} relevant chunks.")

        return {
            "answer": result["result"],
            "search_type": search_type,
            "retrieval_params": {
                "k": k,
                "lambda_mult": lambda_mult if search_type == "mmr" else None
            },
            "source_documents": sources_info,
            "num_sources": len(source_docs)
        }

    except Exception as e:
        logging.error(f"Error during query processing: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process query: {e}")


@app.get("/health", summary="Health check endpoint")
async def health_check():
    """
    Health check endpoint to verify all components are working.
    """
    status = {
        "status": "healthy",
        "components": {
            "embeddings_model": embeddings_model is not None,
            "text_splitter": text_splitter is not None,
            "vectorstore": vectorstore is not None,
            "llm": llm is not None,
            "retriever": retriever is not None,
            "qa_chain": qa_chain is not None
        },
        "database_url": DATABASE_URL.split("@")[0] + "@***",  # Hide credentials
        "model": OLLAMA_MODEL
    }
    
    # Check if any component is missing
    if not all(status["components"].values()):
        status["status"] = "degraded"
        
    return status


@app.get("/collections/info", summary="Get information about stored documents")
async def get_collection_info():
    """
    Get information about documents stored in the vector database.
    """
    if vectorstore is None:
        raise HTTPException(status_code=404, detail="Vector store not initialized.")
    
    try:
        # This is a basic implementation - you might need to adjust based on your PGVector setup
        # Some vector stores provide methods to get collection statistics
        return {
            "collection_name": COLLECTION_NAME,
            "database_url": DATABASE_URL.split("@")[0] + "@***",
            "embedding_model": "all-MiniLM-L6-v2",
            "text_splitter": "SentenceTransformersTokenTextSplitter",
            "search_types": ["similarity", "mmr"],
            "status": "active"
        }
    except Exception as e:
        logging.error(f"Error getting collection info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get collection info: {e}")
