"""
AI Avatar Tutor - Main Application Entry Point
This module initializes and starts the FastAPI application server,
sets up the RAG system, and serves the frontend interface.
"""
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from config.settings import get_settings
from api.routes import router
from utils.logger import get_logger
from rag.document_loader import DocumentLoader
from rag.embedder import Embedder
from rag.vector_store import VectorStore
logger = get_logger(__name__)
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    app = FastAPI(
        title="AI Avatar Tutor",
        description="An AI-powered tutor with voice interaction and avatar video responses",
        version="1.0.0",
        debug=settings.debug,
    )
    app.include_router(router, prefix="/api")
    frontend_path = Path(__file__).parent / "frontend"
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")
    @app.get("/")
    async def serve_frontend() -> FileResponse:
        """Serve the main frontend HTML page."""
        return FileResponse(str(frontend_path / "index.html"))
    @app.on_event("startup")
    async def startup_event() -> None:
        """Initialize the RAG system on application startup."""
        logger.info("Starting AI Avatar Tutor application...")
        try:
            _initialize_rag_system(settings)
            logger.info("Application startup complete.")
        except Exception as e:
            logger.warning(f"RAG initialization skipped: {e}")
    return app
def _initialize_rag_system(settings) -> None:
    """Initialize the RAG system by loading documents and building the vector store."""
    docs_path = Path(settings.docs_path)
    vector_store_path = Path(settings.vector_store_path)
    if vector_store_path.exists() and any(vector_store_path.iterdir()):
        logger.info("Existing vector store found. Loading from disk.")
        return
    if not docs_path.exists():
        docs_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created docs directory at {docs_path}")
        return
    doc_files = list(docs_path.glob("*.pdf")) + list(docs_path.glob("*.txt")) + list(docs_path.glob("*.md"))
    if not doc_files:
        logger.info("No documents found in docs folder. Skipping indexing.")
        return
    logger.info(f"Found {len(doc_files)} documents. Starting indexing...")
    loader = DocumentLoader(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    documents = loader.load_directory(str(docs_path))
    if documents:
        embedder = Embedder(model_name=settings.embedding_model)
        texts = [doc.page_content for doc in documents]
        embeddings = embedder.embed_texts(texts)
        vector_store = VectorStore(store_path=str(vector_store_path))
        vector_store.add_documents(documents, embeddings)
        logger.info(f"Indexed {len(documents)} document chunks into vector store.")
app = create_app()
if __name__ == "__main__":
    settings = get_settings()
    logger.info(f"Starting server on {settings.app_host}:{settings.app_port}")
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
    )