"""
Document loader module for processing various file formats.
Supports PDF, plain text, and markdown files with configurable
chunking for optimal retrieval performance.
"""
from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.logger import get_logger
logger = get_logger(__name__)
class DocumentLoader:
    """Loads and chunks documents from various file formats."""
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        """
        Initialize the document loader with chunking configuration.
        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of overlapping characters between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    def load_directory(self, directory_path: str) -> List[Document]:
        """
        Load all supported documents from a directory.
        Args:
            directory_path: Path to the directory containing documents.
        Returns:
            List of LangChain Document objects with metadata.
        """
        path = Path(directory_path)
        if not path.exists():
            logger.warning(f"Directory does not exist: {directory_path}")
            return []
        all_documents: List[Document] = []
        supported_extensions = {".pdf", ".txt", ".md"}
        for file_path in sorted(path.iterdir()):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    docs = self.load_file(str(file_path))
                    all_documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} chunks from {file_path.name}")
                except Exception as e:
                    logger.error(f"Failed to load {file_path.name}: {e}")
        logger.info(f"Total documents loaded: {len(all_documents)} chunks from {directory_path}")
        return all_documents
    def load_file(self, file_path: str) -> List[Document]:
        """
        Load and chunk a single file.
        Args:
            file_path: Path to the document file.
        Returns:
            List of LangChain Document objects.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        extension = path.suffix.lower()
        if extension == ".pdf":
            return self._load_pdf(path)
        elif extension == ".txt":
            return self._load_text(path)
        elif extension == ".md":
            return self._load_markdown(path)
        else:
            logger.warning(f"Unsupported file type: {extension}")
            return []
    def _load_pdf(self, file_path: Path) -> List[Document]:
        """
        Load a PDF file and split into chunks.
        Args:
            file_path: Path to the PDF file.
        Returns:
            List of Document objects with page metadata.
        """
        try:
            from PyPDF2 import PdfReader
            reader = PdfReader(str(file_path))
            documents: List[Document] = []
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    chunks = self.text_splitter.split_text(text)
                    for chunk_idx, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "source": file_path.name,
                                "page": page_num,
                                "chunk_index": chunk_idx,
                                "file_type": "pdf",
                            },
                        )
                        documents.append(doc)
            return documents
        except Exception as e:
            logger.error(f"Error reading PDF {file_path.name}: {e}")
            raise
    def _load_text(self, file_path: Path) -> List[Document]:
        """
        Load a text file and split into chunks.
        Args:
            file_path: Path to the text file.
        Returns:
            List of Document objects.
        """
        try:
            text = file_path.read_text(encoding="utf-8")
            if not text.strip():
                logger.warning(f"Empty text file: {file_path.name}")
                return []
            chunks = self.text_splitter.split_text(text)
            documents: List[Document] = []
            for chunk_idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": file_path.name,
                        "page": 1,
                        "chunk_index": chunk_idx,
                        "file_type": "txt",
                    },
                )
                documents.append(doc)
            return documents
        except Exception as e:
            logger.error(f"Error reading text file {file_path.name}: {e}")
            raise
    def _load_markdown(self, file_path: Path) -> List[Document]:
        """
        Load a markdown file and split into chunks.
        Args:
            file_path: Path to the markdown file.
        Returns:
            List of Document objects.
        """
        try:
            text = file_path.read_text(encoding="utf-8")
            if not text.strip():
                logger.warning(f"Empty markdown file: {file_path.name}")
                return []
            chunks = self.text_splitter.split_text(text)
            documents: List[Document] = []
            for chunk_idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": file_path.name,
                        "page": 1,
                        "chunk_index": chunk_idx,
                        "file_type": "md",
                    },
                )
                documents.append(doc)
            return documents
        except Exception as e:
            logger.error(f"Error reading markdown file {file_path.name}: {e}")
            raise