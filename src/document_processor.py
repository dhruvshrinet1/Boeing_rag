"""
Document Processing Service
Handles PDF processing with page-level tracking for the RAG system.
"""
import os
import sys
from pathlib import Path
from typing import List, Optional
import pymupdf  # PyMuPDF
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException


class PDFProcessor:
    """
    Processes PDF documents with page-level metadata tracking.
    Each chunk maintains its source page number for citation purposes.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize PDF processor with chunking parameters.

        Args:
            chunk_size: Maximum size of text chunks (increased to 1000 for better context)
            chunk_overlap: Overlap between consecutive chunks to maintain context
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Use separators that preserve table structure better
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]  # Better for tables
        )
        log.info("PDFProcessor initialized", chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def extract_pages_with_metadata(self, pdf_path: str) -> List[Document]:
        """
        Extract text from PDF with page-level metadata.

        Each page is extracted as a separate document with its page number (1-based index).
        This is crucial for accurate page citations in responses.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of Document objects with page_content and metadata including page numbers
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF file not found: {pdf_path}")

            documents = []
            with pymupdf.open(pdf_path) as doc:
                total_pages = doc.page_count
                log.info("Processing PDF", pdf_path=pdf_path, total_pages=total_pages)

                for page_num in range(total_pages):
                    page = doc.load_page(page_num)

                    # Extract text with layout preservation for better table handling
                    text = page.get_text("text")  # Preserves spacing/layout

                    # Extract tables if available
                    try:
                        tables = page.find_tables()
                        if tables:
                            table_text = "\n\n=== TABLES ===\n"
                            for table in tables:
                                # Convert table to text with structure
                                table_data = table.extract()
                                for row in table_data:
                                    table_text += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
                            text += table_text
                    except Exception as e:
                        log.debug("Table extraction skipped", page=page_num + 1, error=str(e))

                    # Skip empty pages
                    if not text.strip():
                        log.warning("Empty page detected", page_number=page_num + 1)
                        continue

                    # Create document with page metadata (1-based index for user-facing page numbers)
                    doc_obj = Document(
                        page_content=text,
                        metadata={
                            "source": os.path.basename(pdf_path),
                            "page": page_num + 1,  # 1-based index as required
                            "total_pages": total_pages
                        }
                    )
                    documents.append(doc_obj)

                log.info("PDF extraction complete",
                        pdf_path=pdf_path,
                        pages_extracted=len(documents))

            return documents

        except Exception as e:
            log.error("Failed to extract PDF pages", error=str(e), pdf_path=pdf_path)
            raise DocumentPortalException(f"PDF extraction failed: {str(e)}", sys)

    def split_documents_with_page_tracking(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks while preserving page number metadata.

        This ensures that even when pages are split into smaller chunks for embedding,
        we can still trace back to the original page number.

        Args:
            documents: List of Document objects with page metadata

        Returns:
            List of chunked Document objects, each maintaining its source page number
        """
        try:
            chunked_documents = []

            for doc in documents:
                # Split the document into chunks
                chunks = self.text_splitter.split_text(doc.page_content)

                # Preserve metadata for each chunk
                for chunk in chunks:
                    chunked_doc = Document(
                        page_content=chunk,
                        metadata=doc.metadata.copy()  # Preserve original page number
                    )
                    chunked_documents.append(chunked_doc)

            log.info("Document chunking complete",
                    original_docs=len(documents),
                    chunked_docs=len(chunked_documents))

            return chunked_documents

        except Exception as e:
            log.error("Failed to chunk documents", error=str(e))
            raise DocumentPortalException(f"Document chunking failed: {str(e)}", sys)

    def process_pdf(self, pdf_path: str) -> List[Document]:
        """
        Complete PDF processing pipeline: extract pages and create chunks.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of processed Document chunks with page metadata
        """
        log.info("Starting PDF processing pipeline", pdf_path=pdf_path)

        # Step 1: Extract pages with metadata
        page_documents = self.extract_pages_with_metadata(pdf_path)

        # Step 2: Split into chunks while maintaining page numbers
        chunked_documents = self.split_documents_with_page_tracking(page_documents)

        log.info("PDF processing pipeline complete",
                pdf_path=pdf_path,
                total_chunks=len(chunked_documents))

        return chunked_documents


class VectorStoreManager:
    """
    Manages FAISS vector store creation and loading.
    Handles embedding and indexing of documents.
    """

    def __init__(self, index_dir: str = "faiss_index"):
        """
        Initialize vector store manager.

        Args:
            index_dir: Directory to store FAISS index files
        """
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_loader = ModelLoader()
        self.embeddings = self.model_loader.load_embeddings()
        log.info("VectorStoreManager initialized", index_dir=str(self.index_dir))

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """
        Create FAISS vector store from documents.

        Args:
            documents: List of Document objects to index

        Returns:
            FAISS vector store object
        """
        try:
            if not documents:
                raise ValueError("No documents provided for vector store creation")

            log.info("Creating FAISS vector store", num_documents=len(documents))

            # Create FAISS index from documents
            vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )

            # Save to disk
            vectorstore.save_local(str(self.index_dir))
            log.info("Vector store created and saved", index_dir=str(self.index_dir))

            return vectorstore

        except Exception as e:
            log.error("Failed to create vector store", error=str(e))
            raise DocumentPortalException(f"Vector store creation failed: {str(e)}", sys)

    def load_vector_store(self) -> Optional[FAISS]:
        """
        Load existing FAISS vector store from disk.

        Returns:
            FAISS vector store object if exists, None otherwise
        """
        try:
            index_file = self.index_dir / "index.faiss"

            if not index_file.exists():
                log.info("No existing vector store found", index_dir=str(self.index_dir))
                return None

            log.info("Loading existing vector store", index_dir=str(self.index_dir))

            vectorstore = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )

            log.info("Vector store loaded successfully")
            return vectorstore

        except Exception as e:
            log.error("Failed to load vector store", error=str(e))
            return None

    def get_or_create_vector_store(self, pdf_path: Optional[str] = None) -> FAISS:
        """
        Load existing vector store or create new one from PDF.

        Args:
            pdf_path: Path to PDF file (required if creating new store)

        Returns:
            FAISS vector store object
        """
        # Try to load existing store first
        vectorstore = self.load_vector_store()

        if vectorstore is not None:
            log.info("Using existing vector store")
            return vectorstore

        # Create new store if none exists
        if pdf_path is None:
            raise DocumentPortalException(
                "No existing vector store found and no PDF path provided", sys
            )

        log.info("Creating new vector store from PDF", pdf_path=pdf_path)

        # Process PDF and create store
        processor = PDFProcessor()
        documents = processor.process_pdf(pdf_path)
        vectorstore = self.create_vector_store(documents)

        return vectorstore


if __name__ == "__main__":
    # Test the document processor
    pdf_path = "data/document_analysis/Boeing B737 Manual-1.pdf"

    if not os.path.exists(pdf_path):
        print(f"PDF not found: {pdf_path}")
        sys.exit(1)

    # Test PDF processing
    processor = PDFProcessor()
    documents = processor.process_pdf(pdf_path)
    print(f"Processed {len(documents)} document chunks")

    # Display sample chunk with metadata
    if documents:
        sample = documents[0]
        print(f"\nSample chunk metadata: {sample.metadata}")
        print(f"Sample content (first 200 chars): {sample.page_content[:200]}...")
