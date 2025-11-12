import os
import sys
from pathlib import Path
from typing import List, Optional
import pymupdf
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException


class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
        )
        log.info("PDFProcessor initialized", chunk_size=chunk_size)

    def extract_pages_with_metadata(self, pdf_path: str) -> List[Document]:
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

            documents = []
            with pymupdf.open(pdf_path) as doc:
                total_pages = doc.page_count
                log.info("Processing PDF", total_pages=total_pages)

                for page_num in range(total_pages):
                    page = doc.load_page(page_num)
                    text = page.get_text("text")

                    try:
                        tables = page.find_tables()
                        if tables:
                            table_text = "\n\n=== TABLES ===\n"
                            for table in tables:
                                table_data = table.extract()
                                for row in table_data:
                                    table_text += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
                            text += table_text
                    except Exception as e:
                        log.debug("Table extraction skipped", page=page_num + 1)

                    if not text.strip():
                        continue

                    doc_obj = Document(
                        page_content=text,
                        metadata={
                            "source": os.path.basename(pdf_path),
                            "page": page_num + 1,
                            "total_pages": total_pages
                        }
                    )
                    documents.append(doc_obj)

                log.info("PDF extracted", pages=len(documents))
            return documents

        except Exception as e:
            log.error("PDF extraction failed", error=str(e))
            raise DocumentPortalException(f"PDF extraction failed: {str(e)}", sys)

    def split_documents_with_page_tracking(self, documents: List[Document]) -> List[Document]:
        try:
            chunked_documents = []

            for doc in documents:
                chunks = self.text_splitter.split_text(doc.page_content)

                for chunk in chunks:
                    chunked_doc = Document(
                        page_content=chunk,
                        metadata=doc.metadata.copy()
                    )
                    chunked_documents.append(chunked_doc)

            log.info("Chunking done", chunks=len(chunked_documents))
            return chunked_documents

        except Exception as e:
            log.error("Chunking failed", error=str(e))
            raise DocumentPortalException(f"Chunking failed: {str(e)}", sys)

    def process_pdf(self, pdf_path: str) -> List[Document]:
        log.info("Processing PDF", path=pdf_path)
        page_documents = self.extract_pages_with_metadata(pdf_path)
        chunked_documents = self.split_documents_with_page_tracking(page_documents)
        return chunked_documents


class VectorStoreManager:
    def __init__(self, index_dir: str = "faiss_index"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_loader = ModelLoader()
        self.embeddings = self.model_loader.load_embeddings()
        log.info("VectorStoreManager initialized", index_dir=str(self.index_dir))

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        try:
            if not documents:
                raise ValueError("No documents provided")

            log.info("Creating vector store", docs=len(documents))

            vectorstore = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )

            vectorstore.save_local(str(self.index_dir))
            log.info("Vector store saved")

            return vectorstore

        except Exception as e:
            log.error("Vector store creation failed", error=str(e))
            raise DocumentPortalException(f"Vector store creation failed: {str(e)}", sys)

    def load_vector_store(self) -> Optional[FAISS]:
        try:
            index_file = self.index_dir / "index.faiss"

            if not index_file.exists():
                log.info("No existing vector store")
                return None

            log.info("Loading vector store")

            vectorstore = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.embeddings,
                allow_dangerous_deserialization=True
            )

            log.info("Vector store loaded")
            return vectorstore

        except Exception as e:
            log.error("Failed to load vector store", error=str(e))
            return None

    def get_or_create_vector_store(self, pdf_path: Optional[str] = None) -> FAISS:
        vectorstore = self.load_vector_store()

        if vectorstore is not None:
            return vectorstore

        if pdf_path is None:
            raise DocumentPortalException("No vector store and no PDF path", sys)

        log.info("Creating new vector store", pdf=pdf_path)

        processor = PDFProcessor()
        documents = processor.process_pdf(pdf_path)
        vectorstore = self.create_vector_store(documents)

        return vectorstore
