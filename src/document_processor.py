import os
import sys
import base64
from pathlib import Path
from typing import List, Optional
import pymupdf
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage
from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException
from prompt.prompt_library import PROMPT_REGISTRY


class PDFProcessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, process_images: bool = True):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.process_images = process_images
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
        )

        # load multimodal LLM for image understanding
        if self.process_images:
            try:
                self.llm = ModelLoader().load_llm()
                log.info("Multimodal LLM loaded for image processing")
            except Exception as e:
                log.warning("Failed to load LLM for images, skipping image processing", error=str(e))
                self.process_images = False

        log.info("PDFProcessor initialized", chunk_size=chunk_size, process_images=self.process_images)

    def _extract_image_info(self, page, page_num: int) -> str:
        """Extract and describe images from a page using multimodal LLM"""
        if not self.process_images:
            return ""

        try:
            image_list = page.get_images()
            if not image_list:
                return ""

            image_descriptions = []
            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = page.parent.extract_image(xref)
                    image_bytes = base_image["image"]

                    # convert to base64 for LLM
                    image_b64 = base64.b64encode(image_bytes).decode('utf-8')

                    # get description from multimodal LLM
                    prompt = PROMPT_REGISTRY["image_description"]

                    # invoke multimodal LLM
                    message = HumanMessage(
                        content=[
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}
                            }
                        ]
                    )

                    response = self.llm.invoke([message])
                    description = response.content if hasattr(response, 'content') else str(response)

                    image_descriptions.append(f"[Image {img_index + 1}]: {description}")
                    log.debug(f"Processed image {img_index + 1} on page {page_num + 1}")

                except Exception as e:
                    log.debug(f"Failed to process image {img_index} on page {page_num + 1}", error=str(e))
                    continue

            if image_descriptions:
                return "\n\n=== IMAGES ===\n" + "\n".join(image_descriptions) + "\n"
            return ""

        except Exception as e:
            log.debug("Image extraction failed", page=page_num + 1, error=str(e))
            return ""

    def extract_pages_with_metadata(self, pdf_path: str) -> List[Document]:
        """Extract text, tables, and images from each page"""
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

            documents = []
            with pymupdf.open(pdf_path) as doc:
                total_pages = doc.page_count
                log.info("Processing PDF", total_pages=total_pages, with_images=self.process_images)

                for page_num in range(total_pages):
                    page = doc.load_page(page_num)
                    text = page.get_text("text")

                    # extract tables
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

                    # extract and describe images
                    if self.process_images:
                        image_text = self._extract_image_info(page, page_num)
                        text += image_text

                    if not text.strip():
                        continue

                    # store with 1-based page numbers
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
        """Split documents into chunks while keeping page metadata"""
        try:
            chunked_documents = []

            for doc in documents:
                chunks = self.text_splitter.split_text(doc.page_content)

                # each chunk keeps original page number
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
        """Extract and chunk PDF in one go"""
        log.info("Processing PDF", path=pdf_path)
        page_documents = self.extract_pages_with_metadata(pdf_path)
        chunked_documents = self.split_documents_with_page_tracking(page_documents)
        return chunked_documents


class VectorStoreManager:
    def __init__(self, index_dir: str = "faiss_index", process_images: bool = True):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.model_loader = ModelLoader()
        self.embeddings = self.model_loader.load_embeddings()
        self.process_images = process_images
        log.info("VectorStoreManager initialized", index_dir=str(self.index_dir))

    def create_vector_store(self, documents: List[Document]) -> FAISS:
        """Create new FAISS index from documents"""
        try:
            if not documents:
                raise ValueError("No documents provided")

            log.info("Creating vector store", docs=len(documents))

            # create embeddings and build index
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
        """Load existing FAISS index if available"""
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
        """Load existing index or create new one"""
        vectorstore = self.load_vector_store()

        if vectorstore is not None:
            return vectorstore

        if pdf_path is None:
            raise DocumentPortalException("No vector store and no PDF path", sys)

        log.info("Creating new vector store", pdf=pdf_path)

        # process PDF and create index
        processor = PDFProcessor(process_images=self.process_images)
        documents = processor.process_pdf(pdf_path)
        vectorstore = self.create_vector_store(documents)

        return vectorstore
