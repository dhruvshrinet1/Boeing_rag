"""
RAG Service for Boeing 737 Manual
Hybrid retrieval (BM25 + Dense) with cross-encoder reranking and query expansion.
"""
import sys
from typing import List, Tuple, Dict, Any
from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from logger import GLOBAL_LOGGER as log
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import PromptType, RAGResponse
from src.query_expander import QueryExpander


class HybridBoeingRAGService:
    """
    Advanced RAG service with:
    - Hybrid retrieval: BM25 (keyword) + Dense (semantic)
    - Cross-encoder reranking for relevance scoring
    - Query expansion with aviation terms
    - Page extraction from LLM answer citations
    """

    def __init__(self, vectorstore: FAISS, top_k: int = 15, rerank_top_k: int = 10):
        """
        Initialize hybrid RAG service with reranking.

        Args:
            vectorstore: FAISS vector store containing indexed manual
            top_k: Number of documents to retrieve from hybrid search
            rerank_top_k: Number of documents to keep after reranking
        """
        try:
            self.vectorstore = vectorstore
            self.top_k = top_k
            self.rerank_top_k = rerank_top_k

            # Load LLM and prompts
            self.llm = ModelLoader().load_llm()
            self.qa_prompt = PROMPT_REGISTRY["context_qa"]

            # Initialize query expander (keyword-based for speed)
            self.query_expander = QueryExpander()

            # Build BM25 index from vectorstore documents
            self._build_bm25_index()

            # Load cross-encoder for reranking
            self._load_reranker()

            # Build RAG chain
            self._build_rag_chain()

            log.info("HybridBoeingRAGService initialized",
                    top_k=top_k,
                    rerank_top_k=rerank_top_k)

        except Exception as e:
            log.error("Failed to initialize HybridBoeingRAGService", error=str(e))
            raise DocumentPortalException("RAG service initialization failed", sys)

    def _build_bm25_index(self):
        """Build BM25 index from all documents in vectorstore."""
        try:
            # Get all documents from vectorstore
            # Note: This accesses the internal docstore - may need adjustment based on FAISS version
            all_docs = list(self.vectorstore.docstore._dict.values())

            # Tokenize documents for BM25
            self.bm25_corpus = [doc.page_content for doc in all_docs]
            tokenized_corpus = [doc.split() for doc in self.bm25_corpus]

            # Create BM25 index
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.bm25_documents = all_docs

            log.info("BM25 index built", num_documents=len(all_docs))

        except Exception as e:
            log.error("Failed to build BM25 index", error=str(e))
            # Fallback: disable BM25 if it fails
            self.bm25 = None
            self.bm25_documents = []
            log.warning("BM25 disabled, using dense retrieval only")

    def _load_reranker(self):
        """Load cross-encoder model for reranking."""
        try:
            from sentence_transformers import CrossEncoder

            # Use a lightweight cross-encoder model
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

            log.info("Cross-encoder reranker loaded successfully")

        except ImportError:
            log.warning("sentence-transformers not installed, reranking disabled")
            self.reranker = None
        except Exception as e:
            log.error("Failed to load reranker", error=str(e))
            self.reranker = None

    def _hybrid_retrieval(self, query: str, k: int) -> List[Tuple[Any, float]]:
        """
        Perform hybrid retrieval combining BM25 and dense search.

        Args:
            query: User query (already expanded)
            k: Number of documents to retrieve

        Returns:
            List of (document, score) tuples
        """
        from collections import defaultdict

        # Dense retrieval (FAISS semantic search)
        dense_results = self.vectorstore.similarity_search_with_score(query, k=k)

        # BM25 retrieval (keyword search)
        bm25_results = []
        if self.bm25 is not None:
            tokenized_query = query.split()
            bm25_scores = self.bm25.get_scores(tokenized_query)

            # Get top-k BM25 results
            top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
            bm25_results = [(self.bm25_documents[i], bm25_scores[i]) for i in top_indices]

        # Combine results using Reciprocal Rank Fusion (RRF)
        doc_scores = defaultdict(lambda: {'dense_rank': float('inf'), 'bm25_rank': float('inf'), 'doc': None})

        # Add dense results (lower FAISS score = better)
        for rank, (doc, score) in enumerate(dense_results):
            doc_key = (doc.metadata.get('page', 0), hash(doc.page_content))
            doc_scores[doc_key]['dense_rank'] = rank
            doc_scores[doc_key]['doc'] = doc

        # Add BM25 results (higher BM25 score = better, so we rank by position)
        for rank, (doc, score) in enumerate(bm25_results):
            doc_key = (doc.metadata.get('page', 0), hash(doc.page_content))
            doc_scores[doc_key]['bm25_rank'] = rank
            if doc_scores[doc_key]['doc'] is None:
                doc_scores[doc_key]['doc'] = doc

        # Calculate RRF score: 1/(rank + 60)
        k_constant = 60
        combined_results = []
        for doc_key, info in doc_scores.items():
            rrf_score = (1 / (info['dense_rank'] + k_constant)) + (1 / (info['bm25_rank'] + k_constant))
            combined_results.append((info['doc'], rrf_score))

        # Sort by RRF score (higher = better)
        combined_results.sort(key=lambda x: x[1], reverse=True)

        log.info("Hybrid retrieval complete",
                dense_results=len(dense_results),
                bm25_results=len(bm25_results),
                combined_results=len(combined_results))

        return combined_results[:k]

    def _rerank_documents(self, query: str, docs_with_scores: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
        """
        Rerank documents using cross-encoder.

        Args:
            query: User query
            docs_with_scores: List of (document, score) from retrieval

        Returns:
            Reranked list of (document, score)
        """
        if self.reranker is None or not docs_with_scores:
            log.info("Reranking skipped (reranker not available)")
            return docs_with_scores

        try:
            # Prepare query-document pairs for reranking
            docs = [doc for doc, _ in docs_with_scores]
            pairs = [[query, doc.page_content] for doc in docs]

            # Get cross-encoder scores
            rerank_scores = self.reranker.predict(pairs)

            # Combine documents with new scores
            reranked = [(docs[i], float(rerank_scores[i])) for i in range(len(docs))]

            # Sort by reranker score (higher = better)
            reranked.sort(key=lambda x: x[1], reverse=True)

            log.info("Reranking complete",
                    original_docs=len(docs_with_scores),
                    reranked_docs=len(reranked))

            return reranked[:self.rerank_top_k]

        except Exception as e:
            log.error("Reranking failed, using original order", error=str(e))
            return docs_with_scores

    def _format_docs(self, documents: List[Any]) -> str:
        """
        Format retrieved documents into a context string.

        Args:
            documents: List of Document objects

        Returns:
            Formatted context string
        """
        formatted_parts = []

        for doc in documents:
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            page = doc.metadata.get('page', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'

            formatted_parts.append(f"[Page {page}]\n{content}")

        return "\n\n---\n\n".join(formatted_parts)

    def _extract_and_clean_pages(self, answer: str) -> Tuple[str, List[int]]:
        """
        Extract page citations from answer and return cleaned answer with page numbers.

        Args:
            answer: LLM answer with page citations like "(Page 43)" or "(Pages 39 and 51)"

        Returns:
            Tuple of (cleaned_answer, sorted_page_numbers)
        """
        import re

        # Extract page numbers from citations
        page_numbers = set()

        # Pattern to match: (Page 43), (Pages 39 and 51), (Page 39, 41, and 43)
        pattern = r'\(Pages?[\s:]+[\d\s,and]+\)'

        matches = re.findall(pattern, answer, re.IGNORECASE)

        for match in matches:
            # Extract all numbers from the citation
            numbers = re.findall(r'\d+', match)
            for num in numbers:
                page_numbers.add(int(num))

        # Remove citations from answer
        clean_answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)

        # Clean up extra whitespace
        clean_answer = re.sub(r'\s+', ' ', clean_answer).strip()
        # Remove double periods that might result from removing citations
        clean_answer = re.sub(r'\.\.+', '.', clean_answer)

        log.info("Extracted pages from answer",
                num_pages=len(page_numbers),
                pages=sorted(list(page_numbers)))

        return clean_answer, sorted(list(page_numbers))

    def _build_rag_chain(self):
        """Build the LCEL RAG chain with hybrid retrieval and reranking."""
        try:
            def retrieve_with_hybrid_and_rerank(question: str) -> Dict[str, Any]:
                """Retrieve documents using hybrid search + reranking"""
                # Step 1: Query expansion
                expanded_query = self.query_expander.expand_with_keywords(question)

                log.info("Query expanded", original=question[:50], expanded_len=len(expanded_query))

                # Step 2: Hybrid retrieval (BM25 + Dense)
                docs_with_scores = self._hybrid_retrieval(expanded_query, k=self.top_k)

                # Step 3: Rerank with cross-encoder
                reranked_docs = self._rerank_documents(question, docs_with_scores)

                # Extract just documents for context
                docs = [doc for doc, _ in reranked_docs]

                return {
                    "documents": docs,
                    "docs_with_scores": reranked_docs,
                    "context": self._format_docs(docs),
                    "input": question
                }

            # Build the chain
            self.chain = (
                RunnableLambda(retrieve_with_hybrid_and_rerank)
                | RunnablePassthrough.assign(
                    answer=lambda x: (
                        self.qa_prompt
                        | self.llm
                        | StrOutputParser()
                    ).invoke({
                        "context": x["context"],
                        "input": x["input"],
                        "chat_history": []
                    })
                )
            )

            log.info("Hybrid RAG chain built successfully")

        except Exception as e:
            log.error("Failed to build RAG chain", error=str(e))
            raise DocumentPortalException("RAG chain building failed", sys)

    def query(self, question: str) -> RAGResponse:
        """
        Process a question and return answer with page citations.

        Args:
            question: User's question about the Boeing 737 manual

        Returns:
            RAGResponse object with answer and page numbers
        """
        try:
            log.info("Processing query", question=question)

            # Invoke the chain
            result = self.chain.invoke(question)

            # Extract answer with page citations
            answer_with_citations = result.get("answer", "")

            # Extract page numbers from citations and clean the answer
            clean_answer, pages = self._extract_and_clean_pages(answer_with_citations)

            log.info("Query processed successfully",
                    question_preview=question[:100],
                    num_pages=len(pages),
                    pages=pages)

            # Create response object
            response = RAGResponse(
                answer=clean_answer,
                pages=pages
            )

            return response

        except Exception as e:
            log.error("Failed to process query", error=str(e), question=question)
            raise DocumentPortalException(f"Query processing failed: {str(e)}", sys)

    def get_retrieval_score_info(self, question: str) -> Dict[str, Any]:
        """
        Get detailed retrieval information for evaluation purposes.

        Args:
            question: User's question

        Returns:
            Dictionary with retrieval details including page numbers and scores
        """
        try:
            # Expand query
            expanded_query = self.query_expander.expand_with_keywords(question)

            # Get hybrid results
            docs_and_scores = self._hybrid_retrieval(expanded_query, k=self.top_k)

            # Rerank
            reranked_docs = self._rerank_documents(question, docs_and_scores)

            retrieval_info = {
                "question": question,
                "expanded_query": expanded_query[:200],
                "num_retrieved": len(reranked_docs),
                "retrieved_pages": [],
                "documents": []
            }

            for doc, score in reranked_docs:
                page = doc.metadata.get('page', 'Unknown')
                retrieval_info["retrieved_pages"].append(page)
                retrieval_info["documents"].append({
                    "page": page,
                    "score": float(score),
                    "content_preview": doc.page_content[:200]
                })

            # Get unique pages
            retrieval_info["unique_pages"] = sorted(list(set(retrieval_info["retrieved_pages"])))

            return retrieval_info

        except Exception as e:
            log.error("Failed to get retrieval info", error=str(e))
            return {"error": str(e)}


if __name__ == "__main__":
    # Test the RAG service
    import os
    from src.document_processor import VectorStoreManager

    # Load or create vector store
    pdf_path = "data/document_analysis/Boeing B737 Manual-1.pdf"
    vs_manager = VectorStoreManager()
    vectorstore = vs_manager.get_or_create_vector_store(pdf_path)

    # Create hybrid RAG service
    rag_service = HybridBoeingRAGService(vectorstore)

    # Test query
    test_question = "What is the fuel capacity of the Boeing 737?"
    response = rag_service.query(test_question)

    print(f"Question: {test_question}")
    print(f"Answer: {response.answer}")
    print(f"Pages: {response.pages}")
