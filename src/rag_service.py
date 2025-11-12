import sys
import re
from typing import List, Tuple, Dict, Any
from collections import defaultdict

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from logger import GLOBAL_LOGGER as log
from prompt.prompt_library import PROMPT_REGISTRY
from model.models import RAGResponse
from src.query_expander import QueryExpander


class HybridBoeingRAGService:
    """Hybrid RAG with BM25 + Dense retrieval and reranking"""

    def __init__(self, vectorstore: FAISS, top_k: int = 20, rerank_top_k: int = 15):
        try:
            self.vectorstore = vectorstore
            self.top_k = top_k
            self.rerank_top_k = rerank_top_k

            self.llm = ModelLoader().load_llm()
            self.qa_prompt = PROMPT_REGISTRY["context_qa"]
            self.query_expander = QueryExpander()

            # setup hybrid search components
            self._build_bm25_index()
            self._load_reranker()
            self._build_rag_chain()

            log.info("HybridBoeingRAGService initialized", top_k=top_k, rerank_top_k=rerank_top_k)
        except Exception as e:
            log.error("Failed to initialize service", error=str(e))
            raise DocumentPortalException("RAG service initialization failed", sys)

    def _build_bm25_index(self):
        """Build BM25 index for keyword search"""
        try:
            all_docs = list(self.vectorstore.docstore._dict.values())
            self.bm25_corpus = [doc.page_content for doc in all_docs]
            tokenized_corpus = [doc.split() for doc in self.bm25_corpus]

            self.bm25 = BM25Okapi(tokenized_corpus)
            self.bm25_documents = all_docs

            log.info("BM25 index built", num_documents=len(all_docs))
        except Exception as e:
            log.error("BM25 index failed", error=str(e))
            # fallback to dense-only if BM25 fails
            self.bm25 = None
            self.bm25_documents = []

    def _load_reranker(self):
        """Load cross-encoder for reranking"""
        try:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            log.info("Reranker loaded")
        except ImportError:
            log.warning("sentence-transformers not installed")
            self.reranker = None
        except Exception as e:
            log.error("Reranker failed to load", error=str(e))
            self.reranker = None

    def _hybrid_retrieval(self, query: str, k: int) -> List[Tuple[Any, float]]:
        """Combine BM25 and dense retrieval using RRF"""
        # dense (semantic) search
        dense_results = self.vectorstore.similarity_search_with_score(query, k=k)

        # BM25 (keyword) search
        bm25_results = []
        if self.bm25 is not None:
            tokenized_query = query.split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
            bm25_results = [(self.bm25_documents[i], bm25_scores[i]) for i in top_indices]

        # track rankings from both methods
        doc_scores = defaultdict(lambda: {'dense_rank': float('inf'), 'bm25_rank': float('inf'), 'doc': None})

        for rank, (doc, score) in enumerate(dense_results):
            doc_key = (doc.metadata.get('page', 0), hash(doc.page_content))
            doc_scores[doc_key]['dense_rank'] = rank
            doc_scores[doc_key]['doc'] = doc

        for rank, (doc, score) in enumerate(bm25_results):
            doc_key = (doc.metadata.get('page', 0), hash(doc.page_content))
            doc_scores[doc_key]['bm25_rank'] = rank
            if doc_scores[doc_key]['doc'] is None:
                doc_scores[doc_key]['doc'] = doc

        # combine using reciprocal rank fusion (RRF)
        k_constant = 60
        combined_results = []
        for doc_key, info in doc_scores.items():
            rrf_score = (1 / (info['dense_rank'] + k_constant)) + (1 / (info['bm25_rank'] + k_constant))
            combined_results.append((info['doc'], rrf_score))

        combined_results.sort(key=lambda x: x[1], reverse=True)

        log.info("Hybrid retrieval done", dense=len(dense_results), bm25=len(bm25_results))
        return combined_results[:k]

    def _rerank_documents(self, query: str, docs_with_scores: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
        """Rerank using cross-encoder for better relevance"""
        if self.reranker is None or not docs_with_scores:
            return docs_with_scores

        try:
            docs = [doc for doc, _ in docs_with_scores]
            pairs = [[query, doc.page_content] for doc in docs]
            rerank_scores = self.reranker.predict(pairs)

            # resort by reranker scores
            reranked = [(docs[i], float(rerank_scores[i])) for i in range(len(docs))]
            reranked.sort(key=lambda x: x[1], reverse=True)

            log.info("Reranking done", docs=len(reranked))
            return reranked[:self.rerank_top_k]
        except Exception as e:
            log.error("Reranking failed", error=str(e))
            return docs_with_scores

    def _format_docs(self, documents: List[Any]) -> str:
        """Format docs with page numbers for LLM context"""
        formatted_parts = []
        for doc in documents:
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            page = doc.metadata.get('page', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
            formatted_parts.append(f"[Page {page}]\n{content}")
        return "\n\n---\n\n".join(formatted_parts)

    def _extract_and_clean_pages(self, answer: str) -> Tuple[str, List[int]]:
        """Extract page citations from LLM answer"""
        page_numbers = set()
        # match patterns like (Page 43) or (Pages 39 and 51)
        pattern = r'\(Pages?[\s:]+[\d\s,and]+\)'
        matches = re.findall(pattern, answer, re.IGNORECASE)

        for match in matches:
            numbers = re.findall(r'\d+', match)
            for num in numbers:
                page_numbers.add(int(num))

        # remove citations from answer text
        clean_answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
        clean_answer = re.sub(r'\s+', ' ', clean_answer).strip()
        clean_answer = re.sub(r'\.\.+', '.', clean_answer)

        return clean_answer, sorted(list(page_numbers))

    def _build_rag_chain(self):
        """Build LCEL chain with hybrid retrieval"""
        try:
            def retrieve_with_hybrid_and_rerank(question: str) -> Dict[str, Any]:
                # expand query with aviation terms
                expanded_queries = self.query_expander.expand_query(question)
                expanded_query = " ".join(expanded_queries)

                # hybrid search
                docs_with_scores = self._hybrid_retrieval(expanded_query, k=self.top_k)

                # rerank for relevance
                reranked_docs = self._rerank_documents(question, docs_with_scores)
                docs = [doc for doc, _ in reranked_docs]

                return {
                    "documents": docs,
                    "docs_with_scores": reranked_docs,
                    "context": self._format_docs(docs),
                    "input": question
                }

            # build chain: retrieve -> generate answer
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

            log.info("RAG chain built")
        except Exception as e:
            log.error("Failed to build chain", error=str(e))
            raise DocumentPortalException("RAG chain building failed", sys)

    def query(self, question: str) -> RAGResponse:
        """Process question and return answer with pages"""
        try:
            result = self.chain.invoke(question)
            answer_with_citations = result.get("answer", "")

            # extract page numbers from citations
            clean_answer, pages = self._extract_and_clean_pages(answer_with_citations)

            return RAGResponse(answer=clean_answer, pages=pages)
        except Exception as e:
            log.error("Query failed", error=str(e))
            raise DocumentPortalException(f"Query processing failed: {str(e)}", sys)

