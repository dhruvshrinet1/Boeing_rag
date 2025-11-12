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
    def __init__(self, vectorstore: FAISS, top_k: int = 15, rerank_top_k: int = 10):
        try:
            self.vectorstore = vectorstore
            self.top_k = top_k
            self.rerank_top_k = rerank_top_k

            self.llm = ModelLoader().load_llm()
            self.qa_prompt = PROMPT_REGISTRY["context_qa"]
            self.query_expander = QueryExpander()

            self._build_bm25_index()
            self._load_reranker()
            self._build_rag_chain()

            log.info("HybridBoeingRAGService initialized", top_k=top_k, rerank_top_k=rerank_top_k)
        except Exception as e:
            log.error("Failed to initialize service", error=str(e))
            raise DocumentPortalException("RAG service initialization failed", sys)

    def _build_bm25_index(self):
        try:
            all_docs = list(self.vectorstore.docstore._dict.values())
            self.bm25_corpus = [doc.page_content for doc in all_docs]
            tokenized_corpus = [doc.split() for doc in self.bm25_corpus]

            self.bm25 = BM25Okapi(tokenized_corpus)
            self.bm25_documents = all_docs

            log.info("BM25 index built", num_documents=len(all_docs))
        except Exception as e:
            log.error("BM25 index failed", error=str(e))
            self.bm25 = None
            self.bm25_documents = []

    def _load_reranker(self):
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
        dense_results = self.vectorstore.similarity_search_with_score(query, k=k)

        bm25_results = []
        if self.bm25 is not None:
            tokenized_query = query.split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
            bm25_results = [(self.bm25_documents[i], bm25_scores[i]) for i in top_indices]

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

        k_constant = 60
        combined_results = []
        for doc_key, info in doc_scores.items():
            rrf_score = (1 / (info['dense_rank'] + k_constant)) + (1 / (info['bm25_rank'] + k_constant))
            combined_results.append((info['doc'], rrf_score))

        combined_results.sort(key=lambda x: x[1], reverse=True)

        log.info("Hybrid retrieval done", dense=len(dense_results), bm25=len(bm25_results))
        return combined_results[:k]

    def _rerank_documents(self, query: str, docs_with_scores: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
        if self.reranker is None or not docs_with_scores:
            return docs_with_scores

        try:
            docs = [doc for doc, _ in docs_with_scores]
            pairs = [[query, doc.page_content] for doc in docs]
            rerank_scores = self.reranker.predict(pairs)

            reranked = [(docs[i], float(rerank_scores[i])) for i in range(len(docs))]
            reranked.sort(key=lambda x: x[1], reverse=True)

            log.info("Reranking done", docs=len(reranked))
            return reranked[:self.rerank_top_k]
        except Exception as e:
            log.error("Reranking failed", error=str(e))
            return docs_with_scores

    def _format_docs(self, documents: List[Any]) -> str:
        formatted_parts = []
        for doc in documents:
            content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
            page = doc.metadata.get('page', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
            formatted_parts.append(f"[Page {page}]\n{content}")
        return "\n\n---\n\n".join(formatted_parts)

    def _extract_and_clean_pages(self, answer: str) -> Tuple[str, List[int]]:
        page_numbers = set()
        pattern = r'\(Pages?[\s:]+[\d\s,and]+\)'
        matches = re.findall(pattern, answer, re.IGNORECASE)

        for match in matches:
            numbers = re.findall(r'\d+', match)
            for num in numbers:
                page_numbers.add(int(num))

        clean_answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
        clean_answer = re.sub(r'\s+', ' ', clean_answer).strip()
        clean_answer = re.sub(r'\.\.+', '.', clean_answer)

        return clean_answer, sorted(list(page_numbers))

    def _build_rag_chain(self):
        try:
            def retrieve_with_hybrid_and_rerank(question: str) -> Dict[str, Any]:
                expanded_queries = self.query_expander.expand_query(question)
                expanded_query = " ".join(expanded_queries)
                docs_with_scores = self._hybrid_retrieval(expanded_query, k=self.top_k)
                reranked_docs = self._rerank_documents(question, docs_with_scores)
                docs = [doc for doc, _ in reranked_docs]

                return {
                    "documents": docs,
                    "docs_with_scores": reranked_docs,
                    "context": self._format_docs(docs),
                    "input": question
                }

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
        try:
            result = self.chain.invoke(question)
            answer_with_citations = result.get("answer", "")
            clean_answer, pages = self._extract_and_clean_pages(answer_with_citations)

            return RAGResponse(answer=clean_answer, pages=pages)
        except Exception as e:
            log.error("Query failed", error=str(e))
            raise DocumentPortalException(f"Query processing failed: {str(e)}", sys)

    def get_retrieval_score_info(self, question: str) -> Dict[str, Any]:
        try:
            expanded_queries = self.query_expander.expand_query(question)
            expanded_query = " ".join(expanded_queries)
            docs_and_scores = self._hybrid_retrieval(expanded_query, k=self.top_k)
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

            retrieval_info["unique_pages"] = sorted(list(set(retrieval_info["retrieved_pages"])))
            return retrieval_info
        except Exception as e:
            log.error("Retrieval info failed", error=str(e))
            return {"error": str(e)}
