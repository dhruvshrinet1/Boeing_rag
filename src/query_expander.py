"""
Query Expansion Service
Expands queries with technical aviation terms and synonyms for better retrieval.
"""
from typing import List, Dict
from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from prompt.prompt_library import PROMPT_REGISTRY


class QueryExpander:
    """
    Expands user queries with aviation-specific terms and context.
    """

    def __init__(self):
        """Initialize query expander with LLM and prompts."""
        self.llm = ModelLoader().load_llm()

        # Load expansion prompt from prompt library
        self.expansion_prompt = PROMPT_REGISTRY["query_expansion"]

        log.info("QueryExpander initialized")

    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with technical terms and synonyms.

        Args:
            query: Original user query

        Returns:
            List of expanded queries (including original)
        """
        try:
            # Generate expansions using LLM
            response = (self.expansion_prompt | self.llm).invoke({"query": query})

            # Parse response
            if hasattr(response, 'content'):
                expanded_text = response.content
            else:
                expanded_text = str(response)

            # Split into individual queries
            expanded_queries = [q.strip() for q in expanded_text.split('\n') if q.strip()]

            # Always include original query first
            all_queries = [query] + expanded_queries[:3]  # Original + max 3 expansions

            log.info("Query expanded",
                    original=query[:50],
                    num_expansions=len(all_queries) - 1)

            return all_queries

        except Exception as e:
            log.warning("Query expansion failed, using original", error=str(e))
            return [query]  # Fallback to original query

    def expand_with_keywords(self, query: str) -> str:
        """
        Quick expansion using keyword mapping (faster than LLM).

        Args:
            query: Original user query

        Returns:
            Expanded query string
        """
        # Aviation-specific keyword mappings
        keyword_map = {
            'takeoff': 'takeoff departure climb',
            'landing': 'landing approach descent',
            'flaps': 'flap flaps slats trailing edge',
            'weight': 'weight mass limit limitation',
            'runway': 'runway field RWY',
            'speed': 'speed velocity V1 V2 Vr',
            'altitude': 'altitude height pressure altitude PA',
            'fuel': 'fuel capacity tank',
            'engine': 'engine thrust N1 EPR',
            'climb': 'climb ascent climb-out',
            'approach': 'approach final descent',
            'wet': 'wet contaminated slush',
            'dry': 'dry uncontaminated',
            'limit': 'limit limitation maximum max',
            'procedure': 'procedure checklist operation',
            'configuration': 'configuration setting flap',
            'retraction': 'retraction retract cleanup',
        }

        expanded = query.lower()

        # Add related keywords
        for keyword, expansion in keyword_map.items():
            if keyword in expanded:
                expanded += f" {expansion}"

        log.info("Query expanded with keywords", original_len=len(query), expanded_len=len(expanded))

        return expanded


class MultiQueryRetriever:
    """
    Retrieves documents using multiple query variations.
    """

    def __init__(self, vectorstore, query_expander: QueryExpander = None):
        """
        Initialize multi-query retriever.

        Args:
            vectorstore: FAISS vector store
            query_expander: QueryExpander instance (optional)
        """
        self.vectorstore = vectorstore
        self.query_expander = query_expander or QueryExpander()
        log.info("MultiQueryRetriever initialized")

    def retrieve_with_expansion(self, query: str, k: int = 15) -> List:
        """
        Retrieve documents using query expansion.

        Args:
            query: Original user query
            k: Number of documents to retrieve per query

        Returns:
            Deduplicated list of documents
        """
        from collections import defaultdict

        # Expand query
        queries = self.query_expander.expand_query(query)

        log.info("Retrieving with multiple queries", num_queries=len(queries))

        # Retrieve for each query
        all_docs = []
        doc_scores = defaultdict(list)  # Track scores for deduplication

        for q in queries:
            docs_with_scores = self.vectorstore.similarity_search_with_score(q, k=k)

            for doc, score in docs_with_scores:
                # Use page + content hash as unique identifier
                doc_key = (doc.metadata.get('page', 0), hash(doc.page_content))
                doc_scores[doc_key].append((doc, score))

        # Deduplicate and average scores
        unique_docs = []
        for doc_key, doc_score_list in doc_scores.items():
            doc = doc_score_list[0][0]  # Take first occurrence of document
            avg_score = sum(score for _, score in doc_score_list) / len(doc_score_list)
            unique_docs.append((doc, avg_score))

        # Sort by average score (lower is better for FAISS)
        unique_docs.sort(key=lambda x: x[1])

        log.info("Multi-query retrieval complete",
                total_retrieved=sum(len(doc_scores[k]) for k in doc_scores),
                unique_docs=len(unique_docs))

        return unique_docs[:k * 2]  # Return up to 2x k documents
