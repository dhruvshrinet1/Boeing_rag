from typing import List
from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from prompt.prompt_library import PROMPT_REGISTRY


class QueryExpander:
    def __init__(self):
        self.llm = ModelLoader().load_llm()
        self.expansion_prompt = PROMPT_REGISTRY["query_expansion"]
        log.info("QueryExpander initialized")

    def expand_query(self, query: str) -> List[str]:
        """Expand query with aviation terms using LLM"""
        try:
            response = (self.expansion_prompt | self.llm).invoke({"query": query})
            expanded_text = response.content if hasattr(response, 'content') else str(response)

            # split response into individual queries
            expanded_queries = [q.strip() for q in expanded_text.split('\n') if q.strip()]
            all_queries = [query] + expanded_queries[:3]  # original + top 3 expansions

            log.info("Query expanded", num_expansions=len(all_queries) - 1)
            return all_queries
        except Exception as e:
            # fallback to original if expansion fails
            log.warning("Query expansion failed", error=str(e))
            return [query]
