"""
Search engine module for retrieving travel information.
"""
import logging
from typing import Dict, Any, List, Optional

from pydantic import SecretStr

import config
from langchain_openai import ChatOpenAI
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

# Configure logging
logging.basicConfig(
    level=logging.INFO if config.DEBUG else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SearchEngine:
    """Search engine for retrieving travel information"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the search engine
        
        Args:
            api_key: Tavily API key (falls back to config if not provided)
        """
        self.tavily_api_key_str = api_key or config.TAVILY_API_KEY
        self.openai_api_key_str = config.OPENROUTER_API_KEY

        # Raise error if Tavily key is missing
        if not self.tavily_api_key_str:
            raise ValueError("Tavily API key is required but not found.")
        # Raise error if OpenAI key is missing for consistency
        if not self.openai_api_key_str:
            raise ValueError("OpenAI API key is required but not found.")

        # Initialize search wrapper - Key is now guaranteed to be non-None
        self.search_wrapper = TavilySearchAPIWrapper(
            tavily_api_key=SecretStr(self.tavily_api_key_str)
        )

        # Initialize LLM - Key is now guaranteed to be non-None
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0.1,
            api_key=SecretStr(self.openai_api_key_str),
            base_url="https://openrouter.ai/api/v1"
        )
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Perform a search with the given query.
        The number of results is determined by the Tavily wrapper 
        initialization.

        Args:
            query: Search query

        Returns:
            List of search results
        """
        logger.info(f"Searching for: {query}")
        
        try:
            # Perform search
            results = self.search_wrapper.results(
                query
            )
            
            logger.info(f"Found {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []
