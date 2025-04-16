"""
Tool registration and initialization for the Plan-and-Execute agent.
"""
import logging
from typing import List, Optional
from langchain_core.tools import Tool
from langchain_core.language_models import BaseLanguageModel

# Import tool modules
from tools.browser_use_tool import get_browser_use_tools
from tools.planning_tool import get_planning_tools

logger = logging.getLogger(__name__)


def initialize_tools(llm: Optional[BaseLanguageModel] = None) -> List[Tool]:
    """
    Initialize all tools and return the tool list.

    Args:
        llm: Language model instance, for tools that require an LLM.

    Returns:
        List containing all tools.
    """
    logger.info("Initializing agent tools...")
    all_tools_list: List[Tool] = []

    # 1. Browser tools
    try:
        browser_tools = get_browser_use_tools(llm)
        all_tools_list.extend(browser_tools)
        logger.info(f"Initialized {len(browser_tools)} browser tools.")
    except Exception as e:
        logger.error(f"Failed to initialize browser tools: {e}", exc_info=True)

    # 2. Planning tools
    try:
        planning_tools = get_planning_tools()
        all_tools_list.extend(planning_tools)
        logger.info(f"Initialized {len(planning_tools)} planning tools.")
    except Exception as e:
        logger.error(f"Failed to initialize planning tools: {e}", exc_info=True)
        
    # Log initialized tools
    tool_names = [tool.name for tool in all_tools_list]
    logger.info(f"Total tools initialized: {len(all_tools_list)}: "
                f"{tool_names}")
    
    return all_tools_list 