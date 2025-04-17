"""
Tool registration and initialization for the Plan-and-Execute agent.
"""
import logging
import os
from typing import List, Optional
from langchain_core.tools import Tool
from langchain_core.language_models import BaseLanguageModel
from langchain_openai import ChatOpenAI

from tools.planning_tool import get_planning_tools
from dotenv import load_dotenv
from pydantic import SecretStr

logger = logging.getLogger(__name__)

# Define global variable for browser tool instance
_browser_tool_instance = None


def initialize_tools(config: dict) -> tuple[list, ChatOpenAI]:
    # Directly get API key from environment variables as a fallback
    load_dotenv()
    
    # Get API key, prioritizing passed config, then environment variables
    api_key = (
        config.get("llm_config", {}).get("api_key", "") or 
        os.getenv("OPENROUTER_API_KEY", "")
    )
    model_name = (
        config.get("llm_config", {}).get("model_name", "") or 
        os.getenv("LLM_MODEL", "openai/gpt-3.5-turbo")
    )
    api_base = (
        config.get("llm_config", {}).get("api_base", "") or 
        "https://openrouter.ai/api/v1"
    )
    
    if not api_key:
        raise ValueError(
            "API key not found. Please set OPENROUTER_API_KEY in your "
            "environment variables or config."
        )
    
    print(f"Using model: {model_name}")
    print(f"API base URL: {api_base}")
    print(f"API key available: {'Yes' if api_key else 'No'}")
    
    # Convert API key to SecretStr type
    api_key_secret = (
        SecretStr(api_key) if isinstance(api_key, str) else api_key
    )
    
    llm = ChatOpenAI(
        temperature=0,
        model=model_name,
        api_key=api_key_secret,
        base_url=api_base,
    )

    # Create browser tool instance - using global singleton pattern
    global _browser_tool_instance
    
    try:
        # Initialize browser tool, using singleton pattern
        # to avoid multiple creations
        if _browser_tool_instance is None:
            from tools.browser_use_tool import BrowserUseTool
            _browser_tool_instance = BrowserUseTool(llm=llm)
            print("Created new browser tool instance")
            
            # Register cleanup function on program exit
            import atexit
            import asyncio
            
            def cleanup_browser():
                print("Cleaning up browser resources...")
                if _browser_tool_instance is not None:
                    try:
                        # Async cleanup requires special handling
                        loop = asyncio.new_event_loop()
                        loop.run_until_complete(
                            _browser_tool_instance.cleanup()
                        )
                        loop.close()
                        print("Browser resources cleaned up successfully")
                    except Exception as e:
                        print(f"Error cleaning up browser resources: {e}")
            
            atexit.register(cleanup_browser)
        
        # Get the tool function set from the tools.browser_use_tool module
        from tools.browser_use_tool import get_browser_use_tools
        browser_tools = get_browser_use_tools(_browser_tool_instance)
        
    except Exception as e:
        logger.error(f"Failed to initialize browser tools: {e}", exc_info=True)
        print(f"**** [DEBUG] ERROR initializing browser tools: {e} ****")
        browser_tools = []

    # Combine all tools
    all_tools = browser_tools

    return all_tools, llm


def initialize_tools_old(llm: Optional[BaseLanguageModel] = None) -> List[Tool]:
    """
    Initialize all tools and return the tool list.

    Args:
        llm: Language model instance, for tools that require an LLM.

    Returns:
        List containing all tools.
    """
    # Use print for earliest possible debug output
    print(
        "++++ [DEBUG] Entering initialize_tools_old ++++"
    )
    logger.info("Initializing agent tools (old)...")
    all_tools_list: List[Tool] = []

    try:
        print("++++ [DEBUG] Calling get_planning_tools... ++++")
        planning_tools = get_planning_tools()
        print(
            "++++ [DEBUG] Returned from get_planning_tools. "
            "Extending list... ++++"
        )
        all_tools_list.extend(planning_tools)
        print("++++ [DEBUG] Planning tools list extended. ++++")
        logger.info(f"Initialized {len(planning_tools)} planning tools.")
    except Exception as e:
        print(
            "**** [DEBUG] ERROR during get_planning_tools or extend: "
            f"{e} ****"
        )
        logger.error(
            f"Failed to initialize planning tools: {e}", exc_info=True
        )

    # Log initialized tools
    tool_names = [tool.name for tool in all_tools_list]
    logger.info(
        f"Total tools initialized: {len(all_tools_list)}: {tool_names}"
    )

    # Exit log
    print("---- [DEBUG] Exiting initialize_tools_old ----")
    return all_tools_list 