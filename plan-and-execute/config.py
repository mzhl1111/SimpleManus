"""
Configuration module for the application.
"""
import os
from dotenv import load_dotenv
from typing import Dict, Any
import logging

load_dotenv()

# Debug mode (Set to True to show log output, including system execution process logs)
DEBUG = False

# Logging level - set to INFO to show informative messages, ERROR to hide warnings
LOG_LEVEL = logging.INFO if DEBUG else logging.ERROR

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Model settings
LLM_MODEL = os.getenv("LLM_MODEL")

# Langgraph settings
LANGGRAPH_CONFIG = {
    "recursion_limit": 100
}

def get_config() -> Dict[str, Any]:
    """Get configuration as dictionary"""
    return {
        "debug": DEBUG,
        "log_level": LOG_LEVEL,
        "llm_model": LLM_MODEL,
        "langgraph_config": LANGGRAPH_CONFIG
    } 
