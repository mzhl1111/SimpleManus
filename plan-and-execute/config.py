"""
Configuration module for the application.
"""
import os
from dotenv import load_dotenv
from typing import Optional, Dict, Any
import logging

load_dotenv()

# Debug mode (Set to False to hide log output, only show the final travel plan result)
DEBUG = False

# Logging level - set to ERROR to hide warnings
LOG_LEVEL = logging.ERROR if not DEBUG else logging.INFO

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

# Set to False to hide log output, only showing the final travel plan result
DEBUG = False 
