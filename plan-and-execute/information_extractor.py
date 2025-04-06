"""
Information extraction module for general-purpose agent.
"""
import logging
import re
from typing import Dict, Any

import config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InformationExtractor:
    """Extract structured information from user input"""
    
    def __init__(self, api_key=None):
        """Initialize the information extractor
        
        Args:
            api_key: OpenAI API key (uses config if not provided)
        """
        self.api_key = api_key or config.OPENROUTER_API_KEY
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0.0,  # Lower temperature for more consistent extraction
            api_key=self.api_key,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Create extraction template using ChatPromptTemplate
        self.extraction_template = ChatPromptTemplate.from_messages([
            ("system", """You are an information extraction expert.
Extract clearly provided information from the user's input.

⚠️ Important! You must respond strictly in the following key-value format, one per line, without any additional text:

ENTITIES: entity1, entity2, ...
FOCUS_AREA: extracted focus or 'Not provided'
TIMEFRAME: extracted timeframe or 'Not provided'
CONSTRAINTS: extracted constraints or 'Not provided'
INTENT: extracted user intent or 'Not provided'

If any field is not explicitly provided in the input, use "Not provided".
Do not use colons (:) in your response except after field names.
Do not add any explanations, preambles or additional text."""),
            ("user", "{input}")
        ])
        
        # Create extraction chain
        self.extraction_chain = self.extraction_template | self.llm
    
    def extract_information(self, text: str) -> Dict[str, Any]:
        """Extract information from text using the extraction template."""
        logger.info(f"Extracting information from: {text}")
        
        try:
            # Call the template with the correct parameter name
            response = self.llm.invoke(
                self.extraction_template.format(input=text)
            )
            
            # Convert response to string
            response_text = str(response.content) if hasattr(response, "content") else str(response)
            logger.debug(f"Raw extraction response: {response_text}")
            
            # Try to extract key-value pairs from the returned text
            info = self._extract_key_value_pairs(response_text)
            
            # If extraction fails, use default values
            if not info:
                logger.warning("Failed to extract key-value pairs, using default values")
                info = self._get_default_info()
            
            logger.info(f"Extracted information: {info}")
            return info
            
        except Exception as e:
            logger.error(f"Error extracting information: {str(e)}")
            return self._get_default_info()
    
    def _extract_key_value_pairs(self, text: str) -> Dict[str, Any]:
        """Extract key-value pairs from response text.
        
        Args:
            text: Response text to parse
            
        Returns:
            Dictionary with extracted information
        """
        # Initialize result with default values
        result = self._get_default_info()
        
        # Clean text
        text = text.strip()
        
        # Define expected keys and their corresponding dictionary keys
        key_mapping = {
            "ENTITIES": "entities",
            "FOCUS_AREA": "focus_area",
            "TIMEFRAME": "timeframe",
            "CONSTRAINTS": "constraints",
            "INTENT": "intent"
        }
        
        # Process each line for key-value pairs
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Look for the first colon as the key-value separator
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip().upper()  # Normalize key to uppercase
                value = parts[1].strip()
                
                # Map the key if it matches one of our expected keys
                if key in key_mapping:
                    dict_key = key_mapping[key]
                    
                    # Handle special case for entities
                    if dict_key == "entities" and "," in value:
                        # Split by comma and strip whitespace
                        entities = [entity.strip() for entity in value.split(",")]
                        # Filter out empty entities
                        entities = [entity for entity in entities if entity]
                        if entities:
                            result[dict_key] = entities
                        else:
                            result[dict_key] = "Not provided"
                    elif value and value.lower() != "not provided":
                        result[dict_key] = value
        
        logger.debug(f"Extracted key-value pairs: {result}")
        return result
    
    def _get_default_info(self):
        """Get default information dictionary
        
        Returns:
            Default information dictionary
        """
        return {
            "entities": "Not provided",
            "focus_area": "Not provided", 
            "timeframe": "Not provided",
            "constraints": "Not provided",
            "intent": "Not provided"
        } 