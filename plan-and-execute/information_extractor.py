"""
Information extraction module for travel planning assistant.
"""
import logging
import json
import re
from typing import Dict, Any

import config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import SecretStr # Import SecretStr for API key

# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InformationExtractor:
    """Extract structured travel information from user input"""
    
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
            api_key=SecretStr(self.api_key) if self.api_key else None,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Create extraction template using ChatPromptTemplate
        self.extraction_template = ChatPromptTemplate.from_messages([
            ("system", """You are a travel information extractor. 
Extract the following from the user's travel request:
- destination: Where they want to travel
- duration: How long they plan to stay
- date: When they plan to travel
- customization_hints: Any preferences about the style or nature of the trip

If any information is not provided, use "Not provided" as the value.
Respond with a JSON object containing only these fields:
{{
  "destination": "extracted destination or 'Not provided'",
  "duration": "extracted duration or 'Not provided'",
  "date": "extracted date or 'Not provided'",
  "customization_hints": "extracted customization hints or 'Not provided'"
}}"""),
            ("user", "{input}")
        ])
        
        # Create extraction chain
        self.extraction_chain = self.extraction_template | self.llm
    
    def extract_information(self, text: str) -> Dict[str, Any]:
        """Extract travel information from text using the extraction template."""
        logger.info(f"Extracting information from: {text}")
        
        try:
            # Call the template with the correct parameter name
            response = self.llm.invoke(
                self.extraction_template.format(input=text)
            )
            
            # Convert response to string
            response_text = str(response.content) if hasattr(response, "content") else str(response)
            logger.debug(f"Raw extraction response: {response_text}")
            
            # Try to extract JSON from the returned text
            info = self._extract_json(response_text)
            
            # If extraction fails, try using regular expressions
            if not info:
                logger.warning("Failed to extract JSON from response, trying regex")
                info = self._extract_with_regex(response_text)
            
            # Use default structure to ensure all necessary fields exist
            result = {
                "destination": "Not provided",
                "duration": "Not provided",
                "date": "Not provided",
                "customization_hints": "Not provided"
            }
            
            # Update fields returned from LLM
            if info:
                for field, value in info.items():
                    result[field] = value
            
            # Special handling: If no destination provided, try intelligent inference from text
            if result["destination"] == "Not provided" and text:
                # Check if there are keywords or patterns indicating destination
                if "sakura" in text.lower() or "cherry blossom" in text.lower():
                    result["destination"] = "Japan"
                    logger.info("Inferred destination 'Japan' from sakura reference")
                elif "northern lights" in text.lower() or "aurora" in text.lower():
                    result["destination"] = "Iceland"
                    logger.info("Inferred destination 'Iceland' from aurora reference")
                # Add more inference rules
            
            # Special handling: If no date provided but there are seasonal hints
            if result["date"] == "Not provided" and text:
                seasons = {
                    "winter": ["winter", "snow", "ski", "december", "january", "february"],
                    "spring": ["spring", "bloom", "blossom", "march", "april", "may", "sakura"],
                    "summer": ["summer", "beach", "hot", "june", "july", "august"],
                    "autumn": ["autumn", "fall", "foliage", "september", "october", "november"]
                }
                
                text_lower = text.lower()
                for season, keywords in seasons.items():
                    if any(keyword in text_lower for keyword in keywords):
                        result["date"] = season.capitalize()
                        logger.info(f"Inferred date '{season}' from seasonal keywords")
                        break
            
            logger.info(f"Extracted information: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting information: {str(e)}")
            return {
                "destination": "Not provided",
                "duration": "Not provided",
                "date": "Not provided",
                "customization_hints": "Not provided"
            }
    
    def _parse_response(self, response):
        """Parse LLM response into structured data
        
        Args:
            response: LLM response text
            
        Returns:
            Dictionary with extracted information
        """
        try:
            # First try to parse as JSON directly
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON in the response
            json_match = re.search(
                r'```(?:json)?\s*({.*?})\s*```', response, re.DOTALL
            )
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Fall back to regex extraction
            return self._extract_with_regex(response)
    
    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from text
        
        Args:
            text: Text containing JSON
            
        Returns:
            Extracted dictionary or None if extraction fails
        """
        try:
            # Try to directly parse the entire response
            data = json.loads(text)
            return data
        except json.JSONDecodeError:
            # Try to find JSON object
            import re
            
            # Try to find JSON object in the entire text
            json_pattern = r'(\{[\s\S]*\})'
            match = re.search(json_pattern, text)
            if match:
                try:
                    json_str = match.group(1)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            # Try to extract from markdown code block
            code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
            match = re.search(code_block_pattern, text)
            if match:
                try:
                    json_str = match.group(1)
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            return None
    
    def _extract_with_regex(self, text):
        """Extract information using regex patterns
        
        Args:
            text: Text to extract from
            
        Returns:
            Dictionary with extracted information
        """
        info = {
            "destination": "Not provided",
            "duration": "Not provided",
            "date": "Not provided",
            "customization_hints": "Not provided"
        }
        
        patterns = {
            "destination": r'"destination"\s*:\s*"([^"]+)"',
            "duration": r'"duration"\s*:\s*"([^"]+)"',
            "date": r'"date"\s*:\s*"([^"]+)"',
            "customization_hints": r'"customization_hints"\s*:\s*"([^"]+)"'
        }
        
        for field, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                if value.lower() not in ('none', 'not provided', 'null'):
                    info[field] = value
        
        return info
    
    def _get_default_info(self):
        """Get default empty information structure
        
        Returns:
            Dictionary with default values
        """
        return {
            "destination": None,
            "duration": None,
            "date": None,
            "customization_hints": None
        } 