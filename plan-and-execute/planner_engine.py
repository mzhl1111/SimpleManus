"""
Planner and executor components for the general-purpose agent.
"""
import json
import logging
from typing import List, Dict, Any

import config
from langchain_openai import ChatOpenAI
from agent_tools import AGENT_TOOLS
from prompt_templates import PLANNING_PROMPT

# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Planner:
    """Component for creating plans to fulfill user requests"""
    
    def __init__(self):
        """Initialize the planner with language model"""
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
    
    def create_plan(self, user_query: str) -> List[Dict[str, Any]]:
        """Generate a plan for fulfilling the user's request
        
        Args:
            user_query: User's original request
            
        Returns:
            List of plan steps, each a dictionary with step details
        """
        logger.info(f"Creating plan for: {user_query}")
        
        # Create prompt with the user query
        formatted_prompt = PLANNING_PROMPT.format(user_query=user_query)
        
        # Generate plan using language model
        response = self.llm.invoke(formatted_prompt)
        
        try:
            # Extract the plan steps
            response_text = response.content
            
            # Log raw response for debugging
            logger.info(f"Raw LLM planning response: {response_text[:200]}...")
            
            # Handle different possible formats in the model's output
            # Clean the response to extract just the JSON part
            json_content = self._extract_json_from_text(response_text)
            
            # Parse the JSON
            parsed_response = json.loads(json_content)
            
            # Extract the steps
            steps = []
            if "steps" in parsed_response:
                steps = parsed_response["steps"]
            else:
                # Assume the response itself is an array of steps
                steps = parsed_response
            
            # Validate the steps format
            validated_steps = self._validate_steps(steps)
            
            # Log the generated plan
            logger.info("Generated plan:")
            for step in validated_steps:
                logger.info(f"Step {step.get('step_id')}: {step.get('description')} - Using tool: {step.get('tool')}")
            
            return validated_steps
        except Exception as e:
            logger.error(f"Error parsing plan: {str(e)}")
            # Create a fallback minimal plan
            default_plan = self._create_default_plan(user_query)
            logger.info("Using default plan due to error:")
            for step in default_plan:
                logger.info(f"Step {step.get('step_id')}: {step.get('description')} - Using tool: {step.get('tool')}")
            return default_plan
    
    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON object from text string that may contain other content"""
        # Look for JSON object start and end
        start_idx = text.find('{')
        if start_idx == -1:
            # Try looking for array
            start_idx = text.find('[')
            if start_idx == -1:
                raise ValueError("No JSON object or array found in response")
        
        # Count braces to find the matching end
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i in range(start_idx, len(text)):
            char = text[i]
            
            # Handle string boundaries
            if char == '"' and not escape_next:
                in_string = not in_string
            
            # Handle escape sequences
            if char == '\\' and not escape_next:
                escape_next = True
            else:
                escape_next = False
            
            # Count braces outside of strings
            if not in_string:
                if char == '{' or char == '[':
                    brace_count += 1
                elif char == '}' or char == ']':
                    brace_count -= 1
                    
                    # If we've found the matching end brace
                    if brace_count == 0:
                        # Extract the JSON content
                        json_content = text[start_idx:i+1]
                        return json_content
        
        raise ValueError("Invalid JSON: unclosed braces or brackets")
    
    def _validate_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean up the plan steps"""
        valid_steps = []
        
        for step in steps:
            # Check required fields
            if not isinstance(step, dict):
                logger.warning(f"Step is not a dictionary, skipping: {step}")
                continue
                
            if "step_id" not in step:
                step["step_id"] = len(valid_steps) + 1
                
            if "description" not in step:
                step["description"] = f"Step {step['step_id']}"
                
            if "tool" not in step:
                logger.warning(f"Step missing tool field, skipping: {step}")
                continue
                
            # Verify tool exists
            tool_name = step["tool"]
            if tool_name not in AGENT_TOOLS:
                logger.warning(f"Unknown tool '{tool_name}', skipping step: {step}")
                continue
            
            # Ensure tool_input exists and is a dictionary
            if "tool_input" not in step or not isinstance(step["tool_input"], dict):
                step["tool_input"] = {}
            
            valid_steps.append(step)
        
        # If no valid steps, create a default plan
        if not valid_steps:
            logger.warning("No valid steps in plan, creating default plan")
            return self._create_default_plan("Help me answer my question")
        
        return valid_steps
    
    def _create_default_plan(self, user_query: str) -> List[Dict[str, Any]]:
        """Create a simple default plan for when planning fails"""
        return [
            {
                "step_id": 1,
                "description": "Extract information from user input",
                "tool": "extract_information",
                "tool_input": {"user_input": user_query}
            },
            {
                "step_id": 2,
                "description": "Search for relevant information",
                "tool": "search_web",
                "tool_input": {"query": user_query}
            },
            {
                "step_id": 3,
                "description": "Generate comprehensive answer",
                "tool": "generate_answer",
                "tool_input": {
                    "question": user_query,
                    "search_results": "{{past_steps.search_web}}",
                    "use_search": True
                }
            }
        ]


class Executor:
    """Executes individual steps in the plan"""
    
    def __init__(self):
        """Initialize the executor"""
        pass
    
    def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step in the plan
        
        Args:
            step: Dictionary containing step details
            
        Returns:
            Result of the step execution
        """
        # Extract step details
        tool_name = step.get("tool", "")
        tool_input = step.get("tool_input", {})
        description = step.get("description", "")
        
        logger.info(f"Executing: {description} using tool '{tool_name}'")
        
        # Check if tool exists
        if tool_name not in AGENT_TOOLS:
            error_msg = f"Unknown tool: {tool_name}"
            logger.error(error_msg)
            return {"error": error_msg}
        
        # Get the tool function
        tool_func = AGENT_TOOLS[tool_name]
        
        try:
            # Execute the tool
            result = tool_func(**tool_input)
            
            logger.info(f"Step execution completed: {description}")
            return result
        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg}


class Replanner:
    """Handles replanning when the original plan fails"""
    
    def __init__(self):
        """Initialize the replanner"""
        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        self.planner = Planner()
    
    def create_new_plan(self, 
                       user_query: str, 
                       extracted_info: Dict[str, Any],
                       past_steps: List[Dict[str, Any]],
                       error_message: str,
                       current_step_index: int) -> Dict[str, Any]:
        """Create a new plan when the current plan fails
        
        Args:
            user_query: User's original request
            extracted_info: Information extracted from user's request
            past_steps: Results of steps executed so far
            error_message: Error message from the failed step
            current_step_index: Index of the failed step
            
        Returns:
            Dictionary containing the new plan
        """
        # Analyze the error and create a replan prompt
        prompt = f"""You are an AI assistant tasked with creating a new plan to fulfill a user's request
after the original plan encountered an error.

USER REQUEST: {user_query}

EXTRACTED INFORMATION:
{json.dumps(extracted_info, indent=2)}

STEPS EXECUTED SO FAR:
{json.dumps(past_steps, indent=2)}

ERROR ENCOUNTERED:
{error_message}

Create a new plan that avoids the error and fulfills the user's request.
The plan should be a sequence of steps that use these available tools:
- extract_information: Extract information from text
- search_web: Search the web for information
- generate_answer: Generate a comprehensive answer to a question
- analyze_with_llm: Analyze text using a language model with specific instructions
- summarize_information: Create a coherent summary from multiple texts
- categorize_user_request: Categorize the type of request the user is making

Return a JSON object with the key "plan" containing an array of steps. Each step should have:
1. step_id: A unique number
2. description: A clear description of the step
3. tool: The name of the tool to use
4. tool_input: Parameters for the tool

Format:
{{
  "plan": [
    {{ "step_id": 1, "description": "...", "tool": "...", "tool_input": {{ ... }} }},
    ...
  ]
}}
"""
        
        try:
            # Generate new plan
            response = self.llm.invoke(prompt)
            response_text = response.content
            
            # Parse the JSON
            json_content = self.planner._extract_json_from_text(response_text)
            parsed_response = json.loads(json_content)
            
            # Extract the plan
            if "plan" in parsed_response:
                new_plan = parsed_response["plan"]
                logger.info(f"Created new plan with {len(new_plan)} steps")
                return {"plan": self.planner._validate_steps(new_plan)}
            else:
                # Create a simple plan that completes the task
                logger.warning("Replan didn't return a 'plan' key, using default")
                return {"plan": self.planner._create_default_plan(user_query)}
            
        except Exception as e:
            logger.error(f"Error creating new plan: {str(e)}")
            return {"plan": self.planner._create_default_plan(user_query)}
