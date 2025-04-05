"""
Planner and executor components for the travel planning assistant.
Follows the Plan-and-Execute pattern.
"""
import logging
from typing import Dict, Any, List, Literal, Optional, Union
from pydantic import BaseModel, Field, ValidationError, SecretStr

import config
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from prompt_templates import PLANNING_PROMPT
from travel_tools import TRAVEL_TOOLS
import json
import re

# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PlanStep(BaseModel):
    """A single step in the travel plan"""
    step_id: int = Field(description="Unique identifier for this step")
    description: str = Field(description="Description of what this step should accomplish")
    tool: str = Field(description="Tool to use for this step (one of the available tools)")
    parameters: Dict[str, Any] = Field(description="Parameters to pass to the tool")


class TravelPlan(BaseModel):
    """A structured travel plan consisting of steps"""
    steps: List[PlanStep] = Field(description="List of steps to execute in order")


class Planner:
    """Generates a structured plan for travel planning"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key_str = api_key or config.OPENROUTER_API_KEY
        if not self.api_key_str:
            raise ValueError("Planner requires OpenAI API Key")

        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0.2,
            api_key=SecretStr(self.api_key_str),
            base_url="https://openrouter.ai/api/v1"
        )

        self.planning_template = PromptTemplate(
            template=PLANNING_PROMPT,
            input_variables=["user_query"]
        )


    def create_plan(self, user_query: str) -> List[Dict[str, Any]]:
        """Create a plan for the travel request."""
        logger.info(f"Creating plan for: {user_query}")
        
        # Check if user_query is None or empty
        if user_query is None or user_query.strip() == "":
            logger.warning("Empty user query received, using default plan")
            return self._default_plan(user_query)
            
        try:
            # Use manual JSON parsing instead of structured output
            response = self.llm.invoke(
                self.planning_template.format(
                    user_query=user_query
                )
            )
            
            # Extract JSON string - more robust parsing method
            logger.info("Received planning response, parsing JSON")
            response_text = response.content
            
            # Try multiple ways to parse JSON
            try:
                # Try to directly parse the entire response
                plan_data = json.loads(response_text)
                if "steps" in plan_data:
                    logger.info(f"Successfully parsed plan with {len(plan_data['steps'])} steps")
                    return plan_data["steps"]
            except json.JSONDecodeError:
                # Try to find JSON object boundaries
                logger.info("Direct JSON parse failed, trying to extract JSON object")
                import re
                
                # Try to extract JSON object from text
                json_pattern = r'(\{[\s\S]*\})'
                match = re.search(json_pattern, response_text)
                if match:
                    try:
                        json_str = match.group(1)
                        plan_data = json.loads(json_str)
                        if "steps" in plan_data:
                            logger.info(f"Successfully extracted and parsed plan with {len(plan_data['steps'])} steps")
                            return plan_data["steps"]
                    except json.JSONDecodeError:
                        logger.warning("Extracted JSON is still invalid")
                
                # Try to extract from markdown code block
                code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
                match = re.search(code_block_pattern, response_text)
                if match:
                    try:
                        json_str = match.group(1)
                        plan_data = json.loads(json_str)
                        if "steps" in plan_data:
                            logger.info(f"Successfully extracted JSON from code block with {len(plan_data['steps'])} steps")
                            return plan_data["steps"]
                    except json.JSONDecodeError:
                        logger.warning("Code block JSON is invalid")
            
            # If all parsing methods fail, use default plan
            logger.error(f"All JSON parsing attempts failed. Response: {response_text[:100]}...")
            return self._default_plan(user_query)
                
        except Exception as e:
            logger.error(f"Error creating plan: {str(e)}. Falling back to default plan.")
            return self._default_plan(user_query)
            
    def _default_plan(self, user_query: str) -> List[Dict[str, Any]]:
        """Create a default plan when automatic planning fails.
        
        Args:
            user_query: The original user query
            
        Returns:
            A list of default plan steps
        """
        logger.info("Creating default travel plan")
        
        # A simple sequential plan with the essential steps
        return [
            {
                "step_id": 1,
                "description": "Extract basic travel information",
                "tool": "extract_travel_info",
                "tool_input": {"user_input": user_query}
            },
            {
                "step_id": 2,
                "description": "Search for attractions",
                "tool": "search_attractions",
                "tool_input": {
                    "destination": "{{travel_info.destination}}",
                    "date": "{{travel_info.date}}"
                }
            },
            {
                "step_id": 3,
                "description": "Search for local tips",
                "tool": "search_local_tips",
                "tool_input": {
                    "destination": "{{travel_info.destination}}",
                    "date": "{{travel_info.date}}"
                }
            },
            {
                "step_id": 4,
                "description": "Generate daily itinerary",
                "tool": "generate_daily_itinerary",
                "tool_input": {
                    "destination": "{{travel_info.destination}}",
                    "duration": "{{travel_info.duration}}",
                    "date": "{{travel_info.date}}",
                    "customization_hints": "{{travel_info.customization_hints}}",
                    "attractions": "{{past_steps.search_attractions}}",
                    "activities": "{{past_steps.search_local_tips}}"
                }
            },
            {
                "step_id": 5,
                "description": "Estimate travel budget",
                "tool": "estimate_budget",
                "tool_input": {
                    "destination": "{{travel_info.destination}}",
                    "duration": "{{travel_info.duration}}",
                    "date": "{{travel_info.date}}"
                }
            }
        ]


class Executor:
    """Executes steps in a travel plan"""

    def __init__(self, api_key: Optional[str] = None):
        self.tools = TRAVEL_TOOLS

    def execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        try:
            tool_name = step.get("tool")
            tool_input = step.get("tool_input", {})
            step_id = step.get("step_id", 0)

            if tool_name not in TRAVEL_TOOLS:
                return {
                    "step_id": step_id,
                    "tool": tool_name,
                    "error": f"Unknown tool: {tool_name}"
                }

            logger.info(f"Executing {tool_name} with input: {tool_input}")
            result = TRAVEL_TOOLS[tool_name](**tool_input)

            return {
                "step_id": step_id,
                "tool": tool_name,
                "result": result
            }
        except Exception as e:
            logger.error(f"Error executing {step.get('tool', 'unknown')}: {str(e)}")
            return {
                "step_id": step.get("step_id", 0),
                "tool": step.get("tool", "unknown"),
                "error": str(e)
            }


class Replanner:
    """Re-evaluates the plan based on execution results"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key_str = api_key or config.OPENROUTER_API_KEY
        if not self.api_key_str:
            raise ValueError("Replanner requires OpenAI API Key")

        self.llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=0.2,
            api_key=SecretStr(self.api_key_str),
            base_url="https://openrouter.ai/api/v1"
        )

    def replan(
        self,
        user_input: str,
        original_plan: List[Dict[str, Any]],
        executed_steps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Re-evaluate the plan based on execution results

        Args:
            user_input: Original user request
            original_plan: The plan steps that were created
            executed_steps: Results from executed steps so far

        Returns:
            Dictionary with action to take and optional new plan or response
        """
        logger.info("Re-evaluating plan based on execution results")
        
        # Format the plans and results for the prompt
        original_plan_str = "\n".join([
            f"{step.get('step_id', i+1)}. {step.get('description', 'Step')} (using {step.get('tool', 'unknown')})"
            for i, step in enumerate(original_plan)
        ])
        
        executed_steps_str = "\n".join([
            f"Step {result.get('step_id')}: " +
            (f"ERROR: {result.get('error')}" if 'error' in result else
             # Limit result string length
             f"Success - {str(result.get('result'))[:100]}...")
            for result in executed_steps
        ])
        
        replan_template = PromptTemplate.from_template("""
        You are a travel planning assistant. Review the original plan and execution
        results to decide what to do next:

        USER REQUEST: {user_input}

        ORIGINAL PLAN:
        {original_plan}

        EXECUTION RESULTS:
        {executed_steps}

        Based on these results, choose ONE of the following actions:
        1. CONTINUE with the current plan
        2. UPDATE the plan (provide a new plan)
        3. COMPLETE the planning (provide a final response)

        IMPORTANT: You should only choose COMPLETE when either all steps have been executed
        or you have at least executed the travel information extraction, attractions search,
        and daily itinerary generation. Budget estimation is especially important and should
        not be skipped.

        Respond with structured JSON data:
        {{
            "action": "continue"|"update"|"complete",
            "plan": [ /* New plan steps if action is "update" */ ],
            "response": "Final response if action is complete"
        }}

        Analyze the executed steps carefully. If a step failed (e.g., search error),
        consider if replanning is needed or if you can complete with available info.
        """)
        
        try:
            # Use manual JSON parsing instead of structured output
            response = self.llm.invoke(
                replan_template.format(
                    user_input=user_input,
                    original_plan=original_plan_str,
                    executed_steps=executed_steps_str
                )
            )
            
            # Extract JSON string - more robust parsing method
            logger.info("Received replanning response, parsing JSON")
            response_text = response.content if hasattr(response, "content") else str(response)
            
            # Try multiple ways to parse JSON
            output = None
            
            # Try to parse the entire response directly
            try:
                logger.info("Attempting direct JSON parse")
                output = json.loads(response_text)
            except json.JSONDecodeError:
                logger.info("Direct JSON parse failed, trying to extract JSON")
                import re
                
                # Try to extract JSON object from text
                json_pattern = r'(\{[\s\S]*\})'
                match = re.search(json_pattern, response_text)
                if match:
                    try:
                        json_str = match.group(1)
                        logger.info(f"Extracted JSON: {json_str[:50]}...")
                        output = json.loads(json_str)
                    except json.JSONDecodeError:
                        logger.warning("Extracted JSON is still invalid")
                
                # Try to extract from markdown code block
                if not output:
                    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
                    match = re.search(code_block_pattern, response_text)
                    if match:
                        try:
                            json_str = match.group(1)
                            logger.info(f"Extracted code block: {json_str[:50]}...")
                            output = json.loads(json_str)
                        except json.JSONDecodeError:
                            logger.warning("Code block JSON is invalid")
            
            # If all parsing methods fail, use default continue action
            if not output:
                logger.error(f"All JSON parsing attempts failed during replanning")
                return {"action": "continue"}
            
            # Validate and process the parsed output
            if "action" not in output:
                logger.error("Parsed JSON missing 'action' field")
                return {"action": "continue"}
            
            action = output.get("action", "").lower()
            
            if action == "update":
                plan = output.get("plan")
                
                # Handle special case where plan is a list of strings
                if isinstance(plan, list):
                    # Check if first element is a string, if so attempt to convert the entire list
                    if len(plan) > 0 and isinstance(plan[0], str):
                        logger.warning("Plan is a list of strings, attempting to convert to proper format")
                        try:
                            # Try to parse each string element and create proper steps
                            converted_plan = []
                            for i, step_str in enumerate(plan):
                                # Remove possible numbering and prefixes
                                clean_str = re.sub(r'^[0-9]+[.:]?\s*', '', step_str)
                                
                                # Extract tool name and description
                                tool_match = re.search(r'(?:using|with)\s+([a-z_]+)', step_str, re.IGNORECASE)
                                tool_name = tool_match.group(1) if tool_match else "unknown_tool"
                                
                                # Create basic step
                                converted_step = {
                                    "step_id": i + 1,
                                    "description": clean_str,
                                    "tool": tool_name,
                                    "tool_input": {}
                                }
                                
                                # Add default parameters based on tool type
                                if tool_name == "extract_travel_info":
                                    converted_step["tool_input"] = {"user_input": "{{travel_info.user_query}}"}
                                elif tool_name in ["search_attractions", "search_local_tips", "search_accommodations"]:
                                    converted_step["tool_input"] = {
                                        "destination": "{{travel_info.destination}}",
                                        "date": "{{travel_info.date}}"
                                    }
                                elif tool_name == "generate_daily_itinerary":
                                    converted_step["tool_input"] = {
                                        "destination": "{{travel_info.destination}}",
                                        "duration": "{{travel_info.duration}}",
                                        "date": "{{travel_info.date}}",
                                        "customization_hints": "{{travel_info.customization_hints}}",
                                        "attractions": "{{past_steps.search_attractions}}",
                                        "activities": "{{past_steps.search_local_tips}}"
                                    }
                                elif tool_name == "estimate_budget":
                                    converted_step["tool_input"] = {
                                        "destination": "{{travel_info.destination}}",
                                        "duration": "{{travel_info.duration}}",
                                        "date": "{{travel_info.date}}"
                                    }
                                
                                converted_plan.append(converted_step)
                                
                            if converted_plan:
                                logger.info(f"Successfully converted string list to {len(converted_plan)} plan steps")
                                plan = converted_plan
                        except Exception as e:
                            logger.error(f"Error converting string list to plan: {str(e)}")
                            # Fall back to default plan
                            plan = None
                
                if not plan or not isinstance(plan, list) or len(plan) == 0:
                    logger.error("Action is 'update' but plan is missing, invalid, or empty")
                    return {"action": "continue"}
                
                # Validate that each step contains necessary fields
                for i, step in enumerate(plan):
                    if not isinstance(step, dict):
                        logger.warning(f"Step {i+1} is not a dictionary, skipping validation")
                        continue
                        
                    # Ensure each step has necessary fields
                    if "tool" not in step:
                        logger.warning(f"Step {i+1} missing 'tool', adding default")
                        step["tool"] = "unknown_tool"
                    
                    if "tool_input" not in step or not isinstance(step["tool_input"], dict):
                        logger.warning(f"Step {i+1} missing 'tool_input', adding empty dict")
                        step["tool_input"] = {}
                    
                    if "step_id" not in step:
                        step["step_id"] = i + 1
                        
                    if "description" not in step:
                        step["description"] = f"Step using {step['tool']}"
                
                logger.info(f"Replanner decided to update the plan with {len(plan)} steps")
                return {"action": "update", "plan": plan}
            elif action == "complete":
                response_text = output.get("response")
                if not response_text or not isinstance(response_text, str):
                    logger.error("Action is 'complete' but response is missing or not a string")
                    return {"action": "continue"}
                logger.info("Replanner decided to complete with final response")
                return {"action": "complete", "response": response_text}
            else:  # continue or any other value
                logger.info("Replanner decided to continue with current plan")
                return {"action": "continue"}
                
        except Exception as e:
            logger.error(f"Error during replanning: {str(e)}. Defaulting to continue.")
            return {"action": "continue"}
