import logging
from typing import Dict, Any, List, Optional

from langchain_openai import ChatOpenAI

from tools.tools_manager import ToolsManager
from error_handler import ErrorHandler

logger = logging.getLogger(__name__)

class Executor:
    """Executor to execute tasks"""

    def __init__(self, tools_config: dict, llm: ChatOpenAI = None):
        """Initialize Executor with tools and LLM"""
        self.tools_manager = ToolsManager(tools_config)
        self.llm = llm or ChatOpenAI(model="gpt-3.5-turbo-0125")
        self.tools = self.tools_manager.get_tools()
        self.error_handler = ErrorHandler()
        self.system_prompt = ("You are a helpful assistant that executes "
                             "tools according to the plan.")
        
    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the next step based on the current state"""
        # Extract state variables
        plan = state.get("plan", [])
        steps_taken = state.get("steps_taken", [])
        current_step_index = state.get("current_step_index", 0)
        state["tool_executions"] = state.get("tool_executions", 0) + 1
        
        # Add stronger safeguard against too many executions
        if state["tool_executions"] > 10:  # Reduced from 15 to 10
            logger.warning("Too many tool executions (>10), forcing termination")
            state["response"] = ("FINAL ANSWER: I've attempted multiple actions "
                               "without resolving your request. Here's what I know: "
                               "Unable to complete the task due to technical limitations.")
            return state
            
        logger.info(f"Current step index: {current_step_index} of {len(plan)}")
        
        # Check if all steps are executed
        if current_step_index >= len(plan):
            logger.info("All steps executed. Generating final response.")
            response = self._generate_final_response(state)
            state["response"] = response
            return state
        
        # Get current step
        current_step = plan[current_step_index]
        logger.info(f"Executing step: {current_step}")
        
        # Execute step
        result = self._execute_step(current_step, state)
        
        # Update steps taken
        steps_taken.append({
            "step": current_step,
            "result": result
        })
        
        # Update state
        state["steps_taken"] = steps_taken
        state["current_step_index"] = current_step_index + 1
        state["response"] = result
        
        return state 

    # Original execute_plan method removed to avoid duplication 