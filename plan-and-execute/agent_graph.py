"""
General purpose assistant using LangGraph for workflow management.
Implements the Plan-and-Execute pattern.
"""
import logging
import operator
from typing import Dict, Any, TypedDict, List, Annotated

from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from pydantic import BaseModel

from planner_engine import Planner, Executor, Replanner
from agent_tools import AGENT_TOOLS

import config

# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Define application state type
class AgentState(TypedDict):
    """State definition for the general purpose assistant"""
    input: str                                      # User's original request
    plan: List[Dict[str, Any]]                      # Current plan steps
    past_steps: Annotated[List[Dict[str, Any]], operator.add]  # Results of executed steps
    current_step_index: int                         # Index of current step
    extracted_info: Dict[str, Any]                  # Extracted information
    response: str                                   # Final response to user
    replan_count: int                               # Count of replanning attempts


class AgentGraph:
    """General purpose assistant using LangGraph with Plan-and-Execute pattern"""
    
    def __init__(self):
        """Initialize the assistant"""
        # Initialize components
        self.planner = Planner()
        self.executor = Executor()
        self.replanner = Replanner()
        
        # Create the graph
        self.graph = self._create_graph()
        
        # Track string plan conversion attempts
        self.string_plan_count = 0
    
    def _create_graph(self) -> CompiledGraph:
        """Create the workflow graph"""
        # Create workflow graph
        workflow = StateGraph(AgentState)
        
        # Add the main nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("replanner", self._replanner_node)
        
        # Set entry point
        workflow.set_entry_point("planner")
        
        # Add edges for the main workflow
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", "replanner")
        
        # Add conditional edges from replanner
        workflow.add_conditional_edges(
            "replanner",
            self._should_continue_or_end,
            {
                "continue": "executor",
                "end": END
            }
        )
        
        # Compile the graph
        return workflow.compile()
    
    def _planner_node(self, state: AgentState) -> Dict[str, Any]:
        """Generate a plan based on user input"""
        try:
            # Extract the user input
            user_input = state["input"]
            logger.info(f"Creating plan for: {user_input}")
            
            # Generate a plan
            plan = self.planner.create_plan(user_input)
            
            # Return the updated state
            return {
                "plan": plan,
                "current_step_index": 0,
                "past_steps": [],
                "replan_count": 0  # Initialize replan counter
            }
        except Exception as e:
            logger.error(f"Error in planner node: {str(e)}")
            # Create a minimal default plan
            return {
                "plan": [
                    {
                        "step_id": 1,
                        "description": "Extract information from user input",
                        "tool": "extract_information",
                        "tool_input": {"user_input": state["input"]}
                    }
                ],
                "current_step_index": 0,
                "past_steps": [],
                "replan_count": 0  # Initialize replan counter
            }
    
    def _executor_node(self, state: AgentState) -> Dict[str, Any]:
        """Execute the current step in the plan"""
        try:
            # Get the current step
            current_index = state["current_step_index"]
            plan = state["plan"]
            
            if current_index >= len(plan):
                logger.warning("No more steps to execute")
                return {}
            
            current_step = plan[current_index]
            
            # Ensure current_step is a dictionary
            if not isinstance(current_step, dict):
                logger.error(f"Step {current_index+1} is not a dictionary: {current_step}")
                return {
                    "past_steps": [{
                        "error": f"Invalid step format: {type(current_step).__name__}",
                        "step_id": current_index + 1,
                        "tool": "unknown"
                    }],
                    "current_step_index": current_index + 1
                }
            
            step_id = current_step.get('step_id', current_index+1)
            description = current_step.get('description', 'Execute step')
            tool_name = current_step.get("tool", "")
            
            logger.info(f"==== EXECUTING STEP {step_id}: {description} ====")
            logger.info(f"Tool: {tool_name}")
            
            if not tool_name:
                logger.error("Missing tool name in step")
                return {
                    "past_steps": [{
                        "error": "Missing tool name in step",
                        "step_id": current_index + 1,
                        "tool": "unknown"
                    }],
                    "current_step_index": current_index + 1
                }
                
            # Process parameters for all tools
            processed_params = {}
            extracted_info = state.get("extracted_info", {})
            past_steps = state.get("past_steps", [])
            
            # Create mapping of past_steps results
            past_results = {}
            for step in past_steps:
                if "tool" in step and "result" in step:
                    past_results[step["tool"]] = step["result"]
            
            tool_input = current_step.get("tool_input", {})
            # Ensure tool_input is a dictionary
            if not isinstance(tool_input, dict):
                logger.warning(f"tool_input is not a dictionary, converting: {tool_input}")
                tool_input = {}
            
            logger.info(f"Original tool_input: {tool_input}")
                
            for param_key, param_value in tool_input.items():
                if isinstance(param_value, str):
                    if "{{extracted_info." in param_value:
                        # Extract field name from extracted_info
                        field = param_value.replace("{{extracted_info.", "").replace("}}", "")
                        processed_params[param_key] = extracted_info.get(field, "Not provided")
                        logger.info(f"Parameter {param_key}: Replaced {{{{extracted_info.{field}}}}} with {processed_params[param_key]}")
                    elif "{{past_steps." in param_value:
                        # Extract tool name from past_steps
                        tool_name_ref = param_value.replace("{{past_steps.", "").replace("}}", "")
                        
                        # Find most recent step result with matching tool name
                        matching_result = None
                        for prev_step in reversed(past_steps):  # Start from most recent step
                            if prev_step.get("tool") == tool_name_ref and "result" in prev_step:
                                matching_result = prev_step["result"]
                                break
                                
                        if matching_result is not None:
                            processed_params[param_key] = matching_result
                            logger.info(f"Parameter {param_key}: Replaced {{{{past_steps.{tool_name_ref}}}}} with result from previous step")
                        else:
                            processed_params[param_key] = None
                            logger.warning(f"Parameter {param_key}: Could not find result for {{{{past_steps.{tool_name_ref}}}}}")
                    else:
                        processed_params[param_key] = param_value
                        logger.info(f"Parameter {param_key}: Using value {param_value}")
                else:
                    processed_params[param_key] = param_value
                    logger.info(f"Parameter {param_key}: Using non-string value")
            
            # Update parameters
            current_step["tool_input"] = processed_params
            logger.info(f"Processed tool_input: {processed_params}")
            
            # Execute the step
            try:
                logger.info(f"Calling tool: {tool_name} with parameters: {processed_params}")
                result = self.executor.execute_step(current_step)
                logger.info(f"Tool execution result type: {type(result)}")
                logger.info(f"Tool execution result (truncated): {str(result)[:200]}...")
            except Exception as e:
                logger.error(f"Error executing step: {str(e)}")
                result = {"error": str(e)}
            
            # Create the result node to add to past_steps
            result_node = {
                "step_id": current_step.get("step_id", current_index + 1),
                "description": current_step.get("description", ""),
                "tool": tool_name,
                "tool_input": current_step.get("tool_input", {}),
                "result": result
            }
            
            # Check if this is an information extraction step, if so, update the state
            if tool_name == "extract_information" and isinstance(result, dict):
                logger.info(f"Extracted information: {result}")
                return {
                    "past_steps": [result_node],
                    "current_step_index": current_index + 1,
                    "extracted_info": result
                }
            
            # Otherwise, just return past_steps and update the current index
            logger.info(f"Completed step {step_id}: {description}")
            return {
                "past_steps": [result_node],
                "current_step_index": current_index + 1
            }
            
        except Exception as e:
            logger.error(f"Error in executor node: {str(e)}")
            return {
                "past_steps": [{
                    "error": str(e),
                    "step_id": state.get("current_step_index", 0) + 1,
                    "tool": "unknown"
                }],
                "current_step_index": state.get("current_step_index", 0) + 1
            }
    
    def _replanner_node(self, state: AgentState) -> Dict[str, Any]:
        """Determine if additional steps are needed or generate the final response"""
        try:
            current_index = state["current_step_index"]
            plan = state["plan"]
            
            # Check if we've reached the end of the plan
            if current_index >= len(plan):
                logger.info("Reached end of plan, generating final response")
                return self._generate_final_response(state)
            
            # Check the last executed step for errors
            past_steps = state.get("past_steps", [])
            if past_steps:
                last_step = past_steps[-1]
                if "error" in last_step:
                    # If there's an error, increment replan count
                    replan_count = state.get("replan_count", 0) + 1
                    
                    # If we've replan too many times, generate a final response with an error
                    if replan_count >= 3:
                        logger.warning(f"Reached maximum replan count ({replan_count}), generating error response")
                        return self._generate_final_response(state, {
                            "error": f"Failed to complete the plan after {replan_count} retries. Last error: {last_step.get('error')}"
                        })
                    
                    # Otherwise, attempt to replan
                    logger.info(f"Step {last_step.get('step_id')} failed, attempting to replan (attempt {replan_count})")
                    
                    # Get the error details
                    error_message = last_step.get("error", "Unknown error")
                    
                    # Generate a new plan starting from the current step
                    replan_result = self.replanner.create_new_plan(
                        state["input"], 
                        state.get("extracted_info", {}),
                        past_steps,
                        error_message,
                        current_index
                    )
                    
                    if not replan_result or not isinstance(replan_result, dict) or "plan" not in replan_result:
                        logger.error(f"Replanner failed to generate a valid plan: {replan_result}")
                        return self._generate_final_response(state, {
                            "error": f"Failed to generate a new plan: {error_message}"
                        })
                    
                    # Extract the new plan
                    new_plan = replan_result.get("plan", [])
                    
                    # Update the plan in the state
                    if new_plan:
                        return {
                            "plan": new_plan,
                            "current_step_index": 0,  # Reset to beginning of new plan
                            "replan_count": replan_count
                        }
                    else:
                        logger.error(f"Replanner returned an empty plan after error: {error_message}")
                        return self._generate_final_response(state, {
                            "error": f"Failed to generate a new plan after error: {error_message}"
                        })
            
            # Continue with the next step in the plan
            return {}
            
        except Exception as e:
            logger.error(f"Error in replanner node: {str(e)}")
            return self._generate_final_response(state, {
                "error": f"Error in plan execution: {str(e)}"
            })
    
    def _generate_final_response(self, state: AgentState, replan_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate a final response based on the completed steps"""
        try:
            # If there's a replan error, use it to generate an error response
            if replan_result and "error" in replan_result:
                error_message = replan_result["error"]
                logger.error(f"Generating error response: {error_message}")
                response = f"I'm sorry, I wasn't able to complete your request due to an error: {error_message}"
                return {
                    "response": response
                }
            
            # If no plan has been executed, return a simple response
            past_steps = state.get("past_steps", [])
            if not past_steps:
                logger.warning("No steps executed, generating generic response")
                return {
                    "response": "I'm sorry, I wasn't able to process your request. Please try again with more details."
                }
            
            # Analyze past steps and generate a coherent response
            user_request = state.get("input", "your request")
            
            # Find the answer generation step result if it exists
            answer = None
            for step in past_steps:
                if step.get("tool") == "generate_answer" and isinstance(step.get("result"), str):
                    answer = step.get("result")
                    break
            
            # If we found an answer from generate_answer, use it
            if answer:
                return {"response": answer}
            
            # Otherwise, try to find any useful information from other steps
            useful_results = []
            for step in past_steps:
                result = step.get("result")
                tool = step.get("tool")
                
                # Skip extraction steps but include other results
                if tool != "extract_information" and result:
                    if isinstance(result, dict) and len(str(result)) > 10:
                        useful_results.append(f"From {tool}: {result}")
                    elif isinstance(result, str) and len(result) > 10:
                        useful_results.append(result)
            
            # If we have useful results, compile them
            if useful_results:
                response = "\n\n".join(["Here's what I found:"] + useful_results)
                return {"response": response}
            
            # If nothing useful found, return a basic response
            basic_response = "\n".join([
                f"In response to your query about '{user_request}', I couldn't find detailed information.",
                "",
                "Please try asking in a different way or provide more details."
            ])
            
            return {"response": basic_response}
            
        except Exception as e:
            logger.error(f"Error generating final response: {str(e)}")
            return {
                "response": f"I'm sorry, I encountered an error while generating your response: {str(e)}"
            }
    
    def _should_continue_or_end(self, state: AgentState) -> str:
        """Determine if the graph should continue or end"""
        # End if we have a response
        if "response" in state and state["response"]:
            return "end"
        return "continue"
    
    def run(self, user_input: str) -> Dict[str, Any]:
        """Run the general purpose assistant with the given input"""
        try:
            # Create the initial state
            initial_state = self._create_initial_state(user_input)
            
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            return final_state
        except Exception as e:
            logger.error(f"Error running assistant: {str(e)}")
            return {
                "error": str(e),
                "response": f"I'm sorry, an error occurred: {str(e)}"
            }
    
    def _create_initial_state(self, user_input: str) -> AgentState:
        """Create the initial state for the graph"""
        return {
            "input": user_input,
            "plan": [],
            "past_steps": [],
            "current_step_index": 0,
            "extracted_info": {},
            "response": "",
            "replan_count": 0
        } 