"""
Travel planning assistant using LangGraph for workflow management.
Implements the Plan-and-Execute pattern.
"""
import logging
import operator
from typing import Dict, Any, TypedDict, List, Annotated

from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from pydantic import BaseModel

from planner_engine import Planner, Executor, Replanner
from travel_tools import TRAVEL_TOOLS

import config

# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Define application state type
class TravelAppState(TypedDict):
    """State definition for the travel planning application"""
    input: str                                      # User's original request
    plan: List[Dict[str, Any]]                      # Current plan steps
    past_steps: Annotated[List[Dict[str, Any]], operator.add]  # Results of executed steps
    current_step_index: int                         # Index of current step
    travel_info: Dict[str, Any]                     # Extracted travel information
    response: str                                   # Final response to user
    replan_count: int                              # Count of replanning attempts


class TravelGraph:
    """Travel planning assistant using LangGraph with Plan-and-Execute pattern"""
    
    def __init__(self):
        """Initialize the travel planning assistant"""
        # Initialize components
        self.planner = Planner()
        self.executor = Executor()
        self.replanner = Replanner()
        
        # Create the graph
        self.graph = self._create_graph()
        
        # Track string plan conversion attempts
        self.string_plan_count = 0
    
    def _create_graph(self) -> CompiledGraph:
        """Create the travel planning workflow graph"""
        # Create workflow graph
        workflow = StateGraph(TravelAppState)
        
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
    
    def _planner_node(self, state: TravelAppState) -> Dict[str, Any]:
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
                        "description": "Extract basic travel information",
                        "tool": "extract_travel_info",
                        "tool_input": {"user_input": state["input"]}
                    }
                ],
                "current_step_index": 0,
                "past_steps": [],
                "replan_count": 0  # Initialize replan counter
            }
    
    def _executor_node(self, state: TravelAppState) -> Dict[str, Any]:
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
            
            logger.info(f"Executing step {current_step.get('step_id', current_index+1)}: "
                        f"{current_step.get('description', 'Execute step')}")
            
            # Process placeholders in parameters
            tool_name = current_step.get("tool", "")
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
                
            # Remove special handling for extract_travel_info, process parameters for all tools
            processed_params = {}
            travel_info = state.get("travel_info", {})
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
                
            for param_key, param_value in tool_input.items():
                if isinstance(param_value, str):
                    if "{{travel_info." in param_value:
                        # Extract field name from travel_info
                        field = param_value.replace("{{travel_info.", "").replace("}}", "")
                        processed_params[param_key] = travel_info.get(field, "Not provided")
                    elif "{{past_steps." in param_value:
                        # Extract tool name from past_steps
                        tool_name = param_value.replace("{{past_steps.", "").replace("}}", "")
                        if tool_name in past_results:
                            processed_params[param_key] = past_results[tool_name]
                        else:
                            processed_params[param_key] = None
                    else:
                        processed_params[param_key] = param_value
                else:
                    processed_params[param_key] = param_value
            
            # Update parameters
            current_step["tool_input"] = processed_params
            
            # Execute the step
            try:
                result = self.executor.execute_step(current_step)
            except Exception as e:
                logger.error(f"Error executing step: {str(e)}")
                result = {
                    "error": str(e),
                    "step_id": current_step.get("step_id", current_index + 1),
                    "tool": current_step.get("tool", "unknown")
                }
            
            # If this is the first step (information extraction), save the travel info
            if current_step.get("tool") == "extract_travel_info":
                # Check result format
                if not isinstance(result, dict):
                    logger.error(f"Expected dict result from extract_travel_info, got {type(result).__name__}")
                    return {
                        "travel_info": {},
                        "past_steps": [{
                            "error": f"Invalid result format: {type(result).__name__}",
                            "step_id": current_step.get("step_id", current_index + 1),
                            "tool": current_step.get("tool", "unknown")
                        }],
                        "current_step_index": current_index + 1
                    }
                
                if "result" in result and result["result"] is not None:
                    # Ensure result is dictionary type
                    if not isinstance(result["result"], dict):
                        logger.error(
                            f"Extraction result is not a dictionary: {type(result['result']).__name__}"
                        )
                        return {
                            "travel_info": {},
                            "past_steps": [{
                                "error": f"Extraction returned non-dictionary: {type(result['result']).__name__}",
                                "step_id": current_step.get("step_id", current_index + 1),
                                "tool": current_step.get("tool", "unknown")
                            }],
                            "current_step_index": current_index + 1
                        }
                    
                    # Normal case: extracted valid travel information
                    travel_info = result["result"]
                    logger.info("========== EXTRACTED TRAVEL INFO ==========")
                    logger.info(f"Destination: {travel_info.get('destination', 'Not provided')}")
                    logger.info(f"Duration: {travel_info.get('duration', 'Not provided')}")
                    logger.info(f"Date: {travel_info.get('date', 'Not provided')}")
                    logger.info(f"Customization: {travel_info.get('customization_hints', 'Not provided')}")
                    logger.info("==========================================")
                    
                    # Save original user query to travel_info for potential future use
                    travel_info["user_query"] = state.get("input", "")
                    
                    return {
                        "travel_info": travel_info,
                        "past_steps": [result],
                        "current_step_index": current_index + 1
                    }
                else:
                    # Error case: No result or missing "result" field
                    error_msg = result.get("error", "Unknown error")
                    logger.error(
                        f"Extraction step failed or returned invalid result format: {error_msg}"
                    )
                    return {
                        "travel_info": {"destination": "Unknown", "user_query": state.get("input", "")},
                        "past_steps": [result],
                        "current_step_index": current_index + 1
                    }
            
            # Otherwise just update past steps and increment the index
            return {
                "past_steps": [result],
                "current_step_index": current_index + 1
            }
        except Exception as e:
            logger.error(f"Error in executor node: {str(e)}")
            # Record the error and move to the next step
            return {
                "past_steps": [{
                    "error": str(e),
                    "step_id": state["current_step_index"] + 1,
                    "tool": "unknown"
                }],
                "current_step_index": state["current_step_index"] + 1
            }
    
    def _replanner_node(self, state: TravelAppState) -> Dict[str, Any]:
        """Re-evaluate the plan based on execution results"""
        try:
            current_index = state["current_step_index"]
            plan = state["plan"]
            past_steps = state.get("past_steps", [])
            travel_info = state.get("travel_info", {})
            
            # Increment replan counter
            state["replan_count"] = state.get("replan_count", 0) + 1
            
            # If replanning attempts exceed limit, force completion
            if state["replan_count"] > 5:  # Maximum allowed 5 replanning attempts
                logger.warning(f"Too many replanning attempts ({state['replan_count']}), forcing completion")
                return self._generate_final_response(state)
            
            # Get destination and ensure safety check
            destination = travel_info.get("destination", "")
            has_valid_destination = bool(destination and isinstance(destination, str) and 
                                       destination.lower() != "not provided")

            # Special handling: Check if only information extraction steps and already have valid destination
            # This handles cases like "pink" being inferred as a specific destination
            if len(plan) == 1 and current_index == 1 and has_valid_destination:
                
                logger.info(f"Successfully extracted destination '{destination}' from vague input. Adding complete travel plan steps.")
                
                # Create complete travel plan steps
                complete_plan = [
                    plan[0],  # Keep original information extraction step
                    {
                        "step_id": 2,
                        "description": f"Search for attractions in {destination} for the specified season",
                        "tool": "search_attractions",
                        "tool_input": {
                            "destination": destination,
                            "date": travel_info.get("date", "Not provided")
                        }
                    },
                    {
                        "step_id": 3,
                        "description": f"Search for local tips relevant to {destination} and the current season",
                        "tool": "search_local_tips",
                        "tool_input": {
                            "destination": destination,
                            "date": travel_info.get("date", "Not provided")
                        }
                    },
                    {
                        "step_id": 4,
                        "description": f"Generate a daily itinerary based on extracted travel info and seasonal content",
                        "tool": "generate_daily_itinerary",
                        "tool_input": {
                            "destination": destination,
                            "duration": travel_info.get("duration", "Not provided"),
                            "date": travel_info.get("date", "Not provided"),
                            "customization_hints": travel_info.get("customization_hints", "Not provided")
                        }
                    },
                    {
                        "step_id": 5,
                        "description": f"Estimate travel budget for the trip",
                        "tool": "estimate_budget",
                        "tool_input": {
                            "destination": destination,
                            "duration": travel_info.get("duration", "Not provided"),
                            "date": travel_info.get("date", "Not provided")
                        }
                    }
                ]
                
                # Return new complete plan, ensure execution starts from step 2 (index 1)
                return {
                    "plan": complete_plan,
                    "current_step_index": 1,  # Execute from step 2
                    "replan_count": 0  # Reset replan counter, because this is new plan
                }

            # Normal case, proceed with step completion check
            if current_index >= len(plan):
                logger.info("All steps completed, generating final response")
                return self._generate_final_response(state)

            # Track string plans to detect recursion
            self.string_plan_count = self.string_plan_count + 1

            # Normal replanning logic
            replan_result = self.replanner.replan(
                state["input"],
                state["plan"],
                state["past_steps"]
            )

            if replan_result["action"] == "complete":
                logger.info("Replanner decided to complete with final response")
                return self._generate_final_response(state, replan_result)
            elif replan_result["action"] == "update":
                logger.info("Replanner decided to update the plan")
                
                # Check if we're getting a string plan
                new_plan = replan_result.get("plan", [])
                if len(new_plan) > 0 and isinstance(new_plan[0], str):
                    # Increment string plan counter
                    self.string_plan_count = self.string_plan_count + 1
                    
                    # If we've tried to handle string plans too many times, force completion
                    if self.string_plan_count > 3:
                        logger.warning(f"Too many string plans ({self.string_plan_count}), forcing completion")
                        return self._generate_final_response(state)
                
                return {
                    "plan": replan_result["plan"],
                    "current_step_index": 0,
                    "past_steps": [],
                    "replan_count": 0  # Reset replan counter, because this is new plan
                }
            else:
                logger.info("Replanner decided to continue with current plan")
                # Return empty dict instead of None
                return {}

        except Exception as e:
            logger.error(f"Error in replanner node: {str(e)}")
            # Return empty dict instead of None
            return {}

    def _generate_final_response(self, state: TravelAppState, replan_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate the final response for the user"""
        travel_info = state.get("travel_info", {})
        past_steps = state.get("past_steps", [])
        
        # Collect results
        results = {}
        for step in past_steps:
            if "tool" in step and "result" in step:
                results[step["tool"]] = step["result"]
        
        # Check if budget estimation needs to be executed
        if "extract_travel_info" in results and "estimate_budget" not in results:
            logger.info("Budget estimation step was not executed, forcing execution now")
            try:
                # Get travel information
                destination = travel_info.get("destination", "Unknown")
                duration = travel_info.get("duration", "1 day") 
                date = travel_info.get("date", "Not specified")
                
                logger.info(f"Travel info for budget estimation: destination={destination}, "
                            f"duration={duration}, date={date}")
                
                from travel_tools import estimate_budget
                # Add recursion depth parameter to avoid circular dependency
                budget_result = estimate_budget(
                    destination=destination,
                    duration=duration,
                    date=date,
                    _recursion_depth=1  # Indicates this is a recursive call
                )
                
                # Add result to results dictionary
                results["estimate_budget"] = budget_result
                logger.info(f"Successfully executed budget estimation: "
                            f"${budget_result.get('estimated_total_usd', 'N/A')}")
            except Exception as e:
                logger.error(f"Error executing forced budget estimation: {str(e)}")
        
        destination = travel_info.get("destination", "Unknown destination")
        response = f"Travel plan for {destination}:\n\n"
        
        # Add attractions list
        if "search_attractions" in results:
            response += "Attractions:\n"
            attractions = results["search_attractions"]
            for i, attraction in enumerate(attractions, 1):
                response += f"{i}. {attraction.get('title', 'Unknown')}\n"
            response += "\n"
        
        # Add itinerary
        if "generate_daily_itinerary" in results:
            response += "Itinerary:\n"
            response += results["generate_daily_itinerary"] + "\n\n"
        elif replan_result and "response" in replan_result:
            # Use replanner's response directly
            response += replan_result["response"] + "\n\n"
        
        # Add budget information
        if "estimate_budget" in results:
            response += "Estimated Budget:\n"
            budget_result = results["estimate_budget"]
            if isinstance(budget_result, dict):
                breakdown = budget_result.get("cost_breakdown", {})
                for k, v in breakdown.items():
                    response += f"- {k.capitalize()}: ${v}\n"
                total = budget_result.get("estimated_total_usd", "N/A")
                response += f"Total: ${total}\n\n"
        
        return {"response": response}

    def _should_continue_or_end(self, state: TravelAppState) -> str:
        """Determine whether to continue execution or end"""
        # If we have a response, we're done
        if "response" in state and state["response"]:
            return "end"
        
        # If we still have steps to execute, continue
        if state["current_step_index"] < len(state["plan"]):
            return "continue"
        
        # If we've executed all steps but don't have a response, end
        return "end"
    
    def run(self, user_input: str) -> Dict[str, Any]:
        """Run the travel planning workflow"""
        try:
            # Set recursion limit through environment variable
            import os
            os.environ["LANGGRAPH_RECURSION_LIMIT"] = "100"
            
            # Initialize state
            initial_state = self._create_initial_state(user_input)
            
            # Run the graph
            result = self.graph.invoke(initial_state)
            return result
        except Exception as e:
            logger.error(f"Error running travel graph: {str(e)}")
            return {
                "error": str(e),
                "response": f"An error occurred: {str(e)}"
            }

    def _create_initial_state(self, user_input: str) -> TravelAppState:
        """
        Create initial state for travel planning app
        
        Args:
            user_input: User's travel query
            
        Returns:
            Initial state for travel graph execution
        """
        logger.info(f"Creating plan for: {user_input}")
        plan = self.planner.create_plan(user_input)
        
        # Create initial state
        return {
            "input": user_input,
            "travel_info": {},
            "plan": plan,
            "current_step_index": 0,
            "past_steps": [],
            "replan_count": 0  # Initialize replan counter
        } 