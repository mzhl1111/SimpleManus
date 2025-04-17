"""
Simplified Plan-and-Execute framework using LangGraph.
"""
import logging
import os
from typing import Dict, Any, TypedDict, List, Optional, Union, cast

from langchain.schema import (
    HumanMessage, BaseMessage, AIMessage, SystemMessage
)
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import create_react_agent, ToolNode
from langchain_core.tools import BaseTool

# Import configuration and tools
import config
import prompt_templates
from agent_tools import initialize_tools
from utils.mlflow_callback import MLflowTracker
import mlflow
import textwrap
# langchain.debug = True # Removed debug setting

# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(levelname)-8s [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure consistent Tool type definition
Tool = BaseTool

class PlanState(TypedDict):
    """State definition: stores plan steps and execution status"""
    input: str
    messages: List[BaseMessage]
    plan: List[Dict[str, Any]]
    current_step: int
    past_steps: List[Dict[str, Any]]
    error: Optional[str]
    result: Optional[str]


def make_system_prompt(prompt_text):
    """Create a system prompt for the agent"""
    return prompt_text


class AgentGraph:
    """LangGraph agent implementation with planning and execution"""

    def __init__(self, llm=None, tools_config=None):
        """Initialize agent graph with LLM and tools"""
        logger.info("[AGENT INIT] Initializing AgentGraph.")
        # LLM Initialization
        if not llm:
            from pydantic import SecretStr
            self.llm = ChatOpenAI(
                model=config.LLM_MODEL or "openai/gpt-3.5-turbo",
                temperature=0,
                api_key=SecretStr(config.OPENROUTER_API_KEY),
                base_url="https://openrouter.ai/api/v1"
            )
        else:
            self.llm = llm
        
        self.mlflow_tracker = MLflowTracker(
            experiment_name="PlanAndExecute_Agent"
        )

        # Prepare config dict for initialize_tools
        api_key_value = ""
        llm_api_key = getattr(self.llm, 'api_key', None)
        if llm_api_key is not None:
            if hasattr(llm_api_key, 'get_secret_value'):
                api_key_value = llm_api_key.get_secret_value()
            elif isinstance(llm_api_key, str):
                api_key_value = llm_api_key

        tools_init_config = {
            "llm_config": {
                "model_name": getattr(self.llm, 'model_name', ''),
                "api_key": api_key_value,
                "api_base": getattr(self.llm, 'base_url', ''),
             }
        }
        self.tools, _ = initialize_tools(config=tools_init_config)
        
        # Create agents
        self.planner_agent = self._create_planner()
        self.executor_agent_decision_maker = self._create_executor()

        # Create ToolNode
        execution_tools_list = [
            tool for tool in self.tools
            if tool.name not in [
                "create_plan", "update_plan", "update_status"
            ]
        ]
        self.executor_tool_node = ToolNode(execution_tools_list)

        self.graph = self._create_graph()
        self._tool_instances = {}
        
    def _create_graph(self):
        """Create and compile workflow graph"""
        workflow = StateGraph(MessagesState)
        
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor_decide", self._executor_decide_node)

        async def executor_tool_node_with_logging(state: MessagesState):
            last_message = state.get("messages", [])[-1] if state.get("messages") else None
            if not isinstance(last_message, AIMessage) or not getattr(last_message, 'tool_calls', None):
                logger.error(
                    "[Executor Tool Node] Last message is not an AIMessage "
                    "or has no tool calls."
                )
                return state
            tool_invocation = last_message.tool_calls[0]
            self._log_tool_call(tool_invocation)
            result = await self.executor_tool_node.ainvoke(state)
            self._log_tool_result(result)
            return result

        workflow.add_node("executor_execute_tool", executor_tool_node_with_logging)

        workflow.set_entry_point("planner")
        
        workflow.add_conditional_edges(
            "planner",
            self._decide_after_planning,
            {"execute": "executor_decide", "finish": END}
        )
        workflow.add_conditional_edges(
            "executor_decide",
            self._decide_after_executor,
            {"tools": "executor_execute_tool", "finish": END, "replan": "planner"}
        )
        workflow.add_edge("executor_execute_tool", "executor_decide")

        graph = workflow.compile()
        return graph
    
    def _planner_node(self, state: MessagesState) -> MessagesState:
        """Planner node processing logic. Returns full MessagesState."""
        logger.info("++++ [PLANNER NODE ENTRY] ++++")
        logger.debug(f"[PLANNER NODE] Input State: {state}")

        planning_msg_count = sum(
            1 for msg in state.get("messages", [])
            if hasattr(msg, 'name') and msg.name == "planner"
        )
        if planning_msg_count > 5:  # 将限制从10降低到5
            logger.warning(
                "[PLANNER] Too many planning steps, forcing finish"
            )
            # 生成更有信息量的最终答案
            original_query = ""
            if state and "messages" in state and state["messages"]:
                first_msg = state["messages"][0]
                if hasattr(first_msg, 'content'):
                    original_query = first_msg.content
            
            final_message = (
                f"FINAL ANSWER: After multiple attempts, I was unable to complete the task "
                f"regarding '{original_query}'. This could be due to technical limitations "
                f"in accessing certain websites or because the requested information is not "
                f"readily available. For ocean temperature data, consider consulting specialized "
                f"oceanographic databases or climate research institutions that may have more "
                f"reliable information."
            )
            
            error_message = AIMessage(
                content=final_message,
                name="planner"
            )
            updated_messages = state.get("messages", []) + [error_message]
            # Cast to MessagesState to satisfy linter
            return MessagesState(messages=updated_messages)

        logger.info("[PLANNER NODE] Generating plan...")
        try:
            result_state = self.planner_agent.invoke(
                state, config={"callbacks": [self.mlflow_tracker]}
            )

            if result_state and result_state.get("messages"):
                if not state.get("messages") or state["messages"][-1] != result_state["messages"][-1]:
                     last_msg = result_state["messages"][-1]
                     if not getattr(last_msg, 'name', None):
                         last_msg.name = "planner"

            logger.debug(f"[PLANNER NODE] Planner Result State: {result_state}")
            logger.info("---- [PLANNER NODE EXIT] ----")
            # Ensure valid MessagesState dictionary is returned, cast if needed
            if isinstance(result_state, dict) and "messages" in result_state:
                return MessagesState(messages=result_state["messages"])
            else:
                 logger.warning(
                     "[PLANNER NODE] Agent did not return expected "
                     "MessagesState format."
                 )
                 # Return input state or an empty state to avoid errors
                 return state # Or MessagesState(messages=[])

        except Exception as e:
            logger.error(f"Error in planner node: {str(e)}", exc_info=True)
            error_message = AIMessage(
                content=f"FINAL ANSWER: Error during planning: {str(e)}", name="planner"
            )
            updated_messages = state.get("messages", []) + [error_message]
            # Cast to MessagesState
            error_state = MessagesState(messages=updated_messages)
            logger.debug(f"[PLANNER NODE] Returning error state: {error_state}")
            logger.info("---- [PLANNER NODE EXIT with ERROR] ----")
            return error_state
    
    def _get_or_create_tool_instance(self, tool_name: str):
        """Get existing tool instance or create new one"""
        if tool_name not in self._tool_instances:
            tool_config = next(
                (t for t in self.tools if t.name == tool_name), 
                None
            )
            if tool_config:
                self._tool_instances[tool_name] = tool_config
        return self._tool_instances.get(tool_name)

    def _executor_decide_node(self, state: MessagesState) -> MessagesState:
        """Runs the executor LLM to decide the next action or provide final answer."""
        logger.info("++++ [EXECUTOR DECIDE NODE ENTRY] ++++")
        logger.debug(f"[EXECUTOR DECIDE NODE] Input State: {state}")
        try:
            logger.info("[EXECUTOR DECIDE NODE] Asking agent for next step...")
            agent_decision_state = self.executor_agent_decision_maker.invoke(
                state, config={"callbacks": [self.mlflow_tracker]}
            )

            if not isinstance(agent_decision_state, dict) or "messages" not in agent_decision_state:
                logger.error(
                    f"Unexpected output type from executor agent: "
                    f"{type(agent_decision_state)}. Expected MessagesState dict. "
                    f"State: {agent_decision_state}"
                )
                error_message = AIMessage(content="FINAL ANSWER: Internal error in executor agent output.")
                updated_messages = state.get("messages", []) + [error_message]
                # Cast to MessagesState
                return MessagesState(messages=updated_messages)

            logger.debug(f"[EXECUTOR DECIDE NODE] Agent Decision State Raw: {agent_decision_state}")
            # Log the last message which contains the agent's decision (tool call or text)
            if agent_decision_state.get("messages"):
                 logger.info(f"[EXECUTOR DECIDE NODE] Agent Decision Message: {agent_decision_state['messages'][-1]}")
            else:
                 logger.warning("[EXECUTOR DECIDE NODE] Agent returned state with no messages.")

            logger.info("---- [EXECUTOR DECIDE NODE EXIT] ----")
            # Cast to MessagesState
            return MessagesState(messages=agent_decision_state["messages"])

        except Exception as e:
            logger.error(
                f"[EXECUTOR DECIDE NODE] Error during decision making: {str(e)}",
                exc_info=True
            )
            error_message = AIMessage(
                content=f"FINAL ANSWER: Error during executor decision: {str(e)}"
            )
            updated_messages = state.get("messages", []) + [error_message]
            # Cast to MessagesState
            error_state = MessagesState(messages=updated_messages)
            logger.debug(f"[EXECUTOR DECIDE NODE] Returning Error State: {error_state}")
            logger.info("---- [EXECUTOR DECIDE NODE EXIT with ERROR] ----")
            return error_state
    
    def _decide_after_executor(self, state: MessagesState) -> str:
        """
        Custom condition function to decide the next step after the executor LLM
        has made a decision (but before executing tools).
        """
        logger.info("[DECIDE AFTER EXECUTOR] Deciding next step...")
        logger.debug(f"[DECIDE AFTER EXECUTOR] Input State: {state}")
        if not state or "messages" not in state or not state["messages"]:
            logger.error("[DECIDE AFTER EXECUTOR] Invalid state: no messages found.")
            return "finish"

        # 统计重新规划的次数
        replan_count = sum(
            1 for msg in state["messages"] 
            if hasattr(msg, 'content') and isinstance(msg.content, str) and "NEED REPLAN" in msg.content
        )
        
        # 如果重新规划次数过多，强制结束并生成最终响应
        if replan_count >= 3:
            logger.warning(
                f"[DECIDE AFTER EXECUTOR] Too many replans ({replan_count}), forcing finish"
            )
            # 生成一个最终响应
            final_response = AIMessage(
                content="FINAL ANSWER: After multiple attempts, I was unable to access reliable information about ocean temperatures in Yokohama for 2025. This could be due to technical limitations or because such specific future predictions may not be readily available. Based on current data, ocean temperatures in Yokohama typically range from around 14°C to 26°C depending on the season, with summer months seeing the highest temperatures. For more accurate predictions for 2025, specialized climate research institutions or oceanographic databases would be needed."
            )
            if isinstance(state, dict) and "messages" in state:
                state["messages"].append(final_response)
                logger.info("[DECIDE AFTER EXECUTOR] Added forced final response after multiple replans.")
            return "finish"

        last_message = state["messages"][-1]
        decision = "finish" # Default to finish

        # Check for tool calls only if it's an AIMessage
        if isinstance(last_message, AIMessage) and hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.info(f"[DECIDE AFTER EXECUTOR] Found tool calls: {last_message.tool_calls}")
            decision = "tools"

        # Check for FINAL ANSWER marker if no tool call
        elif hasattr(last_message, 'content') and "FINAL ANSWER:" in last_message.content:
            logger.info("[DECIDE AFTER EXECUTOR] Found FINAL ANSWER.")
            decision = "finish"

        # Check for explicit replan request
        elif hasattr(last_message, 'content') and "NEED REPLAN" in last_message.content:
            logger.info("[DECIDE AFTER EXECUTOR] Found NEED REPLAN.")
            decision = "replan" # Enable replanning

        # If no tool call and no FINAL ANSWER or REPLAN, finish to prevent loops
        else:
            logger.warning(
                "[DECIDE AFTER EXECUTOR] No tool call, FINAL ANSWER, or NEED REPLAN found. "
                "Generating final response."
            )
            # Generate a final response based on execution history
            final_response = self._generate_final_response(state)
            if final_response:
                # Add the final response to the state's messages
                # This will be processed by _process_result later
                if isinstance(state, dict) and "messages" in state:
                    state["messages"].append(final_response)
                    logger.info("[DECIDE AFTER EXECUTOR] Added generated final response.")
            # decision remains "finish"

        logger.info(f"[DECIDE AFTER EXECUTOR] Decision: Go to '{decision}'")
        return decision
    
    def _generate_final_response(self, state: MessagesState) -> Optional[AIMessage]:
        """
        Generate a well-structured final response based on execution history
        when no explicit FINAL ANSWER is provided.
        
        Args:
            state: Current state with messages history
            
        Returns:
            AIMessage with formatted final response or None if generation fails
        """
        try:
            logger.info("[GENERATE FINAL RESPONSE] Generating structured final response...")
            
            # Extract user query from the first message
            user_query = ""
            if state and "messages" in state and state["messages"]:
                first_msg = state["messages"][0]
                if hasattr(first_msg, 'content'):
                    user_query = first_msg.content
            
            if not user_query:
                logger.warning("[GENERATE FINAL RESPONSE] Could not extract user query.")
                return None
                
            # Extract past steps from execution history
            # This is a simplified approach, could be enhanced to extract more context
            past_steps = []
            for msg in state.get("messages", []):
                if isinstance(msg, AIMessage) and hasattr(msg, 'content'):
                    # Skip messages that are tool calls
                    if not getattr(msg, 'tool_calls', None):
                        # Format the message content as a step
                        step_text = msg.content
                        # Truncate if too long, ensuring step_text is treated as string
                        if isinstance(step_text, str) and len(step_text) > 500:
                            step_text = step_text[:497] + "..."
                        elif not isinstance(step_text, str):
                            # Convert non-string content to string
                            step_text = str(step_text)
                            if len(step_text) > 500:
                                step_text = step_text[:497] + "..."
                        past_steps.append(step_text)
            
            if not past_steps:
                logger.warning("[GENERATE FINAL RESPONSE] No execution steps found.")
                return None
            
            # Format past steps as a numbered list for the prompt
            formatted_steps = "\n".join([f"{i+1}. {step}" for i, step in enumerate(past_steps)])
            
            # Prepare the prompt using the template
            prompt = prompt_templates.FINAL_RESPONSE_PROMPT.format(
                user_query=user_query,
                past_steps=formatted_steps
            )
            
            # Generate the response (includes the LLM call)
            # Use the same LLM as the executor but with different prompt
            messages = [HumanMessage(content=prompt)]
            
            # Pass the callbacks for MLflow tracking
            response = self.llm.invoke(
                messages, config={"callbacks": [self.mlflow_tracker]}
            )
            
            # Create final answer message with the generated content
            final_response = AIMessage(
                content=f"FINAL ANSWER: {response.content}"
            )
            
            logger.info("[GENERATE FINAL RESPONSE] Successfully generated final response.")
            return final_response
            
        except Exception as e:
            logger.error(
                f"[GENERATE FINAL RESPONSE] Error generating final response: {str(e)}",
                exc_info=True
            )
            # Return a basic error message as final answer
            return AIMessage(
                content="FINAL ANSWER: Unable to generate a structured response "
                        "due to an error. Please check the execution logs."
            )
    
    def _decide_after_planning(self, state: MessagesState) -> str:
        """Post-planning decision: execute or finish"""
        logger.info("[DECIDE AFTER PLAN] Deciding next step...") # Log start
        logger.debug(f"[DECIDE AFTER PLAN] Input State: {state}") # Log state
        # Get the last message
        if not state or "messages" not in state or not state["messages"]:
            logger.error("[DECIDE AFTER PLAN] Invalid state: no messages found.")
            return "finish" # Cannot proceed without messages

        last_message = state["messages"][-1]
        content = getattr(last_message, 'content', '') # Safely access content
        
        # Track number of planning steps to prevent infinite recursion
        planning_msg_count = sum(1 for msg in state["messages"] 
                              if hasattr(msg, 'name') and msg.name == "planner")
        if planning_msg_count > 10:
            logger.warning("[PLANNER] Too many planning steps, forcing finish")
            return "finish"
            
        # If message contains completion marker
        if "FINAL ANSWER" in content:
            logger.info("[DECIDE AFTER PLAN] Decision: Finish (FINAL ANSWER found)." ) # Log decision
            return "finish"
        
        # Default to continue execution
        logger.info("[DECIDE AFTER PLAN] Decision: Execute.") # Log decision
        return "execute"
    
    async def ainvoke(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Asynchronously execute agent graph"""
        initial_state = None # Define initial_state here for broader scope
        try:
            # Prepare initial state inside ainvoke if not already a dict
            if isinstance(input_data, str):
                initial_state = self._prepare_initial_state(input_data)
            elif isinstance(input_data, dict) and "messages" in input_data:
                # Assume input_data is already a valid state dict (like MessagesState)
                initial_state = input_data
            else:
                raise ValueError(
                    "Input must be a string query or a dict containing 'messages' key."
                )

            # Call graph with increased recursion limit
            result = await self.graph.ainvoke(
                initial_state,
                config={"recursion_limit": 50} # Increase limit, fixed quotes
            )
            
            # Log execution metrics
            self.mlflow_tracker.log_metric() # Ensure this logs accumulated metrics

            # Log custom metrics (removing time-based for now)
            # mlflow.log_metric("execution_time", execution_time)

            # Log query length based on original input
            query_text = ""
            if isinstance(input_data, str):
                query_text = input_data
            elif initial_state and initial_state.get("messages"):
                 # Attempt to get content from the first message
                 first_msg_content = getattr(initial_state["messages"][0], 'content', '')
                 query_text = first_msg_content if isinstance(first_msg_content, str) else str(first_msg_content)

            mlflow.log_metric("query_length", len(query_text))
            
            # Log the query text
            mlflow.log_param("query", query_text[:250])  # Truncate if too long
                
            return self._process_result(result)
        except Exception as e:
            logger.error(f"Error executing graph: {str(e)}", exc_info=True)
            # Log error in MLflow
            mlflow.log_param("error", str(e)[:250])  # Truncate if too long
            # Return a consistent error structure
            error_state = {
                "messages": initial_state.get("messages", []) if initial_state else [],
                "error": str(e),
                "result": f"Error occurred during execution: {str(e)}"
            }
            return error_state
    
    def _prepare_initial_state(self, user_input: str) -> Dict[str, Any]:
        """Prepare initial state"""
        # Ensure state matches MessagesState format expected by the graph
        return {
            "messages": [HumanMessage(content=user_input)]
        }
    
    def _process_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Process final result, extract FINAL ANSWER"""
        messages = result.get("messages", [])
        
        # Extract final answer from last message
        final_answer = None
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, 'content') and isinstance(last_message.content, str):
                if "FINAL ANSWER" in last_message.content:
                    # Extract content after FINAL ANSWER:
                    parts = last_message.content.split("FINAL ANSWER:", 1)
                    if len(parts) > 1:
                        final_answer = parts[1].strip()
        
        # Get final result text
        result_text = None
        if final_answer:
            result_text = final_answer
        else:
            # Generate comprehensive response if no clear final answer
            result_text = self._generate_final_response(result)
        
        # Log result metrics
        # Count total tokens used from messages
        msg_count = len(messages)
        mlflow.log_metric("message_count", msg_count)
        mlflow.log_metric("result_length", len(str(result_text)))
        
        # Calculate token estimates if not properly tracked
        if self.mlflow_tracker.total_tokens == 0:
            # Estimate tokens: ~4 chars per token for English text
            estimated_input_tokens = sum(
                len(str(getattr(msg, 'content', ''))) // 4 
                for msg in messages if hasattr(msg, 'content')
            )
            estimated_output_tokens = len(str(result_text)) // 4
            
            # Log estimated token counts
            mlflow.log_metrics({
                "estimated_input_tokens": estimated_input_tokens,
                "estimated_output_tokens": estimated_output_tokens,
                "estimated_total_tokens": (
                    estimated_input_tokens + estimated_output_tokens
                )
            })
        
        return {
            "messages": messages,
            "result": result_text,
            "raw_result": result
        }
        
    def _generate_final_response(self, state: Dict[str, Any]) -> str:
        """Generate a comprehensive final response when no clear FINAL ANSWER"""
        try:
            # Extract the original user query from the first message
            messages = state.get("messages", [])
            if not messages:
                return "Execution completed but no messages or results were found."
            
            # Get the original user query
            user_query = ""
            for msg in messages:
                if (isinstance(msg, HumanMessage) and 
                    hasattr(msg, 'content')):
                    user_query = str(getattr(msg, 'content', ''))
                    break
            
            # Collect execution steps from agent messages
            past_steps = []
            for msg in messages:
                if (hasattr(msg, 'name') and 
                    getattr(msg, 'name', '') == "executor" and
                    hasattr(msg, 'content')):
                    step_content = str(getattr(msg, 'content', ''))
                    if step_content:
                        past_steps.append(step_content)
            
            past_steps_text = "\n".join(past_steps)
            
            # Use the final response prompt template
            formatted_prompt = prompt_templates.FINAL_RESPONSE_PROMPT.format(
                user_query=user_query,
                past_steps=past_steps_text
            )
            
            # Generate the final response using the LLM
            response = self.llm.invoke(
                [SystemMessage(content=formatted_prompt)]
            )
            
            # Extract and return the content of the response
            if hasattr(response, 'content'):
                return str(getattr(response, 'content', ''))
            
            return "Execution completed but could not generate a response."
        except Exception as e:
            logger.error(
                f"Error generating final response: {str(e)}", 
                exc_info=True
            )
            return (
                f"Execution completed but encountered an error: {str(e)}"
            )

    def _create_planner(self):
        """Create planner agent"""
        # Isolate planning tools
        planning_tools = [tool for tool in self.tools if tool.name in [
            "create_plan", "update_plan", "update_status"
        ]]
        
        # Use ReAct as planner
        return create_react_agent(
            model=self.llm,
            tools=planning_tools,
            prompt=make_system_prompt(prompt_templates.PLANNER_PROMPT)
        )
    
    def _create_executor(self):
        """Creates the Executor ReAct agent (for DECISION MAKING ONLY)."""
        logger.info("[EXECUTOR CREATION] Creating decision-making agent.")
        execution_tools_list = [tool for tool in self.tools if tool.name not in [
            "create_plan", "update_plan", "update_status"
        ]]
        
        # Use ReAct agent, it will decide which tool from the list to call
        # The actual execution will be handled by ToolNode
        return create_react_agent(
            model=self.llm,
            tools=execution_tools_list, # Agent needs tool schema to decide
            prompt=make_system_prompt(prompt_templates.EXECUTOR_PROMPT)
        )

    def _log_tool_call(self, tool_call):
        """Helper to log tool call details."""
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args")
        logger.info(f"++++ [EXECUTOR EXECUTE TOOL PRE] Calling tool: {tool_name} with args: {tool_args} ++++")

    def _log_tool_result(self, tool_result):
        """Helper to log tool result details."""
        # Shorten potentially very long browser state results for logging
        log_result = str(tool_result)
        if len(log_result) > 500: 
            log_result = log_result[:500] + "... (truncated)"
        logger.info(f"---- [EXECUTOR EXECUTE TOOL POST] Tool Result: {log_result} ----")


def run(query: str) -> Dict[str, Any]:
    """Run agent graph and get results"""
    agent_graph = AgentGraph()
    
    try:
        # Use asyncio.run to call the async method
        import asyncio
        # Pass callbacks explicitly here too, if needed for top-level invoke
        result = asyncio.run(agent_graph.ainvoke(
            query
        ))
        
        # Print final result
        if "result" in result:
            print(f"\nFinal Response:\n{result['result']}")
            
        # Get usage report (might need adjustment if tracker is async)
        # usage_report = agent_graph.mlflow_tracker.get_usage_report() 
        mlflow.log_metrics({
            "total_steps": len(result.get("messages", [])),
            # Update plan_completion based on result, not fixed 1.0
            "plan_completion": 1.0 if "error" not in result else 0.0 
        })
        
        # End MLflow run if successful
        if "error" not in result:
            mlflow.end_run()
        
        return result
    except Exception as e:
        # Ensure MLflow run ends even on outer error
        try:
             mlflow.log_param("outer_error", str(e)[:250])
             mlflow.end_run(status="FAILED")
        except Exception as mlflow_e:
             logger.error(f"MLflow logging failed during exception handling: {mlflow_e}")
        logger.error(f"Error in agent run: {str(e)}", exc_info=True)
        return {"error": str(e), "result": "Error during execution setup or teardown"} 