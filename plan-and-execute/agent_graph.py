"""
Simplified Plan-and-Execute framework using LangGraph.
"""
import logging
from typing import Dict, Any, TypedDict, List, Optional, Union
import time

from langchain.prompts import SystemMessagePromptTemplate
from langchain.schema import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command

# Import configuration and tools
import config
import prompt_templates
from agent_tools import initialize_tools
from utils.mlflow_callback import MLflowTracker
import mlflow

# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(levelname)-8s [%(name)s] %(message)s'
)
logger = logging.getLogger(__name__)


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
        if not llm:
            # Use OpenRouter API
            from pydantic import SecretStr
            model_name = config.LLM_MODEL or "openai/gpt-3.5-turbo"
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=0,
                api_key=SecretStr(config.OPENROUTER_API_KEY),
                base_url="https://openrouter.ai/api/v1"
            )
        else:
            self.llm = llm
        
        # Initialize MLflow tracking
        self.mlflow_tracker = MLflowTracker(experiment_name="PlanAndExecute_Agent")
        
        # Add MLflow tracker to callbacks
        self.llm.callbacks = [self.mlflow_tracker]
        
        # Initialize with MLflow callback
        self.tools = initialize_tools(llm=self.llm)
        
        # Create agents
        self.planner_agent = self._create_planner()
        self.executor_agent = self._create_executor()
        
        # Compile workflow graph
        self.graph = self._create_graph()
        
    def _create_graph(self):
        """Create and compile workflow graph"""
        workflow = StateGraph(MessagesState)
        
        # Add planning and execution nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor", self._executor_node)
        
        # Set entry point to planner
        workflow.set_entry_point("planner")
        
        # Conditional jumps: planner -> executor/end
        workflow.add_conditional_edges(
            "planner",
            self._decide_after_planning,
            {
                "execute": "executor",
                "finish": END
            }
        )
        
        # Conditional jumps: executor -> planner/end
        workflow.add_conditional_edges(
            "executor",
            self._decide_after_execution,
            {
                "replan": "planner",
                "finish": END
            }
        )
        
        # Compile workflow graph
        graph = workflow.compile()
        return graph
    
    def _planner_node(self, state: MessagesState) -> Command:
        """Planner node processing logic"""
        # Call planner agent
        logger.info("[PLANNER NODE] Generating plan...")
        
        # Check if we've already gone through too many planning cycles
        planning_msg_count = sum(
            1 for msg in state["messages"]
            if hasattr(msg, 'name') and msg.name == "planner"
        )
        if planning_msg_count > 10:
            logger.warning("[PLANNER] Too many planning steps, forcing finish")
            return Command(
                update={"messages": state["messages"] + [
                    HumanMessage(
                        content="FINAL ANSWER: I've reached my planning limit.",
                        name="planner"
                    )
                ]},
                goto="finish",
            )
        
        try:
            result = self.planner_agent.invoke(state)
            
            # Wrap as human message to pass to next node
            result["messages"][-1] = HumanMessage(
                content=result["messages"][-1].content,
                name="planner"
            )
            
            # Decide next step based on content
            content = result["messages"][-1].content
            next_node = "execute"
            
            if "FINAL ANSWER" in content:
                next_node = "finish"
            
            return Command(
                update={"messages": result["messages"]},
                goto=next_node,
            )
        except Exception as e:
            logger.error(f"Error in planner node: {str(e)}", exc_info=True)
            return Command(
                update={"messages": state["messages"] + [
                    HumanMessage(
                        content=f"FINAL ANSWER: Error during planning: {str(e)}",
                        name="planner"
                    )
                ]},
                goto="finish",
            )
    
    def _executor_node(self, state: MessagesState) -> Command:
        """Executor node processing logic"""
        # Call executor agent
        logger.info("[EXECUTOR NODE] Executing plan...")
        
        # Check if we've already gone through too many execution cycles
        exec_msg_count = sum(
            1 for msg in state["messages"]
            if hasattr(msg, 'name') and msg.name == "executor"
        )
        
        # Check number of search attempts to prevent infinite searching
        search_attempts = sum(
            1 for msg in state["messages"] 
            if "Executing backup search" in str(msg.content)
        )
        
        # If too many search attempts, force finish with what we know
        if search_attempts > 10:
            logger.warning(f"[EXECUTOR] Too many search attempts ({search_attempts}), forcing finish")
            return Command(
                update={"messages": state["messages"] + [
                    HumanMessage(
                        content=("FINAL ANSWER: I'm unable to find precise information after "
                                "multiple search attempts. Please try a simpler question or "
                                "provide more specific details."),
                        name="executor"
                    )
                ]},
                goto="finish",
            )
            
        if exec_msg_count > 15:
            logger.warning("[EXECUTOR] Too many execution steps, forcing finish")
            return Command(
                update={"messages": state["messages"] + [
                    HumanMessage(
                        content="FINAL ANSWER: I've reached my execution limit.",
                        name="executor"
                    )
                ]},
                goto="finish",
            )
        
        try:
            result = self.executor_agent.invoke(state)
            
            # Wrap as human message to pass to next node
            result["messages"][-1] = HumanMessage(
                content=result["messages"][-1].content,
                name="executor"
            )
            
            # Decide next step based on content
            content = result["messages"][-1].content
            next_node = "replan"
            
            if "FINAL ANSWER" in content:
                next_node = "finish"
            elif "NEED REPLAN" in content:
                next_node = "replan"
            elif not any(marker in content.lower() 
                        for marker in ["continue", "next step", "proceed"]):
                next_node = "finish"
            
            return Command(
                update={"messages": result["messages"]},
                goto=next_node,
            )
        except Exception as e:
            logger.error(f"Error in executor node: {str(e)}", exc_info=True)
            return Command(
                update={"messages": state["messages"] + [
                    HumanMessage(
                        content=f"FINAL ANSWER: Error during execution: {str(e)}",
                        name="executor"
                    )
                ]},
                goto="finish",
            )
    
    def _decide_after_planning(self, state: MessagesState) -> str:
        """Post-planning decision: execute or finish"""
        # Get the last message
        last_message = state["messages"][-1]
        content = last_message.content
        
        # Track number of planning steps to prevent infinite recursion
        planning_msg_count = sum(1 for msg in state["messages"] 
                              if hasattr(msg, 'name') and msg.name == "planner")
        if planning_msg_count > 10:
            logger.warning("[PLANNER] Too many planning steps, forcing finish")
            return "finish"
            
        # If message contains completion marker
        if "FINAL ANSWER" in content:
            return "finish"
        
        # Default to continue execution
        return "execute"
    
    def _decide_after_execution(self, state: MessagesState) -> str:
        """Post-execution decision: replan or finish"""
        # Get the last message
        last_message = state["messages"][-1]
        content = last_message.content
        
        # Track number of execution steps to prevent infinite recursion
        exec_msg_count = sum(1 for msg in state["messages"] 
                          if hasattr(msg, 'name') and msg.name == "executor")
        if exec_msg_count > 15:
            logger.warning("[EXECUTOR] Too many execution steps, forcing finish")
            return "finish"
            
        # If message contains completion marker
        if "FINAL ANSWER" in content:
            return "finish"
            
        # If replanning is explicitly requested
        if "NEED REPLAN" in content:
            return "replan"
        
        # Default to finish if no clear directive is given
        markers = ["continue", "next step", "proceed"]
        if not any(marker in content for marker in markers):
            return "finish"
            
        # Default to return to planning
        return "replan"
    
    async def ainvoke(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Asynchronously execute agent graph"""
        # Prepare initial state
        if isinstance(input_data, str):
            initial_state = self._prepare_initial_state(input_data)
        else:
            initial_state = input_data
            
        # Execute graph
        try:
            # Track start time for performance monitoring
            start_time = time.time()
            
            result = await self.graph.ainvoke(initial_state)
            
            # Log execution metrics
            execution_time = time.time() - start_time
            self.mlflow_tracker.log_metric()
            
            # Log custom metrics
            mlflow.log_metric("execution_time", execution_time)
            mlflow.log_metric("query_length", len(str(input_data)))
            
            # Log the query text
            if isinstance(input_data, str):
                mlflow.log_param("query", input_data[:250])  # Truncate if too long
                
            return self._process_result(result)
        except Exception as e:
            logger.error(f"Error executing graph: {str(e)}", exc_info=True)
            # Log error in MLflow
            mlflow.log_param("error", str(e)[:250])  # Truncate if too long
            return {
                "error": str(e),
                "messages": initial_state.get("messages", []),
                "result": "Error occurred during execution"
            }
    
    def invoke(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Synchronously execute agent graph"""
        import asyncio
        
        # Create event loop for async calls
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(self.ainvoke(input_data))
            return result
        finally:
            loop.close()
    
    def _prepare_initial_state(self, user_input: str) -> Dict[str, Any]:
        """Prepare initial state"""
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
            if "FINAL ANSWER" in last_message.content:
                # Extract content after FINAL ANSWER:
                parts = last_message.content.split("FINAL ANSWER:", 1)
                if len(parts) > 1:
                    final_answer = parts[1].strip()
        
        # Get final result text
        result_text = final_answer or "Execution completed but no clear final result found"
        
        # Log result metrics
        # Count total tokens used from messages
        msg_count = len(messages)
        mlflow.log_metric("message_count", msg_count)
        mlflow.log_metric("result_length", len(str(result_text)))
        
        # Calculate token estimates if not properly tracked
        if self.mlflow_tracker.total_tokens == 0:
            # Estimate tokens: ~4 chars per token for English text
            estimated_input_tokens = sum(len(str(msg.content)) // 4 
                                        for msg in messages if hasattr(msg, 'content'))
            estimated_output_tokens = len(str(result_text)) // 4
            
            # Log estimated token counts
            mlflow.log_metrics({
                "estimated_input_tokens": estimated_input_tokens,
                "estimated_output_tokens": estimated_output_tokens,
                "estimated_total_tokens": estimated_input_tokens + estimated_output_tokens
            })
        
        return {
            "messages": messages,
            "result": result_text,
            "raw_result": result
        }

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
        """Create executor agent"""
        # Filter for execution tools (non-planning tools)
        execution_tools = [tool for tool in self.tools if tool.name not in [
            "create_plan", "update_plan", "update_status"
        ]]
        
        # Use ReAct as executor
        return create_react_agent(
            model=self.llm,
            tools=execution_tools,
            prompt=make_system_prompt(prompt_templates.EXECUTOR_PROMPT)
        )


def run(query: str) -> Dict[str, Any]:
    """Run agent graph and get results"""
    agent_graph = AgentGraph()
    
    try:
        result = agent_graph.invoke(query)
        
        # Print final result
        if "result" in result:
            print(f"\nFinal Response:\n{result['result']}")
            
        # Get usage report
        usage_report = agent_graph.mlflow_tracker.get_usage_report()
        mlflow.log_metrics({
            "total_steps": len(result.get("messages", [])),
            "plan_completion": 1.0  # Indicates plan was completed
        })
        
        # End MLflow run
        mlflow.end_run()
        
        return result
    except Exception as e:
        # End MLflow run in case of error
        mlflow.end_run()
        logger.error(f"Error in agent run: {str(e)}")
        return {"error": str(e), "result": "Error during execution"} 