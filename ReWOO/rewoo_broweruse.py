"""
ReWOO (Reinforcement Learning with Web Operations) Agent Implementation
This module implements an agent that can perform web-based tasks using a combination of LLM reasoning,
web search, and browser automation tools.
"""

from typing import Dict, List, Literal, TypedDict, Tuple, Any
from langchain_openai import ChatOpenAI
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph, START
import re
import json
import os
from dotenv import load_dotenv
from typing import Union
from .tools.browser_use_tool import BrowserUseTool

# Load environment variables for API keys and configurations
load_dotenv()

# Type definitions for the agent's input and state management
class ReWOOInput(TypedDict):
    """Input structure for the ReWOO agent.
    Contains the task description and conversation history."""
    task: str
    messages: List[Union[HumanMessage, AIMessage]]  # Conversation history

class ReWOOState(TypedDict):
    """State management for the ReWOO agent.
    Tracks the current state of task execution, including conversation, plans, and results."""
    messages: List[Union[HumanMessage, AIMessage]]  # Conversation history
    task: str  # Current task description
    plan_string: str  # Generated plan in string format
    steps: List[Tuple[str, str, str, str]]  # List of execution steps
    results: Dict[str, Any]  # Results from each step
    result: str  # Final result

# Initialize core components
# Set up the LLM model using OpenAI API through OpenRouter
model = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
    model_name="google/gemini-2.0-flash-001"
)

# Initialize Tavily search tool for web searches
search = TavilySearchResults(include_answer=True, tavily_api_key=os.getenv("TAVILY_API_KEY"))

# Initialize browser automation tool for web interactions
browser_tool = BrowserUseTool(llm=model)

# Define the planning prompt template that guides the agent's decision-making process
prompt = """For the following task, make plans that can solve the problem step by step. For each plan, indicate \
which external tool together with tool input to retrieve evidence. You can store the evidence into a \
variable #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...)

**IMPORTANT:** You **MUST** output each plan step on a **SINGLE LINE** in the format:
`Plan: <description> #E<step_number> = <ToolName>[<ToolInput>]`
Do **NOT** use multi-line plans or add extra formatting like "Plan 1:".

Tools can be one of the following:
(1) Google[input]: Worker that searches results from Google. Useful when you need to find short
and succinct answers about a specific topic. The input should be a search query.
(2) LLM[input]: A pretrained LLM like yourself. Useful when you need to act with general
world knowledge and common sense. Prioritize it when you are confident in solving the problem
yourself. Input can be any instruction.
(3) GoToURL[url]: Navigates the browser to the specified URL. Input should be a valid URL. Don't include any other text.
(4) GetCurrentState[placeholder]: Retrieves the current state of the browser, including URL, title, tabs, and interactive elements with their indices. Input should be "None" or any placeholder.
(5) ClickElement[index]: Clicks the element identified by the given index (obtained from GetCurrentState).
(6) ExtractContent[goal]: Extracts specific information from the current webpage based on the provided goal.


For example,
Task: What is the name of the director of the highest-grossing movie released in 2023? Go to that director's Wikipedia page and extract the first sentence of their biography.
Plan: Find the highest-grossing movie released in 2023. #E1 = Google[highest grossing movie 2023]
Plan: Identify the director of the movie found in #E1. #E2 = Google[director of #E1]
Plan: Find the Wikipedia page URL for the director identified in #E2. #E3 = Google[#E2 Wikipedia page URL]
Plan: Navigate to the director's Wikipedia page using the URL from #E3. #E4 = GoToURL[#E3]
Plan: Extract the first sentence of the biography from the current page. #E5 = ExtractContent[first sentence of biography]

Begin! 
Describe your plans with rich details. Ensure **EACH STEP** is on a **SINGLE LINE** matching the specified format.

Task: {task}"""

# Regular expression pattern for parsing the generated plans
regex_pattern = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"

async def get_plan(state: ReWOOState) -> Dict:
    """Generate a plan for the given task.
    This function uses the LLM to create a step-by-step plan for executing the task."""
    task = state["task"]
    prompt_template = ChatPromptTemplate.from_messages([("user", prompt)])
    planner = prompt_template | model
    result = await planner.ainvoke({"task": task})
    matches = re.findall(regex_pattern, result.content)
    
    # Initialize messages if not present
    current_messages = state.get("messages", [])
    
    return {
        "messages": current_messages + [AIMessage(content=result.content)],
        "steps": matches,
        "plan_string": result.content
    }

def _get_current_task(state: ReWOOState):
    """Determine which task to execute next in the plan.
    Returns the index of the next task or None if all tasks are completed."""
    if "results" not in state or state["results"] is None:
        return 1
    if len(state["results"]) == len(state["steps"]):
        return None
    else:
        return len(state["results"]) + 1

def extract_content(result):
    """Extract meaningful content from search results.
    Handles both JSON and string formats of search results."""
    if isinstance(result, str):
        try:
            # Try to parse as JSON
            parsed = json.loads(result)
            if isinstance(parsed, list) and len(parsed) > 0:
                # Get content from first result
                return parsed[0].get('content', '')
        except:
            # If not JSON, return as is
            return result
    return str(result)

async def tool_execution(state: ReWOOState) -> Dict:
    """Execute the current tool in the plan.
    This function handles the execution of various tools (Google, LLM, Browser operations)
    and manages the results and state updates."""
    _step = _get_current_task(state)
    _, step_name, tool, original_tool_input = state["steps"][_step - 1] # Renamed to original_tool_input
    _results = (state["results"] or {}) if "results" in state else {}
    
    tool_input = original_tool_input # Start with original input for this execution
    
    # --- Prepare Input using Variable Substitution (with targeted LLM cleaning) ---
    variables = re.findall(r'#E\d+', tool_input)
    
    # Check if the *entire* input is just a variable reference (special handling for GoToURL)
    is_direct_variable_reference = re.match(r"^#E(\d+)$", tool_input.strip())
    
    if is_direct_variable_reference and tool == "GoToURL":
        # Special case: GoToURL with direct variable reference - try using first_url
        prev_step_num = int(is_direct_variable_reference.group(1))
        prev_step_name = f"#E{prev_step_num}"
        if prev_step_name in _results:
            previous_result_data = _results[prev_step_name]
            if isinstance(previous_result_data, dict) and previous_result_data.get('first_url'):
                tool_input = previous_result_data['first_url']
                print(f"INFO: Using first URL from {prev_step_name} results for GoToURL: {tool_input}")
            else:
                # Fallback to substituting the answer/string if no first_url
                print(f"WARN: GoToURL expects URL from {prev_step_name}, but no 'first_url' found. Falling back to substituting answer/string.")
                value_to_substitute = str(previous_result_data.get('answer', previous_result_data)).strip('"' + "'")
                tool_input = value_to_substitute
        else:
             print(f"WARN: Variable {prev_step_name} referenced for GoToURL but not found in results.")
             # Keep original input if variable not found
             tool_input = original_tool_input
    else:
        # General case: Substitute variables within the tool_input string
        temp_input = tool_input # Work on a temporary string
        for var in variables:
            if var in _results:
                previous_result = _results[var]
                value_to_substitute = ""
                
                # *** Targeted LLM Cleaning ***
                # Check if previous result is complex (e.g., from Tavily) and needs cleaning for the current context
                if isinstance(previous_result, dict) and 'answer' in previous_result and len(previous_result['answer']) > 50: # Arbitrary length check for complexity
                    raw_answer = previous_result['answer']
                    # Construct prompt for LLM to extract relevant part
                    extract_prompt = f"""Given the previous result and the context of how it will be used, extract ONLY the specific entity or value needed to replace the variable '{var}'.

Previous Result Answer: {raw_answer}

Current Tool Input Template: {original_tool_input}

Variable to Replace: {var}

What specific, concise value from the 'Previous Result Answer' should replace '{var}' in the 'Current Tool Input Template'? Respond ONLY with the extracted value, nothing else."""
                    
                    try:
                        extracted_value = model.invoke(extract_prompt).content.strip()
                        if extracted_value: # Use LLM result only if it's not empty
                             value_to_substitute = extracted_value.strip('"' + "'")
                             print(f"INFO: LLM extracted '{value_to_substitute}' from {var} for template '{original_tool_input}'")
                        else:
                            print(f"WARN: LLM extraction for {var} returned empty. Falling back to raw answer.")
                            value_to_substitute = raw_answer.strip('"' + "'")
                    except Exception as e:
                        print(f"WARN: LLM extraction for {var} failed ({e}). Falling back to raw answer.")
                        value_to_substitute = raw_answer.strip('"' + "'")
                else:
                    # If not complex or no answer field, use string representation directly
                    value_to_substitute = str(previous_result.get('answer', previous_result)).strip('"' + "'")
                
                # Perform substitution on the temporary string
                temp_input = temp_input.replace(var, value_to_substitute)
                
        tool_input = temp_input # Assign the processed string back
        
    # --- Execute the Tool --- 
    executed_tool = False # Flag to track execution
    
    if tool == "Google":
        # Use tool call invocation to get both search results and answer
        tool_response = search.invoke({
            "args": {'query': tool_input},
            "type": "tool_call",
            "id": f"search_{_step}",
            "name": "tavily"
        })
        
        # Store detailed results including the first URL if available
        if hasattr(tool_response, 'artifact'):
            artifact = tool_response.artifact
            search_results_list = artifact.get('results', [])
            first_url = search_results_list[0].get('url') if search_results_list and isinstance(search_results_list[0], dict) else None
            
            _results[step_name] = {
                'search_results': search_results_list,
                'answer': artifact.get('answer', ''),
                'first_url': first_url, # Store the first URL
                'raw_response': str(tool_response)
            }
        else:
            _results[step_name] = str(tool_response)
            
        print("========Search tool invoked=========")
        print(f"Answer: {_results[step_name].get('answer', 'No answer found')}")
        print(f"Number of search results: {len(_results[step_name].get('search_results', []))}")
        if _results[step_name].get('first_url'):
             print(f"First URL: {_results[step_name]['first_url']}")
        executed_tool = True
        
    elif tool == "LLM":
        result = model.invoke(tool_input)
        _results[step_name] = str(result)
        executed_tool = True
        
    # --- Browser Tool Handling ---
    elif tool == "GoToURL":
        # Input preparation already handled above specifically for GoToURL
        result = await browser_tool.go_to_url(tool_input)
        _results[step_name] = result
        print(f"========Browser Tool: GoToURL invoked=========")
        print(f"Navigating to: {tool_input}") # Print the actual URL used
        print(f"Result: {result}")
        executed_tool = True
        
    elif tool == "GetCurrentState":
        result, = await browser_tool.get_current_state(placeholder="None")  # Unpack the tuple
        _results[step_name] = result
        print(f"========Browser Tool: GetCurrentState invoked=========")
        # Print only a summary to avoid large output
        try:
            state_data = json.loads(result)
            print(f"Current URL: {state_data.get('url')}")
            print(f"Current Title: {state_data.get('title')}")
            print(f"Interactive Elements Count: {len(state_data.get('interactive_elements', '').splitlines())}")
        except Exception as e:
            print(f"Could not parse state: {e}")
            print(f"Raw Result: {result[:500]}...")
        executed_tool = True
        
    elif tool == "ClickElement":
        try:
            index = int(tool_input)
            result = await browser_tool.click_element(index)
            _results[step_name] = result
            print(f"========Browser Tool: ClickElement invoked=========")
            print(f"Result: {result}")
        except ValueError:
             _results[step_name] = f"Error: Invalid index '{tool_input}' for ClickElement. Index must be an integer."
        except Exception as e:
             _results[step_name] = f"Error during ClickElement: {str(e)}"
        executed_tool = True
             
    elif tool == "ExtractContent":
        result = await browser_tool.extract_content(tool_input)
        _results[step_name] = result
        print(f"========Browser Tool: ExtractContent invoked=========")
        print(f"Result: {result[:500]}...") # Print only the start of potentially long content
        executed_tool = True
        
    # --- End Browser Tool Handling ---
    
    if not executed_tool:
        raise ValueError(f"Unknown tool: {tool}")
    
    # Add tool execution results to conversation
    return {
        "messages": state["messages"] + [AIMessage(content=str(_results))],
        "results": _results
    }

async def solve(state: ReWOOState) -> Dict:
    """Generate the final answer based on all evidence collected during task execution.
    This function uses the LLM to synthesize all the collected information into a final answer."""
    plan = ""
    evidence = []
    
    for _plan, step_name, tool, tool_input in state["steps"]:
        # Get the results for this step
        _results = (state["results"] or {}) if "results" in state else {}
        result = _results.get(step_name, "")
        
        # Handle both search results and answers
        if isinstance(result, dict) and 'search_results' in result:
            # For search results, include both the answer and search results
            answer = result.get('answer', '')
            search_results = result.get('search_results', [])
            
            # Format search results content
            search_content = "\n".join(
                f"Source {i+1}: {item.get('content', '')}"
                for i, item in enumerate(search_results[:2])
            )
            
            # Combine answer and search results
            content = f"Answer: {answer}\n\nSearch Results:\n{search_content}"
        else:
            # For LLM results or other types, use as is
            content = str(result)
        
        # Add to the plan with clean evidence
        plan += f"Plan: {_plan}\n"
        plan += f"Evidence: {content}\n\n"
    
    solve_prompt = """Solve the following task or problem. To solve the problem, we have made step-by-step Plan and \
retrieved corresponding Evidence to each Plan. Use them with caution since long evidence might \
contain irrelevant information.

{plan}

Now solve the question or task according to provided Evidence above. Respond with the answer
directly with no extra words.

Task: {task}
Response:"""
    
    prompt = solve_prompt.format(plan=plan, task=state["task"])
    result = model.invoke(prompt)
    
    # Add final answer to conversation
    return {
        "messages": state["messages"] + [AIMessage(content=result.content)],
        "result": result.content
    }

def route_state(state: ReWOOState) -> Literal["solve", "tool"]:
    """Route to next state based on current progress.
    Determines whether to continue with tool execution or move to final solving phase."""
    _step = _get_current_task(state)
    if _step is None:
        return "solve"
    return "tool"

# Create the graph with proper builder pattern
builder = StateGraph(ReWOOState, input=ReWOOInput)

# Add nodes with state initialization
def initialize_state(input_data: ReWOOInput) -> ReWOOState:
    """Initialize the state with all required fields.
    Sets up the initial state for the agent's execution."""
    return {
        "task": input_data["task"],
        "messages": [],
        "plan_string": "",
        "steps": [],
        "results": {},
        "result": ""
    }

# Add nodes to the execution graph
builder.add_node("initialize", initialize_state)  # Add initialize as a named node
builder.add_node("plan", get_plan)
builder.add_node("tool", tool_execution)
builder.add_node("solve", solve)

# Define the execution flow through the graph
builder.add_edge(START, "initialize")
builder.add_edge("initialize", "plan")
builder.add_edge("plan", "tool")
builder.add_conditional_edges(
    "tool",
    route_state,
    {
        "tool": "tool",
        "solve": "solve"
    }
)
builder.add_edge("solve", END)

# Compile the graph into an executable application
graph = builder.compile()
graph.name = "ReWOO Agent"

# For the langgraph dev command
app = graph

# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize with empty messages list
        initial_state = {
            "task": "What is the name of the director of the highest-grossing movie released in 2023? Go to that director's Wikipedia page and briefly describe their biography.",
            "messages": []  # Initialize empty messages list
        }
        final_state = None  # To store the last state
        try:
            async for s in app.astream(initial_state):
                print(s)
                print("---")
                final_state = s # Keep track of the latest state
            # Access the final result safely after the loop
            if final_state and "solve" in final_state and "result" in final_state["solve"]:
                 print(f"Final Result: {final_state['solve']['result']}")
            else:
                 print("Could not determine the final result.")
        finally:
            # --- Add Cleanup ---
            print("Cleaning up browser resources...")
            await browser_tool.cleanup()
            print("Cleanup complete.")
            # --- End Cleanup ---
    
    asyncio.run(main()) 