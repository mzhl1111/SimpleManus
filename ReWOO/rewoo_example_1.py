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
# Load environment variables
load_dotenv()

# Define input and state types
class ReWOOInput(TypedDict):
    """Input for the ReWOO agent."""
    task: str
    messages: List[Union[HumanMessage, AIMessage]]  # Add this field

class ReWOOState(TypedDict):
    """State for the ReWOO agent."""
    messages: List[Union[HumanMessage, AIMessage]]  # Track conversation
    task: str
    plan_string: str
    steps: List[Tuple[str, str, str, str]]
    results: Dict[str, Any]
    result: str

# Initialize tools and model as before
model = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base=os.getenv("OPENROUTER_BASE_URL"),
    model_name="google/gemini-2.0-flash-001"
)

search = TavilySearchResults(include_answer=True, tavily_api_key=os.getenv("TAVILY_API_KEY"))

# Planner prompt
prompt = """For the following task, make plans that can solve the problem step by step. For each plan, indicate \
which external tool together with tool input to retrieve evidence. You can store the evidence into a \
variable #E that can be called by later tools. (Plan, #E1, Plan, #E2, Plan, ...)

Tools can be one of the following:
(1) Google[input]: Worker that searches results from Google. Useful when you need to find short
and succinct answers about a specific topic. The input should be a search query.
(2) LLM[input]: A pretrained LLM like yourself. Useful when you need to act with general
world knowledge and common sense. Prioritize it when you are confident in solving the problem
yourself. Input can be any instruction.

For example,
Task: Thomas, Toby, and Rebecca worked a total of 157 hours in one week. Thomas worked x
hours. Toby worked 10 hours less than twice what Thomas worked, and Rebecca worked 8 hours
less than Toby. How many hours did Rebecca work?
Plan: Given Thomas worked x hours, translate the problem into algebraic expressions and solve
with Wolfram Alpha. #E1 = WolframAlpha[Solve x + (2x − 10) + ((2x − 10) − 8) = 157]
Plan: Find out the number of hours Thomas worked. #E2 = LLM[What is x, given #E1]
Plan: Calculate the number of hours Rebecca worked. #E3 = Calculator[(2 ∗ #E2 − 10) − 8]

Begin! 
Describe your plans with rich details. Each Plan should be followed by only one #E.

Task: {task}"""

# Regex pattern for parsing plans
regex_pattern = r"Plan:\s*(.+)\s*(#E\d+)\s*=\s*(\w+)\s*\[([^\]]+)\]"

async def get_plan(state: ReWOOState) -> Dict:
    """Generate a plan for the given task."""
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
    """Determine which task to execute next."""
    if "results" not in state or state["results"] is None:
        return 1
    if len(state["results"]) == len(state["steps"]):
        return None
    else:
        return len(state["results"]) + 1

def extract_content(result):
    """Extract meaningful content from search results."""
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
    """Execute the current tool in the plan."""
    _step = _get_current_task(state)
    _, step_name, tool, tool_input = state["steps"][_step - 1]
    _results = (state["results"] or {}) if "results" in state else {}
    
    # Handle variable substitution
    variables = re.findall(r'#E\d+', tool_input)
    for var in variables:
        if var in _results:
            # Extract content from previous results
            raw_content = extract_content(_results[var])
            
            # Use LLM to extract relevant entity from previous result
            extract_prompt = f"""Extract the specific entity or value needed for this context.
Previous tool result: {raw_content}
Current tool input: {tool_input}
What specific value from the previous result should replace {var}? Respond ONLY with the value, nothing else."""
            
            try:
                extracted_value = model.invoke(extract_prompt).content
                # Clean up LLM response
                extracted_value = extracted_value.strip('"').split("\n")[0]
            except:
                extracted_value = raw_content  # Fallback to raw content
                
            # Replace the variable in the tool input
            tool_input = tool_input.replace(var, extracted_value)
    
    # Format the search query properly
    if tool == "Google":
        # Use tool call invocation to get both search results and answer
        tool_response = search.invoke({
            "args": {'query': tool_input},
            "type": "tool_call",
            "id": f"search_{_step}",
            "name": "tavily"
        })
        
        # Store both the search results and answer in the results
        if hasattr(tool_response, 'artifact'):
            artifact = tool_response.artifact
            _results[step_name] = {
                'search_results': artifact.get('results', []),
                'answer': artifact.get('answer', ''),
                'raw_response': str(tool_response)
            }
        else:
            _results[step_name] = str(tool_response)
            
        print("========Search tool invoked=========")
        print(f"Answer: {_results[step_name].get('answer', 'No answer found')}")
        print(f"Number of search results: {len(_results[step_name].get('search_results', []))}")
        
    elif tool == "LLM":
        result = model.invoke(tool_input)
        _results[step_name] = str(result)
    else:
        raise ValueError(f"Unknown tool: {tool}")
    
    # Add tool execution results to conversation
    return {
        "messages": state["messages"] + [AIMessage(content=str(_results))],
        "results": _results
    }

async def solve(state: ReWOOState) -> Dict:
    """Generate the final answer based on all evidence."""
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
    """Route to next state based on current progress."""
    _step = _get_current_task(state)
    if _step is None:
        return "solve"
    return "tool"

# Create the graph with proper builder pattern
builder = StateGraph(ReWOOState, input=ReWOOInput)

# Add nodes with state initialization
def initialize_state(input_data: ReWOOInput) -> ReWOOState:
    """Initialize the state with all required fields."""
    return {
        "task": input_data["task"],
        "messages": [],
        "plan_string": "",
        "steps": [],
        "results": {},
        "result": ""
    }

# Add nodes
builder.add_node("initialize", initialize_state)  # Add initialize as a named node
builder.add_node("plan", get_plan)
builder.add_node("tool", tool_execution)
builder.add_node("solve", solve)

# Add edges - correct way
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

# Compile the graph
graph = builder.compile()
graph.name = "ReWOO Agent"

# For the langgraph dev command
app = graph

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize with empty messages list
        initial_state = {
            "task": "what is the hometown of current canadian prime minister",
            "messages": []  # Initialize empty messages list
        }
        async for s in app.astream(initial_state):
            print(s)
            print("---")
        print(s["solve"]["result"])
    
    asyncio.run(main()) 