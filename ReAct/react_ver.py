import asyncio

from dotenv import load_dotenv

from utils.chat_open_router import ChatOpenRouter

# Load environment variables from .env file
load_dotenv()

from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph.message import add_messages
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage
from tools.browser_use_tool import get_browser_use_tools
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from utils.mlflow_callback import MLflowTracker

class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)
llm = ChatOpenRouter(model_name="openai/gpt-4o-mini-2024-07-18")
tools = get_browser_use_tools(llm)
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

# System prompt
sys_msg = SystemMessage(content="You are a helpful assistant interact with browser to finish tasks up on user requests."
                                "If you cannot find wanted information in current web page for few steps, please try "
                                "other links in the search results or redo web searching with new query")

# Node
def assistant(state: MessagesState):
   return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

# Graph
builder = StateGraph(MessagesState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")
react_graph = builder.compile()

# Show
# display(Image(react_graph.get_graph(xray=True).draw_mermaid_png()))


async def main():
    inputs = {
        "messages": [
            (
                "user",
                "help me get the source code of create_react_agent function definition in langgraph package",
            )
        ],
    }
    handler = MLflowTracker(experiment_name="react_ver")
    messages = await react_graph.ainvoke(inputs, {"callbacks": [handler], "recursion_limit": 100})
    handler.log_metric()
    for m in messages['messages']:
        m.pretty_print()

if __name__ == "__main__":
    asyncio.run(main())

