import asyncio

from dotenv import load_dotenv
from langchain_core.tools import Tool

from tools import BrowserUseTool, search_function
from tools.planning_tool import PlanningTool
from utils.chat_open_router import ChatOpenRouter
from typing import Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import MessagesState, END
from langgraph.types import Command

load_dotenv()


def make_system_prompt(suffix: str) -> str:
    return f"""You are a helpful AI assistant collaborating with other assistants.
Use the provided tools to progress toward finishing up the task.
If you are unable to fully answer, that's OK—another assistant with different tools will help where you left off.
Execute what you can to make progress.
Once you or any team member have the final answer or deliverable, prefix your response with "FINAL ANSWER" so the team knows to stop.
{suffix}"""


def get_next_node(last_message: BaseMessage, goto: str):
    if "FINISH" in last_message.content:
        # Any agent decided the work is done
        return END
    return goto


llm = ChatOpenRouter(model_name="openai/gpt-4o-mini-2024-07-18")

planning_tool = PlanningTool()

# Create Langchain tools
planning_tools = [
    Tool(
        name="create_plan",
        func=planning_tool.create_plan,
        description="Create a new plan with a list of steps. **Required:** steps (list of strings)"
    ),
    Tool(
        name="update_plan",
        func=planning_tool.update_plan,
        description="Update an existing plan with a new list of steps. **Required:** steps (list of strings)"
    ),
    Tool(
        name="update_status",
        func=planning_tool.update_status,
        description="Update the status of plan steps. **Required:** statuses (list of 'not_started', 'in_progress', or 'completed')"
    )
]
# Research agent and node
planning_agent = create_react_agent(
    llm,
    tools=planning_tools,
    prompt=make_system_prompt("""You are an expert Planning Agent tasked with solving problems efficiently through structured plans.
Your job is:
1. Analyze requests to understand the task scope
2. Create a clear, actionable plan that makes meaningful progress with the `planning` tools
3. Track and update progress of current planning step with the `planning` tools
5. Update current plans when necessary with the `planning` tools
4. Use `FINISH` to conclude immediately when the task is complete"""
                              ),
)


def planning_node(
        state: MessagesState,
) -> Command[Literal["browser", END]]:
    result = planning_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "browser")
    # wrap in a human message, as not all providers allow
    # AI message at the last position of the input messages list
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="planner"
    )
    return Command(
        update={
            # share internal message history of research agent with other agents
            "messages": result["messages"],
        },
        goto=goto,
    )


browser_tool = BrowserUseTool(llm=llm)

browser_tools = [
    Tool(
        name="Search",
        func=search_function,
        description="Useful for searching the internet for current information."
    ),
    # Tool(
    #     name="Wait",
    #     func=wait,
    #     content="Wait for {input} seconds for other executing actions"
    #
    Tool(
        name="go_to_url",
        func=browser_tool.a_go_to_url,
        coroutine=browser_tool.a_go_to_url,
        description="Navigate to a specific URL in the browser. **Required:** url (string)."
    ),
    Tool(
        name="go_back",
        func=browser_tool.a_go_back,
        coroutine=browser_tool.a_go_back,
        description="Navigate back to the previous page. **No additional parameters required.**"
    ),
    Tool(
        name="click_element",
        func=browser_tool.a_click_element,
        coroutine=browser_tool.a_click_element,
        description="Click an element in the webpage identified by its index. **Required:** index (integer)."
    ),
    Tool(
        name="scroll_down",
        func=browser_tool.a_scroll_down,
        coroutine=browser_tool.a_scroll_down,
        description="Scroll down by a specified amount of pixels. **Required:** scroll_amount (integer)."
    ),
    Tool(
        name="scroll_up",
        func=browser_tool.a_scroll_up,
        coroutine=browser_tool.a_scroll_up,
        description="Scroll up by a specified amount of pixels. **Required:** scroll_amount (integer)."
    ),
    Tool(
        name="scroll_to_text",
        func=browser_tool.a_scroll_to_text,
        coroutine=browser_tool.a_scroll_to_text,
        description="Scroll to a specific text on the page. **Required:** text (string)."
    ),
    Tool(
        name="get_dropdown_options",
        func=browser_tool.a_get_dropdown_options,
        coroutine=browser_tool.a_get_dropdown_options,
        description="Retrieve all dropdown options for a given element. **Required:** index (integer)."
    ),
    Tool(
        name="select_dropdown_option",
        func=browser_tool.a_select_dropdown_option,
        coroutine=browser_tool.a_select_dropdown_option,
        description="Select a dropdown option based on visible text. **Required:** index (integer), text (string)."
    ),
    Tool(
        name="extract_content",
        func=browser_tool.a_extract_content,
        coroutine=browser_tool.a_extract_content,
        description="Extract specific information from a webpage. **Required:** goal (string, describing what to extract)."
    ),
    Tool(
        name="get_current_state",
        func=browser_tool.a_get_current_state,
        coroutine=browser_tool.a_get_current_state,
        description="Retrieve the current browser state. ALWAYS use this tool to get dom element index before using click or dropdown tools. **No additional parameters required.**"
    ),
    Tool(
        name="switch_tab",
        func=browser_tool.a_switch_tab,
        coroutine=browser_tool.a_switch_tab,
        description="Switch to a specific tab using its ID. **Required:** tab_id (integer)."
    ),
    Tool(
        name="open_tab",
        func=browser_tool.a_open_tab,
        coroutine=browser_tool.a_open_tab,
        description="Open a new tab with a given URL. **Required:** url (string)."
    ),
    Tool(
        name="close_tab",
        func=browser_tool.a_close_tab,
        coroutine=browser_tool.a_close_tab,
        description="Close the current tab. **No additional parameters required.**"
    ),
    Tool(
        name="wait",
        func=browser_tool.a_wait,
        coroutine=browser_tool.a_wait,
        description="Pause execution for a specified duration. **Required:** seconds (integer)."
    )
]

browser_agent = create_react_agent(
    llm,
    browser_tools,
    prompt=make_system_prompt(
        "You are an AI agent designed to automate browser tasks. Your goal is to accomplish the ultimate task"
    ),
)


def browser_node(state: MessagesState) -> Command[Literal["planner", END]]:
    result = browser_agent.invoke(state)
    goto = get_next_node(result["messages"][-1], "planner")
    # wrap in a human message, as not all providers allow
    # AI message at the last position of the input messages list
    result["messages"][-1] = HumanMessage(
        content=result["messages"][-1].content, name="browser"
    )
    return Command(
        update={
            # share internal message history of chart agent with other agents
            "messages": result["messages"],
        },
        goto=goto,
    )

from langgraph.graph import StateGraph, START

workflow = StateGraph(MessagesState)
workflow.add_node("planner", planning_node)
workflow.add_node("browser", browser_node)

workflow.add_edge(START, "planner")
graph = workflow.compile()

async def main():
    inputs = {
        "messages": [
            (
                "user",
                "help me get the source code of create_react_agent function in langgraph",
            )
        ],
    }
    await graph.ainvoke(inputs)

if __name__ == "__main__":
    asyncio.run(main())
