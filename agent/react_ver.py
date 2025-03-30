from typing import List
from langchain.agents import Tool
from langchain.llms import openai
from utils import ChatOpenRouter
import asyncio
from googlesearch import search
import asyncio

from browser_use import Browser as BrowserUseBrowser
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig


from tools.browser_use_tool import BrowserUseTool, search_function

llm = ChatOpenRouter(model_name="openai/gpt-4o-mini-2024-07-18")
browser_tool = BrowserUseTool(llm=llm)

tools = [
    Tool(
        name="Search",
        func=search_function,
        description="Useful for searching the internet for current information."
    ),
    Tool(
        name="go_to_url",
        func=browser_tool.go_to_url,
        coroutine=browser_tool.a_go_to_url,
        description="Navigate to a specific URL in the browser. **Required:** url (string)."
    ),
    Tool(
        name="go_back",
        func=browser_tool.go_back,
        coroutine=browser_tool.a_go_back,
        description="Navigate back to the previous page. **No additional parameters required.**"
    ),
    Tool(
        name="click_element",
        func=browser_tool.click_element,
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
        func=browser_tool.scroll_up,
        coroutine=browser_tool.a_scroll_up,
        description="Scroll up by a specified amount of pixels. **Required:** scroll_amount (integer)."
    ),
    Tool(
        name="scroll_to_text",
        func=browser_tool.scroll_to_text,
        coroutine=browser_tool.a_scroll_to_text,
        description="Scroll to a specific text on the page. **Required:** text (string)."
    ),
    Tool(
        name="get_dropdown_options",
        func=browser_tool.get_dropdown_options,
        coroutine=browser_tool.a_get_dropdown_options,
        description="Retrieve all dropdown options for a given element. **Required:** index (integer)."
    ),
    Tool(
        name="select_dropdown_option",
        func=browser_tool.select_dropdown_option,
        coroutine=browser_tool.a_select_dropdown_option,
        description="Select a dropdown option based on visible text. **Required:** index (integer), text (string)."
    ),
    Tool(
        name="extract_content",
        func=browser_tool.extract_content,
        coroutine=browser_tool.a_extract_content,
        description="Extract specific information from a webpage. **Required:** goal (string, describing what to extract)."
    ),
    Tool(
        name="get_current_state",
        func=browser_tool.get_current_state,
        coroutine=browser_tool.a_get_current_state,
        description="Retrieve the current browser state. ALWAYS use this tool to get dom element index before using click or dropdown tools. **No additional parameters required.**"
    ),
    Tool(
        name="switch_tab",
        func=browser_tool.switch_tab,
        coroutine=browser_tool.a_switch_tab,
        description="Switch to a specific tab using its ID. **Required:** tab_id (integer)."
    ),
    Tool(
        name="open_tab",
        func=browser_tool.open_tab,
        coroutine=browser_tool.a_open_tab,
        description="Open a new tab with a given URL. **Required:** url (string)."
    ),
    Tool(
        name="close_tab",
        func=browser_tool.close_tab,
        coroutine=browser_tool.a_close_tab,
        description="Close the current tab. **No additional parameters required.**"
    ),
    Tool(
        name="wait",
        func=browser_tool.wait,
        coroutine=browser_tool.a_wait,
        description="Pause execution for a specified duration. **Required:** seconds (integer)."
    )
]

from langchain.agents import initialize_agent, AgentType

# Create an agent that uses the Zero Shot ReAct prompt.
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,          # Enables detailed logging of the internal chain-of-thought.
    max_iterations=20      # Limits the number of thought/tool-calling cycles.
)


if __name__ == "__main__":
    task = "help me get the source code of create_react_agent function in langgraph"
    result = asyncio.run(agent.arun(task))
    print("Final result:", result)
