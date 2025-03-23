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


def search_function(query: str, num_results: int = 10) -> str:
    """
    Execute a Google search and return a list of URLs.

    Args:
        query (str): The search query to submit to Google.
        num_results (int, optional): The number of search results to return. Default is 10.

    Returns:
        List[str]: A list of URLs matching the search query.
    """
    # Run the search in a thread pool to prevent blocking
    async def fetch_results():
        return list(search(query, num_results=num_results))

    links =  asyncio.run(fetch_results())
    return f"Searched result links: {', '.join(links)}"

from tools.browser_use_tool import BrowserUseTool

llm = ChatOpenRouter(model_name="openai/gpt-4o-mini-2024-07-18")
browser_tool = BrowserUseTool(llm=llm)

tools = [
    Tool(
        name="Search",
        func=search_function,
        description="Useful for searching the internet for current information."
    ),
    # Tool(
    #     name="Wait",
    #     func=wait,
    #     description="Wait for {input} seconds for other executing actions"
    # 
    Tool(
        name="go_to_url",
        func=browser_tool.go_to_url,
        coroutine=browser_tool.go_to_url,
        description="Navigate to a specific URL in the browser. **Required:** url (string)."
    ),
    Tool(
        name="go_back",
        func=browser_tool.go_back,
        coroutine=browser_tool.go_back,
        description="Navigate back to the previous page. **No additional parameters required.**"
    ),
    Tool(
        name="click_element",
        func=browser_tool.click_element,
        coroutine=browser_tool.click_element,
        description="Click an element in the webpage identified by its index. **Required:** index (integer)."
    ),
    Tool(
        name="scroll_down",
        func=browser_tool.scroll_down,
        coroutine=browser_tool.scroll_down,
        description="Scroll down by a specified amount of pixels. **Required:** scroll_amount (integer)."
    ),
    Tool(
        name="scroll_up",
        func=browser_tool.scroll_up,
        coroutine=browser_tool.scroll_up,
        description="Scroll up by a specified amount of pixels. **Required:** scroll_amount (integer)."
    ),
    Tool(
        name="scroll_to_text",
        func=browser_tool.scroll_to_text,
        coroutine=browser_tool.scroll_to_text,
        description="Scroll to a specific text on the page. **Required:** text (string)."
    ),
    Tool(
        name="get_dropdown_options",
        func=browser_tool.get_dropdown_options,
        coroutine=browser_tool.get_dropdown_options,
        description="Retrieve all dropdown options for a given element. **Required:** index (integer)."
    ),
    Tool(
        name="select_dropdown_option",
        func=browser_tool.select_dropdown_option,
        coroutine=browser_tool.select_dropdown_option,
        description="Select a dropdown option based on visible text. **Required:** index (integer), text (string)."
    ),
    Tool(
        name="extract_content",
        func=browser_tool.extract_content,
        coroutine=browser_tool.extract_content,
        description="Extract specific information from a webpage. **Required:** goal (string, describing what to extract)."
    ),
    Tool(
        name="get_current_state",
        func=browser_tool.get_current_state,
        coroutine=browser_tool.get_current_state,
        description="Retrieve the current browser state. ALWAYS use this tool to get dom element index before using click or dropdown tools. **No additional parameters required.**"
    ),
    Tool(
        name="switch_tab",
        func=browser_tool.switch_tab,
        coroutine=browser_tool.switch_tab,
        description="Switch to a specific tab using its ID. **Required:** tab_id (integer)."
    ),
    Tool(
        name="open_tab",
        func=browser_tool.open_tab,
        coroutine=browser_tool.open_tab,
        description="Open a new tab with a given URL. **Required:** url (string)."
    ),
    Tool(
        name="close_tab",
        func=browser_tool.close_tab,
        coroutine=browser_tool.close_tab,
        description="Close the current tab. **No additional parameters required.**"
    ),
    Tool(
        name="wait",
        func=browser_tool.wait,
        coroutine=browser_tool.wait,
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
    task = "get today's date and list all the movies that are playing on theaters tomorrow"
    result = asyncio.run(agent.arun(task))
    print("Final result:", result)
