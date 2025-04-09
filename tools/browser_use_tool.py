import asyncio
import base64
import json
from typing import Generic, Optional, TypeVar

from langchain_core.tools import Tool
from pydantic import BaseModel, ConfigDict

from browser_use import Browser as BrowserUseBrowser
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.dom.service import DomService
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from googlesearch import search

_BROWSER_DESCRIPTION = """
Interact with a web browser to perform various actions such as navigation, element interaction, content extraction, and tab management. This tool provides a comprehensive set of browser automation capabilities:

Navigation:
- 'go_to_url': Go to a specific URL in the current tab
- 'go_back': Go back
- 'refresh': Refresh the current page
- 'web_search': Search the query in the current tab, the query should be a search query like humans search in web, concrete and not vague or super long. More the single most important items.

Element Interaction:
- 'click_element': Click an element by index
- 'input_text': Input text into a form element
- 'scroll_down'/'scroll_up': Scroll the page (with optional pixel amount)
- 'scroll_to_text': If you dont find something which you want to interact with, scroll to it
- 'send_keys': Send strings of special keys like Escape,Backspace, Insert, PageDown, Delete, Enter, Shortcuts such as `Control+o`, `Control+Shift+T` are supported as well. This gets used in keyboard.press.
- 'get_dropdown_options': Get all options from a dropdown
- 'select_dropdown_option': Select dropdown option for interactive element index by the text of the option you want to select

Content Extraction:
- 'extract_content': Extract page content to retrieve specific information from the page, e.g. all company names, a specifc content, all information about, links with companies in structured format or simply links

Tab Management:
- 'switch_tab': Switch to a specific tab
- 'open_tab': Open a new tab with a URL
- 'close_tab': Close the current tab

Utility:
- 'wait': Wait for a specified number of seconds
"""

Context = TypeVar("Context")


class BrowserUseTool(BaseModel, Generic[Context]):
    name: str = "browser_use"
    description: str = _BROWSER_DESCRIPTION

    lock: asyncio.Lock = Field(default_factory=asyncio.Lock, exclude=True)
    model_config = ConfigDict(arbitrary_types_allowed=True)
    browser: Optional[BrowserUseBrowser] = Field(default=None, exclude=True)
    context: Optional[BrowserContext] = Field(default=None, exclude=True)
    dom_service: Optional[DomService] = Field(default=None, exclude=True)

    # Context for generic functionality
    tool_context: Optional[Context] = Field(default=None, exclude=True)

    llm: Optional[ChatOpenAI] = Field(default=None, exclude=True)

    async def _ensure_browser_initialized(self) -> BrowserContext:
        """Ensure browser and context are initialized."""
        if self.browser is None:
            browser_config_kwargs = {"headless": False, "disable_security": True}

            self.browser = BrowserUseBrowser(BrowserConfig(**browser_config_kwargs))

        if self.context is None:
            context_config = BrowserContextConfig()
            self.context = await self.browser.new_context(context_config)
            self.dom_service = DomService(await self.context.get_current_page())

        return self.context

    async def a_go_to_url(self, url: str):
        if not url:
            return "go to url Error: URL is required for 'go_to_url'"
        try:
            context = await self._ensure_browser_initialized()
            page = await context.get_current_page()
            await page.goto(url)
            await page.wait_for_load_state()
            return f"Navigated to {url}"
        except Exception as e:
            return f"go to url Error: {str(e)}"

    def go_to_url(self, url: str):
        return asyncio.run(self.a_go_to_url(url))

    async def a_go_back(self, placeholder):
        try:
            context = await self._ensure_browser_initialized()
            await context.go_back()
        except Exception as e:
            return f"go back Error {str(e)}"

    def go_back(self, placeholder):
        return asyncio.run(self.a_go_back(placeholder))

    async def a_click_element(self, index):
        try:
            context = await self._ensure_browser_initialized()
            element = await context.get_dom_element_by_index(index)
            if not element:
                return f"click element Error: Element with index {index} not found"
            download_path = await context._click_element_node(element)
            output = f"Clicked element at index {index}"
            if download_path:
                output += f" - Downloaded file to {download_path}"
            return output
        except Exception as e:
            return f"click element Error {str(e)}"

    def click_element(self, index):
        return asyncio.run(self._click_element_node(index))

    async def a_input_text(self, index, text):
        if index is None or not text:
            return "Input text Error: Index and text are required for 'input_text' action"

        context = await self._ensure_browser_initialized()
        element = await context.get_dom_element_by_index(index)
        if not element:
            return f"Input text Error Element with index {index} not found"
        await context._input_text_element_node(element, text)
        return f"Successfully input '{text}' into element at index {index}"

    async def input_text(self, index, text):
        return asyncio.run(self.a_input_text(index, text))

    async def a_scroll_down(self, scroll_amount):
        amount = 0
        try:
            context = await self._ensure_browser_initialized()
            amount = (scroll_amount if scroll_amount else context.config.browser_window_size["height"])
            await context.execute_javascript(f"window.scrollBy(0, {amount});")
            return f"scrolled down for {amount} pixels"
        except Exception as e:
            return f"scroll_down Error: {str(e)}"

    def scroll_down(self, scroll_amount):
        return asyncio.run(self.a_scroll_down(scroll_amount))

    async def a_scroll_up(self, scroll_amount):
        direction = -1
        amount = 0
        try:
            context = await self._ensure_browser_initialized()
            amount = (scroll_amount if scroll_amount else context.config.browser_window_size["height"])
            await context.execute_javascript(f"window.scrollBy(0, {direction * amount});")
            return f"scrolled up for {amount} pixels"
        except Exception as e:
            return f"scroll_up Error: {str(e)}"

    def scroll_up(self, scroll_amount):
        return asyncio.run(self.a_scroll_up(scroll_amount))

    async def a_scroll_to_text(self, text):
        if not text:
            return f"scroll to text Error: Text is required for 'scroll_to_text' action"

        try:
            context = await self._ensure_browser_initialized()
            page = await context.get_current_page()
            try:
                locator = page.get_by_text(text, exact=False)
                await locator.scroll_into_view_if_needed()
                return f"Scrolled to text: '{text}'"
            except Exception as e:
                return f"scroll_to_text Error {str(e)}"
        except Exception as e:
            return f"scroll_to_text Error {str(e)}"

    def scroll_to_text(self, text):
        return asyncio.run(self.a_scroll_to_text(text))

    async def a_get_dropdown_options(self, index):
        if index is None:
            return f"get dropdown option Error Index is required for 'get_dropdown_options' tool"
        try:
            context = await self._ensure_browser_initialized()
            element = await context.get_dom_element_by_index(index)
            if not element:
                return "get dropdown options Error: Element with index {index} not found"
            page = await context.get_current_page()
            options = await page.evaluate(
                """
                        (xpath) => {
                            const select = document.evaluate(xpath, document, null,
                                XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                            if (!select) return null;
                            return Array.from(select.options).map(opt => ({
                                text: opt.text,
                                value: opt.value,
                                index: opt.index
                            }));
                        }
                    """,
                element.xpath,
            )
            return f"Dropdown options: {options}"
        except Exception as e:
            return f"get dropdown options Error: {str(e)}"

    def get_dropdown_options(self, index: int):
        return asyncio.run(self.async_get_dropdown_options(index))

    async def a_select_dropdown_option(self, index, text):
        if index is None or not text:
            return f"select dropdown option Error Index is required for 'get_dropdown_options' tool"

        try:
            context = await self._ensure_browser_initialized()
            element = await context.get_dom_element_by_index(index)
            page = await context.get_current_page()
            await page.select_option(element.xpath, label=text)
            return f"Selected option '{text}' from dropdown at index {index}"
        except Exception as e:
            return f"select dropdown options Error: {str(e)}"

    def select_dropdown_option(self, index: int):
        return asyncio.run(self.async_select_dropdown_options(index))

    async def a_extract_content(self, goal):
        if not goal:
            return "Goal is required for 'extract_content' action"
        context = await self._ensure_browser_initialized()
        page = await context.get_current_page()
        try:
            # Get page content and convert to markdown for better processing
            html_content = await page.content()

            try:
                import markdownify

                content = markdownify.markdownify(html_content)
            except ImportError:
                # Fallback if markdownify is not available
                content = html_content

            # Format the prompt with the goal and content
            max_content_length = min(20000, len(content))
            messages = [
                (
                    "system",
                    "Your task is to extract the content of the page. You will be given a page and a goal, and you should extract all relevant information around this goal from the page. If the goal is vague, summarize the page. Respond in json format.",
                ),
                ("human", "Extraction goal: {goal} \n\n Page content:\n{page}"),
            ]
            prompt = ChatPromptTemplate.from_messages(messages)

            chain = prompt | self.llm
            response = await chain.ainvoke(
                {
                    "goal": goal,
                    "page": content[:max_content_length]
                }
            )

            return f"Extracted content: {response.content}"
        except Exception as e:
            # Provide a more helpful error message
            error_msg = f"Failed to extract content: {str(e)}"
            try:
                # Try to return a portion of the page content as fallback
                return f"{error_msg}\nHere's a portion of the page content:\n{content[:20000]}..."
            except:
                return error_msg

    def extract_content(self, goal):
        return asyncio.run(self.a_extract_content(goal))

    async def a_get_current_state(self, placeholder):
        """
        Get the current browser state as a ToolResult.
        If context is not provided, uses self.context.
        """
        try:
            ctx = await self._ensure_browser_initialized()
            state = await ctx.get_state()

            viewport_height = 0
            if hasattr(state, "viewport_info") and state.viewport_info:
                viewport_height = state.viewport_info.height
            elif hasattr(ctx, "config") and hasattr(ctx.config, "browser_window_size"):
                viewport_height = ctx.config.browser_window_size.get("height", 0)

            # Take a screenshot for the state
            page = await ctx.get_current_page()

            await page.bring_to_front()
            await page.wait_for_load_state()

            state_info = {
                "url": state.url,
                "title": state.title,
                "tabs": [tab.model_dump() for tab in state.tabs],
                "help": "[0], [1], [2], etc., represent clickable indices corresponding to the elements listed. Clicking on these indices will navigate to or interact with the respective content behind them.",
                "interactive_elements_with_index": (
                    [f"INDEX[{k}]: {v.clickable_elements_to_string()}" for k, v in state.selector_map.items()]
                    if state.selector_map
                    else ""
                ),
                "scroll_info": {
                    "pixels_above": getattr(state, "pixels_above", 0),
                    "pixels_below": getattr(state, "pixels_below", 0),
                    "total_height": getattr(state, "pixels_above", 0)
                                    + getattr(state, "pixels_below", 0)
                                    + viewport_height,
                },
                "viewport_height": viewport_height,
            }

            return json.dumps(state_info, indent=4, ensure_ascii=False),
        except Exception as e:
            return f"get current state Error: Failed to get browser state: {str(e)}"

    def get_current_state(self, placeholder):
        return asyncio.run(self.a_get_current_state(placeholder))

    async def a_switch_tab(self, tab_id: int):
        context = await self._ensure_browser_initialized()
        await context.switch_to_tab(tab_id)
        page = await context.get_current_page()
        await page.wait_for_load_state()
        return f"Switched to tab {tab_id}"

    def switch_tab(self, tab_id):
        return asyncio.run(self.a_switch_tab(tab_id))

    async def a_open_tab(self, url: str):
        if not url:
            return f"open tab Error: URL is required for 'open_tab' action"
        context = await self._ensure_browser_initialized()
        await context.create_new_tab(url)
        return f"Opened new tab with {url}"

    def open_tab(self, url: str):
        return asyncio.run(self.a_open_tab(url))

    async def a_close_tab(self, placeholder):
        context = await self._ensure_browser_initialized()
        await context.close_current_tab()
        return "Closed current tab"

    def close_tab(self, placeholder):
        return asyncio.run(self.a_close_tab(placeholder))

    async def a_wait(self, seconds: int):
        seconds_to_wait = int(seconds) if seconds is not None else 3
        await asyncio.sleep(seconds_to_wait)
        return f"Waited for {seconds_to_wait} seconds"

    def wait(self, seconds: int):
        return asyncio.run(self.a_wait(seconds))

    async def cleanup(self):
        """Clean up browser resources."""
        async with self.lock:
            if self.context is not None:
                await self.context.close()
                self.context = None
                self.dom_service = None
            if self.browser is not None:
                await self.browser.close()
                self.browser = None

    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        if self.browser is not None or self.context is not None:
            try:
                asyncio.run(self.cleanup())
            except RuntimeError:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.cleanup())
                loop.close()


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

    links = asyncio.run(fetch_results())
    return f"Searched result links: {', '.join(links)}"


def get_browser_use_tools(llm):
    browser_tool = BrowserUseTool(llm=llm)
    return [
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
            name="input_text",
            func=browser_tool.input_text,
            coroutine=browser_tool.a_input_text,
            description="Input text into a form element. **Required:** index (integer), text (string)."
        ),

        Tool(
            name="scroll_down",
            func=browser_tool.scroll_down,
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
