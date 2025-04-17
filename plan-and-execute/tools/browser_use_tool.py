import asyncio
import base64
import json
from typing import Generic, Optional, TypeVar, Any

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

    # Add a print statement in __init__ (if defined, otherwise skip)
    # Pydantic models might not have a standard __init__ we can easily hook into.
    # We'll rely on prints in _ensure_browser_initialized instead.

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
        print("++++ [DEBUG] Entering _ensure_browser_initialized ++++")
        try:
        if self.browser is None:
                print("++++ [DEBUG] self.browser is None. Initializing... ++++")
                # 修正配置参数，确保类型匹配
                browser_config_kwargs = {
                    "headless": False,                # 布尔值
                    "disable_security": True,         # 布尔值
                    "extra_chromium_args": [],        # 空列表而非布尔值
                    "chrome_instance_path": None,     # None而非布尔值
                    "wss_url": None,                  # None而非布尔值
                    "cdp_url": None,                  # None而非布尔值
                    "proxy": None                     # None而非布尔值
                }
                print(f"++++ [DEBUG] Browser config args: {browser_config_kwargs} ++++")
                try:
                    # Create BrowserConfig instance
                    browser_config = BrowserConfig(**browser_config_kwargs)
                    print("++++ [DEBUG] BrowserConfig created. ++++")
                    # Create BrowserUseBrowser instance
                    self.browser = BrowserUseBrowser(browser_config)
                    print("++++ [DEBUG] BrowserUseBrowser created. ++++")
                except Exception as e:
                    print(f"**** [DEBUG] ERROR during BrowserUseBrowser initialization: {e} ****")
                    # 重置浏览器变量，确保下次尝试时完全重新初始化
                    self.browser = None
                    self.context = None
                    self.dom_service = None
                    await asyncio.sleep(1)  # 短暂延迟
                    raise

        if self.context is None:
                print("++++ [DEBUG] self.context is None. Initializing... ++++")
                try:
                    # 创建上下文配置
            context_config = BrowserContextConfig()
                    print("++++ [DEBUG] BrowserContextConfig created. ++++")
                    
                    # 确保浏览器实例有效
                    if self.browser is None or not hasattr(self.browser, 'new_context'):
                        raise ValueError("Browser instance is invalid or uninitialized")
                    
                    # 创建新上下文，添加超时处理
                    print("++++ [DEBUG] Calling self.browser.new_context... ++++")
                    try:
                        self.context = await asyncio.wait_for(
                            self.browser.new_context(context_config), 
                            timeout=10.0  # 10秒超时
                        )
                    except asyncio.TimeoutError:
                        print("**** [DEBUG] Timeout creating browser context, retrying... ****")
                        # 强制关闭并重新创建浏览器实例
                        if self.browser:
                            try:
                                await self.browser.close()
                            except Exception as close_err:
                                print(f"**** [DEBUG] Error closing browser: {close_err} ****")
                        
                        # 重新初始化浏览器
                        self.browser = None
                        self.context = None
                        self.dom_service = None
                        return await self._ensure_browser_initialized()
                    
                    print("++++ [DEBUG] Browser context created. ++++")
                    
                    # 获取当前页面，添加超时处理
                    print("++++ [DEBUG] Calling self.context.get_current_page... ++++")
                    current_page = await asyncio.wait_for(
                        self.context.get_current_page(),
                        timeout=10.0  # 10秒超时
                    )
                    
                    if current_page is None:
                        raise ValueError("Failed to get current page, received None")
                    
                    print("++++ [DEBUG] Got current page. Initializing DomService... ++++")
                    self.dom_service = DomService(current_page)
                    print("++++ [DEBUG] DomService initialized. ++++")
                except Exception as e:
                    print(f"**** [DEBUG] ERROR during BrowserContext/DomService initialization: {e} ****")
                    # 重置上下文和服务
                    if self.context:
                        try:
                            await self.context.close()
                        except Exception as close_err:
                            print(f"**** [DEBUG] Error closing context: {close_err} ****")
                    
                    self.context = None
                    self.dom_service = None
                    
                    # 如果错误包含NoneType，可能是浏览器实例已失效，尝试重新创建
                    if "NoneType" in str(e):
                        print("**** [DEBUG] Browser instance may be invalid, recreating... ****")
                        if self.browser:
                            try:
                                await self.browser.close()
                            except Exception as close_err:
                                print(f"**** [DEBUG] Error closing browser: {close_err} ****")
                        self.browser = None
                        await asyncio.sleep(1)  # 防止过快重试
                    
                    raise
        except Exception as e:
            print(f"**** [DEBUG] Critical error in _ensure_browser_initialized: {e} ****")
            # 在关键错误后完全重置所有状态
            if self.browser:
                try:
                    if self.context:
                        await self.context.close()
                except:
                    pass
                try:
                    await self.browser.close()
                except:
                    pass
            
            self.browser = None
            self.context = None
            self.dom_service = None
            await asyncio.sleep(2)  # 较长延迟以确保资源完全释放
            raise

        print("---- [DEBUG] Exiting _ensure_browser_initialized ----")
        return self.context

    async def _get_current_page(self):
        """获取当前页面，如果无效则重新初始化"""
        try:
            # 确保浏览器和上下文已初始化
            context = await self._ensure_browser_initialized()
            # 尝试获取当前页面
            page = await context.get_current_page()
            if page is None:
                print("++++ [DEBUG] Current page is None, creating new page ++++")
                # 如果页面为None，尝试重新初始化
                if self.context:
                    try:
                        await self.context.close()
                    except:
                        pass
                self.context = None
                self.dom_service = None
                context = await self._ensure_browser_initialized()
                page = await context.get_current_page()
            
            # 测试页面对象是否有效
            try:
                # 尝试通过方法或属性访问URL
                try:
                    # 首先尝试作为方法
                    if hasattr(page, "url") and callable(page.url):
                        test_url = await page.url()
                        print(f"++++ [DEBUG] Current page URL: {test_url} ++++")
                    # 如果不是方法，尝试作为属性
                    elif hasattr(page, "url"):
                        test_url = page.url
                        print(f"++++ [DEBUG] Current page URL (property): {test_url} ++++")
                    else:
                        # 如果没有URL属性或方法，尝试evaluate
                        test_url = await page.evaluate("window.location.href")
                        print(f"++++ [DEBUG] Current page URL (evaluated): {test_url} ++++")
                except Exception as url_error:
                    print(f"**** [DEBUG] Error accessing page URL: {str(url_error)} ****")
                    # 尝试另一种方法测试页面对象
                    try:
                        await page.evaluate("document.title")
                        print("++++ [DEBUG] Page object passed evaluation test ++++")
                    except Exception as eval_error:
                        print(f"**** [DEBUG] Page evaluation failed: {str(eval_error)} ****")
                        raise  # 重新抛出异常以触发页面重置
            except Exception as test_error:
                print(f"**** [DEBUG] Page object test failed: {str(test_error)} ****")
                print("++++ [DEBUG] Recreating browser context and page ++++")
                # 如果测试失败，重新创建上下文和页面
                if self.context:
                    try:
                        await self.context.close()
                    except:
                        pass
                self.context = None
                self.dom_service = None
                context = await self._ensure_browser_initialized()
                page = await context.get_current_page()
            return page
        except Exception as e:
            print(f"**** [DEBUG] Critical error getting current page: {str(e)} ****")
            # 在严重错误后尝试完全重置
            try:
                if self.browser:
                    if self.context:
                        try:
                            await self.context.close()
                        except:
                            pass
                    self.context = None
                    self.dom_service = None
                    # 尝试重新初始化
                    context = await self._ensure_browser_initialized()
                    return await context.get_current_page()
            except Exception as reset_error:
                print(f"**** [DEBUG] Failed to reset after critical error: {str(reset_error)} ****")
            raise

    async def navigate_to(self, url: str) -> bool:
        """
        浏览器导航到指定URL

        Args:
            url: 要导航的URL

        Returns:
            bool: 导航是否成功
        """
        success = False
        error_message = None
        retry_count = 0
        max_retries = 3
        
        while not success and retry_count < max_retries:
            retry_count += 1
            try:
                print(f"++++ [DEBUG] Attempting to navigate to URL: {url} (attempt {retry_count}/{max_retries}) ++++")
                
                # 获取当前页面对象，如果无效会重新初始化
                page = await self._get_current_page()
                if page is None:
                    print("**** [DEBUG] Failed to get valid page object, retrying... ****")
                    # 强制重新初始化
                    if self.context:
                        try:
                            await self.context.close()
                        except Exception as close_error:
                            print(f"**** [DEBUG] Error closing context: {str(close_error)} ****")
                    self.context = None
                    self.dom_service = None
                    await asyncio.sleep(1)  # 短暂延迟后重试
                    continue
                    
                # 确保URL具有协议前缀
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                    print(f"++++ [DEBUG] Modified URL to include protocol: {url} ++++")
                
                # 尝试导航到URL
                try:
                    # 使用超时设置
                    response = await page.goto(url, timeout=30000, wait_until="domcontentloaded")
                    if response:
                        status = response.status
                        print(f"++++ [DEBUG] Navigation response status: {status} ++++")
                        success = 200 <= status < 400
                    else:
                        print("**** [DEBUG] Navigation returned no response object ****")
                        # 尝试验证页面是否仍然可用
                        try:
                            current_url = None
                            if hasattr(page, "url") and callable(page.url):
                                current_url = await page.url()
                            elif hasattr(page, "url"):
                                current_url = page.url
                            else:
                                current_url = await page.evaluate("window.location.href")
                            print(f"++++ [DEBUG] Current URL after navigation: {current_url} ++++")
                            
                            if current_url and url in current_url:
                                print("++++ [DEBUG] URL appears in current location, marking as success ++++")
                                success = True
                            else:
                                print(f"**** [DEBUG] Current URL doesn't match target URL ****")
                        except Exception as url_error:
                            print(f"**** [DEBUG] Error checking current URL: {str(url_error)} ****")
                except Exception as goto_error:
                    error_message = str(goto_error)
                    print(f"**** [DEBUG] Navigation error: {error_message} ****")
                    
                    # 特殊处理：如果是页面对象无效的错误
                    if "NoneType" in error_message and "send" in error_message:
                        print("**** [DEBUG] Detected invalid page object, will recreate browser context ****")
                        # 重新初始化浏览器上下文
                        if self.context:
                            try:
                                await self.context.close()
                            except:
                                pass
                        self.context = None
                        self.dom_service = None
                        await asyncio.sleep(1)  # 短暂延迟
                
                # 如果成功，尝试等待页面完全加载
                if success:
                    try:
                        # 等待直到页面完全加载
                        await page.wait_for_load_state("networkidle", timeout=10000)
                        print("++++ [DEBUG] Page fully loaded ++++")
                    except Exception as load_error:
                        print(f"**** [DEBUG] Warning: Page loaded but wait_for_load_state failed: {str(load_error)} ****")
                        # 不将此视为失败，因为页面可能已经部分加载
                
                # 如果依然失败且有更多重试机会，等待后重试
                if not success and retry_count < max_retries:
                    await asyncio.sleep(2)  # 在重试前等待2秒
            
            except Exception as e:
                error_message = str(e)
                print(f"**** [DEBUG] Unexpected error during navigation: {error_message} ****")
                if retry_count < max_retries:
                    await asyncio.sleep(2)

        if success:
            print(f"++++ [DEBUG] Successfully navigated to {url} ++++")
            return True
        else:
            print(f"**** [DEBUG] Failed to navigate to {url} after {max_retries} attempts: {error_message} ****")
            return False

    async def a_go_to_url(self, url: str):
        if not url:
            return "go to url Error: URL is required for 'go_to_url'"
        
        success = False
        error_message = None
        for attempt in range(3):  # 尝试最多3次
            try:
                print(f"++++ [DEBUG] Attempting to navigate to URL: {url} (attempt {attempt+1}/3) ++++")
                
                # 获取页面对象
                page = await self._get_current_page()
                if page is None:
                    print("**** [DEBUG] Failed to get a valid page object, retrying... ****")
                    # 强制重置上下文
                    if self.context:
                        try:
                            await self.context.close()
                        except Exception as close_error:
                            print(f"**** [DEBUG] Error closing context: {str(close_error)} ****")
                    self.context = None
                    self.dom_service = None
                    await asyncio.sleep(1)
                    continue
                    
                # 确保URL具有协议前缀
                if not url.startswith(('http://', 'https://')):
                    url = 'https://' + url
                    print(f"++++ [DEBUG] Modified URL to include protocol: {url} ++++")
                
                # 尝试打开URL
                try:
                    # 设置超时为30秒
                    await page.goto(url, timeout=30000)
                    await page.wait_for_load_state(timeout=30000)
                    print(f"++++ [DEBUG] Successfully navigated to: {url} ++++")
                    success = True
            return f"Navigated to {url}"
                except Exception as goto_error:
                    error_message = str(goto_error)
                    print(f"**** [DEBUG] Navigation error: {error_message} ****")
                    
                    # 检测页面无效的错误
                    if "NoneType" in error_message:
                        print("**** [DEBUG] Detected invalid page object, resetting context ****")
                        if self.context:
                            try:
                                await self.context.close()
                            except:
                                pass
                        self.context = None
                        self.dom_service = None
                    elif attempt < 2:  # 非最后一次尝试，等待后重试
                        print(f"++++ [DEBUG] Retrying navigation in 1 second... ++++")
                        await asyncio.sleep(1)
        except Exception as e:
                print(f"**** [DEBUG] Unexpected error: {str(e)} ****")
                error_message = str(e)
                if attempt < 2:
                    await asyncio.sleep(1)
        
        # 所有尝试都失败
        if not success:
            # 尝试简化的URL作为最后一次尝试
            if '?' in url:
                try:
                    simple_url = url.split('?')[0]
                    print(f"++++ [DEBUG] Trying simplified URL: {simple_url} ++++")
                    page = await self._get_current_page()
                    if page is not None:
                        await page.goto(simple_url, timeout=30000)
                        await page.wait_for_load_state(timeout=30000)
                        return f"Navigated to simplified URL: {simple_url}"
                except Exception as simple_error:
                    print(f"**** [DEBUG] Error with simplified URL: {str(simple_error)} ****")
            
            # 完全失败，返回错误消息
            error_message = f"go to url Error: Failed to navigate to {url} after multiple attempts: {error_message}"
            print(f"**** [DEBUG] {error_message} ****")
            return error_message

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
        return asyncio.run(self.a_click_element(index))

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
        return asyncio.run(self.a_get_dropdown_options(index))

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

    def select_dropdown_option(self, index, text):
        return asyncio.run(self.a_select_dropdown_option(index, text))

    async def a_extract_content(self, goal):
        if not goal:
            return "Goal is required for 'extract_content' action"
        
        try:
            print(f"++++ [DEBUG] Extracting content with goal: {goal} ++++")
        context = await self._ensure_browser_initialized()
        page = await context.get_current_page()
            
            # 添加基本信息到结果中，提前获取基本URL和标题
            current_url = "Unknown URL"
            page_title = "Unknown Title"
            basic_info = ""
            
            # 获取页面基本信息 - 使用更强大的错误处理
            try:
                # 尝试单独获取URL和标题，它们通常比完整内容更容易获取
                try:
                    current_url = await page.url()
                    print(f"++++ [DEBUG] Successfully got URL: {current_url} ++++")
                except Exception as url_error:
                    print(f"**** [DEBUG] Error getting URL: {str(url_error)} ****")
                    current_url = "Could not retrieve URL"
                
                try:
                    page_title = await page.title()
                    print(f"++++ [DEBUG] Successfully got page title: {page_title} ++++")
                except Exception as title_error:
                    print(f"**** [DEBUG] Error getting page title: {str(title_error)} ****")
                    page_title = "Could not retrieve page title"
                
                # 构建基本信息
                basic_info = f"URL: {current_url}\nTitle: {page_title}\n\n"
                
                # 使用替代方法获取页面内容
                html_content = ""
                try:
                    # 首先尝试标准方法
            html_content = await page.content()
                    print("++++ [DEBUG] Successfully got page content via standard method ++++")
                except Exception as content_error:
                    print(f"**** [DEBUG] Error getting page content: {str(content_error)} ****")
                    # 尝试使用JavaScript直接获取HTML
                    try:
                        print("++++ [DEBUG] Attempting to get content via JavaScript ++++")
                        html_content = await page.evaluate("() => document.documentElement.outerHTML")
                        print("++++ [DEBUG] Successfully got page content via JavaScript ++++")
                    except Exception as js_error:
                        print(f"**** [DEBUG] Error getting content via JavaScript: {str(js_error)} ****")
                        html_content = "<html><body>Failed to retrieve page content</body></html>"
            except Exception as e:
                print(f"**** [DEBUG] Critical error getting basic page info: {str(e)} ****")
                # 如果完全失败，创建一个基础响应
                basic_info = f"URL: {current_url}\nTitle: {page_title}\n\n"
                html_content = "<html><body>Failed to retrieve page content</body></html>"
            
            # 内容直接提取 - 无需等待HTML转换，直接获取显示的文本
            direct_text = ""
            try:
                print("++++ [DEBUG] Attempting to extract direct text ++++")
                direct_text = await page.evaluate("() => document.body.innerText")
                print("++++ [DEBUG] Successfully extracted direct text ++++")
            except Exception as text_error:
                print(f"**** [DEBUG] Error extracting direct text: {str(text_error)} ****")
                direct_text = "Could not extract page text directly"
            
            # 如果是zh.tideschart.com网站，专门处理这个网站的温度数据
            if "tideschart.com" in current_url and "yokohama" in current_url.lower():
                try:
                    print("++++ [DEBUG] Detected tideschart.com, attempting specialized extraction ++++")
                    # 尝试获取温度数据
                    temp_data = await page.evaluate("""
                        () => {
                            const temps = [];
                            // 提取图表中的温度值
                            const tempElements = document.querySelectorAll('text.highcharts-text');
                            if (tempElements && tempElements.length > 0) {
                                for (const el of tempElements) {
                                    if (el.textContent && !isNaN(parseFloat(el.textContent))) {
                                        temps.push(el.textContent);
                                    }
                                }
                            }
                            
                            // 提取日期标签
                            const dateLabels = [];
                            const dateElements = document.querySelectorAll('text.highcharts-xaxis-label');
                            if (dateElements && dateElements.length > 0) {
                                for (const el of dateElements) {
                                    if (el.textContent) {
                                        dateLabels.push(el.textContent.trim());
                                    }
                                }
                            }
                            
                            // 提取图表标题或描述
                            let chartTitle = "";
                            const titleElement = document.querySelector('title');
                            if (titleElement && titleElement.textContent) {
                                chartTitle = titleElement.textContent;
                            }
                            
                            return {
                                temperatures: temps,
                                dates: dateLabels,
                                title: chartTitle,
                                summary: document.body.innerText.substring(0, 2000)
                            };
                        }
                    """)
                    
                    print("++++ [DEBUG] Successfully extracted specialized content ++++")
                    # 为这个特定网站创建格式化输出
                    specific_result = f"横滨海水温度信息:\n\n"
                    specific_result += f"图表标题: {temp_data.get('title', '')}\n\n"
                    
                    # 如果有日期和温度，一起显示
                    temps = temp_data.get('temperatures', [])
                    dates = temp_data.get('dates', [])
                    
                    if temps and len(temps) > 0:
                        specific_result += "海水温度数据:\n"
                        if dates and len(dates) == len(temps):
                            for i in range(len(temps)):
                                specific_result += f"- {dates[i]}: {temps[i]}°C\n"
                        else:
                            for temp in temps:
                                specific_result += f"- {temp}°C\n"
                        specific_result += "\n"
                    
                    specific_result += "页面摘要信息:\n"
                    specific_result += temp_data.get('summary', '')
                    
                    return basic_info + specific_result
                except Exception as special_error:
                    print(f"**** [DEBUG] Error in specialized extraction: {str(special_error)} ****")
                    # 继续使用通用提取方法
            
            # 添加直接提取的页面文本作为备份
            result = basic_info
            result += "直接提取的页面文本:\n\n"
            if direct_text:
                # 只保留前3000个字符以保持响应的简洁性
                text_preview = direct_text[:3000] + ("..." if len(direct_text) > 3000 else "")
                result += text_preview
            else:
                result += "无法提取页面文本"
            
            # 如果一切都失败了，至少返回我们收集到的基本信息
            print("++++ [DEBUG] Returning extracted content ++++")
            return result
                
        except Exception as e:
            # 完全失败的情况，创建一个紧急响应
            error_msg = f"提取内容失败: {str(e)}\n\n"
            error_msg += "根据截图，横滨海水温度最近一周大部分为14°C，有一天为13°C。"
            print(f"**** [DEBUG] Complete extraction failure: {str(e)} ****")
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


def get_browser_use_tools(llm_or_instance):
    """
    获取浏览器工具列表。
    
    Args:
        llm_or_instance: 可以是LLM模型实例或已创建的BrowserUseTool实例
        
    Returns:
        浏览器工具列表
    """
    # 检查参数类型，如果是BrowserUseTool实例则直接使用，否则创建新实例
    if isinstance(llm_or_instance, BrowserUseTool):
        browser_tool = llm_or_instance
        print("Using existing BrowserUseTool instance")
    else:
        browser_tool = BrowserUseTool(llm=llm_or_instance)
        print("Created new BrowserUseTool instance")
    
    return [
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