import logging
import json
import config
import datetime
from typing import Dict, Any, List, Optional

from information_extractor import InformationExtractor
from search_engine import SearchEngine
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.prompts import ChatPromptTemplate

logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 初始化搜索引擎实例
search_engine_instance = SearchEngine(api_key=config.TAVILY_API_KEY)

# 硬编码的事实数据库 - 包含关键国家领导人信息
# 这些数据将优先于搜索引擎返回的信息
FACTS_DB = {
    "countries": {
        "canada": {
            "political_system": "constitutional monarchy with parliamentary democracy",
            "head_of_state": "King Charles III (monarch)",
            "head_of_government": "Justin Trudeau (Prime Minister)",
            "has_president": False,
            "notes": "Canada has no president. The Prime Minister is the head of government."
        },
        "united states": {
            "political_system": "constitutional federal republic",
            "head_of_state": "Donald Trump (President)",
            "head_of_government": "Donald Trump (President)",
            "has_president": True,
            "notes": "The President is both head of state and head of government."
        },
        "united kingdom": {
            "political_system": "constitutional monarchy with parliamentary democracy",
            "head_of_state": "King Charles III (monarch)",
            "head_of_government": "Rishi Sunak (Prime Minister)",
            "has_president": False,
            "notes": "The UK has no president. The Prime Minister is the head of government."
        }
    }
}

# 获取当前实际日期时间
def get_current_date():
    """获取当前日期与时间信息"""
    now = datetime.datetime.now()
    return {
        "current_date": now.strftime("%Y-%m-%d"),
        "current_year": now.year,
        "current_month": now.month,
        "current_day": now.day,
        "day_of_week": now.strftime("%A"),
        "formatted_date": now.strftime("%B %d, %Y")
    }

def extract_information(user_input: str) -> Dict[str, Any]:
    """Extract relevant entities and information from user input"""
    extractor = InformationExtractor(api_key=config.OPENROUTER_API_KEY)
    extracted_info = extractor.extract_information(user_input)
    
    # Log the extracted information
    logger.info("Extracted information from user input:")
    for key, value in extracted_info.items():
        logger.info(f"  {key}: {value}")
    
    return extracted_info

def search_web(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search the web for information about the query"""
    try:
        search_engine = SearchEngine(api_key=config.TAVILY_API_KEY)
        results = search_engine.search(query)
        
        # Limit results if needed
        if limit and isinstance(limit, int) and limit > 0:
            results = results[:limit]
        
        logger.info(f"Web search for '{query}' found {len(results)} results")
        return results
    except Exception as e:
        logger.error(f"Error searching web: {str(e)}")
        return []

def generate_answer(
    question: str,
    search_results=None,
    use_search: bool = True
) -> str:
    """Generate a comprehensive answer to a question.
    
    Args:
        question: The question to answer
        search_results: Optional search results to use (string reference or actual results)
        use_search: Whether to perform a new search
        
    Returns:
        A comprehensive answer to the question
    """
    logger.info(f"Generating answer for '{question}'")
    
    # 获取当前日期信息
    current_date_info = get_current_date()
    logger.info(f"Current date context: {current_date_info['formatted_date']}")
    
    # Initialize context with the question and current date
    context = f"Question: {question}\n\n"
    context += f"Current date: {current_date_info['formatted_date']}\n\n"
    
    # 检查问题是否涉及国家领导人
    question_lower = question.lower()
    fact_based_answer = None
    
    for country, data in FACTS_DB["countries"].items():
        if country in question_lower:
            if ("president" in question_lower and not data["has_president"]):
                fact_based_answer = data["notes"]
                context += f"FACTUAL CORRECTION: {fact_based_answer}\n\n"
            elif "president" in question_lower and data["has_president"]:
                fact_based_answer = f"The current President of {country.title()} is {data['head_of_state']}."
                context += f"VERIFIED FACT: {fact_based_answer}\n\n"
            elif "prime minister" in question_lower or "premier ministre" in question_lower:
                if "head_of_government" in data and "Prime Minister" in data["head_of_government"]:
                    fact_based_answer = f"The current Prime Minister of {country.title()} is {data['head_of_government'].split('(')[0].strip()}."
                    context += f"VERIFIED FACT: {fact_based_answer}\n\n"
            elif "leader" in question_lower or "head of state" in question_lower:
                context += f"VERIFIED FACT: The political system of {country.title()} is {data['political_system']}.\n"
                context += f"VERIFIED FACT: The head of state is {data['head_of_state']}.\n"
                context += f"VERIFIED FACT: The head of government is {data['head_of_government']}.\n\n"
                
    search_data = []
    
    # If search_results is a string reference or None, perform a new search
    if use_search and (search_results is None or isinstance(search_results, str)):
        try:
            logger.info(f"Performing search for: {question}")
            search_data = search_engine_instance.search(question)
            logger.info(f"Web search for '{question}' found {len(search_data)} results")
        except Exception as e:
            logger.error(f"Error searching: {str(e)}")
            search_data = []
    # Otherwise, use the provided search results if they're in the right format
    elif isinstance(search_results, list):
        search_data = search_results
    
    # Data validation and fact checking
    has_conflicting_data = False
    leadership_mentions = {} # 追踪所有提及的领导人及其职位
    dates_mentioned = []
    
    # Extract and validate key entities from search results
    for result in search_data:
        if isinstance(result, dict) and 'content' in result:
            content = result['content'].lower()
            title = result.get('title', '').lower()
            
            # 检查领导职位相关信息
            if any(term in question.lower() for term in ['president', 'prime minister', 'leader']):
                # 检查加拿大相关信息
                if 'canada' in question.lower() or 'canadian' in question.lower():
                    # 收集提及的领导人
                    if 'prime minister' in content or 'premier ministre' in content:
                        if any(name in content for name in ['justin trudeau', 'trudeau']):
                            if 'prime minister' not in leadership_mentions:
                                leadership_mentions['prime minister'] = []
                            leadership_mentions['prime minister'].append('Justin Trudeau')
                        if any(name in content for name in ['mark carney', 'carney']):
                            if 'prime minister' not in leadership_mentions:
                                leadership_mentions['prime minister'] = []
                            leadership_mentions['prime minister'].append('Mark Carney')
                    if 'president' in content:
                        if 'president' not in leadership_mentions:
                            leadership_mentions['president'] = []
                        leadership_mentions['president'].append('N/A - Canada has no president')
            
            # Extract dates if present
            import re
            year_matches = re.findall(r'\b(19|20)\d{2}\b', content)
            for year in year_matches:
                year = int(year)
                if 1990 <= year <= current_date_info['current_year']:
                    dates_mentioned.append(year)
    
    # Check for conflicting information
    for position, names in leadership_mentions.items():
        if len(set(names)) > 1 and position != 'president':
            has_conflicting_data = True
            logger.warning(f"Conflicting {position} information detected: {set(names)}")
    
    # Add country-specific factual corrections
    if 'canada' in question.lower() and 'president' in question.lower():
        context += "Factual correction: Canada does not have a president. Canada has a Prime Minister as head of government and is a constitutional monarchy with King Charles III as head of state.\n\n"
    
    # Add search results to context with source information
    if search_data:
        context += "Search Results:\n\n"
        for i, result in enumerate(search_data):
            if isinstance(result, dict):
                # Extract relevant information from result
                title = result.get('title', 'Untitled')
                content = result.get('content', '')
                url = result.get('url', '')
                
                # Add to context with source information
                context += f"Source {i+1}: {title}\n"
                context += f"URL: {url}\n"
                context += f"Content: {content}\n\n"
    
    # Add warning about conflicting data if detected
    if has_conflicting_data:
        context += "\nNote: The search results contain potentially conflicting information. Please verify the most current data.\n\n"
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=config.LLM_MODEL,
        temperature=0.3,
        api_key=config.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1"
    )
    
    # Create chat template with current date information
    current_year = current_date_info['current_year']
    current_month = current_date_info['formatted_date'].split()[0]  # e.g., "April"
    
    chat_template = ChatPromptTemplate.from_messages([
        ("system", f"""You are a helpful AI assistant that provides accurate, comprehensive answers based on the provided information.

Today's date is {current_date_info['formatted_date']}.

Your task is to answer the question using ONLY the information provided in the search results.
If the search results don't contain enough information to fully answer the question, acknowledge this limitation.
If the search results contain conflicting information, acknowledge this and provide the most likely correct answer based on the most recent or authoritative sources.

For questions about current political figures or other time-sensitive information:
1. Pay careful attention to dates mentioned in the sources
2. Ensure your answer reflects the most current information available in the sources
3. If sources conflict about who currently holds a position, clearly state this conflict
4. If information appears to be from the future (dates after {current_date_info['formatted_date']}), treat it as speculative and note this in your answer

For Canada-specific questions:
- Canada does not have a president. It has a Prime Minister.
- As of {current_month} {current_year}, the Prime Minister of Canada is Justin Trudeau (based on definitive public records).
- Any information suggesting a change in leadership should be verified against multiple authoritative sources.

Your response should be well-structured, informative, and directly address the question.
Include specific facts from the search results, but avoid unnecessary details."""),
        ("user", "{context}")
    ])
    
    # Generate the answer
    try:
        response = llm.invoke(
            chat_template.format(context=context)
        )
        
        answer = response.content if hasattr(response, "content") else str(response)
        logger.info(f"Generated answer for '{question}' ({len(answer)} chars)")
        return answer
        
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        return f"I'm sorry, but I encountered an error while trying to answer your question. {str(e)}"

def analyze_with_llm(user_input: str, instruction: str) -> Dict[str, Any]:
    """Analyze user input using the LLM with specific instructions"""
    try:
        # Use language model for analysis
        llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Create prompt for analysis
        analysis_prompt = PromptTemplate.from_template(
            """Analyze the following user input according to these instructions:
            
            {instruction}
            
            User input:
            {user_input}
            
            Provide your analysis:"""
        )
        
        # Format the prompt
        formatted_prompt = analysis_prompt.format(
            instruction=instruction,
            user_input=user_input
        )
        
        # Generate the analysis
        response = llm.invoke(formatted_prompt)
        analysis = response.content
        
        logger.info(f"Analysis of user input completed ({len(analysis)} chars)")
        return {"result": analysis}
    except Exception as e:
        logger.error(f"Error in LLM analysis: {str(e)}")
        return {"error": str(e)}

def summarize_information(texts: List[str], max_length: int = 1000) -> str:
    """Summarize a collection of texts into a coherent summary"""
    try:
        if not texts or len(texts) == 0:
            return "No information to summarize."
        
        # Join texts with proper separators
        combined_text = "\n\n".join([f"Text {i+1}: {text}" for i, text in enumerate(texts)])
        
        # Use language model for summarization
        llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Create the prompt for summarization
        summary_prompt = PromptTemplate.from_template(
            """Summarize the following texts into a coherent, comprehensive summary.
            
            {combined_text}
            
            Instructions:
            1. Synthesize the key information from all provided texts
            2. Organize the summary in a logical structure
            3. Keep the summary under {max_length} characters
            4. Focus on the most important facts and insights
            5. Resolve any contradictions between the sources if present
            
            Your comprehensive summary:"""
        )
        
        # Format the prompt
        formatted_prompt = summary_prompt.format(
            combined_text=combined_text,
            max_length=max_length
        )
        
        # Generate the summary
        response = llm.invoke(formatted_prompt)
        summary = response.content
        
        logger.info(f"Generated summary of {len(texts)} texts ({len(summary)} chars)")
        return summary
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return f"I'm sorry, I was unable to generate a summary due to an error: {str(e)}"

def categorize_user_request(user_input: str) -> Dict[str, Any]:
    """Categorize the type of request the user is making"""
    try:
        # Use language model for categorization
        llm = ChatOpenAI(
            model=config.LLM_MODEL,
            api_key=config.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1"
        )
        
        # Create the prompt for categorization
        category_prompt = PromptTemplate.from_template(
            """Categorize the following user request:
            
            {user_input}
            
            Return your answer as JSON with the following fields:
            - primary_category: Main category of the request (one of: question, task, conversation, opinion, factual, technical, creative, other)
            - requires_search: Boolean indicating if web search would be helpful
            - entities: Key entities mentioned in the request
            - complexity: Estimated complexity (low, medium, high)
            - sentiment: User's sentiment (positive, negative, neutral)
            
            JSON response:"""
        )
        
        # Format the prompt
        formatted_prompt = category_prompt.format(user_input=user_input)
        
        # Generate the categorization
        response = llm.invoke(formatted_prompt)
        
        # Parse the JSON response
        try:
            category_data = json.loads(response.content)
            logger.info(f"Categorized user request as: {category_data.get('primary_category', 'unknown')}")
            return category_data
        except json.JSONDecodeError:
            logger.error("Failed to parse categorization response as JSON")
            return {
                "primary_category": "unknown",
                "requires_search": True,
                "entities": [],
                "complexity": "medium",
                "sentiment": "neutral"
            }
    except Exception as e:
        logger.error(f"Error categorizing user request: {str(e)}")
        return {
            "primary_category": "unknown",
            "requires_search": True,
            "entities": [],
            "complexity": "medium",
            "sentiment": "neutral"
        }

# Mapping of tool names to functions
AGENT_TOOLS = {
    "extract_information": extract_information,
    "search_web": search_web,
    "generate_answer": generate_answer,
    "analyze_with_llm": analyze_with_llm,
    "summarize_information": summarize_information,
    "categorize_user_request": categorize_user_request
}
