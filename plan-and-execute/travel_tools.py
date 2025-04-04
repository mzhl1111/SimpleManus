import logging
from typing import Dict, Any, List, Optional

from pydantic import SecretStr

import config
from information_extractor import InformationExtractor
from search_engine import SearchEngine
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_travel_info(user_input: str) -> Dict[str, Any]:
    extractor = InformationExtractor(api_key=config.OPENAI_API_KEY)
    extracted_info = extractor.extract_information(user_input)
    
    # If no destination was extracted, try to infer one
    destination = extracted_info.get("destination", "")
    if not destination or (isinstance(destination, str) and destination.lower() == "not provided"):
        suggested_destination = suggest_destination(user_input)
        if suggested_destination:
            extracted_info["destination"] = suggested_destination
            logger.info(f"Suggested destination: {suggested_destination}")
    
    # If there's a destination but no date, search for the best time
    destination = extracted_info.get("destination", "")
    date = extracted_info.get("date", "")
    if destination and (not date or (isinstance(date, str) and date.lower() == "not provided")):
        best_time = search_best_time(destination)
        if best_time and best_time != "current date":
            extracted_info["date"] = best_time
            logger.info(f"Found best travel time: {best_time}")
    
    # If there's a destination but no duration, search for recommended duration
    duration = extracted_info.get("duration", "")
    if destination and (not duration or (isinstance(duration, str) and duration.lower() == "not provided")):
        recommended_duration = search_recommended_duration(destination)
        if recommended_duration:
            extracted_info["duration"] = recommended_duration
            logger.info(f"Found recommended duration: {recommended_duration}")
    
    return extracted_info

def search_best_time(destination: str) -> str:
    """Search for the best time to visit a destination"""
    if not destination:
        return "current date"
    
    logger.info(f"Searching for best time to visit {destination}")
    try:
        search_engine = SearchEngine(api_key=config.TAVILY_API_KEY)
        query = f"best time of year to visit {destination} for tourism"
        results = search_engine.search(query)
        
        if results and len(results) > 0:
            result_text = results[0].get("content", "")
            # Try to extract month or season from results
            months = ["January", "February", "March", "April", "May", "June", 
                     "July", "August", "September", "October", "November", 
                     "December"]
            seasons = ["Spring", "Summer", "Fall", "Autumn", "Winter"]
            
            # Check if any month or season is in the results
            found_time = None
            for month in months:
                if month in result_text:
                    found_time = month
                    break
            
            if not found_time:
                for season in seasons:
                    if season in result_text:
                        found_time = season
                        break
            
            if found_time:
                logger.info(f"Found best time to visit {destination}: {found_time}")
                return found_time
        
        logger.info(f"Could not determine best time for {destination}, using current date")
        return "current date"
    except Exception as e:
        logger.error(f"Error searching best time: {str(e)}")
        return "current date"

def search_attractions(destination: str, date: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    if not destination:
        return []
    
    # Adjust search quantity based on trip duration
    try:
        if isinstance(limit, str) and limit.isdigit():
            limit = int(limit)
    except:
        pass
    
    # If no date is provided or date is "Not provided", search for best travel time
    if not date or date.lower() == "not provided":
        date = search_best_time(destination)
    
    date_str = f" in {date}" if date and date.lower() != "not provided" else ""
    query = f"top {limit} tourist attractions in {destination}{date_str}"
    
    # Add log entry to show seasonal search
    logger.info(f"Searching attractions with seasonal query: '{query}'")
    
    try:
        search_engine = SearchEngine(api_key=config.TAVILY_API_KEY)
        results = search_engine.search(query)
        # Add log entry to show number of results found
        logger.info(f"Found {len(results)} attractions for {destination}{date_str}")
        return results
    except Exception as e:
        logger.error(f"Error searching attractions: {str(e)}")
        return []

def search_accommodations(destination: str, date: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    if not destination:
        return []
    
    # If no date is provided or date is "Not provided", search for best travel time
    if not date or date.lower() == "not provided":
        date = search_best_time(destination)
    
    date_str = f" in {date}" if date and date.lower() != "not provided" else ""
    query = f"best hotels in {destination}{date_str}"
    try:
        search_engine = SearchEngine(api_key=config.TAVILY_API_KEY)
        return search_engine.search(query)
    except Exception as e:
        logger.error(f"Error searching accommodations: {str(e)}")
        return []

def search_activities(destination: str, date: Optional[str] = None, interests: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    if not destination:
        return []
    
    # Adjust search quantity based on trip duration
    try:
        if isinstance(limit, str) and limit.isdigit():
            limit = int(limit)
    except:
        pass
    
    # If no date is provided or date is "Not provided", search for best travel time
    if not date or date.lower() == "not provided":
        date = search_best_time(destination)
    
    date_str = f" in {date}" if date and date.lower() != "not provided" else ""
    query = f"top {limit} things to do in {destination}{date_str}"
    if interests and interests.lower() != "not provided":
        query += f" for {interests}"
    try:
        search_engine = SearchEngine(api_key=config.TAVILY_API_KEY)
        return search_engine.search(query)
    except Exception as e:
        logger.error(f"Error searching activities: {str(e)}")
        return []

def search_local_tips(destination: str, date: Optional[str] = None, limit: int = 5) -> List[Dict[str, Any]]:
    if not destination:
        return []
    
    # If no date is provided or date is "Not provided", search for best travel time
    if not date or date.lower() == "not provided":
        date = search_best_time(destination)
    
    date_str = f" in {date}" if date and date.lower() != "not provided" else ""
    query = f"local tips and advice for travelers in {destination}{date_str}"
    
    # Add log entry to show seasonal search
    logger.info(f"Searching local tips with seasonal query: '{query}'")
    
    try:
        search_engine = SearchEngine(api_key=config.TAVILY_API_KEY)
        results = search_engine.search(query)
        # Add log entry to show number of results found
        logger.info(f"Found {len(results)} local tips for {destination}{date_str}")
        return results
    except Exception as e:
        logger.error(f"Error searching local tips: {str(e)}")
        return []

def generate_daily_itinerary(destination: str, duration: Optional[str] = None,
                             date: Optional[str] = None,
                             attractions: Optional[List[Dict[str, Any]]] = None,
                             activities: Optional[List[Dict[str, Any]]] = None,
                             customization_hints: Optional[str] = None) -> str:

    if not destination:
        return "Cannot generate itinerary without a destination."
    try:
        # Parameter type checking and conversion
        # Ensure destination is a string
        if not isinstance(destination, str):
            destination = str(destination)
            
        # Ensure duration is a string
        if duration is not None:
            if isinstance(duration, int):
                duration = f"{duration} days"
            elif not isinstance(duration, str):
                duration = str(duration)
                
        # Ensure date is a string
        if date is not None and not isinstance(date, str):
            date = str(date)
            
        # Ensure customization_hints is a string
        if customization_hints is not None and not isinstance(customization_hints, str):
            customization_hints = str(customization_hints)
        
        # Ensure attractions is a list
        if attractions is not None and not isinstance(attractions, list):
            attractions = []
            
        # Ensure activities is a list
        if activities is not None and not isinstance(activities, list):
            activities = []
        
        # If no date is provided or date is "Not provided", search for best travel time
        if not date or date.lower() == "not provided":
            date = search_best_time(destination)
        
        # Print original input parameters
        logger.info(f"Generating itinerary for {destination}, duration={duration}, date={date}")
        logger.info(f"Customization hints: {customization_hints}")
        
        # Print attractions and activities' details
        logger.info(f"Received attractions (count: {len(attractions) if attractions else 0}):")
        if attractions:
            for i, attraction in enumerate(attractions):
                logger.info(f"  {i+1}. {attraction.get('title', 'Unknown')} - {attraction.get('content', '')[:100]}...")
        
        logger.info(f"Received activities (count: {len(activities) if activities else 0}):")
        if activities:
            for i, activity in enumerate(activities):
                logger.info(f"  {i+1}. {activity.get('title', 'Unknown')} - {activity.get('content', '')[:100]}...")
        
        attractions_list = attractions or []
        activities_list = activities or []
        attractions_text = "\n".join([f"- {a.get('title', 'Unknown')}" for a in attractions_list])
        activities_text = "\n".join([f"- {a.get('title', 'Unknown')}" for a in activities_list])
        
        duration_text = f" for {duration}" if duration and duration.lower() != "not provided" else ""
        date_text = f" in {date}" if date and date.lower() != "not provided" else ""
        customization_hints_text = customization_hints or "No specific preferences provided."
        
        # Adjust planning based on trip duration
        if duration and duration.lower() != "not provided":
            try:
                days = int(''.join(filter(str.isdigit, duration)))
                if days > 1:
                    day_planning = "Create a detailed day-by-day plan for all {} days of the trip."
                    day_planning = day_planning.format(days)
                else:
                    day_planning = "Create a single day plan that best captures the destination."
            except:
                day_planning = "Create a day-by-day plan based on the appropriate duration for this trip."
        else:
            day_planning = "Create a day-by-day plan that seems appropriate for this destination."

        prompt_template = (
            "Generate a detailed daily itinerary for a trip to {destination}{duration_text}{date_text}.\n\n"
            "Preferences: {customization_hints_text}\n\n"
            "Attractions to include:\n{attractions_text}\n\n"
            "Activities to consider:\n{activities_text}\n\n"
            "{day_planning}\n"
            "Include morning, afternoon, and evening activities with estimated times and practical logistics.\n"
            "Consider the appropriate seasonal activities since the trip is{date_text}."
        )

        # Generate final prompt content and print
        formatted_prompt = prompt_template.format(
            destination=destination,
            duration_text=duration_text,
            date_text=date_text,
            customization_hints_text=customization_hints_text,
            attractions_text=attractions_text or "No specific attractions provided.",
            activities_text=activities_text or "No specific activities provided.",
            day_planning=day_planning
        )
        
        logger.info("=========== FINAL PROMPT ===========")
        logger.info(formatted_prompt)
        logger.info("===================================")

        prompt = PromptTemplate.from_template(prompt_template)
        llm = ChatOpenAI(
            model=config.LLM_MODEL, 
            temperature=0.7,
            api_key=SecretStr(config.OPENAI_API_KEY)
        )
        response = llm.invoke(prompt.format(
            destination=destination,
            duration_text=duration_text,
            date_text=date_text,
            customization_hints_text=customization_hints_text,
            attractions_text=attractions_text or "No specific attractions provided.",
            activities_text=activities_text or "No specific activities provided.",
            day_planning=day_planning
        ))
        if hasattr(response, 'content'):
            return str(response.content)
        else:
            logger.error(f"Itinerary generation returned unexpected type: {type(response)}")
            return "Failed to generate itinerary due to LLM response format."

    except Exception as e:
        logger.error(f"Error generating itinerary: {str(e)}")
        return f"Failed to generate itinerary: {str(e)}"

def _simple_budget_estimate(destination: str, duration: str = "1 day", num_people: int = 1) -> Dict[str, Any]:
    """Simple budget estimation without using search, adjusting base prices according to destination type"""
    import re
    
    # Parse number of days
    match = re.search(r"(\d+)", str(duration))
    days = int(match.group(1)) if match else 1
    
    # Determine cost level based on destination type
    # Convert destination to lowercase and remove extra spaces
    dest_lower = str(destination).lower().strip()
    
    # Default values - medium cost city
    hotel_cost = 100  # per night
    food_cost = 40    # per person per day
    transport_cost = 15  # per person per day
    ticket_cost = 20   # per attraction
    
    # High cost cities list
    high_cost_cities = ["new york", "tokyo", "london", "paris", "dubai", "sydney", 
                        "singapore", "hong kong", "geneva", "zurich", "venice", "rome", 
                        "san francisco", "los angeles", "miami", "tel aviv"]
    
    # Low cost cities list
    low_cost_cities = ["bangkok", "delhi", "cairo", "mexico city", "bucharest", 
                       "hanoi", "delhi", "istanbul", "kathmandu", "quito", "lima", 
                       "manila", "jakarta"]
    
    # Adjust prices based on destination
    for city in high_cost_cities:
        if city in dest_lower or dest_lower in city:
            hotel_cost = 250
            food_cost = 80
            transport_cost = 30
            ticket_cost = 40
            break
            
    for city in low_cost_cities:
        if city in dest_lower or dest_lower in city:
            hotel_cost = 50
            food_cost = 20
            transport_cost = 8
            ticket_cost = 10
            break
    
    # Calculate total costs
    attraction_count = 3  # Assume visiting 3 attractions per day
    
    food_total = food_cost * days * num_people
    transport_total = transport_cost * days * num_people
    ticket_total = ticket_cost * attraction_count * days * num_people
    hotel_total = hotel_cost * (days if days > 0 else 1)  # At least one night stay
    
    total = round(food_total + transport_total + ticket_total + hotel_total, 2)

    return {
        "destination": destination,
        "duration": f"{days} day(s)",
        "people": num_people,
        "cost_breakdown": {
            "food": round(food_total, 2),
            "transport": round(transport_total, 2),
            "tickets": round(ticket_total, 2),
            "hotel": round(hotel_total, 2)
        },
        "estimated_total_usd": total
    }

def extract_cost_from_results(results: List[Dict[str, Any]], default_value: float) -> float:
    """Extract price information from search results"""
    if not results:
        return default_value
    
    # Combine all result content
    all_text = " ".join([result.get("content", "") for result in results if result.get("content")])
    
    # Use regular expressions to extract price values
    import re
    
    # Try to find various price expressions: $50, USD 50, 50 dollars, etc.
    patterns = [
        r'\$(\d+(?:\.\d+)?)',  # $50 or $50.99
        r'(\d+(?:\.\d+)?)\s*(?:USD|dollars)',  # 50 USD or 50 dollars
        r'(?:USD|US\$)\s*(\d+(?:\.\d+)?)',  # USD 50 or US$ 50
        r'(\d+(?:\.\d+)?)\s*per\s*(?:day|night)',  # 50 per day/night
        r'(?:cost|price)\D+(\d+(?:\.\d+)?)'  # cost around 50 or price of 50
    ]
    
    found_prices = []
    
    for pattern in patterns:
        matches = re.findall(pattern, all_text, re.IGNORECASE)
        found_prices.extend([float(price) for price in matches if price])
    
    if found_prices:
        # Handle outliers:
        # 1. If there's only one price, return it directly
        if len(found_prices) == 1:
            return found_prices[0]
            
        # 2. If there are multiple prices, return the median to avoid extreme values
        found_prices.sort()
        mid_idx = len(found_prices) // 2
        if len(found_prices) % 2 == 0:
            return (found_prices[mid_idx-1] + found_prices[mid_idx]) / 2
        else:
            return found_prices[mid_idx]
    
    # If no valid prices found, return default value
    return default_value

def suggest_destination(user_input: str) -> str:
    """Infer travel destination from user input"""
    logger.info(f"Trying to suggest destination from: {user_input}")
    try:
        search_engine = SearchEngine(api_key=config.TAVILY_API_KEY)
        # Modify query to be more directive
        query = f"best travel destination related to {user_input}"
        results = search_engine.search(query)
        
        if results and len(results) > 0:
            result_text = results[0].get("content", "")
            logger.info(f"Search result for destination suggestion: {result_text[:100]}...")
            
            # Use LLM to extract destination name
            try:
                from langchain_openai import ChatOpenAI
                
                # Ensure API key is not None
                api_key = config.OPENAI_API_KEY
                if not api_key:
                    raise ValueError("OpenAI API key is required")
                
                llm = ChatOpenAI(
                    model=config.LLM_MODEL,
                    temperature=0,
                    api_key=SecretStr(api_key)
                )
                
                prompt = (
                    "Extract a specific travel destination name (city or country) from the following text. If there are multiple options, choose the one most suitable for travel.\n\n"
                    f"Text: {result_text}\n\n"
                    "Return only the destination name, without any other text. If you cannot determine a clear destination, return 'Unknown'."
                )
                
                response = llm.invoke(prompt)
                content = ""
                
                # Safely access response content
                if hasattr(response, 'content'):
                    content = str(response.content)
                
                if content and content.strip() and content.strip().lower() != "unknown":
                    destination = content.strip()
                    logger.info(f"LLM extracted destination: {destination}")
                    return destination
                    
            except Exception as e:
                logger.error(f"Error using LLM to extract destination: {str(e)}")
                
            # Backup plan: If LLM extraction fails, try using rules
            import re
            # Try to extract place names in quotes
            quote_match = re.search(r'"([^"]+)"', result_text)
            if quote_match:
                destination = quote_match.group(1)
                logger.info(f"Found destination in quotes: {destination}")
                return destination
                
            # Try to extract common place name forms
            cities_match = re.search(r'(visit|travel to|go to|explore) ([A-Z][a-z]+ ?[A-Z]?[a-z]*)', result_text)
            if cities_match:
                destination = cities_match.group(2)
                logger.info(f"Found destination in text pattern: {destination}")
                return destination
                
            # Try to extract place names directly
            for line in result_text.split('\n'):
                if re.search(r'^[A-Z][a-z]+', line.strip()):
                    potential_dest = line.strip().split(',')[0].split('.')[0]
                    logger.info(f"Found potential destination at line start: {potential_dest}")
                    return potential_dest
        
        logger.warning("Could not suggest a destination")
        return "Paris"  # Return a default popular destination when inference fails
    except Exception as e:
        logger.error(f"Error suggesting destination: {str(e)}")
        return "Paris"  # Default popular destination

def search_recommended_duration(destination: str) -> str:
    """Search for recommended number of days to visit a destination"""
    if not destination:
        return "1 day"
    
    logger.info(f"Searching for recommended duration to visit {destination}")
    try:
        search_engine = SearchEngine(api_key=config.TAVILY_API_KEY)
        query = f"how many days should I spend in {destination} for tourism"
        results = search_engine.search(query)
        
        if results and len(results) > 0:
            result_text = results[0].get("content", "")
            
            # Try to extract number of days from results
            import re
            # Look for number+day pattern
            days_match = re.search(r'(\d+)[\s-]*(days?|nights?)', result_text, re.IGNORECASE)
            if days_match:
                days = days_match.group(1)
                logger.info(f"Found recommended duration for {destination}: {days} days")
                return f"{days} days"
                
            # Look for specific number+duration pattern
            duration_match = re.search(r'(a|one|two|three|four|five|six|seven)[\s-]*(day|week|month)', 
                                      result_text, re.IGNORECASE)
            if duration_match:
                word_to_number = {
                    "a": "1", "one": "1", "two": "2", "three": "3", 
                    "four": "4", "five": "5", "six": "6", "seven": "7"
                }
                number = word_to_number.get(duration_match.group(1).lower(), "1")
                unit = duration_match.group(2).lower()
                
                if unit == "week":
                    days = int(number) * 7
                    result = f"{days} days"
                elif unit == "month":
                    days = int(number) * 30
                    result = f"{days} days"
                else:
                    result = f"{number} {unit}"
                
                logger.info(f"Found recommended duration for {destination}: {result}")
                return result
        
        logger.info(f"Could not determine recommended duration for {destination}, using default")
        return "3 days"  # Default recommended duration
    except Exception as e:
        logger.error(f"Error searching recommended duration: {str(e)}")
        return "3 days"

def estimate_budget(destination: str,
                    duration: str = "1 day",
                    date: Optional[str] = None,
                    num_people: int = 1,
                    itinerary: Optional[str] = None,
                    _recursion_depth: int = 0) -> Dict[str, Any]:
    """Estimate travel budget for a trip based on real data from search.
    
    Args:
        destination: The travel destination
        duration: Trip duration (e.g., "3 days" or 3)
        date: Travel date or season
        num_people: Number of travelers
        itinerary: Generated travel itinerary (optional, not used)
        _recursion_depth: Internal parameter to detect recursive calls
    
    Returns:
        Dictionary with cost breakdown and estimated total
    """
    
    # Validate and normalize input parameters
    if not destination or not isinstance(destination, str):
        # If destination is missing or invalid, use default
        if not destination:
            logger.debug("Missing destination parameter in estimate_budget, using default")
        else:
            logger.debug(f"Converting non-string destination '{destination}' to string")
        destination = str(destination) if destination else "Paris"
    
    # Handle duration parameter - convert int to string if needed
    if isinstance(duration, int) or (isinstance(duration, str) and duration.isdigit()):
        # Convert numeric duration to string format
        days = int(duration) if isinstance(duration, int) else int(duration)
        duration = f"{days} day{'s' if days != 1 else ''}"
        logger.debug(f"Converted numeric duration to string format: {duration}")
    elif not duration or not isinstance(duration, str):
        # Handle invalid duration
        logger.debug(f"Invalid duration format: {duration}, using default")
        duration = "1 day"
    
    if date is not None and not isinstance(date, str):
        # Convert non-string date to string
        logger.debug(f"Converting non-string date '{date}' to string")
        date = str(date)
    
    if not isinstance(num_people, int) or num_people < 1:
        # Fix invalid number of people
        logger.debug(f"Invalid num_people parameter: {num_people}, using default")
        num_people = 1
    
    logger.info(f"Estimating budget for {destination}, duration={duration}, date={date}, people={num_people}")
    
    # Detect recursive call, if it's a recursive call, use simple estimation
    if _recursion_depth > 0:
        logger.warning(f"Detected recursive budget estimation (depth: {_recursion_depth}), using simple estimate")
        return _simple_budget_estimate(destination, duration, num_people)

    try:
        # Parse duration days
        import re
        match = re.search(r"(\d+)", str(duration))
        days = int(match.group(1)) if match else 1
        
        # Prevent recursion, use simple estimation as default value
        simple_estimate = _simple_budget_estimate(destination, duration, num_people)
        default_hotel = simple_estimate["cost_breakdown"]["hotel"] / (days if days > 0 else 1)
        default_food = simple_estimate["cost_breakdown"]["food"] / (days * num_people)
        default_transport = simple_estimate["cost_breakdown"]["transport"] / (days * num_people)
        default_ticket = simple_estimate["cost_breakdown"]["tickets"] / (3 * days * num_people)  # Assume 3 attractions per day
        
        try:
            # Search for actual consumption data based on destination
            search_engine = SearchEngine(api_key=config.TAVILY_API_KEY)
            
            # Search hotel prices
            hotel_query = f"average hotel cost per night in {destination}"
            hotel_results = search_engine.search(hotel_query)
            hotel_cost = extract_cost_from_results(hotel_results, default_hotel)
            logger.info(f"Searched hotel costs in {destination}: ${hotel_cost}/night")
            
            # Search food prices
            food_query = f"average cost of meals per day in {destination} for tourists"
            food_results = search_engine.search(food_query)
            food_cost = extract_cost_from_results(food_results, default_food)
            logger.info(f"Searched food costs in {destination}: ${food_cost}/day")
            
            # Search transportation prices
            transport_query = f"average daily transportation cost in {destination} for tourists"
            transport_results = search_engine.search(transport_query)
            transport_cost = extract_cost_from_results(transport_results, default_transport)
            logger.info(f"Searched transportation costs in {destination}: ${transport_cost}/day")
            
            # Search attraction ticket prices
            tickets_query = f"average cost of tourist attractions in {destination}"
            tickets_results = search_engine.search(tickets_query)
            ticket_cost = extract_cost_from_results(tickets_results, default_ticket)
            attraction_count = 3  # Assume visiting 3 attractions per day
            logger.info(f"Searched attraction costs in {destination}: ${ticket_cost}/attraction")
        except Exception as e:
            logger.error(f"Error searching costs, falling back to simple estimate: {str(e)}")
            return simple_estimate
        
        # Calculate total costs
        food_total = food_cost * days * num_people
        transport_total = transport_cost * days * num_people
        ticket_total = ticket_cost * attraction_count * days * num_people
        hotel_total = hotel_cost * (days if days > 0 else 1)  # At least one night stay
        
        total = round(food_total + transport_total + ticket_total + hotel_total, 2)

        return {
            "destination": destination,
            "duration": f"{days} day(s)",
            "season": date or "Not specified",
            "people": num_people,
            "cost_breakdown": {
                "food": round(food_total, 2),
                "transport": round(transport_total, 2),
                "tickets": round(ticket_total, 2),
                "hotel": round(hotel_total, 2)
            },
            "estimated_total_usd": total
        }

    except Exception as e:
        logger.error(f"Error estimating budget: {str(e)}")
        # If error occurs, fall back to simple estimation
        return _simple_budget_estimate(destination, duration, num_people)

TRAVEL_TOOLS = {
    "extract_travel_info": extract_travel_info,
    "search_attractions": search_attractions,
    "search_accommodations": search_accommodations,
    "search_activities": search_activities,
    "search_local_tips": search_local_tips,
    "generate_daily_itinerary": generate_daily_itinerary,
    "estimate_budget": estimate_budget
}
