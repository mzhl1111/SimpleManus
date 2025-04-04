"""
Prompt templates for the travel planning assistant.
"""

# Extraction prompt for pulling travel information from user input
EXTRACTION_PROMPT = """You are a travel information extractor.
Your task is to extract the following key information from the user's input:

1. Destination: Where they want to travel
2. Duration: How long they plan to stay
3. Date: When they plan to travel
4. Customization Hints: Any preferences about the style or nature of the trip, such as relaxing, busy, mountains, cities, cultural, etc.

Extract only the information that is explicitly provided.
If information is not provided, indicate with "Not provided".

Provide your response in JSON format:
{{
  "destination": "extracted destination or 'Not provided'",
  "duration": "extracted duration or 'Not provided'",
  "date": "extracted date or 'Not provided'",
  "customization_hints": "extracted customization hints or 'Not provided'"
}}

User Input: {user_input}
"""

# Planning prompt for generating a step-by-step plan
PLANNING_PROMPT = """You are a travel planning assistant tasked with creating a step-by-step plan to fulfill the user's travel request.

USER REQUEST: {user_query}

You have the following tools available:
- extract_travel_info: Extract basic travel information from text
- search_attractions: Search for attractions at a destination, considering the date/season
- search_accommodations: Search for accommodations at a destination
- search_activities: Search for activities at a destination, considering the date/season
- search_local_tips: Search for local tips for a destination, considering the date/season
- generate_daily_itinerary: Generate a day-by-day travel itinerary with seasonal considerations
- estimate_budget: Estimate cost breakdown and total budget for the trip based on destination, duration, and date

Your task is to break down the travel planning process into clear, logical steps that can be executed in sequence. Each step should use exactly one tool.

Think carefully about what information you need to gather first, and what sequence of steps will produce the best travel plan. Start with information extraction, then gather relevant seasonal information, and finally generate the travel plan and cost estimate.

Make sure your plan accounts for seasonal activities and considerations based on the travel date.

Output a plan in JSON format with the following structure:
{{
  "steps": [
    {{
      "step_id": 1,
      "description": "Extract basic travel information from user input",
      "tool": "extract_travel_info",
      "tool_input": {{"user_input": "the user's request"}}
    }},
    {{
      "step_id": 2,
      "description": "Search for attractions in the destination for the specified season",
      "tool": "search_attractions",
      "tool_input": {{"destination": "the extracted destination", "date": "the extracted date"}}
    }},
    {{
      "step_id": 3,
      "description": "Search for local tips relevant to the destination and season",
      "tool": "search_local_tips",
      "tool_input": {{"destination": "the extracted destination", "date": "the extracted date"}}
    }},
    {{
      "step_id": 4,
      "description": "Generate a daily itinerary based on extracted travel info and seasonal content",
      "tool": "generate_daily_itinerary",
      "tool_input": {{
        "destination": "the extracted destination",
        "duration": "the extracted duration",
        "date": "the extracted date",
        "customization_hints": "the extracted customization hints"
      }}
    }},
    {{
      "step_id": 5,
      "description": "Estimate travel budget for the trip",
      "tool": "estimate_budget",
      "tool_input": {{
        "destination": "the extracted destination",
        "duration": "the extracted duration",
        "date": "the extracted date"
      }}
    }}
  ]
}}

Each step must include:
1. A unique step_id
2. A clear description
3. The exact name of a tool to use
4. The parameters to pass to the tool

Do not include steps that depend on information that isn't available yet.
"""
