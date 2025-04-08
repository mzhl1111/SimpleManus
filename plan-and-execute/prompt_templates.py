"""
Prompt templates for the general-purpose assistant.
"""

# Extraction prompt for information from user input
EXTRACTION_PROMPT = """You are an information extraction expert.
Extract clearly provided information from the user's input.

WARNING: Important! You must respond strictly in the following key-value format, one per line, without any additional text:

ENTITIES: entity1, entity2, ...
FOCUS_AREA: extracted focus or 'Not provided'
TIMEFRAME: extracted timeframe or 'Not provided'
CONSTRAINTS: extracted constraints or 'Not provided'
INTENT: extracted user intent or 'Not provided'

If any field is not explicitly provided in the input, use "Not provided".
Do not use colons (:) in your response except after field names.
Do not add any explanations, preambles or additional text.

User Input: {user_input}
"""

# Planning prompt for generating a step-by-step plan
PLANNING_PROMPT = """You are an AI assistant tasked with creating a step-by-step plan to fulfill the user's request.

USER REQUEST: {user_query}

You have the following tools available:
- extract_information: Extract key information from text
- search_web: Search the web for information
- generate_answer: Generate a comprehensive answer to a question
- analyze_with_llm: Analyze text using a language model with specific instructions
- summarize_information: Create a coherent summary from multiple texts
- categorize_user_request: Categorize the type of request the user is making

Your task is to break down the process of answering the user's request into clear, logical steps that can be executed in sequence. Each step should use exactly one tool.

Think carefully about what information you need to gather first, and what sequence of steps will produce the best response. Start with information extraction, then gather relevant information, and finally generate a response.

Output a plan in JSON format with the following structure:
{{
  "steps": [
    {{
      "step_id": 1,
      "description": "Extract key information from user input",
      "tool": "extract_information",
      "tool_input": {{"user_input": "the user's request"}}
    }},
    {{
      "step_id": 2,
      "description": "Categorize the type of request",
      "tool": "categorize_user_request",
      "tool_input": {{"user_input": "the user's request"}}
    }},
    {{
      "step_id": 3,
      "description": "Search for relevant information",
      "tool": "search_web",
      "tool_input": {{"query": "specific search query based on the request"}}
    }},
    {{
      "step_id": 4,
      "description": "Generate a comprehensive answer",
      "tool": "generate_answer",
      "tool_input": {{
        "question": "the original question",
        "search_results": "{{past_steps.search_web}}",
        "use_search": true
      }}
    }}
  ]
}}

Each step must include:
1. A unique step_id
2. A clear description
3. The exact name of a tool to use
4. The parameters to pass to the tool

Create a plan that will result in the most helpful, accurate, and complete response for the user. Adjust the steps based on the specific type of request:
- For factual questions, focus on search and information synthesis
- For subjective questions, include analysis steps
- For complex topics, break down information gathering into multiple targeted searches
- For creative requests, use appropriate analysis tools before generating content

Do not include steps that depend on information that isn't available yet.
"""
