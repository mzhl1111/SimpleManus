# Travel Planning Assistant

An intelligent travel planning assistant based on the Plan-and-Execute pattern, capable of automatically generating complete travel plans from simple user inputs, including destination recommendations, itinerary arrangements, and budget estimates.

## Project Architecture

This project adopts the Plan-and-Execute pattern, using LangGraph for workflow management. Core components include:

```
┌───────────────┐      ┌───────────────┐      ┌───────────────┐
│   Planner     │──────▶   Executor    │──────▶   Replanner   │
└───────────────┘      └───────────────┘      └───────────────┘
```

- **Planner**: Generates travel plan steps based on user input
- **Executor**: Executes each plan step, calling the appropriate tool functions
- **Replanner**: Analyzes execution results, decides whether to continue executing, modify the plan, or complete the output

### Core File Structure

- `travel_graph.py`: Main workflow management, implements state graph and node logic
- `planner_engine.py`: Contains specific implementations of Planner, Executor, and Replanner
- `travel_tools.py`: Various travel tool functions such as information extraction, attraction search, budget estimation, etc.
- `information_extractor.py`: Extracts travel information from user input
- `search_engine.py`: Search engine interface for obtaining real travel data
- `prompt_templates.py`: Prompt template library for generating high-quality LLM prompts
- `config.py`: System configuration
- `run_travel_app.py`: Main entry program

## Technology Stack

- **LangGraph**: LangChain-based workflow engine for building Plan-and-Execute processes
- **OpenAI**: Uses the GPT-4o model for natural language understanding and generation
- **Tavily API**: For real-time search of travel information and price data
- **LangSmith**: For tracking and monitoring LLM applications
- **Python**: Core development language, version 3.9+

## Installation Guide

1. Clone the repository

```bash
git clone <repository-url>
cd travel-plan-assistant
```

2. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Configure environment variables
   Copy `.env.example` to `.env`, and fill in the following API keys:

```
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="your_langsmith_api_key_here"
LANGSMITH_PROJECT="travel-plan-assistant"

OPENAI_API_KEY="your_openai_api_key_here"
TAVILY_API_KEY="your_tavily_api_key_here"
```

## Usage Instructions

Run the main program:

```bash
python run_travel_app.py
```

Input examples:

- "Seattle, sakura, 3 days"
- "beach vacation in Miami"
- "safari for 7 days"
- "gray"

The system will automatically:

1. Extract travel information (destination, duration, date, custom requirements)
2. Search for popular attractions
3. Generate daily itinerary
4. Estimate travel budget
5. Provide a complete travel plan

## Key Features

- **Intelligent Information Extraction**: Even with vague user inputs (like "pink"), the system can intelligently infer reasonable travel arrangements
- **Real-time Price Estimation**: Obtains real travel price data through the search engine
- **Dynamic Itinerary Generation**: Automatically generates reasonable daily itinerary arrangements based on destination and duration
- **Error Handling and Recovery**: Even if certain steps fail, the system can still provide reasonable travel recommendations

## Configuration Parameters

The following parameters can be adjusted in `config.py`:

- `DEBUG`: Set to False to hide system execution process logs, showing only the final travel plan results
- `LLM_MODEL`: Choose the LLM model to use, default is "gpt-4o"
