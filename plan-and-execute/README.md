# Plan-and-Execute AI Agent

A general-purpose AI agent that uses the Plan-and-Execute pattern to answer questions and solve problems.

## Overview

This repository implements a framework for a general-purpose AI agent that:

1. Takes a user's question or request
2. Breaks it down into a sequence of steps using LLM planning
3. Executes the steps in order, gathering information as needed
4. Synthesizes the results into a coherent response

The agent uses LangGraph for workflow management and can handle a wide variety of queries by dynamically creating execution plans tailored to each request.

## Features

- **Plan-and-Execute Architecture**: Creates detailed plans for addressing user requests, then executes them step by step
- **Real-time Web Search**: Uses Tavily API to retrieve up-to-date information
- **Extraction and Analysis**: Pulls relevant details from user input
- **Smart Error Handling**: Detects and recovers from failures by replanning
- **Conflict Resolution**: Identifies and manages conflicting information from sources
- **Tool System**: Flexible toolkit of specialized components for different tasks

## Getting Started

### Prerequisites

- Python 3.8+
- OpenRouter API key (or OpenAI API key)
- Tavily API key (for search functionality)

### Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy the `.env.example` file to `.env` and add your API keys:
   ```
   cp .env.example .env
   # Edit .env file to add your API keys
   ```

### Usage

Run the agent with:

```
python run_app.py
```

Then enter your question when prompted.

## How It Works

The agent follows the Plan-and-Execute pattern using a structured workflow:

1. **Planning**: When a user submits a query, the planner breaks it down into steps
2. **Execution**: Each step is executed in sequence, using tools like:
   - Information extraction
   - Web search
   - Text analysis
   - Answer generation
3. **Replanning**: If a step fails, the agent replans to work around the failure
4. **Response Generation**: Results are synthesized into a final answer

## Architecture

The system consists of several key components:

- **AgentGraph**: The core workflow manager that orchestrates the plan-and-execute pattern
- **Planner**: Creates execution plans for user queries
- **Executor**: Runs individual steps in the plan
- **Replanner**: Handles failures by creating new plans
- **Agent Tools**: Collection of functions that perform specific tasks:
  - `extract_information`: Identifies key entities and concepts using key-value format
  - `search_web`: Searches for information on the web
  - `generate_answer`: Creates comprehensive answers with fact verification
  - `analyze_with_llm`: Analyzes text with specific instructions
  - `summarize_information`: Synthesizes multiple texts
  - `categorize_user_request`: Categorizes the type of request

## Advanced Features

### Information Processing

The system performs several key information handling tasks:

1. **Extraction**: Pulls structured data from user inputs, recognizing:

   - Key entities
   - Timeframes
   - Special constraints
   - User intent and focus areas

2. **Search and Retrieval**: Performs targeted web searches to get the most current information

3. **Verification and Integration**: Compares information from multiple sources to identify potential conflicts

4. **Contextual Analysis**: Considers the current date when evaluating time-sensitive information

5. **Structured Response Generation**: Creates comprehensive answers that directly address the user's query

## Extending the Agent

You can extend the agent by adding new tools:

1. Create a new function in `agent_tools.py`
2. Add it to the `AGENT_TOOLS` dictionary
3. The planner will automatically be able to use your new tool

## License

[MIT License](LICENSE)

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Uses [Tavily](https://tavily.com/) for web search
