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

- **Dynamic Planning**: Creates custom execution plans for each user query
- **Web Search Integration**: Searches the web for relevant information
- **Information Extraction**: Identifies key entities and themes in user queries
- **Answer Generation**: Synthesizes information into coherent, comprehensive answers
- **Error Handling**: Detects failures and replans when necessary
- **Time Awareness**: Includes current date/time information to maintain temporal context
- **Fact Verification**: Uses a hardcoded facts database for key information like political leaders
- **Data Consistency**: Identifies and flags conflicting information in search results
- **Structured Parsing**: Employs robust key-value extraction for consistent information processing

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

### Fact Verification Database

The system includes a hardcoded facts database (`FACTS_DB`) that contains verified information about key topics like political leaders. This helps ensure accurate responses even when search results contain outdated or conflicting information.

### Time Awareness

The agent maintains awareness of the current date and time, allowing it to:

- Properly contextualize "current" events
- Flag potentially outdated information
- Indicate when information may be speculative (future-dated)

### Robust Information Extraction

Information extraction uses a reliable key-value format that is:

- More robust than JSON parsing
- Less prone to syntax errors
- Easier to extract partial information
- More consistently produced by LLMs

### Data Consistency Checking

The system detects conflicting information in search results and:

- Flags potential inconsistencies to the user
- Prioritizes verified facts from the facts database
- Provides appropriate caveats when information reliability is uncertain

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
