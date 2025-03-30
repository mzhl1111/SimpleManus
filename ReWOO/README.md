# ReWOO (Reasoning WithOut Observation)

A framework for implementing reasoning agents that can plan and execute tasks using external tools without direct observation.

## Overview

This project contains three main components:

1. **ReWOO Framework Demo** (`rewoo_example_1.py`): A demonstration of the ReWOO framework that implements a reasoning agent using the search tool.
2. **Tavily Search API Demo** (`tavily_example_1.py`): A simple example showing how to use the Tavily search API with answer generation.
3. **(WORK IN PROGRESS) ReWOO Browser Use** (`rewoo_browseruse.py`): An advanced implementation of the ReWOO framework that combines web search, LLM reasoning, and browser automation to perform complex web-based tasks.

## Features

### Core Features
- Task planning and execution
- Integration with external tools (search, browser automation)
- State management using LangGraph
- Conversation tracking
- Variable substitution in tool inputs

### Browser Automation Features
- Web navigation and interaction
- Dynamic content extraction
- Element interaction (clicking, form filling)
- State tracking of browser sessions
- Automatic cleanup of browser resources

## Requirements

- Python 3.x
- Required packages:
  - langchain
  - langchain-openai
  - langchain-community
  - langgraph
  - python-dotenv
  - browser-use-tool (custom package for browser automation)

## Environment Setup

Create a `.env` file with the following API keys:
```
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_BASE_URL=your_openrouter_base_url
TAVILY_API_KEY=your_tavily_api_key
```

## Usage

### ReWOO Framework Demo

Either run the `rewoo_example_1.py` file directly or edit the `langgraph.json` file to identify the graph and run the langgraph dev server with `langgraph dev`, which pops up a web interface for interacting with the agent.

### Tavily Search Demo

The `tavily_example_1.ipynb` file is a Jupyter notebook that demonstrates how to use the Tavily search API with answer generation.

### ReWOO Browser Use

The `rewoo_browseruse.py` implements an advanced agent that can:
1. Plan complex web-based tasks
2. Execute multi-step operations
3. Navigate websites
4. Extract information
5. Interact with web elements

#### Available Tools
1. **Google[input]**: Search for information using Tavily API
2. **LLM[input]**: Use language model for reasoning and decision making
3. **GoToURL[url]**: Navigate to specific web pages
4. **GetCurrentState[placeholder]**: Get current browser state
5. **ClickElement[index]**: Click on specific web elements
6. **ExtractContent[goal]**: Extract information from web pages

#### Example Usage
```python
initial_state = {
    "task": "What is the name of the director of the highest-grossing movie released in 2023? Go to that director's Wikipedia page and briefly describe their biography.",
    "messages": []
}
```

The agent will:
1. Plan the steps needed to complete the task
2. Execute each step using appropriate tools
3. Track progress and maintain state
4. Generate a final answer based on collected evidence

## Project Status

This is a work in progress. Currently, the framework supports:
- Basic task planning following the ReWOO framework
- Search tool integration using the Tavily API
- Advanced browser automation capabilities
- Complex multi-step web interactions
- State management and conversation tracking
- Variable substitution and result chaining

## Future Improvements
- Enhanced error handling and recovery
- Support for more browser automation features
- Improved state management
- Better handling of dynamic web content
- Additional tool integrations
