# SimpleManus: Multi-Agent Browser Automation Framework

SimpleManus is a Python-based framework for building and testing multi-agent architectures with browser automation capabilities. It enables the creation of intelligent agents that can collaborate to complete complex web-based tasks through a combination of planning and execution.

## Features

- **Multiple Agent Architectures**: Supports three different agent architectures:
  - **ReAct**: Single agent that alternates between reasoning and taking actions
  - **Plan and Execute**: Two-phase approach with planning and execution agents
  - **ReWOO**: Reasoning Without Observation architecture
- **Browser Automation**: Comprehensive set of browser interaction tools including:
  - Navigation (URL navigation, back/forward)
  - Element interaction (clicking, text input, scrolling)
  - Content extraction
  - Tab management
  - Form handling (dropdowns, text inputs)
- **Planning Capabilities**: Agents can create, update, and track progress of task plans
- **Asynchronous Operations**: Built with async/await for efficient browser operations
- **Integration with LLMs**: Leverages OpenAI's GPT models for intelligent decision making

## Quick Start

Here's a simple example of how to use the ReAct architecture to automate a web task:

```python
from ReAct.multi_agent_ver import get_multi_agent_graph
import asyncio

async def run_task(task_description):
    graph = get_multi_agent_graph()
    inputs = {
        "messages": [
            ("user", task_description)
        ]
    }
    await graph.ainvoke(inputs)

# Example: Find and book a flight
asyncio.run(run_task("Help me find and book a flight from Boston to New York"))
```

## Architecture

The framework consists of several key components:

1. **Planning Agent**: Creates and manages task plans
2. **Browser Agent**: Executes browser automation tasks
3. **Browser Tools**: Comprehensive set of browser interaction capabilities
4. **State Management**: Tracks and manages the state of multi-agent interactions

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SimpleManus.git
cd SimpleManus
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Set up environment variables:
Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your-api-key
```

## Running Experiments

The `experiments` folder contains various test cases and experiments for each agent architecture. To run experiments:

1. Navigate to the experiments directory:
```bash
cd experiments
```

2. Run specific experiments:
```bash
# Run ReAct experiments
python react_experiments.py

# Run Plan and Execute experiments
python plan_execute_experiments.py

# Run ReWOO experiments
python rewoo_experiments.py
```

Each experiment file contains multiple test cases that demonstrate different capabilities of the agents. You can modify the test cases or add new ones to test specific scenarios.

## Browser Tools

The framework provides a comprehensive set of browser automation tools:

- `go_to_url`: Navigate to a specific URL
- `click_element`: Click elements on the page
- `input_text`: Enter text into form fields
- `scroll_down/scroll_up`: Scroll the page
- `get_dropdown_options`: Get options from dropdown menus
- `select_dropdown_option`: Select options from dropdowns
- `extract_content`: Extract specific information from pages
- `switch_tab`: Manage browser tabs
- And more...

## Development

The project uses Poetry for dependency management and follows a modular structure:

- `ReAct/`: Implementation of ReAct architecture
- `plan_and_execute/`: Implementation of Plan and Execute architecture
- `ReWOO/`: Implementation of ReWOO architecture
- `tools/`: Browser automation tools and utilities
- `utils/`: Helper functions and utilities
- `experiments/`: Test cases and experiments for each architecture

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.