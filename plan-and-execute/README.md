# Plan-and-Execute Agent using LangGraph

This project implements a Plan-and-Execute agent framework using LangGraph and LangChain. The agent can break down complex tasks into smaller steps (plan) and then execute those steps using various tools.

## Features

- **Planning:** Uses an LLM to generate a sequence of steps to achieve a given objective.
- **Execution:** Executes the planned steps using available tools (e.g., web browser interaction, search).
- **Tool Integration:** Leverages LangChain tools for functionalities like web browsing and search.
- **Decision Making:** Employs a ReAct-style agent for deciding the next action during execution.
- **MLflow Tracking:** Integrates with MLflow to log experiment runs, parameters, metrics, and artifacts.

## Setup

1.  **Clone the Repository:**

    ```bash
    git clone <your-repository-url>
    cd plan-and-execute
    ```

2.  **Create and Activate Virtual Environment:**

    - **Windows (PowerShell):**
      ```powershell
      python -m venv venv
      .\venv\Scripts\Activate.ps1
      ```
    - **macOS/Linux:**
      ```bash
      python3 -m venv venv
      source venv/bin/activate
      ```

3.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    - Copy the example environment file:
      ```bash
      cp .env.example .env
      ```
    - Open the `.env` file and fill in your API keys and other configurations:
      - `OPENROUTER_API_KEY`: Your API key for OpenRouter (or another LLM provider).
      - `LLM_MODEL`: (Optional) Specify the LLM model to use (defaults are set).
      - `LANGSMITH_API_KEY`: (Optional) Your LangSmith API key for tracing.
      - `LANGSMITH_PROJECT`: (Optional) Your LangSmith project name.

## Usage

Run the main application script `run_app.py` with your query:

```bash
python run_app.py --question "Your question or task for the agent"
```

For example:

```bash
python run_app.py --question "What is the current ocean temperature in Yokohama?"
```

The agent will then plan and execute the steps to answer your question. The final result will be printed to the console.

## MLflow Tracking

This project uses MLflow to track agent runs.

- Experiment data is stored locally in the `mlruns` directory.
- To view the MLflow UI:
  1.  Make sure you are in the project's root directory.
  2.  Run the command:
      ```bash
      mlflow ui
      ```
  3.  Open your web browser and navigate to `http://127.0.0.1:5000` (or the address provided by the command).

You can inspect parameters (like the query), metrics (like token counts, execution time), and potentially artifacts for each run.

## Tools

The agent utilizes several tools, including:

- **Planning Tools:** `create_plan`, `update_plan`, `update_status` used by the planner agent.
- **Browser Tools:** Tools for navigating web pages, extracting content, clicking elements, inputting text, scrolling, etc. (Managed by `BrowserUseTool`).
- **(Potentially) Search Tools:** Standard web search capabilities.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details (if available).
