
def search_function(query: str) -> str:
    # Here you would implement logic to perform a search.
    # For this example, we simply return a dummy response.
    return f"Simulated search results for: {query}"

def python_function(code: str) -> str:
    # WARNING: Evaluating code can be dangerous. In a production system,
    # use proper sandboxing.
    try:
        # Evaluate the code (here we use eval for simplicity)
        result = eval(code)
        return str(result)
    except Exception as e:
        return f"Error executing code: {e}"

from langchain.agents import Tool

tools = [
    Tool(
        name="Search",
        func=search_function,
        description="Useful for searching the internet for current information."
    ),
    Tool(
        name="Python",
        func=python_function,
        description="Executes Python code and returns the result."
    )
]

from langchain.llms import OpenAI
from langchain.agents import initialize_agent, AgentType

# Initialize the language model with a lower temperature for deterministic responses.
llm = OpenAI(temperature=0)

# Create an agent that uses the Zero Shot ReAct prompt.
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,          # Enables detailed logging of the internal chain-of-thought.
    max_iterations=10      # Limits the number of thought/tool-calling cycles.
)


if __name__ == "__main__":
    task = "What is the sum of the numbers from 1 to 10 and provide your reasoning?"
    result = agent.run(task)
    print("Final result:", result)
