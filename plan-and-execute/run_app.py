"""
Main entry point for the general-purpose assistant.
Implements the Plan-and-Execute pattern with LangGraph.
"""
import logging
from typing import Dict, Any

import config
from agent_graph import AgentGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO if config.DEBUG else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_assistant(user_input: str) -> Dict[str, Any]:
    """Run the general-purpose assistant with Plan-and-Execute pattern
    
    Args:
        user_input: User's query or request
        
    Returns:
        Final state with response
    """
    try:
        # Initialize the agent graph
        agent_graph = AgentGraph()
        
        # Run the graph with user input
        final_state = agent_graph.run(user_input)
        
        return final_state
    except Exception as e:
        logger.error(f"Error running assistant: {str(e)}")
        return {
            "error": str(e),
            "response": f"An error occurred: {str(e)}"
        }


def main():
    """Main entry point"""
    print("====== AI Assistant ======")
    print("I can help you answer questions, solve problems, and provide information on almost any topic.")
    print("Some examples of what you can ask:")
    print(" - How do I prepare for a job interview?")
    print(" - Explain quantum computing in simple terms")
    print(" - What's the best way to learn a new language?")
    
    # Get user input
    user_input = input("\nYour question: ").strip()
    
    if not user_input:
        print("Please provide a question or request.")
        return
    
    print("\nAnalyzing your request...")
    print("Generating a response...\n")
    
    # Run the assistant
    final_state = run_assistant(user_input)
    
    # Display the response or error
    if "response" in final_state and final_state["response"]:
        print("\n====== Your Answer ======\n")
        print(final_state["response"])
        print("\n========================\n")
    else:
        error_msg = final_state.get("error", "Unknown error")
        print(f"\nFailed to generate a response: {error_msg}")

    print(
        "Thank you for using the AI Assistant!"
    )


if __name__ == "__main__":
    main() 