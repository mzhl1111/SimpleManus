"""
Main entry point for the travel planning assistant.
Implements the Plan-and-Execute pattern with LangGraph.
"""
import logging
from typing import Dict, Any

import config
from travel_graph import TravelGraph

# Configure logging
logging.basicConfig(
    level=logging.INFO if config.DEBUG else logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_travel_assistant(user_input: str) -> Dict[str, Any]:
    """Run the travel planning assistant with Plan-and-Execute pattern
    
    Args:
        user_input: User's travel request/query
        
    Returns:
        Final state with response
    """
    try:
        # Initialize the travel graph
        travel_graph = TravelGraph()
        
        # Run the graph with user input
        final_state = travel_graph.run(user_input)
        
        return final_state
    except Exception as e:
        logger.error(f"Error running travel assistant: {str(e)}")
        return {
            "error": str(e),
            "response": f"An error occurred: {str(e)}"
        }


def main():
    """Main entry point"""
    print("====== Travel Planning Assistant ======")
    print("Please describe your travel requirements, for example:")
    print(" - 7-day Japan tour")
    print(" - Weekend trip to Paris in June with $1000 budget")
    print(" - Family vacation to Disney World for 5 days")
    
    # Get user input
    user_input = input("\nYour travel requirements: ").strip()
    
    if not user_input:
        print("Please provide your travel requirements.")
        return
    
    print("\nAnalyzing your travel requirements...")
    print("Generating your travel plan...\n")
    
    # Run the assistant
    final_state = run_travel_assistant(user_input)
    
    # Display the plan or error
    if "response" in final_state and final_state["response"]:
        print("\n====== Your Travel Plan ======\n")
        print(final_state["response"])
        print("\n========================\n")
    else:
        error_msg = final_state.get("error", "Unknown error")
        print(f"\nFailed to generate travel plan: {error_msg}")

    print(
        "Thank you for using the Travel Planning Assistant! Have a great trip!"
    )


if __name__ == "__main__":
    main() 