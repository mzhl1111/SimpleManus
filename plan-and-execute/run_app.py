#!/usr/bin/env python
"""
Plan-and-Execute AI Assistant application.
"""
import argparse
import datetime
import logging
import os
from typing import Optional, Dict, Any

from dotenv import load_dotenv

# Import the necessary function for running the agent
from agent_graph import run as run_agent
import config

# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(levelname)-8s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure output directory exists
os.makedirs("outputs", exist_ok=True)


def save_output(content: str, filename: Optional[str] = None) -> str:
    """Save output content to a file"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if not filename:
        filename = f"outputs/assistant_output_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    
    return filename


def format_final_output(result: Dict[str, Any]) -> str:
    """Format final output"""
    # Basic output header
    output = """
====== Plan-and-Execute AI Assistant ======
"""
    
    # If there's error information
    if "error" in result and result["error"]:
        output += f"\nError: {result['error']}\n"
    
    # Final result
    result_content = result.get("result", "No result generated")
    output += f"\nFinal Response:\n{result_content}\n"
    
    return output


def create_output_content(question: str, result: Dict[str, Any], 
                          filename: str) -> str:
    """Create complete output content"""
    # Output header
    output = f"""
====== Plan-and-Execute AI Assistant ======
Output will be saved to: {filename}
Running with question from command line:
"{question}"

Analyzing request...
Generating a response...

"""
    
    # If there's error information
    if "error" in result and result["error"]:
        output += f"Error: {result['error']}\n\n"
    
    # Final result
    result_content = result.get("result", "No result generated")
    output += f"Final Response:\n{result_content}\n\n"
    
    # Completion information
    output += f"Complete output saved to: {filename}\n"
    
    return output


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Plan-and-Execute AI Assistant"
    )
    parser.add_argument(
        "--question", 
        type=str,
        default="help me get the source code of create_react_agent function.",
        help="Question to answer"
    )
    return parser.parse_args()


def check_dependencies():
    """Check if required dependencies are installed"""
    missing_deps = []
    
    # Check for MLflow
    try:
        import mlflow
    except ImportError:
        missing_deps.append("mlflow")
    
    # Handle missing dependencies
    if missing_deps:
        print("Warning: The following optional dependencies are missing:")
        for dep in missing_deps:
            print(f"  - {dep}")
        print("To install missing dependencies, run: pip install " + " ".join(missing_deps))
        print("Some features may be disabled.\n")
        
    return len(missing_deps) == 0


def main():
    """Main function"""
    print("=== Starting Plan-and-Execute Assistant Application ===")
    
    # Load environment variables
    load_dotenv()
    
    # Check dependencies
    has_all_deps = check_dependencies()
    
    # Configure MLflow if available
    if has_all_deps:
        try:
            import mlflow
            # Set MLflow tracking URI - use a local directory if not specified
            mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
            mlflow.set_tracking_uri(mlflow_uri)
            logger.info(f"MLflow tracking configured with URI: {mlflow_uri}")
        except Exception as e:
            logger.warning(f"Failed to configure MLflow: {e}")
    
    # Parse command line arguments
    args = parse_arguments()
    question = args.question
    
    # Create output filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"outputs/assistant_output_{timestamp}.txt"
    
    # Initialize output
    initial_output = f"""
====== Plan-and-Execute AI Assistant ======
Output will be saved to: {output_filename}
Running with question from command line:
"{question}"
Analyzing request...
Generating a response...
"""
    print(initial_output)
    
    try:
        # Log start of execution
        logger.info(f"[MAIN] Starting with question: {question}")
        
        # Run agent graph
        logger.info("[MAIN] Running agent graph...")
        result = run_agent(question)
        logger.info("[MAIN] Agent graph execution finished.")
        
        # Format output
        final_output = create_output_content(question, result, output_filename)
        
        # Save output
        save_output(final_output, output_filename)
        
        # Print final response
        print(f"Final Response:\n{result.get('result', 'No result generated')}")
        print(f"Complete output saved to: {output_filename}")
        
    except Exception as e:
        logger.error(f"Error in main: {e}", exc_info=True)
        error_output = f"""
====== Plan-and-Execute AI Assistant ======
Error occurred during execution: {str(e)}
"""
        print(error_output)
        save_output(error_output, output_filename)
    
    logger.info("=== Plan-and-Execute Assistant Application Finished ===")


if __name__ == "__main__":
    main() 