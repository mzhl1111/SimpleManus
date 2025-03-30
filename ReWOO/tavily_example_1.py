"""
Example 1: Using TavilySearchResults with include_answer parameter
to get both search results and answer in a single API call, with tool call invocation
https://python.langchain.com/api_reference/_modules/langchain_community/tools/tavily_search/tool.html#TavilySearchResults 
"""

import os
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults

# Load API key from .env file
load_dotenv()

# Verify API key is loaded
if os.getenv("TAVILY_API_KEY") is None:
    print("Tavily API key is not set in .env file")
    exit(1)

# Create the search tool with include_answer=True
search_tool = TavilySearchResults(
    max_results=1,
    include_answer=True,  # This will include an answer in the response
    # include_raw_content=True,  # Optional: includes parsed HTML of search results
    # include_images=True,  # Optional: includes related images
)

# Execute the search
query = "When will be the next canadian federal election?"

# Tool call invocation (to get artifact with answer)
print("METHOD 2: TOOL CALL INVOCATION")
tool_response = search_tool.invoke({
    "args": {'query': query},
    "type": "tool_call",
    "id": "search_1",
    "name": "tavily"
})

print(f"Tool response type: {type(tool_response)}")
print(f"Tool response attributes: {dir(tool_response)[:10]}")
print("\n" + "-"*50 + "\n")

# Access the artifact which contains all the detailed information
if hasattr(tool_response, 'artifact'):
    artifact = tool_response.artifact
    
    print("ARTIFACT CONTENTS:")
    print(f"Artifact type: {type(artifact)}")
    print(f"Artifact keys: {artifact.keys() if hasattr(artifact, 'keys') else 'Not a dict'}")
    
    # Print the answer from the artifact
    print("\nAnswer from artifact:")
    print(artifact.get("answer", "No answer found"))
    
    # Print the search results from the artifact
    print("\nSearch Results from artifact:")
    results = artifact.get("results", [])
    for i, result in enumerate(results, 1):
        print(f"{i}. {result.get('title', 'No title')}")
        print(f"   URL: {result.get('url', 'No URL')}")
        print(f"   Content: {result.get('content', 'No content')[:150]}...\n")
    
    # Print image URLs if available
    images = artifact.get("images", [])
    if images:
        print("\nImage URLs from artifact:")
        for i, image_url in enumerate(images, 1):
            print(f"{i}. {image_url}")
            
    # Print follow-up questions if available
    follow_up = artifact.get("follow_up_questions", None)
    if follow_up:
        print("\nFollow-up questions:")
        for i, question in enumerate(follow_up, 1):
            print(f"{i}. {question}")
            
    # Print response time
    print(f"\nResponse time: {artifact.get('response_time', 'N/A')} seconds")
else:
    print("No artifact found in the tool response") 