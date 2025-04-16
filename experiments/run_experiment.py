#!/usr/bin/env python3
"""
Script to create and track an experiment with MLflow.
This script sets up an experiment and logs metrics in a format that can be collected
by the experiment_metrics collection script.
"""
import asyncio
import os

import dotenv
import mlflow
from tqdm import tqdm

from ReAct.react_ver import get_react_graph
from utils.mlflow_callback import MLflowTracker
from langchain.globals import set_verbose


async def create_and_log_experiment(experiment_name, task, get_graph_func=get_react_graph):
    inputs = {
        "messages": [
            (
                "user",
                task,
            )
        ],
    }


    handler = MLflowTracker(experiment_name=experiment_name)
    graph = get_graph_func()
    try:
        # Changed from ainvoke to regular invoke
        messages = await graph.ainvoke(inputs, {"callbacks": [handler], "recursion_limit": 50})
    except Exception as e:
        handler.log_metric()
        handler.log_success(0)
        print(str(e))
    else:
        handler.log_metric()
        handler.log_success(1)
        for m in messages['messages']:
            m.pretty_print()

    finally:
        mlflow.end_run()




def main():
    # Hard-coded experiment names to create and track
    experiments_to_track = (
        "Travel Planning", """Plan a 7 day trip from Vancouver to tokyo, you need to include details about attractions ,food, transportation, and hotel""",
        # "Academic Paper Aggregation", "Academic Paper Aggregation: Traverse first 3 reference links from the paper 'Attention is all you need''s wiki page to compile a bibliography of related works",
        # "Product Comparison" , """Compare Alienware x16 and Macbook Pro 16inch M4 extract detailed specifications and possiblely benchmark score for for each item, such as processor type, RAM, GPU, storage, display size, and price."""
    )

    agent = "React"
    exp_name, t = experiments_to_track
    print(f"\nTracking experiment: {agent} {exp_name}")
    try:
        asyncio.run(create_and_log_experiment(experiment_name=f"{agent} {exp_name}", task=t))
        print(f"Successfully tracked experiment '{exp_name}'")
    except Exception as e:
        print(f"Failed to track experiment '{exp_name}': {str(e)}")


if __name__ == "__main__":
    dotenv.load_dotenv()
    mlflow.set_tracking_uri("http://localhost:5001")

    main()