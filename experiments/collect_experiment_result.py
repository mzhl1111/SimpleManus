#!/usr/bin/env python3
"""
Script to collect metrics from all runs in specified experiments.
Usage: python collect_metrics.py <experiment_name1> [experiment_name2 experiment_name3 ...]
"""

import sys
import argparse
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient


def get_experiment_metrics(experiment_name):
    """
    Collects metrics from all runs in the specified experiment.

    Args:
        experiment_name (str): Name of the MLflow experiment

    Returns:
        pandas.DataFrame: DataFrame with experiment name, run IDs and their metrics
    """
    # Initialize MLflow client
    client = MlflowClient()

    # Get experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Error: Experiment '{experiment_name}' not found")
        return None

    experiment_id = experiment.experiment_id
    print(f"Found experiment '{experiment_name}' with ID: {experiment_id}")

    # Get all runs for this experiment
    runs = client.search_runs(experiment_ids=[experiment_id])
    if not runs:
        print(f"No runs found for experiment '{experiment_name}'")
        return None

    print(f"Found {len(runs)} runs for experiment '{experiment_name}'")

    # Collect all metrics from all runs
    all_metrics = {}
    metric_keys = set()

    # First pass: collect all metric keys across all runs
    for run in runs:
        run_id = run.info.run_id
        metrics = client.get_run(run_id).data.metrics
        all_metrics[run_id] = metrics
        metric_keys.update(metrics.keys())

    # Create a DataFrame with experiment_name and run_id as first columns and all metrics as additional columns
    df_data = []

    for run_id, metrics in all_metrics.items():
        row = {'experiment_name': experiment_name, 'run_id': run_id}
        for key in metric_keys:
            row[key] = metrics.get(key, None)  # Use None for missing metrics
        df_data.append(row)

    # Create DataFrame and set experiment_name and run_id as the first columns
    metrics_df = pd.DataFrame(df_data)
    if not metrics_df.empty:
        # Ensure experiment_name and run_id are the first columns
        cols = ['experiment_name', 'run_id'] + [col for col in metrics_df.columns if
                                                col not in ['experiment_name', 'run_id']]
        metrics_df = metrics_df[cols]

    return metrics_df


# !/usr/bin/env python3
"""
Script to collect metrics from all runs in specified experiments.
Edit the experiment_names list in the main function to specify which experiments to collect metrics from.
"""

import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient


def get_experiment_metrics(experiment_name):
    """
    Collects metrics from all runs in the specified experiment.

    Args:
        experiment_name (str): Name of the MLflow experiment

    Returns:
        pandas.DataFrame: DataFrame with experiment name, run IDs and their metrics
    """
    # Initialize MLflow client
    client = MlflowClient()

    # Get experiment by name
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Error: Experiment '{experiment_name}' not found")
        return None

    experiment_id = experiment.experiment_id
    print(f"Found experiment '{experiment_name}' with ID: {experiment_id}")

    # Get all runs for this experiment
    runs = client.search_runs(experiment_ids=[experiment_id])
    if not runs:
        print(f"No runs found for experiment '{experiment_name}'")
        return None

    print(f"Found {len(runs)} runs for experiment '{experiment_name}'")

    # Collect all metrics from all runs
    all_metrics = {}
    metric_keys = set()

    # First pass: collect all metric keys across all runs
    for run in runs:
        run_id = run.info.run_id
        metrics = client.get_run(run_id).data.metrics
        all_metrics[run_id] = metrics
        metric_keys.update(metrics.keys())

    # Create a DataFrame with experiment_name and run_id as first columns and all metrics as additional columns
    df_data = []

    for run_id, metrics in all_metrics.items():
        row = {'experiment_name': experiment_name, 'run_id': run_id}
        for key in metric_keys:
            row[key] = metrics.get(key, None)  # Use None for missing metrics
        df_data.append(row)

    # Create DataFrame and set experiment_name and run_id as the first columns
    metrics_df = pd.DataFrame(df_data)
    if not metrics_df.empty:
        # Ensure experiment_name and run_id are the first columns
        cols = ['experiment_name', 'run_id'] + [col for col in metrics_df.columns if
                                                col not in ['experiment_name', 'run_id']]
        metrics_df = metrics_df[cols]

    return metrics_df


def main():
    # Hard-coded list of experiment names to collect metrics from
    experiment_names = [
        "experiment1",  # Replace with your actual experiment names
        "experiment2",
        "experiment3"
    ]

    # Hard-coded output file path (set to None to disable CSV output)
    output_file = "experiment_metrics.csv"

    # Initialize an empty list to store DataFrames for each experiment
    all_dfs = []

    # Process each experiment
    for exp_name in experiment_names:
        # Get metrics DataFrame for this experiment
        metrics_df = get_experiment_metrics(exp_name)
        if metrics_df is not None:
            all_dfs.append(metrics_df)

    # Combine all DataFrames
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Display DataFrame
        print("\nMetrics Summary:")
        print(combined_df)

        # Save to CSV if output path is provided
        if output_file:
            combined_df.to_csv(output_file, index=False)
            print(f"\nMetrics saved to {output_file}")
    else:
        print("No metrics data collected from any experiment.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()