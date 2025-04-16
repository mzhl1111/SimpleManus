import time
import asyncio
import mlflow
from typing import Dict, Any, AsyncIterator
from langchain_core.callbacks import BaseCallbackHandler
from langgraph.graph import StateGraph, END


class MLflowTracker(BaseCallbackHandler):
    def __init__(self, experiment_name="LangGraph_Streaming_Agent"):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.start_times = {}
        self.streaming_chunks = 0

        # Set up MLflow experiment
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run()
        self.run_id = self.run.info.run_id

        # Record stream start time
        self.stream_start_time = time.time()

    def on_llm_end(self, response, **kwargs):
        run_id = kwargs.get("run_id")
        if run_id in self.start_times:
            elapsed = time.time() - self.start_times[run_id]
            mlflow.log_metric(f"execution_time_{run_id}", elapsed)

        # Extract token usage if available
        if hasattr(response, "llm_output"):
            usage = response.llm_output["token_usage"]
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += prompt_tokens + completion_tokens

    def get_usage_report(self):
        report = {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "streaming_chunks": self.streaming_chunks,
            "run_id": self.run_id
        }
        return report

    def log_metric(self):
        total_streaming_time = time.time() - self.stream_start_time
        mlflow.log_metric("total_run_time", total_streaming_time)
        mlflow.log_metrics({
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens
        })

    def log_success(self, result):
        mlflow.log_metric("success", result)
