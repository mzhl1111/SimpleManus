import time
import mlflow
import tiktoken
from typing import Dict
from langchain_core.callbacks import BaseCallbackHandler


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
        
        # Initialize tiktoken encoder for accurate counting
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text: str) -> int:
        """Count tokens accurately using tiktoken"""
        if not text:
            return 0
        try:
            return len(self.encoding.encode(text))
        except Exception:
            # Fallback to character estimation if tiktoken fails
            return len(text) // 4  # Rough estimate: ~4 chars per token

    def on_llm_end(self, response, **kwargs):
        run_id = kwargs.get("run_id")
        if run_id in self.start_times:
            elapsed = time.time() - self.start_times[run_id]
            mlflow.log_metric(f"execution_time_{run_id}", elapsed)

        # Extract token usage if available
        token_usage = {}
        
        # Try different formats for token tracking
        if hasattr(response, "llm_output") and response.llm_output:
            # Standard LangChain format
            if "token_usage" in response.llm_output:
                token_usage = response.llm_output["token_usage"]
        
        # Try to extract from generations for OpenRouter format
        if not token_usage and hasattr(response, "generations"):
            for gen_list in response.generations:
                for gen in gen_list:
                    if hasattr(gen, "generation_info") and gen.generation_info:
                        # OpenRouter often puts token info in generation_info
                        if "token_usage" in gen.generation_info:
                            token_usage = gen.generation_info["token_usage"]
                        # Sometimes directly in generation_info
                        elif "prompt_tokens" in gen.generation_info:
                            token_usage = {
                                "prompt_tokens": gen.generation_info.get("prompt_tokens", 0),
                                "completion_tokens": gen.generation_info.get("completion_tokens", 0),
                                "total_tokens": gen.generation_info.get("total_tokens", 0)
                            }
        
        # If no token usage found in response, calculate it manually
        if not token_usage:
            # Get prompt and completion text
            prompt_text = ""
            completion_text = ""
            
            # Extract prompt from input messages
            if hasattr(response, "prompt") and response.prompt:
                prompt_text = response.prompt
            
            # Extract completions from generations
            if hasattr(response, "generations"):
                for gen_list in response.generations:
                    for gen in gen_list:
                        if hasattr(gen, "text"):
                            completion_text += gen.text
            
            # Count tokens using tiktoken
            prompt_tokens = self.count_tokens(prompt_text)
            completion_tokens = self.count_tokens(completion_text)
            total_tokens = prompt_tokens + completion_tokens
            
            token_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        
        # If we have token usage information, log it
        if token_usage:
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            total_tokens = token_usage.get("total_tokens", 0) or (prompt_tokens + completion_tokens)
            
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += total_tokens
            
            # Log individual call token usage
            mlflow.log_metrics({
                f"prompt_tokens_{run_id}": prompt_tokens,
                f"completion_tokens_{run_id}": completion_tokens,
                f"total_tokens_{run_id}": total_tokens
            })
        
        # Even if we don't find token info, log the completion
        self.streaming_chunks += 1

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
