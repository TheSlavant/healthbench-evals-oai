#!/usr/bin/env python
"""
Example script to run HealthBench evaluations using a local consensus.jsonl file.
"""

import argparse
import subprocess
from pathlib import Path
from dotenv import load_dotenv
import os
import sys
from sampler.ai_sdk_sampler import AISDKSampler

def run_healthbench(
    input_file="data/consensus_10.jsonl",
    examples=None,
    n_threads=10,
    n_repeats=1,
    model="gpt-4.1",
    temperature=1.0,
    max_tokens=2048,
    api_url="http://localhost:3000/api/sample",
    prompt=None
):
    """
    Run HealthBench evaluation with specified parameters
    
    Args:
        input_file (str): Path to local jsonl file containing evaluation examples
        examples (int, optional): Number of examples to run
        n_threads (int): Number of threads to run
        n_repeats (int): Number of times to repeat each example (for averaging)
        model (str): Model to evaluate
        temperature (float): Temperature for sampling
        max_tokens (int): Maximum tokens for generation
        api_url (str): API URL for AI SDK sampler
        prompt (str, optional): Custom prompt string to pass through the AI SDK sampler
        
    Returns:
        subprocess.CompletedProcess: Result of the subprocess run
    """
    # Make sure the input file exists
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        return None
    
    # Initialize the AISDKSampler
    sampler = AISDKSampler(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_url=api_url,
        prompt=prompt,
    )
    
    # Set environment variables for the sampler configuration
    os.environ["AI_SDK_MODEL"] = model
    os.environ["AI_SDK_TEMPERATURE"] = str(temperature)
    os.environ["AI_SDK_MAX_TOKENS"] = str(max_tokens)
    os.environ["AI_SDK_API_URL"] = api_url
    os.environ["AI_SDK_N_REPEATS"] = str(n_repeats)
    
    # Set the prompt environment variable if provided
    if prompt:
        os.environ["AI_SDK_PROMPT"] = prompt
    
    # Construct command
    cmd = [
        "python", "-m", "healthbench_eval",
        "--custom_input_path", str(input_path),
    ]
    
    # Add optional arguments
    if examples:
        cmd.extend(["--examples", str(examples)])
    if n_threads:
        cmd.extend(["--n-threads", str(n_threads)])
    if n_repeats > 1:
        cmd.extend(["--n-repeats", str(n_repeats)])
    
    # Run command
    print(f"Running: {' '.join(cmd)}")
    return subprocess.run(cmd)

def run_healthbench_cli():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Run HealthBench with local file")
    parser.add_argument(
        "--input_file", 
        type=str, 
        default="data/consensus_10.jsonl",
        help="Path to local jsonl file containing evaluation examples"
    )
    parser.add_argument(
        "--examples", 
        type=int, 
        help="Number of examples to run"
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=10,
        help="Number of threads to run",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=1,
        help="Number of times to repeat each example (for averaging)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1",
        help="Model to evaluate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens for generation",
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://localhost:3000/api/sample",
        help="API URL for AI SDK sampler",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt string to pass through the AI SDK sampler",
    )
    args = parser.parse_args()
    
    run_healthbench(
        input_file=args.input_file,
        examples=args.examples,
        n_threads=args.n_threads,
        n_repeats=args.n_repeats,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_url=args.api_url,
        prompt=args.prompt
    )

def main():
  run_healthbench(
    model="auto",
    input_file="data/consensus_1.jsonl",
    examples=None,
    n_repeats=1,
    n_threads=17,
    api_url="http://localhost:3000/api/sample",
    temperature=1.0, #TODO: Figure how we pass this to model server side
    max_tokens=2048, #TODO: Figure how we pass this to model server side
    prompt=None,
  )
    
if __name__ == "__main__":
    main() 