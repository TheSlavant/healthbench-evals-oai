#!/usr/bin/env python
"""
Script to run HealthBench simulations with pre-computed reference scores.
"""

import argparse
import subprocess
import json
from pathlib import Path
from dotenv import load_dotenv
import os
import sys
from sampler.simulate_sampler import SimulateSampler

# Hardcoded reference files for each model
MODEL_REFERENCE_FILES = {
    "gpt-4.1": "data/references/gpt-4.1_results.jsonl",
    "o3": "data/references/o3_results.jsonl",
    "grok-3": "data/references/grok-3_results.jsonl",
}

def load_reference_scores(model: str, dialog_value: str):
    """
    Load reference scores for a given model and dialog.
    
    Args:
        model: The model name
        dialog_value: The dialog content to match in the reference file
        
    Returns:
        Dictionary of scores from the reference file or None if not found
    """
    if model not in MODEL_REFERENCE_FILES:
        print(f"Warning: No reference file found for model {model}")
        return None
        
    ref_file = MODEL_REFERENCE_FILES[model]
    if not os.path.exists(ref_file):
        print(f"Warning: Reference file {ref_file} does not exist")
        return None
        
    with open(ref_file, 'r') as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                # Check if this is the matching dialog (prompt)
                if entry.get("prompt") == dialog_value:
                    return entry.get("scores", {})
            except json.JSONDecodeError:
                continue
                
    print(f"Warning: No matching dialog found in reference file for {model}")
    return None

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Run HealthBench simulations with reference scores")
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
        default=170,
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
        help="Model to simulate (used for the API call)",
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
        default=1024,
        help="Maximum tokens for generation",
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://localhost:3000/api/sample",
        help="API URL for AI SDK sampler",
    )
    args = parser.parse_args()
    
    # Make sure the input file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file {input_path} does not exist.")
        return
    
    # Create data/references directory if it doesn't exist
    references_dir = Path("data/references")
    references_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the SimulateSampler
    sampler = SimulateSampler(
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_url=args.api_url,
    )
    
    # Set environment variables for the sampler configuration
    os.environ["SIMULATE_MODEL"] = args.model
    os.environ["SIMULATE_TEMPERATURE"] = str(args.temperature)
    os.environ["SIMULATE_MAX_TOKENS"] = str(args.max_tokens)
    os.environ["SIMULATE_API_URL"] = args.api_url
    os.environ["SIMULATE_N_REPEATS"] = str(args.n_repeats)
    
    # Construct command
    cmd = [
        "python", "-m", "healthbench_eval",
        "--custom_input_path", str(input_path),
    ]
    
    # Add optional arguments
    if args.examples:
        cmd.extend(["--examples", str(args.examples)])
    if args.n_threads:
        cmd.extend(["--n-threads", str(args.n_threads)])
    if args.n_repeats > 1:
        cmd.extend(["--n-repeats", str(args.n_repeats)])
    
    # Run command
    print(f"Running simulation: {' '.join(cmd)}")
    subprocess.run(cmd)
    
if __name__ == "__main__":
    main() 