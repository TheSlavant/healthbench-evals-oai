import os
import json
import asyncio
import subprocess
import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime
from colorama import Fore, Style, init

import aiohttp
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Import run_healthbench from hb.py
from hb import run_healthbench

# Initialize colorama
init(autoreset=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format=f"{Fore.CYAN}[%(asctime)s]{Style.RESET_ALL} %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("prompt_evolution")

# Load environment variables, e.g. OPENAI_API_KEY
load_dotenv()

BASE_PROMPT = """Select the best model to continue the conversation. Analyze the messages up to this point, if any, and decide which model will provide the best next response. Return only the model name and the reason why this model was chosen.

Select the model based only on the descriptions below and nothing else:

""" + """grok-3
- The user's message contains reducible uncertainty—there is missing health-related context that could change the advice, and an extra question (or two) would resolve it.
- The task requires understanding the user's personal context (e.g., history, lifestyle, prior answers) and integrating it before responding.
- A correct answer hinges on the assistant's ability to identify which contextual details are important and explicitly request them.
- The situation calls for active context-seeking behavior rather than an immediate recommendation; Grok 3 excels here (largest margin over other models in \"Any reducible uncertainty – Context-seeking\").
- The assistant must hedge responsibly while information is incomplete – Grok 3 leads in hedging for both reducible and irreducible uncertainty, avoiding over-confidence.
- The health question is not an emergency (those route to o3), but still serious enough that guessing could be harmful.
- The conversation is long-form or conversational, where follow-up questions, clarifications, and step-by-step reasoning improve safety and usefulness.
- The user has provided health data that is insufficient or ambiguous (e.g., partial lab values, unclear symptom timelines), so the assistant must surface what's missing before advising.
- The topic involves global or situational healthcare nuances that are unclear; Grok 3 will surface those contextual gaps instead of assuming a single standard.
- Don't choose Grok 3 when uncertainty is irreducible or absent. Irreducible uncertainty means questions where no additional user context can meaningfully change the answer (e.g., long-term prognosis probabilities given all known data, inherently stochastic outcomes).

""" + """gpt-4.1
- The user explicitly wants a short, precise answer—speed and clarity are higher-priority than deep reasoning or additional questions.
- All necessary health context is already known (e.g., after a long exchange); no clarification is needed before giving advice.
- The request is non-emergency (critical or time-sensitive cases are routed to o3).
- The task is a \"query requiring a simple response – appropriate depth\" where GPT-4.1 leads its peers.
- The reply should show high emotional intelligence (EQ) while staying concise—ideal for sensitive health topics that still need brevity.
- The conversation has reached the point where the user just wants the final recommendation or summary rather than further exploration AND the health question involves minimal or no uncertainty, such that a definitive answer is possible without hedging or context-seeking.

""" + """o3
- The health issue is potentially urgent or an emergency; decisive, high-accuracy guidance is required immediately.
- The task's margin for error is near zero—a wrong answer could cause significant harm, so maximum factual precision matters more than conversational nuance.
- The problem needs broad, general-intelligence reasoning across multiple medical domains or complex differential diagnoses.
- The request involves interpreting or generating images (e.g., photos of rashes, X-rays) alongside text.
- All essential facts appear to be present (no big context gaps), yet the situation is critical enough that conservative, accuracy-first handling is preferred.
- No other model's specialty clearly applies; by default, routing falls to o3 unless uncertainty (→ Grok 3) or pure brevity/EQ (→ GPT-4.1) dominates.
- The user explicitly asks for the model that gives the \"most accurate\" or \"safest possible\" answer in a high-stakes health scenario.
- A quick, confident recommendation is needed without additional clarifying questions, and speed is secondary to correctness.
- The user is asking to analyze a file or an image, or their message implies that analyzing a file or an image is relevant – o3 is best at reasoning over files."""

@dataclass
class Config:
    openai_api_key: str
    openai_model: str = "gpt-4.1"

async def generate_variation(prompt: str, grade: Optional[str], config: Config, client: AsyncOpenAI) -> str:
    """Use OpenAI to create a variant of the given prompt."""
    logger.info(f"{Fore.YELLOW}Generating prompt variation with {Fore.MAGENTA}{config.openai_model}{Fore.YELLOW}...")
    
    system_prompt = f"""You mutate routing prompts for a health-related router.
    Generate an improvement on the provided prompt based on the grade this prompt received. 
    Here is the grade: {grade}""" if grade else "You mutate routing prompts for a health-related router. Generate an improvement of the provided prompt. Do not include any other text in your response."
    
    start_time = time.time()
    completion = await client.chat.completions.create(
        model=config.openai_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    elapsed = time.time() - start_time
    
    variation = completion.choices[0].message.content.strip()
    logger.info(f"{Fore.GREEN}Variation generated in {Fore.CYAN}{elapsed:.2f}s")
    
    # Log a preview of the variation
    preview = variation[:100] + "..." if len(variation) > 100 else variation
    logger.info(f"{Fore.BLUE}Variation preview: {Fore.WHITE}{preview}")
    
    return variation

def run_healthbench_evaluation(prompt: str) -> None:
    """Run the healthbench evaluation with the given prompt using the imported function"""
    logger.info(f"{Fore.YELLOW}Running healthbench evaluation...")
    start_time = time.time()
    
    run_healthbench(
        model="auto",
        input_file="data/consensus_1.jsonl",
        examples=None,
        n_repeats=1,
        n_threads=17,
        temperature=1.0,
        max_tokens=2048,
        prompt=prompt
    )
    
    elapsed = time.time() - start_time
    logger.info(f"{Fore.GREEN}Evaluation completed in {Fore.CYAN}{elapsed:.2f}s")

async def process_prompt(prompt: str, grade: Optional[str], config: Config, client: AsyncOpenAI) -> Tuple[str, dict]:
    variation = await generate_variation(prompt, grade, config, client)
    
    logger.info(f"{Fore.YELLOW}Processing prompt variation...")
    run_healthbench_evaluation(variation)
    
    logger.info(f"{Fore.YELLOW}Loading evaluation results...")
    with open("grade.json", "r") as f:
        grade_data = json.load(f)[-1]
    
    logger.info(f"{Fore.GREEN}Evaluation grade: {Fore.WHITE}{grade_data}")
    return variation, grade_data

async def run_evolutionary_loop(start_prompt: str, config: Config, client: AsyncOpenAI) -> None:
    logger.info(f"{Fore.CYAN}Starting evolutionary prompt optimization process")
    logger.info(f"{Fore.BLUE}Base prompt length: {Fore.WHITE}{len(start_prompt)} chars")
    
    curr_prompt = start_prompt
    grade = None
    
    # Create a log directory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"evolution_logs_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    logger.info(f"{Fore.MAGENTA}Created log directory: {Fore.WHITE}{log_dir}")
    
    # Save the initial prompt
    with open(f"{log_dir}/prompt_initial.txt", "w") as f:
        f.write(curr_prompt)
    
    for iteration in range(10):
        logger.info(f"{Fore.MAGENTA}{'='*50}")
        logger.info(f"{Fore.CYAN}Starting iteration {Fore.WHITE}{iteration+1}/10")
        
        # Process the current prompt
        iteration_start = time.time()
        curr_prompt, grade = await process_prompt(curr_prompt, grade, config, client)
        iteration_time = time.time() - iteration_start
        
        logger.info(f"{Fore.GREEN}Iteration {iteration+1} completed in {Fore.CYAN}{iteration_time:.2f}s")
        
        # Save this iteration's prompt and grade
        with open(f"{log_dir}/prompt_iteration_{iteration+1}.txt", "w") as f:
            f.write(curr_prompt)
        with open(f"{log_dir}/grade_iteration_{iteration+1}.json", "w") as f:
            json.dump(grade, f, indent=2)
        
        logger.info(f"{Fore.YELLOW}Previous Grade: {Fore.WHITE}{grade}")
        logger.info(f"{Fore.YELLOW}New prompt saved to: {Fore.WHITE}{log_dir}/prompt_iteration_{iteration+1}.txt")
        
        # Print a visual progress bar
        progress = (iteration + 1) / 10
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        logger.info(f"{Fore.CYAN}Progress: {Fore.WHITE}|{bar}| {progress*100:.1f}%")

def main() -> None:
    logger.info(f"{Fore.MAGENTA}{'='*50}")
    logger.info(f"{Fore.CYAN}HealthBench Prompt Evolution Tool {Fore.YELLOW}v1.0")
    logger.info(f"{Fore.MAGENTA}{'='*50}")
    
    config = Config(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    )
    
    logger.info(f"{Fore.YELLOW}Initializing with model: {Fore.WHITE}{config.openai_model}")
    client = AsyncOpenAI(api_key=config.openai_api_key)
    
    try:
        start_time = time.time()
        asyncio.run(run_evolutionary_loop(BASE_PROMPT, config, client))
        total_time = time.time() - start_time
        
        logger.info(f"{Fore.MAGENTA}{'='*50}")
        logger.info(f"{Fore.GREEN}Evolutionary process completed in {Fore.CYAN}{total_time:.2f}s")
        logger.info(f"{Fore.YELLOW}Check the evolution_logs_* directory for all outputs")
        logger.info(f"{Fore.MAGENTA}{'='*50}")
    
    except Exception as e:
        logger.error(f"{Fore.RED}Error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
