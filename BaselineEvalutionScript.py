import numpy as np
import argparse
import random
import copy
import json
import os
import re
import time
from datetime import datetime
from typing import List, Any, Optional, Dict, Tuple, Union

# Transformers/Torch for execution
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- Reusable Helper Functions ---

def load_dataset(path: str, input_key: str, target_scores_key: str) -> List[Dict[str, Any]]:
    """Loads dataset from a JSON Lines file, using specified keys."""
    data = []
    print(f"Loading dataset from: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                try:
                    if line.strip():
                        record = json.loads(line)
                        # Basic validation
                        if input_key not in record or target_scores_key not in record:
                            # print(f"Warning: Skipping line {line_num+1}: Missing '{input_key}' or '{target_scores_key}'.")
                            continue
                        if not isinstance(record.get(target_scores_key), dict):
                            # print(f"Warning: Skipping line {line_num+1}: '{target_scores_key}' is not a dictionary.")
                            continue
                        data.append(record)
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON on line {line_num+1} in {path}.")
                except Exception as e:
                    print(f"Warning: Error processing line {line_num+1} in {path}: {e}.")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {path}")
        raise
    except Exception as e:
        print(f"Error loading dataset from {path}: {e}")
        raise
    print(f"Loaded {len(data)} valid records from {path} using keys '{input_key}' and '{target_scores_key}'.")
    return data

def get_correct_answer(target_scores: Dict[str, float]) -> Optional[str]:
    """Finds the key corresponding to the 1.0 score. (Unchanged)"""
    if not isinstance(target_scores, dict): return None
    for key, score in target_scores.items():
        try:
                if float(score) == 1.0: return key
        except (ValueError, TypeError): continue
    print(f"Warning: No key found with score 1.0 in target_scores: {target_scores}")
    return None

def get_evaluation_sample(full_dataset: List[Dict], sample_size: Optional[int], seed: Optional[int]) -> List[Dict]:
    """Returns the specific sample to be used for evaluation based on seed and size."""
    full_dataset_size = len(full_dataset)
    if sample_size is None or sample_size >= full_dataset_size:
        print(f"Using full dataset ({full_dataset_size} examples).")
        return full_dataset
    else:
        current_rng = random.Random(seed) # Seeded generator
        run_seeds = [current_rng.randint(1, 2**31 - 1) for _ in range(1)]
        actual_sample_size = min(sample_size, full_dataset_size)
        print(f"Creating sample of size {actual_sample_size} (Seed: {seed}).")
        return current_rng.sample(full_dataset, actual_sample_size)

# --- Standalone Transformer Execution Function ---
# (Adapted from PromptIndividual.execute_prompt_with_transformer)

# Global cache for models and tokenizers
_hf_model_cache: Dict[str, Any] = {}
_hf_tokenizer_cache: Dict[str, Any] = {}

def execute_transformer_prompt(
    prompt_text: str,
    model_id: str,
    exec_config: Dict[str, Any],
    use_cache: bool = True # Enable caching by default for efficiency
) -> Tuple[Optional[str], int, int]:
    """
    Executes a prompt using a specified Hugging Face transformer model.

    Args:
        prompt_text: The complete text prompt to send to the model.
        model_id: The Hugging Face identifier for the model.
        exec_config: Dictionary with execution parameters like 'device',
                     'max_new_tokens', 'quantization', 'temperature', etc.
        use_cache: Whether to use the global cache for models/tokenizers.

    Returns:
        A tuple containing:
        - The generated response text (str) or None if error.
        - Input token count (int).
        - Output token count (int).
    """
    tokenizer = None
    model = None
    device = exec_config.get('device', 'auto')
    quantization = exec_config.get('quantization') # e.g., '4bit', '8bit', None
    cache_key = f"{model_id}_{device}_{quantization}"

    # --- 1. Get Model and Tokenizer (with Caching) ---
    if use_cache and cache_key in _hf_model_cache:
        model = _hf_model_cache[cache_key]
        tokenizer = _hf_tokenizer_cache[cache_key]
    else:
        print(f"  Loading model/tokenizer: {model_id} (Quant: {quantization}, Device: {device})...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
                trust_remote_code=True
            )
            model.eval()
            if use_cache:
                _hf_model_cache[cache_key] = model
                _hf_tokenizer_cache[cache_key] = tokenizer
            print(f"  Model {model_id} loaded.")
        except Exception as e:
            print(f"  Error loading model/tokenizer {model_id}: {e}")
            return None, 0, 0 # Return error state

    # --- 2. Tokenize Input ---
    try:
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(prompt_text, return_tensors="pt", padding=False, truncation=False)
        input_ids = inputs["input_ids"].to(model.device)
        input_token_count = input_ids.shape[1]
    except Exception as e:
        print(f"  Error tokenizing prompt: {e}")
        return None, 0, 0

    # --- 3. Generate Response ---
    output_token_count = 0
    response_text = None
    try:
        with torch.no_grad():
            generation_config = {
                "max_new_tokens": exec_config.get('max_new_tokens', 10), # Default low
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "temperature": exec_config.get('temperature', 0.1),
                "do_sample": exec_config.get('do_sample', False),
                # Add other relevant params from exec_config if needed (top_k, top_p)
            }
            generation_config = {k: v for k, v in generation_config.items() if v is not None}

            outputs = model.generate(input_ids, **generation_config)
            output_token_count = outputs.shape[1] - input_token_count
            if output_token_count < 0: output_token_count = 0
            generated_ids = outputs[0, input_token_count:]
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    except Exception as e:
        print(f"  Error during model generation: {e}")
        return None, input_token_count, 0 # Return input tokens even if generation fails

    return response_text.strip() if response_text else None, input_token_count, output_token_count


# --- Baseline Prompt Formatting Functions ---

def format_zero_shot(input_text: str) -> str:
    """Creates a simple zero-shot prompt."""
    # Basic instruction suitable for the yes/no implicature task
    instruction = "Analyze the conversation. Does the second speaker's response imply 'yes' or 'no'? Respond with only 'yes' or 'no'."
    return f"{instruction}\n\nConversation:\n{input_text}\n\nAnswer:"

def format_contextual(input_text: str) -> str:
    """Creates a prompt with fixed context."""
    context = "You are evaluating conversational implicatures. Determine if the second speaker's response implies 'yes' or 'no' to the underlying question or statement."
    instruction = "Respond with only 'yes' or 'no'."
    return f"{context}\n\n{instruction}\n\nConversation:\n{input_text}\n\nAnswer:"

# --- Baseline Configuration ---

BASELINE_MODELS = [
    # Include the same models used in the evolution for fair comparison
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", # Check availability/requirements
    "Qwen/Qwen2.5-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "microsoft/Phi-4-mini-instruct",
    "google/gemma-3-1b-it",
    # Add others if used in main experiment
]

# Map names to formatting functions
BASELINE_PROMPT_FORMATTERS = {
    "ZeroShot": format_zero_shot,
    # "Contextual": format_contextual,
}


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Simplified Baseline Prompt Evaluation")
    parser.add_argument("--dataset", required=True, help="Path to the evaluation dataset (JSON Lines)")
    parser.add_argument("--sample_data_size", type=int, default=None, help="Evaluation dataset sample size (None=full)")
    parser.add_argument("--seed", type=int, required=True, help="Seed for evaluation sampling (MUST match main experiment)")
    parser.add_argument("--output_dir", default=".", help="Directory to save baseline results")
    parser.add_argument("--drive_dir", default="PromptEvolutionResults/Baselines_Simple", help="Subdirectory name within MyDrive for Colab output")
    # Add arguments for transformer_config if needed
    parser.add_argument("--max_new_tokens", type=int, default=10, help="Max new tokens for generation")
    parser.add_argument("--device", default="auto", help="Device for transformer ('auto', 'cuda', 'cpu')")

    args = parser.parse_args()

    # --- Detect Colab and Mount Drive ---
    # (Same logic as before)
    running_in_colab = False; google_drive_mount_point = "/content/drive"
    try: import google.colab; running_in_colab = True; print("Colab detected.")
    except ImportError: print("Not in Colab.")
    if running_in_colab:
        try: from google.colab import drive; drive.mount(google_drive_mount_point, force_remount=True); print("Drive mounted.")
        except Exception as e: print(f"Drive mount error: {e}. Saving locally."); running_in_colab = False

    # --- Determine Effective Output Directory ---
    if running_in_colab: effective_output_dir = os.path.join(google_drive_mount_point, 'MyDrive', args.drive_dir)
    else: effective_output_dir = args.output_dir
    print(f"Output directory: {effective_output_dir}")
    try: os.makedirs(effective_output_dir, exist_ok=True); print("Output directory ensured.")
    except OSError as e: print(f"Error creating output dir: {e}. Exiting."); exit()

    # --- Setup Evaluation ---
    transformer_config = {
        'device': args.device,
        'max_new_tokens': args.max_new_tokens,
        'quantization': None, # Add arg if needed
        'use_cache': False, # Keep false for baseline consistency? Or True for speed? Let's use True.
        'temperature': 0.1,
        'do_sample': False,
    }
    print(f"Using Transformer Config: {transformer_config}")

    # Load dataset and get sample
    try:
        full_dataset = load_dataset(args.dataset, 'input', 'target_scores')
        evaluation_sample = get_evaluation_sample(full_dataset, args.sample_data_size, args.seed)
        print(f"Using evaluation sample of size {len(evaluation_sample)} based on seed {args.seed}.")
    except Exception as e:
        print(f"Fatal Error: Could not load dataset sample: {e}")
        exit()

    # --- Run Baseline Evaluations ---
    baseline_results = []
    start_baseline_time = time.time()

    print("\n--- Starting Simplified Baseline Evaluations ---")
    for model_id in BASELINE_MODELS:
        print(f"\n--- Evaluating Model: {model_id} ---")
        # Clear cache if desired between models, or keep for potential reuse if models share components
        # _hf_model_cache.clear()
        # _hf_tokenizer_cache.clear()

        for prompt_name, format_func in BASELINE_PROMPT_FORMATTERS.items():
            print(f"  Evaluating Prompt Format: {prompt_name}")

            total_correct = 0; total_input_tokens = 0; total_output_tokens = 0
            total_evaluated = 0; failed_executions = 0
            total_attempted = len(evaluation_sample)

            for i, example in enumerate(evaluation_sample):
                input_text = example.get('input')
                target_scores = example.get('target_scores')
                correct_answer = get_correct_answer(target_scores)

                if not isinstance(input_text, str) or correct_answer is None:
                    failed_executions += 1; continue

                # Format the prompt using the current function
                prompt_string = format_func(input_text)

                # Execute the prompt
                try:
                    response, in_tokens, out_tokens = execute_transformer_prompt(
                        prompt_text=prompt_string,
                        model_id=model_id,
                        exec_config=transformer_config,
                        use_cache=True # Use cache within model eval loop
                    )
                    total_input_tokens += in_tokens
                    total_output_tokens += out_tokens
                    total_evaluated += 1
                    print(model_id,prompt_name,response.strip().lower(),correct_answer.lower())
                    if response is not None and response.strip().lower() == correct_answer.lower():
                        total_correct += 1

                except Exception as e:
                    failed_executions += 1
                    print(f"    Error executing example {i+1}/{total_attempted}: {e}")

            # Calculate results for this model/prompt combo
            if total_evaluated > 0:
                accuracy = total_correct / total_evaluated
                avg_tokens = (total_input_tokens + total_output_tokens) / total_evaluated
            else:
                accuracy = 0.0
                avg_tokens = float('inf')

            print(f"    Result: Acc={accuracy:.4f}, AvgTokens={avg_tokens:.2f} (Evaluated={total_evaluated}/{total_attempted})")

            # Store results
            result_entry = {
                "baseline_prompt_name": prompt_name,
                "model_name": model_id,
                "accuracy": accuracy,
                "average_tokens": avg_tokens,
                # Optionally store example formatted prompt
                "example_prompt_format": format_func("SAMPLE_INPUT_TEXT")
            }
            baseline_results.append(result_entry)

    end_baseline_time = time.time()
    print(f"\nTotal baseline evaluation time: {end_baseline_time - start_baseline_time:.2f} seconds.")

    # --- Save Baseline Results ---
    if baseline_results:
        print(f"\n--- Saving Baseline Results ({len(baseline_results)} entries) ---")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_base = os.path.splitext(os.path.basename(args.dataset))[0]
        sampling_info = f"sample{args.sample_data_size}" if args.sample_data_size else "full"
        seed_info = f"seed{args.seed}" if args.seed is not None else "noseed"
        base_filename = f"baseline_simple_{dataset_base}_{sampling_info}_{seed_info}_{timestamp}"

        # Save JSON
        results_filename = os.path.join(effective_output_dir, f"{base_filename}_results.json")
        try:
            with open(results_filename, 'w', encoding='utf-8') as f:
                json.dump(baseline_results, f, indent=4)
            print(f"Baseline results saved to: {results_filename}")
        except Exception as e: print(f"Error saving baseline JSON: {e}")

        # Save CSV
        try:
            csv_filename = os.path.join(effective_output_dir, f"{base_filename}_results.csv")
            header = "BaselineName,Model,Accuracy,AvgTokens"
            with open(csv_filename, 'w', encoding='utf-8') as f:
                f.write(header + '\n')
                for entry in baseline_results:
                     f.write(f"{entry['baseline_prompt_name']},{entry['model_name']},{entry['accuracy']:.6f},{entry['average_tokens']:.2f}\n")
            print(f"Baseline results summary saved to: {csv_filename}")
        except Exception as e: print(f"Error saving baseline CSV: {e}")
    else:
        print("\nNo baseline results generated.")

    print("\n--- Simplified Baseline Experiment Complete ---")

# python BaselineEvalutionScript.py \
#     --dataset "testsets/implicatures.json" \
#     --sample_data_size 3 \
#     --seed 42 \
#     --output_dir "output" \
#     --max_new_tokens 1 \
#     --device "auto"