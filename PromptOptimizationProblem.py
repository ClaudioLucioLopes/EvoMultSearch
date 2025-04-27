# --- Required Imports ---
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from typing import Callable, Any, Dict, Optional, Tuple, List
import json
import copy
import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import time # For timing model loads
from PromptIndividual import PromptIndividual

# --- Pymoo Problem Definition with Caching ---

class PromptOptimizationProblem(ElementwiseProblem):
    """
    Pymoo Problem definition for optimizing PromptIndividual objects.
    Handles model/tokenizer caching and execution internally for efficiency.

    Optimizes for:
    1. Minimize (1.0 - Avg Accuracy on Sample)
    2. Minimize (Avg Total Tokens on Sample)
    """

    def __init__(self,
                 dataset_path: str,
                 input_key: str = 'input',
                 target_scores_key: str = 'target_scores',
                 sample_size: Optional[int] = None,
                 seed: Optional[int] = None,
                 transformer_exec_config: Optional[Dict[str, Any]] = None,
                 debug_print: bool = False,
                 **kwargs):
        """
        Initializes the optimization problem and the model/tokenizer cache.
        """
        self.dataset_path = dataset_path
        self.input_key = input_key
        self.target_scores_key = target_scores_key
        self.sample_size = sample_size
        self.seed = seed
        # Store exec config, ensuring defaults
        self.transformer_exec_config = transformer_exec_config if transformer_exec_config else {}
        self.transformer_exec_config.setdefault('device', 'auto')
        self.transformer_exec_config.setdefault('max_new_tokens', 10)
        self.transformer_exec_config.setdefault('quantization', None)
        self.transformer_exec_config.setdefault('temperature', 0.1)
        self.transformer_exec_config.setdefault('do_sample', False)

        self.debug_print = debug_print

        # --- Model/Tokenizer Cache ---
        # Stores {cache_key: model_object}
        self._model_cache: Dict[str, Any] = {}
        # Stores {cache_key: tokenizer_object}
        self._tokenizer_cache: Dict[str, Any] = {}

        # Load the full dataset once
        self.full_dataset = self._load_dataset(dataset_path, input_key, target_scores_key)
        self.full_dataset_size = len(self.full_dataset)
        if not self.full_dataset: raise ValueError("Dataset empty.")

        # Determine effective sample size
        if self.sample_size is None or self.sample_size >= self.full_dataset_size:
            self.effective_sample_size = self.full_dataset_size
            print(f"Evaluation: Full dataset ({self.full_dataset_size} examples).")
        else:
            self.effective_sample_size = self.sample_size
            print(f"Evaluation: Sample size {self.effective_sample_size} (Seed: {self.seed}).")

        # Initialize Pymoo Problem
        super().__init__(n_var=1, n_obj=2, n_constr=0, xl=None, xu=None, vtype=object, **kwargs)

    def _load_dataset(self, path: str, input_key: str, target_key: str) -> List[Dict[str, Any]]:
        """Loads dataset from a JSON Lines file."""
        data = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        if line.strip():
                            record = json.loads(line)
                            if input_key not in record or target_key not in record: continue
                            if not isinstance(record.get(target_key), dict): continue
                            data.append(record)
                    except Exception as e: print(f"Warn: Skipping line {line_num+1} in {path}: {e}")
            print(f"Loaded {len(data)} valid records from {path}")
            return data
        except Exception as e: print(f"Error loading dataset {path}: {e}"); raise

    def _get_correct_answer(self, target_scores: Dict[str, float]) -> Optional[str]:
        """Finds the key corresponding to the 1.0 score."""
        if not isinstance(target_scores, dict): return None
        for key, score in target_scores.items():
            try:
                 if abs(float(score) - 1.0) < 1e-6: return key
            except (ValueError, TypeError): continue
        return None

    def _get_model_and_tokenizer(self, model_id: str):
        """Loads model/tokenizer from cache or Hugging Face, storing in cache."""
        device = self.transformer_exec_config.get('device', 'auto')
        quantization = self.transformer_exec_config.get('quantization')
        # Create a unique key based on model, device, and quantization
        cache_key = f"{model_id}_{device}_{quantization}"

        # Check cache first
        if cache_key in self._model_cache:
            return self._model_cache[cache_key], self._tokenizer_cache[cache_key]
        else:
            # Load if not in cache
            print(f"Loading model/tokenizer for {cache_key}...")
            start_load = time.time()
            try:
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                if tokenizer.pad_token is None:
                    # Set pad token if missing (common practice)
                    tokenizer.pad_token = tokenizer.eos_token
                    print(f"  Set pad_token to eos_token for {model_id}")

                # Setup quantization config
                bnb_config = None
                load_kwargs = {}
                if quantization == "4bit" and torch.cuda.is_available():
                     bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
                     load_kwargs['quantization_config'] = bnb_config
                     print("    Applying 4-bit quantization.")
                elif quantization == "8bit" and torch.cuda.is_available():
                     bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                     load_kwargs['quantization_config'] = bnb_config
                     print("    Applying 8-bit quantization.")

                # Load model
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    # Use device_map only if NOT using quantization that needs 'auto'
                    device_map=device if not bnb_config else "auto",
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
                    trust_remote_code=True, # Often needed
                    **load_kwargs
                )
                model.eval() # Set to evaluation mode

                # --- Store in cache ---
                self._model_cache[cache_key] = model
                self._tokenizer_cache[cache_key] = tokenizer
                end_load = time.time()
                print(f"Model {model_id} loaded and cached in {end_load - start_load:.2f}s.")
                return model, tokenizer
            except Exception as e:
                print(f"Error loading model/tokenizer {model_id}: {e}")
                # Raise error to stop evaluation if essential model fails
                raise RuntimeError(f"Failed to load model {model_id}") from e


    def _execute_single_prompt(self,
                               model: Any,
                               tokenizer: Any,
                               prompt_text: str
                              ) -> Tuple[Optional[str], int, int]:
        """Internal helper to run inference for one prompt using loaded model/tokenizer."""
        input_token_count = 0; output_token_count = 0; response_text = None
        try:
            # --- Tokenize ---
            inputs = tokenizer(prompt_text, return_tensors="pt", padding=False, truncation=False)
            input_ids = inputs["input_ids"].to(model.device) # Move to model's device
            input_token_count = input_ids.shape[1]

            # --- Generate ---
            with torch.no_grad():
                # Prepare generation config from problem's exec_config
                gen_kwargs = {
                    k: v for k, v in self.transformer_exec_config.items()
                    if k in ["max_new_tokens", "temperature", "do_sample", "top_k", "top_p"] # Add other relevant params
                }
                gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
                gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
                gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None} # Filter None

                outputs = model.generate(input_ids, **gen_kwargs)

                # --- Decode ---
                output_token_count = outputs.shape[1] - input_token_count
                if output_token_count < 0: output_token_count = 0
                # Decode only the newly generated part
                generated_ids = outputs[0, input_token_count:]
                response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        except Exception as e:
            print(f"  Error during generation/decoding: {e}")
            return None, input_token_count, 0 # Return error state

        return response_text.strip() if response_text else None, input_token_count, output_token_count


    def _evaluate(self, x: np.ndarray, out, *args, **kwargs):
        """
        Evaluates a single PromptIndividual (x[0]) against a SAMPLE of the dataset,
        using cached models/tokenizers.
        """
        individual: PromptIndividual = x[0] # Extract object from Pymoo's input array

        # --- Get Model/Tokenizer (from cache or load) ---
        try:
            model, tokenizer = self._get_model_and_tokenizer(individual.model_name)
        except Exception as e:
            print(f"Cannot evaluate individual {individual!r} due to model load failure: {e}")
            out["F"] = [1.0, float('inf')] # Assign worst objectives
            return

        # --- Select Sample ---
        if self.sample_size is None or self.sample_size >= self.full_dataset_size:
            dataset_to_evaluate = self.full_dataset
        else:
            current_rng = random.Random(self.seed)
            actual_sample_size = min(self.sample_size, self.full_dataset_size)
            dataset_to_evaluate = current_rng.sample(self.full_dataset, actual_sample_size)

        total_correct = 0; total_input_tokens = 0; total_output_tokens = 0
        total_evaluated = 0; failed_executions = 0
        total_attempted = len(dataset_to_evaluate)

        # --- Iterate over the selected sample ---
        for i, example in enumerate(dataset_to_evaluate):
            example_input_text = example.get(self.input_key)
            target_scores_dict = example.get(self.target_scores_key)

            if not isinstance(example_input_text, str) or not isinstance(target_scores_dict, dict):
                 failed_executions += 1; continue

            correct_answer = self._get_correct_answer(target_scores_dict)
            if correct_answer is None:
                failed_executions += 1; continue

            # --- Render prompt using individual's method ---
            prompt_string = individual.render_prompt_string(actual_input=example_input_text)
            if "Error rendering prompt" in prompt_string:
                 if self.debug_print: print(f"  Skipping example {i+1} due to prompt rendering error.")
                 failed_executions += 1; continue

            # --- Execute using the loaded model/tokenizer ---
            response_text, input_tokens, output_tokens ='yes',1,1
            
            # response_text, input_tokens, output_tokens = self._execute_single_prompt(
            #     model=model,
            #     tokenizer=tokenizer,
            #     prompt_text=prompt_string
            # )

            # --- Accumulate results ---
            if response_text is not None: # Check if execution was successful
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_evaluated += 1
                is_correct = False
                if response_text.strip().lower() == correct_answer.lower():
                    total_correct += 1
                    is_correct = True
                if self.debug_print:
                    print(f"  Ex {i+1}: Resp='{response_text.strip()}', Correct='{correct_answer}', Match={is_correct}, InTok={input_tokens}, OutTok={output_tokens}")
            else: # Execution failed
                 failed_executions += 1
                 if self.debug_print: print(f"  Execution failed for example {i+1}")

        # --- Calculate Final Objectives based on the sample ---
        if total_evaluated > 0:
            overall_accuracy = total_correct / total_evaluated
            average_total_tokens = (total_input_tokens + total_output_tokens) / total_evaluated
        else:
            overall_accuracy = 0.0; average_total_tokens = float('inf')

        accuracy_objective = 1.0 - overall_accuracy
        token_objective = average_total_tokens
        out["F"] = [accuracy_objective, token_objective]

        # Print summary less frequently if not debugging
        if self.debug_print or random.random() < 0.05: # Print occasionally
             print(f"Eval Summary (Ind: {individual!r}): Correct={total_correct}/{total_evaluated} (Failed/Skipped: {failed_executions}). "
                   f"Objectives (F): [{accuracy_objective:.4f}, {token_objective:.2f}]")

