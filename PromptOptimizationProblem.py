# --- Required Imports ---
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from typing import Callable, Any, Dict, Optional, Tuple, List
import json
import copy
import random # Added for sampling
import torch # Needed for PromptIndividual's method if not already imported
from PromptIndividual import ALL_COMPONENT_NAMES, PromptIndividual


# --- Pymoo Problem Definition with Sampling ---

class PromptOptimizationProblem(ElementwiseProblem):
    """
    Pymoo Problem definition for optimizing PromptIndividual objects based on
    evaluation against a SAMPLE of a dataset in JSON Lines format.

    Optimizes for two objectives:
    1. Maximize Average Accuracy across the sample (minimizing 1.0 - accuracy).
    2. Minimize Average Total Tokens (input + output) across the sample.

    Uses sample_size and seed for reproducible sampling during evaluation.
    Assumes the optimization algorithm handles PromptIndividual objects as variables.
    """

    def __init__(self,
                 dataset_path: str,
                 input_key: str = 'input',
                 target_scores_key: str = 'target_scores',
                 sample_size: Optional[int] = None, # Max number of examples to sample
                 seed: Optional[int] = None, # Seed for reproducible sampling
                 transformer_exec_config: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """
        Initializes the optimization problem.

        Args:
            dataset_path: Path to the dataset file (JSON Lines format expected).
            input_key: Key for input text in dataset records. Defaults to 'input', it represents the user question
            target_scores_key: Key for target scores dictionary. Defaults to 'target_scores',answer to user question
            sample_size: The number of examples to randomly sample from the dataset
                         for each evaluation. If None or >= dataset size, the full
                         dataset is used.
            seed: An integer seed for the random number generator used for sampling,
                  ensuring reproducibility if provided. If None, sampling will
                  be non-deterministic run-to-run. Using a fixed seed means the
                  SAME sample is used for EVERY evaluation call.
            transformer_exec_config: Fixed arguments for execute_prompt_with_transformer.
            **kwargs: Additional arguments passed to the pymoo Problem constructor.
        """
        self.dataset_path = dataset_path
        self.input_key = input_key
        self.target_scores_key = target_scores_key
        self.sample_size = sample_size
        self.seed = seed
        self.transformer_exec_config = transformer_exec_config if transformer_exec_config else {}

        # Load the full dataset once during initialization
        self.full_dataset = self._load_dataset(dataset_path, input_key, target_scores_key)
        self.full_dataset_size = len(self.full_dataset)

        if not self.full_dataset:
            raise ValueError(f"Dataset loaded from {dataset_path} is empty or failed to load.")

        # Determine the effective sample size for reporting/sanity checks
        if self.sample_size is None or self.sample_size >= self.full_dataset_size:
            self.effective_sample_size = self.full_dataset_size
            print(f"Evaluation will use the full dataset ({self.full_dataset_size} examples).")
        else:
            self.effective_sample_size = self.sample_size
            print(f"Evaluation will use a sample of size {self.effective_sample_size} from the dataset "
                  f"(Seed: {'Provided' if self.seed is not None else 'None'}).")


        super().__init__(n_var=1, n_obj=2, n_constr=0, xl=None, xu=None, vtype=object, **kwargs)

    def _load_dataset(self, path: str, input_key: str, target_scores_key: str) -> List[Dict[str, Any]]:
        """Loads dataset from a JSON Lines file, using specified keys. (Unchanged)"""
        data = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        if line.strip():
                            record = json.loads(line)
                            if input_key not in record or target_scores_key not in record:
                                print(f"Warning: Skipping line {line_num+1} in {path}: Missing '{input_key}' or '{target_scores_key}'.")
                                continue
                            if not isinstance(record.get(target_scores_key), dict):
                                print(f"Warning: Skipping line {line_num+1} in {path}: '{target_scores_key}' is not a dictionary.")
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

    def _get_correct_answer(self, target_scores: Dict[str, float]) -> Optional[str]:
        """Finds the key corresponding to the 1.0 score. (Unchanged)"""
        if not isinstance(target_scores, dict): return None
        for key, score in target_scores.items():
            try:
                 if float(score) == 1.0: return key
            except (ValueError, TypeError): continue
        print(f"Warning: No key found with score 1.0 in target_scores: {target_scores}")
        return None

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluates a single PromptIndividual against a SAMPLE of the dataset.
        """
        individual: 'PromptIndividual' = x[0]
        # print("-"*40)
        # print("\n\n",individual)

        # --- Select Sample ---
        if self.sample_size is None or self.sample_size >= self.full_dataset_size:
            dataset_to_evaluate = self.full_dataset
            eval_description = f"full dataset ({len(dataset_to_evaluate)} examples)"
        else:
            # Use seeded random generator for reproducible sampling IF seed is provided
            # Note: Using the same seed for every evaluate call means the sample is
            # identical for every individual evaluation. This might be desired for
            # direct comparison but reduces exploration of the dataset during opt.
            # If None, uses standard random, non-reproducible run-to-run.
            current_rng = random.Random(self.seed) # Seeded generator
            actual_sample_size = min(self.sample_size, self.full_dataset_size)
            dataset_to_evaluate = current_rng.sample(self.full_dataset, actual_sample_size)
            eval_description = f"sample of {len(dataset_to_evaluate)} (Seed: {self.seed})"

        total_correct = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_evaluated = 0 # Count successful executions within the sample
        total_attempted = len(dataset_to_evaluate) # Number of examples in the sample
        failed_executions = 0

        
        print(f"\nEvaluating Individual (Rule: {individual.get_rule_identifier()}, Model: {individual.model_name}) on {eval_description}...")

        # --- Iterate over the selected sample ---
        for i, example in enumerate(dataset_to_evaluate):
            eval_individual = individual.copy()
            example_input_text = example.get(self.input_key)
            target_scores_dict = example.get(self.target_scores_key)

            if not isinstance(example_input_text, str) or not isinstance(target_scores_dict, dict):
                 print(f"  Skipping example {i+1}/{total_attempted}: Invalid data type for configured keys.")
                 failed_executions += 1
                 continue

            eval_individual.input_content = example_input_text
            correct_answer = self._get_correct_answer(target_scores_dict)

            if correct_answer is None:
                print(f"  Skipping example {i+1}/{total_attempted}: Could not determine correct answer.")
                failed_executions += 1
                continue

            try:
                response_text, input_tokens, output_tokens = eval_individual.execute_prompt_with_transformer(
                    **self.transformer_exec_config
                )
                # response_text, input_tokens, output_tokens = 'yes', random.randint(0, 1),random.randint(0, 1)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_evaluated += 1
                # ind=eval_individual
                # print({
                #    "rule_identifier": ind.get_rule_identifier(),
                #     "model_name": ind.model_name,
                #     "context_content": ind.context_content,
                #     "req_content": ind.req_content,
                #     "instr_content": ind.instr_content,
                #     "examples_content": ind.examples_content,
                #     "cot_content": ind.cot_content,
                #     "input_content": ind.input_content # Include placeholder input for context
                # })
                # print("response_text: ",response_text)
                # print("correct_answer: ",correct_answer)
                if response_text is not None and response_text.strip().lower() == correct_answer.lower():
                    total_correct += 1

            except Exception as e:
                failed_executions += 1
                print(f"  Error executing example {i+1}/{total_attempted} for individual: {e}")

        # --- Calculate Final Objectives based on the sample ---
        if total_evaluated > 0:
            print('total_correct / total_evaluated',total_correct, total_evaluated)
            overall_accuracy = total_correct / total_evaluated
            average_total_tokens = (total_input_tokens + total_output_tokens) / total_evaluated
        else:
            print(f"  Warning: All {total_attempted} example executions failed or were skipped in this sample.")
            overall_accuracy = 0.0
            average_total_tokens = float('inf')

        accuracy_objective =  1-overall_accuracy
        token_objective = average_total_tokens
        out["F"] = [accuracy_objective, token_objective]

        print(f"Evaluation Summary (Sample): Correct={total_correct}/{total_evaluated} (Failed/Skipped: {failed_executions}). "
              f"Objectives (F): [{accuracy_objective:.4f}, {token_objective:.2f}]")


