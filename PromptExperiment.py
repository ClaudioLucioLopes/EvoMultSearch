import numpy as np
import argparse
import random
import copy
import json
import os
import re
import time
from typing import List, Any, Optional, Dict, Tuple, Union

from PromptSampling import PromptSampling,create_content_dict_from_bnf
from PromptOptimizationProblem import PromptOptimizationProblem
from PromptMutator import PromptMutator
from PromptCrossover import PromptCrossover
from PromptIndividual import ALL_RULE_IDENTIFIERS, PromptIndividual, PROMPT_TEMPLATE_STRING
from PromptDuplication import PromptDuplicateElimination
from jinja2 import Template


from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


# --- Main Experiment Function ---
def run_optimization_experiment(
    bnf_filepath: str,
    dataset_filepath: str,
    num_runs: int = 5,
    pop_size: int = 30,
    n_gen: int = 10,
    cx_prob_model: float = 0.9,
    cx_prob_attr_swap: float = 0.9,
    mut_prob_model: float = 0.1, # Specific internal probabilities for PromptMutator
    mut_prob_param: float = 0.2, # Specific internal probabilities for PromptMutator
    tournament_size: int = 5,
    sample_data_size: Optional[int] = None, # Samples for evaluation (None=full dataset)
    sample_seed: Optional[int] = None, # Seed for evaluation sampling
    exec_config: Optional[Dict[str, Any]] = None,
    input_key: str = 'input',
    target_key: str = 'target_scores'
):
    """
    Runs the multi-objective prompt optimization experiment multiple times.

    Args:
        bnf_filepath: Path to the BNF definition file for prompt components.
        dataset_filepath: Path to the evaluation dataset (JSON Lines).
        num_runs: Number of independent optimization runs (default: 5).
        pop_size: Population size (default: 30).
        n_gen: Number of generations (default: 10).
        cx_prob_model:  Crossover probability model (default: 0.9).
        cx_prob_attr_swap:  Crossover probability attribute swap (default: 0.9).
        mut_prob_model: Internal probability for model mutation within PromptMutator.
        mut_prob_param: Internal probability for parameter mutation within PromptMutator.
        tournament_size: Size for tournament selection (default: 5).
        sample_data_size: Number of samples to use for evaluation from dataset.
        sample_seed: Seed for evaluation sampling reproducibility.
        exec_config: Dictionary with execution params for transformer models.
        input_key: Key for input data in the dataset JSON.
        target_key: Key for target scores data in the dataset JSON.

    Returns:
        Tuple: (all_solutions, all_objectives) containing lists of all non-dominated
               individuals and their objectives found across all runs.
    """

    # --- 1. Initial Setup ---
    print("--- Experiment Setup ---")
    models_to_use = [
        # Use smaller models that might fit typical research hardware
        #"Qwen/Qwen1.5-0.5B-Chat", # Example smaller model
        #"google/gemma-2b-it", # Example smaller model
        # Add others carefully based on available resources
        # "meta-llama/Llama-3-8B-Instruct", # Larger
        # "mistralai/Mistral-7B-Instruct-v0.2", # Larger
        # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", # Check availability/requirements
        "Qwen/Qwen2.5-1.5B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        # "microsoft/Phi-4-mini-instruct",
        # "google/gemma-3-1b-it"
    ]
    print(f"Using models: {models_to_use}")

    tag_map = { # Map BNF tags to dictionary keys
        'context': 'context', 'req': 'req', 'instr': 'instr',
        'example': 'examples', 'sbs': 'cot'
    }
    try:
        available_contents = create_content_dict_from_bnf(bnf_filepath, tag_map)
        print(f"Loaded content options for keys: {list(available_contents.keys())}")
        # Ensure all components needed by mutator/sampler are present, add empty list if not
        for comp in ['context', 'req', 'instr', 'examples', 'cot']:
            if comp not in available_contents:
                 print(f"Warning: Content for '{comp}' not found in BNF file. Using empty list.")
                 available_contents[comp] = [None] # Add at least None option
    except Exception as e:
        print(f"Fatal Error: Could not load or parse BNF file '{bnf_filepath}': {e}")
        return [], []

    # Default execution config if none provided
    if exec_config is None:
        exec_config = {'device': 'auto', 'max_new_tokens': 1, 'use_cache': False}
    print(f"Transformer execution config: {exec_config}")

    all_run_solutions = []
    all_run_objectives = []

     # --- Generate Seeds for Each Run ---
    run_seeds = []
    if sample_seed is not None:
        print(f"Using main seed {sample_seed} to generate seeds for {num_runs} runs.")
        master_rng = random.Random(sample_seed)
        # Generate seeds in a reasonable range for Pymoo/random
        run_seeds = [master_rng.randint(1, 2**31 - 1) for _ in range(num_runs)]
        print(f"Generated run seeds: {run_seeds}")
    else:
        print(f"No main seed provided. Each run will use a different random seed.")
        run_seeds = [None] * num_runs # Pass None to use potentially random seeds per run


    # --- 2. Run Experiments Loop ---
    for run in range(num_runs):
        print(f"\n--- Starting Run {run + 1}/{num_runs} ---")
        start_time = time.time()
        run_specific_seed = run_seeds[run]

        # a) Initialize Problem for this run
        problem = PromptOptimizationProblem(
            dataset_path=dataset_filepath,
            input_key=input_key,
            target_scores_key=target_key,
            sample_size=sample_data_size,
            seed=run_specific_seed,
            transformer_exec_config=exec_config
        )

        # b) Initialize Operators
        sampling = PromptSampling(
            available_rules=ALL_RULE_IDENTIFIERS,
            available_models=models_to_use,
            available_contents=available_contents
        )
        # Note: Pymoo's mutation probability acts *per individual*.
        # The internal probabilities control *what* changes if mutation occurs.
        mutation = PromptMutator(
                 available_models=models_to_use,
                 possible_content_values=available_contents, # Use dict here now
                 prob_model=mut_prob_model,
                 prob_parameter=mut_prob_param
             )

        crossover = PromptCrossover(
                 prob_model=cx_prob_model, # Reuse model mutation prob for model crossover? Or define separate cx_prob_model?
                 prob_attr_swap=cx_prob_attr_swap # Standard uniform swap probability for attributes
             )
        
        # selection = TournamentSelection(pressure=tournament_size)
        # TournamentSelection()

        # # Instantiate the custom duplicate elimination
        duplicate_elimination_strategy = PromptDuplicateElimination()

        # c) Initialize Algorithm
        # NSGA-II handles elitism automatically
        # Using elementwise=True because Problem inherits from ElementwiseProblem
        # eliminate_duplicates=True helps manage population diversity with custom objects
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=sampling,
            # selection=selection,
            crossover=crossover,
            mutation=mutation,
            eliminate_duplicates=duplicate_elimination_strategy,
            # eliminate_duplicates=True,
            # n_offsprings=pop_size # Often set equal to pop_size
        )
        

        # d) Run Optimization
        # try:
        res = minimize(problem,
                        algorithm,
                        termination=('n_gen', n_gen),
                        seed=run_specific_seed, # Seed for this specific Pymoo run
                        save_history=False, # Set to True to save history
                        verbose=True,
                        # Pymoo handles elementwise evaluation internally now
                        )

        # e) Store Results from this run
        # res.opt contains the Population object of non-dominated solutions
        unique_solutions_single_run = []
        unique_objectives_list_single_run = []
        
        if res.opt is not None and len(res.opt):
                run_solutions = res.opt.get("X") # Get the individuals
                run_objectives = res.opt.get("F") # Get the objectives
                 # --- Deduplication Step ---
                deduplicator = PromptDuplicateElimination() # Use your custom class

                for i, current_sol in enumerate(run_solutions):
                    is_duplicate = False
                    for existing_sol in unique_solutions_single_run:
                        if deduplicator.is_equal(current_sol, existing_sol):
                            is_duplicate = True
                            break
                    if not is_duplicate:
                        unique_solutions_single_run.append(current_sol)
                        unique_objectives_list_single_run.append(run_objectives[i])
        else:
                print(f"Run {run + 1} finished. No non-dominated solutions found.")

        # except Exception as e:
        #     print(f"Error during optimization run {run + 1}: {e}")
        #     # Decide whether to continue other runs or stop

        end_time = time.time()
        print(f"Run {run + 1} duration: {end_time - start_time:.2f} seconds.")
        print(f"Run {run + 1} finished. Found {len(unique_solutions_single_run)} non-duplicated solutions.")
        print(f"Run {run + 1} finished. Found {len(unique_objectives_list_single_run)} objectives non-duplicated solutions.")
        all_run_solutions.extend(unique_solutions_single_run)
        all_run_objectives.extend(unique_objectives_list_single_run) # Append the array of objectives


    # --- 3. Aggregate and Find Final Pareto Front ---
    if not all_run_solutions:
        print("\nNo solutions found across any run.")
        return [], []
    print(type(all_run_objectives),len(all_run_objectives),len(all_run_solutions))
    unique_solutions_across_runs = []
    unique_objectives_across_runs = []

    for i, current_sol in enumerate(all_run_solutions):
        is_duplicate = False
        # Check against already added unique solutions
        print('Starting:', current_sol[0],'\n')
        for existing_sol in unique_solutions_across_runs:
            if deduplicator.is_equal(current_sol[0], existing_sol):
                print(existing_sol,current_sol[0],deduplicator.is_equal(current_sol[0], existing_sol))
                is_duplicate = True
                break
        # If it's not a duplicate of any existing unique solution, add it
        if not is_duplicate:
            print('Entrou: ', current_sol[0])
            unique_solutions_across_runs.append(current_sol[0])
            unique_objectives_across_runs.append(all_run_objectives[i])
        print('-'*40)
        print(unique_solutions_across_runs)
        print('-'*40)
        
    print('-'*40)
    print(len(unique_objectives_across_runs),len(unique_solutions_across_runs))
    print(unique_objectives_across_runs,unique_solutions_across_runs)

    print(f"\n--- Aggregating results from {num_runs} runs ---")
    # Combine objectives from all runs into a single NumPy array

    # Convert unique objectives list back to NumPy array
    unique_objectives_across_runs = np.array(unique_objectives_across_runs)
    
    combined_objectives = np.vstack(unique_objectives_across_runs)
    # all_run_solutions is already a flat list of individuals
    
    print(f"Total non-duplicated solutions found across runs: {len(unique_solutions_across_runs)}")
    

    # # Perform non-dominated sorting on the combined results
    # nds = NonDominatedSorting()
    # # Pass objectives, get ranked indices
    # # We expect objectives to be shape (n_solutions, n_objectives)
    # front_indices = nds.do(combined_objectives, only_non_dominated_front=True) # Get only the first front indices

    # Extract the globally non-dominated solutions and their objectives
    # final_solutions = [unique_solutions_across_runs[i] for i in front_indices]
    # final_objectives = combined_objectives[front_indices]

    final_solutions = unique_solutions_across_runs
    final_objectives = combined_objectives

    print(f"Found {len(final_solutions)} globally non-dominated solutions after merging.")

    return final_solutions, final_objectives

# --- Helper Function to Convert PromptIndividual to Dict ---
def prompt_individual_to_dict(ind: 'PromptIndividual') -> Dict[str, Any]:
    """Converts a PromptIndividual object to a serializable dictionary."""
    if not isinstance(ind, PromptIndividual):
        return {"error": "Not a PromptIndividual object"}
    template_context: Dict[str, Any] = {
                "rule_identifier": ind.get_rule_identifier(),
                "context_content": ind.context_content,
                "req_content": ind.req_content,
                "instr_content": ind.instr_content,
                "examples_content": ind.examples_content,
                "cot_content": ind.cot_content,
                "input_content": ind.input_content
            }
    prompt_template = Template(PROMPT_TEMPLATE_STRING)
    rendered_prompt = prompt_template.render(template_context)
    return {
        "rule_identifier": ind.get_rule_identifier(),
        "model_name": ind.model_name,
        "context_content": ind.context_content,
        "req_content": ind.req_content,
        "instr_content": ind.instr_content,
        "examples_content": ind.examples_content,
        "cot_content": ind.cot_content,
        "rendered_prompt": rendered_prompt
    }

# --- Main Experiment Function (run_optimization_experiment - unchanged) ---
# def run_optimization_experiment(...): ... # As defined previously

# --- Main Execution Block (Modified for Saving) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Prompt Optimization Experiment")
    parser.add_argument("--bnf_filepath", help="Path to the BNF definition file (e.g., implicatures.txt)")
    parser.add_argument("--dataset", default="implicatures.json", help="Path to the evaluation dataset (JSON Lines)")
    parser.add_argument("--runs", type=int, default=5, help="Number of independent runs")
    parser.add_argument("--pop_size", type=int, default=30, help="Population size")
    parser.add_argument("--gens", type=int, default=10, help="Number of generations")
    parser.add_argument("--cx_prob_model", type=float, default=0.9, help="Crossover model probability")
    parser.add_argument("--cx_prob_attr_swap", type=float, default=0.9, help="Crossover attribute swap probability")
    parser.add_argument("--mut_prob_model", type=float, default=0.1, help="Mutation model probability ")
    parser.add_argument("--mut_prob_param", type=float, default=0.1, help="Mutation probability per attribute")
    parser.add_argument("--sample_data_size", type=int, default=None, help="Evaluation dataset sample size (None=full)")
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation sampling")
    parser.add_argument("--output_dir", default=".", help="Directory to save output files")
    # Add more args for other parameters if needed

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # Example execution config (customize as needed)
    transformer_config = {
        'device': 'auto', 'max_new_tokens': 1, 'quantization': None,
        'use_cache': False, 'temperature': 0.1, 'do_sample': False,
    }

    # --- Run the Experiment ---
    start_run_time = time.time()
    final_pareto_solutions, final_pareto_objectives = run_optimization_experiment(
        bnf_filepath=args.bnf_filepath,
        dataset_filepath=args.dataset,
        num_runs=args.runs,
        pop_size=args.pop_size,
        n_gen=args.gens,
        cx_prob_model=args.cx_prob_model,
        cx_prob_attr_swap=args.cx_prob_attr_swap,
        mut_prob_model=args.mut_prob_model, 
        mut_prob_param=args.mut_prob_param, 
        tournament_size=5,
        sample_data_size=args.sample_data_size,
        sample_seed=args.seed,
        exec_config=transformer_config,
        input_key='input',      # Assuming default keys
        target_key='target_scores'
    )
    end_run_time = time.time()
    print(f"\nTotal experiment execution time: {end_run_time - start_run_time:.2f} seconds.")
    print(final_pareto_solutions)
    # --- Save Final Results ---
    if final_pareto_solutions:
        print(f"\n--- Saving Final Pareto Front ({len(final_pareto_solutions)} solutions) ---")

        # Create informative base filename
        bnf_base = os.path.splitext(os.path.basename(args.bnf_filepath))[0]
        dataset_base = os.path.splitext(os.path.basename(args.dataset))[0]
        sampling_info = f"sample{args.sample_data_size}" if args.sample_data_size else "full"
        base_filename = f"pareto_pop{args.pop_size}_gen{args.gens}_runs{args.runs}_{bnf_base}_{dataset_base}_{sampling_info}"

        # 1. Save Objectives (CSV)
        objectives_filename = os.path.join(args.output_dir, f"{base_filename}_objectives.csv")
        try:
            header = "Objective1_(1-Accuracy),Objective2_(TotalTokens)"
            np.savetxt(objectives_filename, final_pareto_objectives, delimiter=',', header=header, comments='')
            print(f"Objectives saved to: {objectives_filename}")
        except Exception as e:
            print(f"Error saving objectives to {objectives_filename}: {e}")

        # 2. Save Solutions (JSON)
        solutions_filename = os.path.join(args.output_dir, f"{base_filename}_solutions.json")
        try:
            solutions_data = [prompt_individual_to_dict(sol) for sol in final_pareto_solutions]
            with open(solutions_filename, 'w', encoding='utf-8') as f:
                json.dump(solutions_data, f, indent=4)
            print(f"Solutions saved to: {solutions_filename}")
        except Exception as e:
            print(f"Error saving solutions to  {solutions_filename}: {e}")


    print("\n--- Experiment Complete ---")
