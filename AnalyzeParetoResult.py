import numpy as np
import json
import matplotlib.pyplot as plt
import argparse
import os
from typing import List, Dict, Any, Tuple

# Import Pymoo's NonDominatedSorting
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

def load_results(objectives_csv_path: str, solutions_json_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Loads ALL objectives and solutions from the experiment output files.
    Performs basic validation on alignment and format.
    """
    print(f"Loading objectives from: {objectives_csv_path}")
    if not os.path.exists(objectives_csv_path):
        raise FileNotFoundError(f"Objectives file not found: {objectives_csv_path}")
    try:
        # Load all objectives, skipping header
        objectives_data = np.loadtxt(
            objectives_csv_path,
            delimiter=',',
            skiprows=1,
            dtype=float
        )
        # Handle single row case
        if objectives_data.ndim == 1:
            if objectives_data.shape[0] >= 2: # Need at least 2 objectives
                 objectives_data = objectives_data.reshape(1, -1)
            else: # Cannot perform NDS if only one objective value exists
                 print(f"Warning: Only {objectives_data.shape[0]} value found in objectives file row. Cannot perform NDS.")
                 objectives_data = np.array([]) # Treat as empty
        elif objectives_data.shape[0] == 0:
             objectives_data = np.array([]) # Treat as empty if no data rows

    except Exception as e:
        print(f"Error loading objectives CSV: {e}")
        raise ValueError(f"Could not load objectives from {objectives_csv_path}") from e

    print(f"Loading solutions from: {solutions_json_path}")
    if not os.path.exists(solutions_json_path):
         raise FileNotFoundError(f"Solutions file not found: {solutions_json_path}")
    try:
        with open(solutions_json_path, 'r', encoding='utf-8') as f:
            solutions_data = json.load(f)
    except Exception as e:
        print(f"Error loading solutions JSON: {e}")
        raise ValueError(f"Could not load solutions from {solutions_json_path}") from e

    # --- Verification ---
    if objectives_data.size > 0 and objectives_data.shape[0] != len(solutions_data):
        # Allow proceeding if objectives are empty (maybe only solutions saved?)
        # but raise error if both exist and counts mismatch
        raise ValueError(f"CRITICAL: Mismatch between number of objectives ({objectives_data.shape[0]}) "
                         f"and number of solutions ({len(solutions_data)}). Files are misaligned.")
    if objectives_data.size > 0 and objectives_data.shape[1] != 2:
         print(f"Warning: Expected 2 objectives per row, found {objectives_data.shape[1]}. Assuming columns are [1-Accuracy, Tokens].")

    print(f"Loaded {len(solutions_data)} solutions and {objectives_data.shape[0]} objective points.")
    return objectives_data, solutions_data

def shorten_model_name(full_name: str) -> str:
    """Creates a shorter, more readable model name."""
    # (Implementation remains the same as before)
    if not isinstance(full_name, str): return "Unknown"
    parts = full_name.split('/')
    name = parts[-1]
    name = name.replace("-Instruct", "").replace("-Chat", "").replace("-it", "")
    name = name.replace("Qwen2.5", "Qwen2").replace("Llama-3.2", "Llama3")
    name = name.replace("OpenMath-Nemotron", "Nemotron")
    return name

def plot_pareto_front(objectives: np.ndarray, solutions: List[Dict[str, Any]], output_filename: str):
    """
    Generates and saves a scatter plot of the Pareto front points,
    labeling each point with model and truncated start/end of rendered prompt
    (excluding the input placeholder text).
    Plots Accuracy vs Tokens and saves as EPS with larger labels.
    """
    if objectives.size == 0 or not solutions:
        print("No data to plot.")
        return

    print("Generating Pareto front plot with detailed labels (EPS format, Accuracy Axis)...")
    accuracy = 1.0 - objectives[:, 0] # Y-axis
    tokens = objectives[:, 1]           # X-axis

    plt.figure(figsize=(17, 12))
    scatter = plt.scatter(tokens, accuracy, marker='o', s=90, alpha=0.75, edgecolors='k', label='Pareto Optimal Solutions')

    # Add labels to points
    for i, sol in enumerate(solutions):
        model_short = shorten_model_name(sol.get('model_name', 'N/A'))

        # --- Generate Start/End Prompt Snippet (excluding placeholder) ---
        rendered = sol.get('rendered_prompt', None)
        rendered_snippet = "N/A"
        placeholder_text = "PLACEHOLDER_INPUT_CONTENT" # Define the text to remove
        # print('rendered: ',rendered)
        if rendered:
             # Clean newlines and strip whitespace
             cleaned_prompt = rendered.replace('\\n', ' ').replace('\n', ' ').strip()
             # --- REMOVE PLACEHOLDER ---
             cleaned_prompt_no_placeholder = cleaned_prompt.replace(placeholder_text, "").strip()
             # Handle potential double spaces left after removal
             cleaned_prompt_no_placeholder = ' '.join(cleaned_prompt_no_placeholder.split())

            #  print(cleaned_prompt,cleaned_prompt_no_placeholder)
             # Define lengths for snippet
             start_len = 30 # Characters from start
             end_len = 30   # Characters from end

             # Create snippet from the cleaned prompt *without* the placeholder
             if len(cleaned_prompt_no_placeholder) > start_len + end_len + 3:
                 rendered_snippet = f"{cleaned_prompt_no_placeholder[:start_len]}...\n{cleaned_prompt_no_placeholder[-end_len:]}"
                #  print("-----"*40)
             elif len(cleaned_prompt_no_placeholder) > 0:
                 rendered_snippet = cleaned_prompt_no_placeholder # Us///////////////e full cleaned prompt if short
             else:
                 rendered_snippet = "[Prompt structure only]" # Indicate if only placeholder was present

        # --- End Snippet Generation ---

        # Create multi-line label (Model + Prompt Snippet)
        label = f"{model_short}\n'{rendered_snippet}'"

        # Add text slightly offset from the point with increased font size
        plt.text(tokens[i] + 0.2, accuracy[i], label, fontsize=10, ha='left', va='center', rotation=0)

    # --- Update axis labels and title for Accuracy ---
    plt.xlabel("Average Total Tokens per Evaluation", fontsize=12)
    plt.ylabel("Average Accuracy", fontsize=12)
    plt.title("Calculated Pareto Front: Accuracy vs. Token Cost", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Dynamic axis limits based on accuracy
    y_min = max(0, np.min(accuracy) - 0.05) if accuracy.size > 0 else 0
    y_max = min(1.05, np.max(accuracy) + 0.05) if accuracy.size > 0 else 1.05
    x_min = max(0, np.min(tokens) - 5) if tokens.size > 0 else 0
    x_padding = max(15, np.ptp(tokens) * 0.1) if tokens.size > 1 else 15
    x_max = np.max(tokens) + x_padding if tokens.size > 0 else 100
    plt.ylim(bottom=y_min, top=y_max)
    plt.xlim(left=x_min, right=x_max)
    plt.legend(fontsize=11)
    plt.tick_params(axis='both', which='major', labelsize=11)
    plt.tight_layout()

    # --- Save as EPS ---
    if not output_filename.lower().endswith(".eps"):
        output_filename = os.path.splitext(output_filename)[0] + ".eps"

    try:
        plt.savefig(output_filename, format='eps', dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_filename}")
    except Exception as e:
        print(f"Error saving plot to {output_filename}: {e}")
    # plt.show()



# def plot_pareto_front(objectives: np.ndarray, solutions: List[Dict[str, Any]], output_filename: str):
#     """Generates and saves a scatter plot of the given Pareto front points."""
#     # (Implementation remains the same as before)
#     if objectives.size == 0 or not solutions:
#         print("No data to plot.")
#         return

#     print("Generating Pareto front plot...")
#     accuracy = 1.0 - objectives[:, 0]
#     tokens = objectives[:, 1]

#     plt.figure(figsize=(12, 8))
#     scatter = plt.scatter(tokens, accuracy, marker='o', s=60, alpha=0.7, edgecolors='k', label='Pareto Optimal Solutions')

#     texts = []
#     for i, sol in enumerate(solutions):
#         model_short = shorten_model_name(sol.get('model_name', 'N/A'))
#         rule = sol.get('rule_identifier', 'N/A')
#         label = f"{model_short}\n{rule}"
#         plt.text(tokens[i] + 0.1, accuracy[i] + 0.001, label, fontsize=8, ha='left', va='bottom')

#     plt.xlabel("Average Total Tokens per Evaluation")
#     plt.ylabel("Average Accuracy")
#     plt.title("Calculated Pareto Front: Accuracy vs. Token Cost")
#     plt.grid(True, linestyle='--', alpha=0.6)
#     # Dynamic axis limits
#     y_min = max(0, np.min(accuracy) - 0.05) if accuracy.size > 0 else 0
#     y_max = min(1.05, np.max(accuracy) + 0.05) if accuracy.size > 0 else 1.05
#     x_min = max(0, np.min(tokens) - 5) if tokens.size > 0 else 0
#     x_max = np.max(tokens) + 5 if tokens.size > 0 else 100
#     plt.ylim(bottom=y_min, top=y_max)
#     plt.xlim(left=x_min, right=x_max)
#     plt.legend()
#     plt.tight_layout()

#     try:
#         plt.savefig(output_filename, dpi=300, bbox_inches='tight')
#         print(f"Plot saved to: {output_filename}")
#     except Exception as e:
#         print(f"Error saving plot to {output_filename}: {e}")
#     # plt.show()


def analyze_results(objectives: np.ndarray, solutions: List[Dict[str, Any]]):
    """Provides commentary on the Pareto front results."""
    # (Implementation remains the same as before)
    if objectives.size == 0 or not solutions:
        print("\n--- Analysis ---")
        print("No non-dominated solutions to analyze.")
        return

    print("\n--- Analysis of Calculated Pareto Front ---")
    accuracy = 1.0 - objectives[:, 0]
    tokens = objectives[:, 1]

    # Find best accuracy point
    best_acc_idx = np.argmax(accuracy)
    best_acc_sol = solutions[best_acc_idx]
    best_acc_val = accuracy[best_acc_idx]
    best_acc_tokens = tokens[best_acc_idx]
    print(f"\n1. Highest Accuracy Solution:")
    print(f"   - Accuracy: {best_acc_val:.4f}")
    print(f"   - Avg Tokens: {best_acc_tokens:.2f}")
    print(f"   - Model: {best_acc_sol.get('model_name')}")
    print(f"   - Rule: {best_acc_sol.get('rule_identifier')}")

    # Find lowest token point
    min_token_idx = np.argmin(tokens)
    min_token_sol = solutions[min_token_idx]
    min_token_val = tokens[min_token_idx]
    min_token_acc = accuracy[min_token_idx]
    print(f"\n2. Lowest Token Cost Solution:")
    print(f"   - Avg Tokens: {min_token_val:.2f}")
    print(f"   - Accuracy: {min_token_acc:.4f}")
    print(f"   - Model: {min_token_sol.get('model_name')}")
    print(f"   - Rule: {min_token_sol.get('rule_identifier')}")

    # Find potential "knee" point
    norm_f0 = objectives[:, 0]
    min_t, max_t = np.min(tokens), np.max(tokens)
    if max_t > min_t: norm_f1 = (tokens - min_t) / (max_t - min_t)
    else: norm_f1 = np.zeros_like(tokens)
    norm_distances = np.sqrt(norm_f0**2 + norm_f1**2)
    knee_idx = np.argmin(norm_distances)
    knee_sol = solutions[knee_idx]
    knee_acc = accuracy[knee_idx]
    knee_tokens = tokens[knee_idx]
    print(f"\n3. Potential 'Knee' Solution (Balanced Trade-off):")
    print(f"   - Accuracy: {knee_acc:.4f}")
    print(f"   - Avg Tokens: {knee_tokens:.2f}")
    print(f"   - Model: {knee_sol.get('model_name')}")
    print(f"   - Rule: {knee_sol.get('rule_identifier')}")

    # General Commentary
    print("\n4. General Observations:")
    print(f"   - The Pareto front shows the trade-off between maximizing accuracy and minimizing token usage.")
    model_counts = {}; rule_counts = {}
    for sol in solutions:
        mn = sol.get('model_name', 'N/A'); ri = sol.get('rule_identifier', 'N/A')
        model_counts[mn] = model_counts.get(mn, 0) + 1
        rule_counts[ri] = rule_counts.get(ri, 0) + 1
    print(f"   - Models appearing on the front: {model_counts}")
    print(f"   - Rules appearing on the front: {rule_counts}")
    print(f"   - Examine the plot to see which models/rules dominate different regions.")


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment results: find Pareto front from objectives and plot.")
    parser.add_argument("objectives_csv", help="Path to the objectives CSV file (e.g., pareto_..._objectives.csv)")
    parser.add_argument("solutions_json", help="Path to the solutions JSON file (e.g., pareto_..._solutions.json)")
    parser.add_argument("--output_plot", default="calculated_pareto_front.png", help="Filename for the output plot image.")

    args = parser.parse_args()

    try:
        # Load ALL results first
        all_objectives, all_solutions = load_results(args.objectives_csv, args.solutions_json)

        if all_objectives.size == 0:
            print("No objective data loaded, cannot perform NDS.")
        else:
            # --- Perform Non-Dominated Sorting ---
            print("\nPerforming non-dominated sorting on loaded objectives...")
            nds = NonDominatedSorting()
            # Assuming lower values are better for both objectives
            front_indices = nds.do(all_objectives, only_non_dominated_front=True)

            if front_indices is None or len(front_indices) == 0:
                print("No non-dominated solutions found in the provided data.")
            else:
                # --- Filter results based on NDS ---
                final_pareto_objectives = all_objectives[front_indices]
                # Ensure solutions list is indexed correctly
                final_pareto_solutions = [all_solutions[i] for i in front_indices]
                print(f"Identified {len(final_pareto_solutions)} non-dominated solutions.")

                # --- Plot and Analyze the calculated Pareto Front ---
                plot_pareto_front(final_pareto_objectives, final_pareto_solutions, args.output_plot)
                analyze_results(final_pareto_objectives, final_pareto_solutions)

    except FileNotFoundError as e:
        print(f"\nError: Input file not found. {e}")
    except ValueError as e:
        print(f"\nError: Problem loading or processing data. {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


# python AnalyzeParetoResult.py path/to/your/pareto_..._objectives.csv path/to/your/pareto_..._solutions.json --output_plot my_pareto_plot.png
# python AnalyzeParetoResult.py /home/claudiolucio/Projetos/EvoMult-MP-Dados/pareto_pop30_gen10_runs5_implicatures_implicatures_sample50_objectives.csv  /home/claudiolucio/Projetos/EvoMult-MP-Dados/pareto_pop30_gen10_runs5_implicatures_implicatures_sample50_solutions.json --output_plot /home/claudiolucio/Projetos/EvoMult-MP-Dados/pareto_pop30_gen10_runs5_implicatures_implicatures_sample50_solutions.png