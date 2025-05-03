import numpy as np
import json
import matplotlib.pyplot as plt
import argparse
import os
from typing import List, Dict, Any, Tuple

# Import Pymoo's NonDominatedSorting
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# --- Helper Functions (load_results, shorten_model_name - remain the same) ---

def load_results(objectives_csv_path: str, solutions_json_path: str) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Loads ALL objectives and solutions from the experiment output files.
    Performs basic validation on alignment and format.
    """
    print(f"Loading objectives from: {objectives_csv_path}")
    if not os.path.exists(objectives_csv_path):
        raise FileNotFoundError(f"Objectives file not found: {objectives_csv_path}")
    try:
        # Load all objectives, skipping header. Assumes column 0 is (1 - Accuracy)
        objectives_data = np.loadtxt(
            objectives_csv_path, delimiter=',', skiprows=1, dtype=float
        )
        if objectives_data.ndim == 1:
            if objectives_data.shape[0] >= 2: objectives_data = objectives_data.reshape(1, -1)
            else: objectives_data = np.array([])
        elif objectives_data.shape[0] == 0: objectives_data = np.array([])
    except Exception as e:
        raise ValueError(f"Could not load objectives from {objectives_csv_path}") from e

    print(f"Loading solutions from: {solutions_json_path}")
    if not os.path.exists(solutions_json_path):
         raise FileNotFoundError(f"Solutions file not found: {solutions_json_path}")
    try:
        with open(solutions_json_path, 'r', encoding='utf-8') as f:
            solutions_data = json.load(f)
    except Exception as e:
        raise ValueError(f"Could not load solutions from {solutions_json_path}") from e

    if objectives_data.size > 0 and objectives_data.shape[0] != len(solutions_data):
        raise ValueError(f"CRITICAL: Mismatch between number of objectives ({objectives_data.shape[0]}) "
                         f"and number of solutions ({len(solutions_data)}). Files are misaligned.")
    if objectives_data.size > 0 and objectives_data.shape[1] != 2:
         print(f"Warning: Expected 2 objectives per row, found {objectives_data.shape[1]}. Assuming [1-Accuracy, Tokens].")

    print(f"Loaded {len(solutions_data)} solutions and {objectives_data.shape[0]} objective points.")
    return objectives_data, solutions_data

def shorten_model_name(full_name: str) -> str:
    """Creates a shorter, more readable model name."""
    # (Implementation remains the same)
    if not isinstance(full_name, str): return "Unknown"
    parts = full_name.split('/')
    name = parts[-1]
    name = name.replace("-Instruct", "").replace("-Chat", "").replace("-it", "")
    name = name.replace("Qwen2.5-1.5B", "Qwen1.5B").replace("Llama-3.2-1B", "Llama1B")
    name = name.replace("OpenMath-Nemotron-1.5B", "Nemotron1.5B")
    name = name.replace("DeepSeek-R1-Distill-Qwen-1.5B", "DS-Qwen1.5B")
    name = name.replace("Phi-4-mini", "Phi4mini")
    name = name.replace("gemma-3-1b", "gemma3-1b")
    return name

# --- Plotting Function (plot_pareto_front - remains the same) ---
def plot_pareto_front(objectives: np.ndarray, solutions: List[Dict[str, Any]], output_filename: str):
    """
    Generates and saves a scatter plot of the Pareto front points,
    labeling each point with model and truncated start/end of rendered prompt.
    Plots Accuracy vs Tokens and saves as EPS with larger labels.
    """
    # (Implementation remains the same as previous version)
    if objectives.size == 0 or not solutions:
        print("No data to plot.")
        return

    print("Generating Pareto front plot with detailed labels (EPS format, Accuracy Axis)...")
    accuracy = 1.0 - objectives[:, 0] # Y-axis
    tokens = objectives[:, 1]           # X-axis

    plt.figure(figsize=(17, 12))
    scatter = plt.scatter(tokens, accuracy, marker='o', s=90, alpha=0.75, edgecolors='k', label='Pareto Optimal Solutions')

    for i, sol in enumerate(solutions):
        model_short = shorten_model_name(sol.get('model_name', 'N/A'))
        rendered = sol.get('rendered_prompt', None)
        rendered_snippet = "N/A"; placeholder_text = "PLACEHOLDER_INPUT_CONTENT"
        if rendered:
             cleaned_prompt = rendered.replace('\\n', ' ').replace('\n', ' ').strip()
             cleaned_prompt_no_placeholder = cleaned_prompt.replace(placeholder_text, "").strip()
             cleaned_prompt_no_placeholder = ' '.join(cleaned_prompt_no_placeholder.split())
             start_len = 20; end_len = 20
             if len(cleaned_prompt_no_placeholder) > start_len + end_len + 3:
                 rendered_snippet = f"{cleaned_prompt_no_placeholder[:start_len]}...{cleaned_prompt_no_placeholder[-end_len:]}"
             elif len(cleaned_prompt_no_placeholder) > 0:
                 rendered_snippet = cleaned_prompt_no_placeholder
             else: rendered_snippet = "[Prompt structure only]"
        label = f"{model_short}\n'{rendered_snippet}'"
        plt.text(tokens[i] + 0.2, accuracy[i], label, fontsize=10, ha='left', va='center', rotation=0)

    plt.xlabel("Average Total Tokens per Evaluation", fontsize=12)
    plt.ylabel("Average Accuracy", fontsize=12)
    plt.title("Calculated Pareto Front: Accuracy vs. Token Cost", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    y_min = max(0, np.min(accuracy) - 0.05) if accuracy.size > 0 else 0
    y_max = min(1.05, np.max(accuracy) + 0.05) if accuracy.size > 0 else 1.05
    x_min = max(0, np.min(tokens) - 5) if tokens.size > 0 else 0
    x_padding = max(15, np.ptp(tokens) * 0.1) if tokens.size > 1 else 15
    x_max = np.max(tokens) + x_padding if tokens.size > 0 else 100
    plt.ylim(bottom=y_min, top=y_max)
    plt.xlim(left=x_min, right=x_max)
    plt.legend(fontsize=11); plt.tick_params(axis='both', which='major', labelsize=11)
    plt.tight_layout()

    if not output_filename.lower().endswith(".eps"): output_filename = os.path.splitext(output_filename)[0] + ".eps"
    try:
        plt.savefig(output_filename, format='eps', dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_filename}")
    except Exception as e: print(f"Error saving plot to {output_filename}: {e}")
    # plt.show()


# --- Analysis Function (analyze_results - remains the same) ---
def analyze_results(objectives: np.ndarray, solutions: List[Dict[str, Any]]):
    """Provides commentary on the Pareto front results."""
    # (Implementation remains the same as before)
    if objectives.size == 0 or not solutions:
        print("\n--- Analysis ---"); print("No non-dominated solutions to analyze."); return

    print("\n--- Analysis of Calculated Pareto Front ---")
    accuracy = 1.0 - objectives[:, 0]; tokens = objectives[:, 1]

    best_acc_idx = np.argmax(accuracy); best_acc_sol = solutions[best_acc_idx]
    best_acc_val = accuracy[best_acc_idx]; best_acc_tokens = tokens[best_acc_idx]
    print(f"\n1. Highest Accuracy Solution:"); print(f"   - Accuracy: {best_acc_val:.4f} (Obj0: {objectives[best_acc_idx, 0]:.4f})")
    print(f"   - Avg Tokens: {best_acc_tokens:.2f}"); print(f"   - Model: {best_acc_sol.get('model_name')}"); print(f"   - Rule: {best_acc_sol.get('rule_identifier')}")

    min_token_idx = np.argmin(tokens); min_token_sol = solutions[min_token_idx]
    min_token_val = tokens[min_token_idx]; min_token_acc = accuracy[min_token_idx]
    print(f"\n2. Lowest Token Cost Solution:"); print(f"   - Avg Tokens: {min_token_val:.2f}")
    print(f"   - Accuracy: {min_token_acc:.4f} (Obj0: {objectives[min_token_idx, 0]:.4f})")
    print(f"   - Model: {min_token_sol.get('model_name')}"); print(f"   - Rule: {min_token_sol.get('rule_identifier')}")

    norm_f0 = objectives[:, 0]; min_t, max_t = np.min(tokens), np.max(tokens)
    if max_t > min_t: norm_f1 = (tokens - min_t) / (max_t - min_t)
    else: norm_f1 = np.zeros_like(tokens)
    norm_distances = np.sqrt(norm_f0**2 + norm_f1**2); knee_idx = np.argmin(norm_distances)
    knee_sol = solutions[knee_idx]; knee_acc = accuracy[knee_idx]; knee_tokens = tokens[knee_idx]
    print(f"\n3. Potential 'Knee' Solution (Balanced Trade-off):"); print(f"   - Accuracy: {knee_acc:.4f} (Obj0: {objectives[knee_idx, 0]:.4f})")
    print(f"   - Avg Tokens: {knee_tokens:.2f}"); print(f"   - Model: {knee_sol.get('model_name')}"); print(f"   - Rule: {knee_sol.get('rule_identifier')}")

    print("\n4. General Observations:"); print(f"   - The Pareto front shows the trade-off between maximizing accuracy and minimizing token usage.")
    model_counts = {}; rule_counts = {}
    for sol in solutions:
        mn = sol.get('model_name', 'N/A'); ri = sol.get('rule_identifier', 'N/A')
        model_counts[mn] = model_counts.get(mn, 0) + 1; rule_counts[ri] = rule_counts.get(ri, 0) + 1
    print(f"   - Models appearing on the front: {model_counts}"); print(f"   - Rules appearing on the front: {rule_counts}")
    print(f"   - Examine the plot to see which models/rules dominate different regions.")


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze experiment results: filter zero-accuracy, find Pareto front, plot.")
    parser.add_argument("objectives_csv", help="Path to the objectives CSV file (e.g., pareto_..._objectives.csv)")
    parser.add_argument("solutions_json", help="Path to the solutions JSON file (e.g., pareto_..._solutions.json)")
    parser.add_argument("--output_plot", default="calculated_pareto_front_filtered.eps", help="Filename for the output plot image (EPS format).")

    args = parser.parse_args()

    try:
        # Load ALL results first
        all_objectives, all_solutions = load_results(args.objectives_csv, args.solutions_json)

        if all_objectives.size == 0:
            print("No objective data loaded, cannot perform filtering or NDS.")
        else:
            # --- Filter out solutions with zero accuracy ---
            print(f"\nFiltering out solutions with zero accuracy (Objective 0 == 1.0)...")
            # Objective 0 is 1 - Accuracy. Accuracy is 0 when Objective 0 is 1.0.
            # Use a small tolerance for floating point comparison
            accuracy_tolerance = 1e-9
            non_zero_acc_indices = np.where(all_objectives[:, 0] < (1.0 - accuracy_tolerance))[0]

            if len(non_zero_acc_indices) == 0:
                print("No solutions with non-zero accuracy found. Cannot proceed.")
            else:
                filtered_objectives = all_objectives[non_zero_acc_indices]
                filtered_solutions = [all_solutions[i] for i in non_zero_acc_indices]
                print(f"Filtered down to {len(filtered_solutions)} solutions with non-zero accuracy.")

                # --- Perform Non-Dominated Sorting on FILTERED Results ---
                print("\nPerforming non-dominated sorting on filtered objectives...")
                nds = NonDominatedSorting()
                # Pass the filtered objectives to NDS
                front_indices_filtered = nds.do(filtered_objectives, only_non_dominated_front=True)

                if front_indices_filtered is None or len(front_indices_filtered) == 0:
                    print("No non-dominated solutions found in the filtered data.")
                else:
                    # --- Extract final results using indices relative to the FILTERED set ---
                    final_pareto_objectives = filtered_objectives[front_indices_filtered]
                    final_pareto_solutions = [filtered_solutions[i] for i in front_indices_filtered]
                    print(f"Identified {len(final_pareto_solutions)} non-dominated solutions after filtering.")

                    # --- Plot and Analyze the FINAL Pareto Front ---
                    plot_pareto_front(final_pareto_objectives, final_pareto_solutions, args.output_plot)
                    analyze_results(final_pareto_objectives, final_pareto_solutions)

    except FileNotFoundError as e: print(f"\nError: Input file not found. {e}")
    except ValueError as e: print(f"\nError: Problem loading or processing data. {e}")
    except Exception as e: print(f"\nAn unexpected error occurred: {e}")

