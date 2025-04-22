from PromptOptimizationProblem import PromptOptimizationProblem
from PromptIndividual import PromptIndividual   

# --- Ensure PromptIndividual class and constants are defined ---

# 1. Define Path and Keys for Dataset
DATASET_FILE = "testsets/implicatures.json" # Ensure this file exists
INPUT_DATA_KEY = 'input'
TARGET_DATA_KEY = 'target_scores'

# 2. Define fixed execution parameters
EXEC_CONFIG = {
    'device': 'auto', 'max_new_tokens': 2, 'quantization': None,
    'use_cache': False, 'temperature': 0.1, 'do_sample': False,
}

# 3. Instantiate the Problem with Sampling Parameters
SAMPLE_N = 5  # Number of examples to sample
RANDOM_SEED = 42 # Seed for reproducibility

try:
    problem = PromptOptimizationProblem(
        dataset_path=DATASET_FILE,
        input_key=INPUT_DATA_KEY,
        target_scores_key=TARGET_DATA_KEY,
        sample_size=SAMPLE_N, # Use sampling
        seed=RANDOM_SEED,     # Use a seed
        transformer_exec_config=EXEC_CONFIG
    )
except Exception as e:
    print(f"Failed to initialize problem: {e}")
    exit()

# 4. Create a Sample PromptIndividual
sample_individual = PromptIndividual(
    rule_identifier="zero_1",
    model_name="meta-llama/Llama-3.2-1B-Instruct", # Ensure available
    context=None,
    req="Analyze the conversation.",
    instr="Respond ONLY one time 'yes' or 'no', do not expand your answer.",
    examples=None, cot=None,
    input_content="Placeholder"
)

# 5. Evaluate the sample individual (will use the sample specified in Problem init)
print("\n--- Evaluating Sample Individual (using sampling) ---")
results = {}
try:
    problem._evaluate(sample_individual, results)
    print("\n--- Evaluation Result (Sample Individual) ---")
    if "F" in results:
        print(f"Calculated Objectives (F) based on sample: {results['F']}")
        print(f"  Objective 1 (Sample Avg Accuracy): {results['F'][0]:.4f} -> (Sample Avg Acc: {results['F'][0]:.4f})")
        print(f"  Objective 2 (Sample Avg Total Tokens):   {results['F'][1]:.2f}")
    else:
        print("Evaluation failed to produce objectives.")
except Exception as e:
    print(f"\nAn error occurred during the evaluation example: {e}")


except Exception as e:
    print(f"Failed to initialize or evaluate problem with full dataset: {e}")