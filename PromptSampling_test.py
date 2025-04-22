# --- Make sure PromptIndividual class and constants are defined ---

from PromptSampling import PromptSampling, create_content_dict_from_bnf, tag_to_dict_key_map
from PromptIndividual import ALL_RULE_IDENTIFIERS, PromptIndividual

# Example Data for Sampling
AVAILABLE_HF_MODELS = [
    "mistralai/Mistral-7B-Instruct-v0.2",
    "google/gemma-2-9b-it",
    "meta-llama/Llama-3-8B-Instruct",
]

# Use the same content dictionary structure as the Mutator example
# Note: 'input' key is excluded here, as it's handled separately
# # --- Specify the filename ---
bnf_filename = "grammars/implicatures.txt" # Make sure this file exists and is updated

AVAILABLE_CONTENTS_DICT = create_content_dict_from_bnf(bnf_filename, tag_to_dict_key_map)

# Instantiate the Sampling operator
sampling_op = PromptSampling(
    available_rules=ALL_RULE_IDENTIFIERS, # Use all defined rules
    available_models=AVAILABLE_HF_MODELS,
    available_contents=AVAILABLE_CONTENTS_DICT
)

# Define a dummy problem (only needed for the method signature)
# In a real run, this would be your actual PromptOptimizationProblem instance
class DummyProblem: pass
problem_instance = DummyProblem()

# Generate an initial population
population_size = 10
print(f"\n--- Generating Initial Population (Size: {population_size}) ---")
initial_population_array = sampling_op._do(problem_instance, population_size) # Call _do directly for demo

# Print the generated individuals
print(f"\n--- Generated Population (Array Shape: {initial_population_array.shape}) ---")
for i, ind in enumerate(initial_population_array):
    print(f"\nIndividual {i+1}:")
    if isinstance(ind[0], PromptIndividual):
        print(f"  Type: {type(ind[0])}")
        print(f"  Rule: {ind[0].get_rule_identifier()}")
        print(f"  Model: {ind[0].model_name}")
        print(f"  Context: '{ind[0].context_content}'")
        print(f"  Req: '{ind[0].req_content}'")
        print(f"  Instr: '{ind[0].instr_content}'")
        print(f"  Examples: '{ind[0].examples_content}'")
        print(f"  COT: '{ind[0].cot_content}'")
        print(f"  Input: '{ind[0].input_content}'") # Should be placeholder
    else:
        print(f"  Unexpected item in population: {ind}")

# Verify return type
print(f"\nPopulation Array Type: {type(initial_population_array)}")
print(f"Population Array Dtype: {initial_population_array.dtype}")