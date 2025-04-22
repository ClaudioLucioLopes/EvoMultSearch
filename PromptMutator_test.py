from PromptIndividual import PromptIndividual
from PromptMutator import PromptMutator
from typing import List, Any, Optional, Dict, Tuple, Union
from PromptSampling import tag_to_dict_key_map, create_content_dict_from_bnf
from PromptIndividual import PromptIndividual
import numpy as np


AVAILABLE_HF_MODELS = [
"meta-llama/Llama-3.2-1B-Instruct"]


bnf_filename = "grammars/implicatures.txt" # Make sure this file exists and is updated

POSSIBLE_CONTENTS_DICT = create_content_dict_from_bnf(bnf_filename, tag_to_dict_key_map)


# Instantiate the Mutator
mutator = PromptMutator(
    available_models=AVAILABLE_HF_MODELS,
    possible_content_values=POSSIBLE_CONTENTS_DICT, # Pass the dictionary
    prob_model=0.5,
    prob_parameter=0.8 # Increased parameter mutation probability for demo
)


# Create an initial PromptIndividual
initial_individual = PromptIndividual(
    rule_identifier="zero_3", 
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    context="Envision being a participant in a leadership training program, responding to questions about leadership styles.",
    req=None,
    instr=None,
    examples=None,
    cot=None,
    input_content="PLACEHOLDER_INPUT_CONTENT"
)

print("Initial Individual:")
print(initial_individual)
print("-" * 40)

problem = None
X = np.array([[initial_individual]]).reshape(1, 1)

# Perform mutation multiple times
for i in range(10):
    all_mutations = mutator._do(problem,X)
    for mutated_copy in all_mutations: # More attempts to see parameter changes
        print("\nResulting Individual:")
        print(mutated_copy[0])
        print(f"  Context: '{mutated_copy[0].context_content}'")
        print(f"  Req: '{mutated_copy[0].req_content}'")
        print(f"  Instr: '{mutated_copy[0].instr_content}'")
        print(f"  Examples: '{mutated_copy[0].examples_content}'")
        print(f"  COT: '{mutated_copy[0].cot_content}'")
        print(f"  Input: '{mutated_copy[0].input_content}'") # Verify input remains unchanged by param mutation
        print("-" * 40)

