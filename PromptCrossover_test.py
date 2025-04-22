from PromptIndividual import PromptIndividual
from PromptCrossover import PromptCrossover
import numpy as np
from typing import List, Any, Optional, Dict, Tuple, Union

# Create two distinct parent individuals
parent_A = PromptIndividual(
    rule_identifier="zerocot_1",
    model_name="model-alpha", # Different model
    context="Context A",
    req="Request A: Summarize.",
    instr="Instruction A: Be brief.",
    examples=None,
    cot="COT A: Step 1...",
    input_content="Input A: Text to summarize."
)

parent_B = PromptIndividual(
    rule_identifier="combi_2", # Different rule
    model_name="model-beta",  # Different model
    context="Context B",
    req="Request B: Translate.",
    instr="Instruction B: To German.",
    examples="Examples B: hello->hallo",
    cot="COT B: Think about grammar...",
    input_content="Input B: Text to translate."
)

crossover_op = PromptCrossover(
    prob_model=0.6,      # 60% chance to swap model names
    prob_attr_swap=0.5   # 50% chance per attribute to swap (uniform)
)

problem = None
X = np.array([[parent_A], [parent_B]]).reshape(2, 1, 1)
print(X.shape)
print(X)

for i in range(3):
    print(f"\nCrossover Attempt {i+1}:")
    Y = crossover_op._do(problem, X)
    offspring_1 = Y[0, 0, 0]
    offspring_2 = Y[1, 0, 0]

    print("\nOffspring 1:")
    print(offspring_1)
    print(f"  Model: {offspring_1.model_name}, Rule: {offspring_1.get_rule_identifier()}") # Rule should match parent A initially
    print(f"  Context: '{offspring_1.context_content}'")
    print(f"  Req: '{offspring_1.req_content}'")
    print(f"  Instr: '{offspring_1.instr_content}'")
    print(f"  Examples: '{offspring_1.examples_content}'")
    print(f"  COT: '{offspring_1.cot_content}'")
    print(f"  Input: '{offspring_1.input_content}'")

    print("\nOffspring 2:")
    print(offspring_2)
    print(f"  Model: {offspring_2.model_name}, Rule: {offspring_2.get_rule_identifier()}") # Rule should match parent B initially
    print(f"  Context: '{offspring_2.context_content}'")
    print(f"  Req: '{offspring_2.req_content}'")
    print(f"  Instr: '{offspring_2.instr_content}'")
    print(f"  Examples: '{offspring_2.examples_content}'")
    print(f"  COT: '{offspring_2.cot_content}'")
    print(f"  Input: '{offspring_2.input_content}'")
    print("-" * 40)