import numpy as np
from pymoo.core.duplicate import DuplicateElimination
from typing import Any
from PromptIndividual import PromptIndividual

class PromptDuplicateElimination(DuplicateElimination):
    """
    Custom duplicate elimination strategy for PromptIndividual objects.

    Considers two PromptIndividuals duplicates if all their defining attributes
    (rule, model, context, req, instr, examples, cot, input) are identical.

    This class should be instantiated and passed to the `eliminate_duplicates`
    parameter of algorithms like NSGA2 when working with PromptIndividual objects.
    """
    def is_equal(self, indiv_a: Any, indiv_b: Any) -> bool:
        if not isinstance(indiv_a, PromptIndividual) or not isinstance(indiv_b, PromptIndividual):
            return False

        # Compare all relevant attributes for equality.
        # Return False immediately if any attribute differs.

        # Compare internal rule identifier
        if indiv_a._rule_identifier != indiv_b._rule_identifier:
            return False

        # Compare model name
        if indiv_a.model_name != indiv_b.model_name:
            return False

        # Compare content attributes
        if indiv_a.context_content != indiv_b.context_content:
            return False
        if indiv_a.req_content != indiv_b.req_content:
            return False
        if indiv_a.instr_content != indiv_b.instr_content:
            return False
        if indiv_a.examples_content != indiv_b.examples_content:
            return False
        if indiv_a.cot_content != indiv_b.cot_content:
            return False

        # # Compare input content (even if it's often a placeholder, it defines the object state)
        # if indiv_a.input_content != indiv_b.input_content:
        #     return False

        # If all checks passed, the individuals are considered duplicates
        return True
