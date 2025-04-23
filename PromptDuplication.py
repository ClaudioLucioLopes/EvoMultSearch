import numpy as np
from pymoo.core.duplicate import DuplicateElimination
from typing import Any
from PromptIndividual import PromptIndividual

class PromptDuplicateElimination(DuplicateElimination):
    """
    Custom duplicate elimination strategy for PromptIndividual objects.
    Considers two PromptIndividuals duplicates if their defining, evolvable
    attributes (rule, model, context, req, instr, examples, cot) are identical.
    The 'input_content' attribute is intentionally ignored.
    """
    def is_equal(self, indiv_a: Any, indiv_b: Any) -> bool:
        if not isinstance(indiv_a, PromptIndividual) or not isinstance(indiv_b, PromptIndividual):
            return False
        # Compare all relevant attributes EXCEPT input_content
        return (indiv_a._rule_identifier == indiv_b._rule_identifier and
                indiv_a.model_name == indiv_b.model_name and
                indiv_a.context_content == indiv_b.context_content and
                indiv_a.req_content == indiv_b.req_content and
                indiv_a.instr_content == indiv_b.instr_content and
                indiv_a.examples_content == indiv_b.examples_content and
                indiv_a.cot_content == indiv_b.cot_content)