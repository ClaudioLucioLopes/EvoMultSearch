import random
import copy
import numpy as np
from typing import List, Any, Optional, Dict, Tuple, Union
from PromptIndividual import ALL_COMPONENT_NAMES, PromptIndividual
from pymoo.core.crossover import Crossover

class PromptCrossover(Crossover):
    """
    Performs crossover between two PromptIndividual parents.

    Handles model name crossover based on a specific probability, and
    content attribute crossover using uniform crossover independently.
    The rule identifier is not crossed over.
    """

    def __init__(self,
                 prob_model: float = 0.5,
                 prob_attr_swap: float = 0.5):
        """
        Initializes the PromptCrossover operator.

        Args:
            prob_model: The probability (0.0 to 1.0) of swapping the model_name
                        between the offspring.
            prob_attr_swap: The probability (0.0 to 1.0) for swapping each
                            individual content attribute during uniform crossover.
        """
        super().__init__(n_parents=2, n_offsprings=2,)
        if not (0.0 <= prob_model <= 1.0):
            raise ValueError("prob_model must be between 0.0 and 1.0")
        if not (0.0 <= prob_attr_swap <= 1.0):
             raise ValueError("prob_attr_swap must be between 0.0 and 1.0")

        self.prob_model = prob_model
        self.prob_attr_swap = prob_attr_swap


        self.prob_model = prob_model
        self.prob_attr_swap = prob_attr_swap

        # List of content attribute names to be potentially swapped
        # **MODIFIED**: Exclude 'input' from the list comprehension
        self.swappable_attribute_names = [
            f"{name}_content" for name in ALL_COMPONENT_NAMES if name != 'input'
        ]
        if not self.swappable_attribute_names:
             print("Warning: No swappable attributes defined (excluding 'input'). "
                   "Attribute crossover will have no effect.")

    def _do(self, problem, X, **kwargs):
        """
        Executes the crossover operation.

        Args:
            problem: The optimization problem instance (unused here but required by signature).
            X: A numpy array containing pairs of parent individuals.
               Shape: (n_parents, n_matings, n_var) -> (2, n_matings, 1)
            **kwargs: Additional arguments.

        Returns:
            A numpy array containing pairs of offspring individuals.
            Shape: (n_offsprings, n_matings, n_var) -> (2, n_matings, 1)
        """
        # Number of pairs participating in crossover (matings)
        # n_parents = number of parents involved (should be 2)
        # n_matings = number of pairs to be crossed over
        # n_var = number of variables per individual (1, the object itself)
        n_parents, n_matings, n_var = X.shape

        # Initialize the output array for offspring
        # Shape: (n_offsprings, n_matings, n_var)
        Y = np.full((self.n_offsprings, n_matings, n_var), None, dtype=object)

        # Loop through each pair of parents (matings)
        for i in range(n_matings):
            # Get the two parents for this mating
            parent1: PromptIndividual = X[0, i, 0] # Access the object
            parent2: PromptIndividual = X[1, i, 0] # Access the object

            # Create deep copies for offspring
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()

            # 1. Model Crossover
            if random.random() < self.prob_model:
                offspring1.model_name, offspring2.model_name = offspring2.model_name, offspring1.model_name
                # Optional print for debugging:
                # print(f"  [CX Run {i}] Model names SWAPPED.")

            # 2. Attribute Crossover (Uniform, excluding input)
            for attr_name in self.swappable_attribute_names:
                val1 = getattr(parent1, attr_name, None) # Get original parent values
                val2 = getattr(parent2, attr_name, None)
                # Only consider swapping if BOTH parent values are not None
                if val1 is not None and val2 is not None:
                    # Apply the swap probability only to attributes present in both parents
                    if random.random() < self.prob_attr_swap:
                        # Perform the swap on offspring (values are guaranteed non-None here)
                        setattr(offspring1, attr_name, val2)
                        setattr(offspring2, attr_name, val1)
                        # print(f"  [CX Run {i}] Attribute '{attr_name}' SWAPPED (both parents non-None).") # Debug
                # --- END MODIFIED LOGIC ---
                # else: at least one parent has None for this attribute, so no swap occurs

            # Store the generated offspring in the output array Y
            Y[0, i, 0] = offspring1
            Y[1, i, 0] = offspring2

        return Y