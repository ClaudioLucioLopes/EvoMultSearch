import random
import copy
from typing import List, Any, Optional, Dict, Tuple, Union
from PromptIndividual import ALL_COMPONENT_NAMES, PromptIndividual
from pymoo.core.mutation import Mutation

class PromptMutator(Mutation):
    """
    Mutates PromptIndividual objects by potentially changing the model name
    or one of the content parameters (excluding 'input') based on given
    probabilities, using component-specific value pools.
    """

    def __init__(self,
                 available_models: List[str],
                 possible_content_values: Dict[str, List[str]], 
                 prob_model: float = 0.1,
                 prob_parameter: float = 0.2):
        """
        Initializes the PromptMutator.

        Args:
            available_models: A list of valid Hugging Face model IDs to choose from.
            possible_content_values: A dictionary where keys are component names
                                     (e.g., 'context', 'req', 'instr', 'examples', 'cot' -
                                     EXCLUDING 'input') and values are lists of
                                     potential new string values for that component.
            prob_model: The probability (0.0 to 1.0) of attempting to mutate the model_name.
            prob_parameter: The probability (0.0 to 1.0) of attempting to mutate one
                            of the content parameters specified in possible_content_values.
        """
        super().__init__(prob=prob_parameter)

        # --- Validation ---
        if not (0.0 <= prob_model <= 1.0):
            raise ValueError("prob_model must be between 0.0 and 1.0")
        if not (0.0 <= prob_parameter <= 1.0):
            raise ValueError("prob_parameter must be between 0.0 and 1.0")
        if not available_models:
            raise ValueError("available_models list cannot be empty.")
        if not isinstance(possible_content_values, dict):
             raise TypeError("possible_content_values must be a dictionary.")
        if not possible_content_values:
             raise ValueError("possible_content_values dictionary cannot be empty.")

        # Validate the dictionary structure and content
        valid_component_names = set(ALL_COMPONENT_NAMES)
        if 'input' in valid_component_names:
             valid_component_names.remove('input') # Exclude 'input' from allowed keys

        self.content_attribute_names: List[str] = []
        for component_name, value_list in possible_content_values.items():
            if component_name not in valid_component_names:
                raise ValueError(f"Invalid component name '{component_name}' in possible_content_values. "
                                 f"Allowed names (excluding 'input'): {valid_component_names}")
            if not isinstance(value_list, list) or not value_list:
                raise ValueError(f"Value for component '{component_name}' must be a non-empty list.")
            if not all(isinstance(v, str) or v is None for v in value_list):
                 # Allow strings or None in the lists
                 raise TypeError(f"All values for component '{component_name}' must be strings or None.")
            # Add the corresponding attribute name (e.g., 'context_content') to our list
            self.content_attribute_names.append(f"{component_name}_content")

        if not self.content_attribute_names:
             raise ValueError("No valid mutable component attributes derived from possible_content_values keys.")
        # --- End Validation ---

        self.available_models = available_models
        # Store the dictionary directly
        self.possible_content_values: Dict[str, List[Optional[str]]] = possible_content_values
        self.prob_model = prob_model
        self.prob_parameter = prob_parameter


    def _do(self, problem, X, **kwargs):
        """
        Executes the mutation operation on the population X.
        This method defines *what* happens if mutation is applied.

        Args:
            problem: The optimization problem instance.
            X: A numpy array containing the individuals to be mutated.
               Shape: (n_individuals, n_var) -> (n_individuals, 1)
            **kwargs: Additional arguments.

        Returns:
            The mutated population array (modified in-place or a new array).
            Pymoo's base class handles returning the correct object based on prob.
            This method should return the potentially modified X.
        """
        # Iterate through each individual in the population array X
        # The base class's __call__ method handles the 'prob' check,
        # so _do is called only for individuals selected for mutation.
        for i in range(len(X)):
            individual: PromptIndividual = X[i, 0] # Get the individual object
            mutated_individual = individual.copy() # Work on a copy
            mutated = False
            
            # 1. Attempt Model Mutation (based on internal prob_model)
            if random.random() < self.prob_model:
                current_model = mutated_individual.model_name
                potential_new_models = [m for m in self.available_models if m != current_model]
                if potential_new_models:
                    new_model = random.choice(potential_new_models)
                    mutated_individual.model_name = new_model
                    mutated = True
                    # print(f"  [Mutation Ind {i}] Model changed.") # Debug

            # 2. Attempt Parameter Mutation (based on internal prob_parameter)
            if random.random() < self.prob_parameter and self.content_attribute_names:
                attr_to_mutate = random.choice(mutated_individual.get_active_component_attrs())
                component_name = attr_to_mutate.replace("_content", "")
                current_value = getattr(mutated_individual, attr_to_mutate, None)
                value_pool = self.possible_content_values.get(component_name, []) # Use .get for safety

                if value_pool: # Check if pool is not empty
                    potential_new_values = [v for v in value_pool if v != current_value]
                    if not potential_new_values:
                        potential_new_values = value_pool # Fallback if only one value exists

                    new_value = random.choice(potential_new_values)
                    if new_value != current_value:
                        setattr(mutated_individual, attr_to_mutate, new_value)
                        mutated = True
                        # print(f"  [Mutation Ind {i}] Attribute '{attr_to_mutate}' changed.") # Debug

            # If any mutation occurred, update the individual in the population array
            if mutated:
                X[i, 0] = mutated_individual

        # Return the modified population array
        return X