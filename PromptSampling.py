import re
import json
import numpy as np
from pymoo.core.sampling import Sampling
from typing import List, Any, Optional, Dict, Tuple, Union
import random
from PromptIndividual import ALL_COMPONENT_NAMES, PromptIndividual,RULE_TO_COMPONENTS
from PromptOptimizationProblem import PromptOptimizationProblem
import re

# --- Sampling Class Definition ---

class PromptSampling(Sampling):
    """
    Pymoo Sampling class to generate an initial population of PromptIndividual objects.
    Randomly selects rule identifiers, model names, and content for each component
    from provided lists/dictionaries.
    """

    def __init__(self,
                 available_rules: List[str],
                 available_models: List[str],
                 available_contents: Dict[str, List[Optional[str]]]):
        """
        Initializes the PromptSampling operator.

        Args:
            available_rules: A list of valid rule identifiers (strings) to choose from.
            available_models: A list of valid model name strings (e.g., HF IDs).
            available_contents: A dictionary where keys are component names
                                (e.g., 'context', 'req', 'instr', 'examples', 'cot')
                                and values are lists of potential string values
                                (or None) for that specific component.
                                It's recommended to exclude 'input' here, as its
                                content is typically set during evaluation.
        """
        super().__init__() # Call parent constructor

        # --- Validation ---
        if not available_rules:
            raise ValueError("available_rules list cannot be empty.")
        if not available_models:
            raise ValueError("available_models list cannot be empty.")
        if not isinstance(available_contents, dict) or not available_contents:
            raise ValueError("available_contents must be a non-empty dictionary.")

        # Validate content dictionary structure
        valid_component_names = set(ALL_COMPONENT_NAMES)
        for component_name, value_list in available_contents.items():
            if component_name not in valid_component_names:
                 print(f"Warning: Key '{component_name}' in available_contents is not in ALL_COMPONENT_NAMES.")
            if not isinstance(value_list, list) or not value_list:
                raise ValueError(f"Value for component '{component_name}' in available_contents must be a non-empty list.")
            if not all(isinstance(v, str) or v is None for v in value_list):
                 raise TypeError(f"All values for component '{component_name}' must be strings or None.")
        # --- End Validation ---

        self.available_rules = available_rules
        self.available_models = available_models
        self.available_contents = available_contents
        # Define components for which content needs to be sampled
        self.content_components_to_sample = list(available_contents.keys())


    def _do(self, problem: 'PromptOptimizationProblem', n_samples: int, **kwargs) -> np.ndarray:
        """
        Generate `n_samples` PromptIndividual objects.

        Args:
            problem: The optimization problem instance (provides context, though not strictly used here).
            n_samples: The number of individuals to generate.
            **kwargs: Additional arguments.

        Returns:
            A NumPy array of dtype=object containing the generated PromptIndividual instances.
        """
        population =  np.full((n_samples, 1), None, dtype=object)

        for i_sample in range(n_samples):
            # 1. Choose Rule Identifier
            rule_id = random.choice(self.available_rules)

            # 2. Choose Model Name
            model_name = random.choice(self.available_models)

            # 3. Choose Content for specified components
            content_dict = {}
            for component_name in ALL_COMPONENT_NAMES:
                if component_name == 'input':
                    # Assign placeholder for input_content
                    content_dict['input'] = "PLACEHOLDER_INPUT_CONTENT"
                elif component_name in RULE_TO_COMPONENTS.get(rule_id, []):
                    # Select randomly from the provided list for this component
                    content_dict[component_name] = random.choice(self.available_contents[component_name])
                else:
                     # Assign None if no content pool provided for other components
                     content_dict[component_name] = None


            # 4. Create PromptIndividual instance
            try:
                individual = PromptIndividual(
                    rule_identifier=rule_id,
                    model_name=model_name,
                    context=content_dict.get('context'),
                    req=content_dict.get('req'),
                    instr=content_dict.get('instr'),
                    examples=content_dict.get('examples'),
                    cot=content_dict.get('cot'),
                    input_content=content_dict['input'] # Use placeholder
                )
                population[i_sample, 0] = individual
            except Exception as e:
                print(f"Error creating PromptIndividual during sampling: {e}")
                print(f"  Rule: {rule_id}, Model: {model_name}, Content Dict: {content_dict}")
                # Decide how to handle errors: skip, retry, raise? Let's skip for now.
                continue # Skip adding this individual if creation failed

        # Ensure we generated the requested number, handle potential errors during creation
        if len(population) != n_samples:
             print(f"Warning: Generated {len(population)} samples instead of the requested {n_samples} due to errors.")
             # Optionally, you could add logic here to generate more samples to reach n_samples

        # Convert the list of objects to a NumPy array of objects
        # This is the format Pymoo algorithms typically expect
        return np.array(population, dtype=object)



def create_content_dict_from_bnf(filepath: str, tag_map: Dict[str, str]) -> Dict[str, List[Optional[str]]]:
    """
    Reads a BNF definition file, parses content options for specified tags,
    and returns the resulting dictionary. Relies solely on the file content
    for definitions based on the tag_map.

    Args:
        filepath: The path to the file containing BNF definitions.
        tag_map: A dictionary mapping BNF tags (e.g., 'context', 'instr') to the
                 desired keys in the output dictionary (e.g., 'context', 'instr').

    Returns:
        A dictionary where keys are the target keys from tag_map and values
        are lists of option strings for the corresponding component.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        Exception: For other file reading or parsing errors.
    """
    print(f"Attempting to load and parse BNF definitions from: {filepath}")
    result_dict: Dict[str, List[Optional[str]]] = {} # Allow Optional[str] for None values

    try:
        # --- File Reading ---
        with open(filepath, 'r', encoding='utf-8') as f:
            bnf_content = f.read()
        print("File read successfully.")

        # --- Parsing Logic ---
        # Regex to find definitions: <tag> ::= options... (until next tag or EOF)
        pattern = re.compile(r"<(\w+)>\s*::=\s*(.*?)(?=\n\s*<\w+>|\Z)", re.DOTALL | re.MULTILINE)

        for match in pattern.finditer(bnf_content):
            tag_name = match.group(1)
            options_text = match.group(2)

            # Check if the parsed tag is one we are interested in via the map
            if tag_name in tag_map:
                target_key = tag_map[tag_name]
                # Split options by '|', strip whitespace, filter empty
                options = [opt.strip() for opt in options_text.split('|') if opt.strip()]

                if options:
                     # Initialize list if key doesn't exist
                     if target_key not in result_dict:
                         result_dict[target_key] = []
                     # Extend list with parsed options
                     result_dict[target_key].extend(options)
                     print(f"  Parsed {len(options)} options for tag '<{tag_name}>' -> key '{target_key}'.")
                else:
                    print(f"Warning: No valid options found for tag '<{tag_name}>' in file.")

        print("BNF content parsing complete.")

        # --- REMOVED: Manual addition of 'instr' options ---
        # The parsing loop above will now handle <instr> if defined in the file
        # and included in the tag_map.

        # --- Optional: Add None to value lists if desired ---
        # Can be useful for mutation/sampling to allow removing a component
        # for key in result_dict:
        #      if None not in result_dict[key]:
        #           result_dict[key].append(None)

        # Final check for expected keys based on tag_map
        for expected_key in tag_map.values():
            if expected_key not in result_dict:
                print(f"Warning: Key '{expected_key}' (from tag_map) was not found or parsed from the file '{filepath}'.")


        return result_dict

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        raise
    except Exception as e:
        print(f"Error reading or parsing file {filepath}: {e}")
        raise


# # --- Mapping from BNF tags to desired dictionary keys ---
# # Ensure this map includes all tags you expect to parse from the file
tag_to_dict_key_map = {
    'context': 'context',
    'req': 'req',
    'example': 'examples', # Tag name containing example definitions
    'sbs': 'cot',         # Map <sbs> tag to 'cot' key
    'instr': 'instr'      # Map <instr> tag to 'instr' key
}

# # --- Specify the filename ---
# bnf_filename = "grammars/implicatures.txt" # Make sure this file exists and is updated

# AVAILABLE_CONTENTS_DICT = create_content_dict_from_bnf(bnf_filename, tag_to_dict_key_map)
# print(AVAILABLE_CONTENTS_DICT)


