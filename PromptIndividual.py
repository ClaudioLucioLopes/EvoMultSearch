# --- Required Mappings (MUST BE UPDATED in your code) ---

PROMPT_TEMPLATE_STRING = r"""
{# <START_OF_SYSTEM_PROMPT> #}
{# Context always goes in the system prompt if provided #}
{% if context_content is not none %}{{context_content}}{% endif %}
{# <END_OF_SYSTEM_PROMPT> #}

{# <START_OF_USER> #}
{# --- Rule: zero_1 --- #}
{% if rule_identifier == "zero_1" %}
{{ req_content }} {{ instr_content }}\n{{ input_content }}
{# --- Rule: zero_2 --- #}
{% elif rule_identifier == "zero_2" %}
{{ input_content }}\n{{ req_content }} {{ instr_content }}
{# --- Rule: zero_3 (Context is handled in SYSTEM) --- #}
{% elif rule_identifier == "zero_3" %}
{{ input_content }}
{# --- Rule: cot_1 --- #}
{% elif rule_identifier == "cot_1" %}
{{ req_content }} {{ instr_content }} For instance:\n{{ examples_content }}\n{{ input_content }}
{# --- Rule: cot_2 --- #}
{% elif rule_identifier == "cot_2" %}
{{ req_content }} {{ instr_content }}\n{{ input_content }} For instance:\n{{ examples_content }}
{# --- Rule: cot_3 --- #}
{% elif rule_identifier == "cot_3" %}
{{ input_content }}\n{{ req_content }} {{ instr_content }} For instance:\n{{ examples_content }}
{# --- Rule: cot_4 (Context is handled in SYSTEM) --- #}
{% elif rule_identifier == "cot_4" %}
{{ input_content }} For instance:\n{{ examples_content }}
{# --- Rule: zerocot_1 --- #}
{% elif rule_identifier == "zerocot_1" %}
{{ req_content }} {{ instr_content }}\n{{ input_content }}\n{{ cot_content }}
{# --- Rule: zerocot_2 --- #}
{% elif rule_identifier == "zerocot_2" %}
{{ input_content }}\n{{ req_content }} {{ instr_content }}\n{{ cot_content }}
{# --- Rule: zerocot_3 (Context is handled in SYSTEM) --- #}
{% elif rule_identifier == "zerocot_3" %}
{{ input_content }}\n{{ cot_content }}
{# --- Rule: zerocot_4 --- #}
{% elif rule_identifier == "zerocot_4" %}
{{ cot_content }}\n{{ req_content }} {{ instr_content }}\n{{ input_content }}
{# --- Rule: zerocot_5 --- #}
{% elif rule_identifier == "zerocot_5" %}
{{ cot_content }}\n{{ input_content }}\n{{ req_content }} {{ instr_content }}
{# --- Rule: zerocot_6 (Context is handled in SYSTEM) --- #}
{% elif rule_identifier == "zerocot_6" %}
{{ cot_content }}\n{{ input_content }}
{# --- Rule: zerocot_7 --- #}
{% elif rule_identifier == "zerocot_7" %}
{{ req_content }} {{ instr_content }}\n{{ cot_content }}\n{{ input_content }}
{# --- Rule: combi_1 --- #}
{% elif rule_identifier == "combi_1" %}
{{ req_content }} {{ instr_content }} For instance:\n{{ examples_content }}\n{{ cot_content }}\n{{ input_content }}
{# --- Rule: combi_2 --- #}
{% elif rule_identifier == "combi_2" %}
Consider these examples:\n{{ examples_content }}\n{{ req_content }} {{ instr_content }}\n{{ input_content }}\n{{ cot_content }}
{# --- Rule: combi_3 --- #}
{% elif rule_identifier == "combi_3" %}
Consider these examples:\n{{ examples_content }}\n{{ input_content }}\n{{ req_content }} {{ instr_content }}\n{{ cot_content }}
{# --- Rule: combi_4 (Context is handled in SYSTEM) --- #}
{% elif rule_identifier == "combi_4" %}
Consider these examples:\n{{ examples_content }}\n{{ input_content }}\n{{ cot_content }}
{# --- Rule: combi_5 --- #}
{% elif rule_identifier == "combi_5" %}
Consider these examples:\n{{ examples_content }}\n{{ cot_content }}\n{{ req_content }} {{ instr_content }}\n{{ input_content }}
{# --- Rule: combi_6 --- #}
{% elif rule_identifier == "combi_6" %}
Consider these examples:\n{{ examples_content }}\n{{ cot_content }}\n{{ input_content }}\n{{ req_content }} {{ instr_content }}
{# --- Rule: combi_7 (Context is handled in SYSTEM) --- #}
{% elif rule_identifier == "combi_7" %}
Consider these examples:\n{{ examples_content }}\n{{ cot_content }}\n{{ input_content }}
{# --- Rule: combi_8 --- #}
{% elif rule_identifier == "combi_8" %}
Consider these examples:\n{{ examples_content }}\n{{ req_content }} {{ instr_content }}\n{{ cot_content }}\n{{ input_content }}
{# --- Rule: combi_9 --- #}
{% elif rule_identifier == "combi_9" %}
{{ req_content }} {{ instr_content }} For instance:\n{{ examples_content }}\n{{ input_content }}\n{{ cot_content }}
{# --- Rule: combi_10 --- #}
{% elif rule_identifier == "combi_10" %}
{{ req_content }} {{ instr_content }}\n{{ input_content }} For instance:\n{{ examples_content }}\n{{ cot_content }}
{# --- Rule: combi_11 --- #}
{% elif rule_identifier == "combi_11" %}
{{ input_content }}\n{{ req_content }} {{ instr_content }} For instance:\n{{ examples_content }}\n{{ cot_content }}
{# --- Rule: combi_12 (Context is handled in SYSTEM) --- #}
{% elif rule_identifier == "combi_12" %}
{{ input_content }} For instance:\n{{ examples_content }}\n{{ cot_content }}
{# --- Fallback/Error --- #}
{% else %}
Error: Invalid or missing rule_identifier '{{ rule_identifier }}' provided to the template.
{% endif %}
{# <END_OF_USER> #}
"""

# Change 'sbs' to 'cot' in this list
ALL_COMPONENT_NAMES = ['context', 'req', 'instr', 'examples', 'cot', 'input']

# Change 'sbs' to 'cot' in the *values* (lists) of this dictionary
RULE_TO_COMPONENTS = {
    "zero_1": ['req', 'instr', 'input'],
    "zero_2": ['input', 'req', 'instr'],
    "zero_3": ['context', 'input'],
    "cot_1": ['req', 'instr', 'examples', 'input'],
    "cot_2": ['req', 'instr', 'input', 'examples'],
    "cot_3": ['input', 'req', 'instr', 'examples'],
    "cot_4": ['context', 'input', 'examples'],
    "zerocot_1": ['req', 'instr', 'input', 'cot'], 
    "zerocot_2": ['input', 'req', 'instr', 'cot'],
    "zerocot_3": ['context', 'input', 'cot'],     
    "zerocot_4": ['cot', 'req', 'instr', 'input'], 
    "zerocot_5": ['cot', 'input', 'req', 'instr'], 
    "zerocot_6": ['cot', 'context', 'input'],      
    "zerocot_7": ['req', 'instr', 'cot', 'input'], 
    "combi_1": ['req', 'instr', 'examples', 'cot', 'input'], 
    "combi_2": ['examples', 'req', 'instr', 'input', 'cot'], 
    "combi_3": ['examples', 'input', 'req', 'instr', 'cot'], 
    "combi_4": ['examples', 'context', 'input', 'cot'],      
    "combi_5": ['examples', 'cot', 'req', 'instr', 'input'], 
    "combi_6": ['examples', 'cot', 'input', 'req', 'instr'], 
    "combi_7": ['examples', 'cot', 'context', 'input'],      
    "combi_8": ['examples', 'req', 'instr', 'cot', 'input'], 
    "combi_9": ['req', 'instr', 'examples', 'input', 'cot'], 
    "combi_10": ['req', 'instr', 'input', 'examples', 'cot'], 
    "combi_11": ['input', 'req', 'instr', 'examples', 'cot'], 
    "combi_12": ['context', 'input', 'examples', 'cot'],      
}
ALL_RULE_IDENTIFIERS = list(RULE_TO_COMPONENTS.keys())


# --- PromptIndividual Class Definition ---
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from jinja2 import Environment, Template # Needed for template processing
from typing import Tuple, Any, Optional, Dict, List, Union 

class PromptIndividual:
    """
    Represents a single prompt instance intended for use with pymoo.

    Stores the prompt's structure internally (_rule_identifier) and content
    for components including context, request, instruction, examples,
    chain-of-thought (cot), and a mandatory input.
    """
    def __init__(self, rule_identifier: str,
                 model_name: str, # Name of the target model
                 context: Union[str, None], # Python 3.10+: context: str | None
                 req: Union[str, None],
                 instr: Union[str, None],
                 examples: Union[str, None],
                 cot: Union[str, None], # Renamed from sbs
                 input_content: str):   # Changed: Cannot be None
        """
        Initializes a PromptIndividual.

        Parameters
        ----------
        rule_identifier : str
            Identifier for the prompt structure (key in RULE_TO_COMPONENTS). Stored internally.
        model_name : str
            Identifier for the target language model (e.g., "Llama3.2-1B"). Cannot be empty.
        context : str or None
            Content for the <context> component.
        req : str or None
            Content for the <req> component.
        instr : str or None
            Content for the <instr> component.
        examples : str or None
            Content for the <examples> component.
        cot : str or None
            Content for the <cot> (Chain-of-Thought) component. Replaces 'sbs'.
        input_content : str
            Content for the <input> component. This cannot be None.
        """
        # Validate the rule_identifier before storing it internally
        if rule_identifier not in RULE_TO_COMPONENTS:
             raise ValueError(f"Unknown rule_identifier provided: {rule_identifier}")

        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string.")


        # Ensure input_content is provided
        if input_content is None:
            raise ValueError("input_content cannot be None.")
        # Optional: Add check if input_content is an empty string if that's invalid too
        # if not isinstance(input_content, str) or len(input_content) == 0:
        #    raise ValueError("input_content must be a non-empty string.")


        # Store rule_identifier with a single underscore to denote internal/protected use
        self._rule_identifier: str = rule_identifier

        # Store the target model name (public attribute)
        self.model_name: str = model_name

        # Content attributes remain public
        self.context_content: Optional[str] = context
        self.req_content: Optional[str] = req
        self.instr_content: Optional[str] = instr
        self.examples_content: Optional[str] = examples
        self.cot_content: Optional[str] = cot 
        self.input_content: str = input_content 

    def get_rule_identifier(self) -> str:
        """Returns the internal rule identifier."""
        return self._rule_identifier

    def get_active_component_attrs(self) -> List[str]:
        """
        Returns the list of full attribute names (e.g., 'req_content', 'cot_content')
        corresponding to the components that are active/used by this
        individual's internal rule_identifier, based on the RULE_TO_COMPONENTS mapping.
        """
        # Access the internal attribute
        # Uses the externally defined RULE_TO_COMPONENTS map (which MUST be updated)
        required_short_names = RULE_TO_COMPONENTS.get(self._rule_identifier, [])
        # Convert to full attribute names (like 'req_content', 'cot_content')
        return [f"{name}_content" for name in required_short_names]

    def __repr__(self) -> str:
        """Provides a concise string representation, hiding the internal rule ID."""
        active_attrs_list = self.get_active_component_attrs()
        # Ensure ALL_COMPONENT_NAMES used here is the updated one (with 'cot')
        active_short_names = {attr.replace('_content', '') for attr in active_attrs_list}

        content_repr_parts = []
        # Uses the externally defined ALL_COMPONENT_NAMES list (which MUST be updated)
        for short_name in ALL_COMPONENT_NAMES:
            if short_name in active_short_names:
                attr_name = f"{short_name}_content"
                has_content = getattr(self, attr_name, None) is not None
                content_repr_parts.append(f"{short_name}={'Yes' if has_content else 'No'}")

        content_summary = ", ".join(content_repr_parts)
        return (f"PromptIndividual(Model: {self.model_name}, "
                f"Rule: {self._rule_identifier}, " # Use internal directly in repr
                f"ActiveComponents: [{content_summary}])")
    
    def copy(self):
        import copy
        return copy.deepcopy(self)
    
    def execute_prompt_with_transformer(self, device: str = "auto",
                                        max_new_tokens: int = 512,**generation_kwargs: Any # Pass additional args to model.generate (e.g., temperature, do_sample)
    ) -> Tuple[str, int, int]:
        """
        Constructs a prompt from PromptIndividual, executes it using a Hugging Face
        transformer model, and returns the response text and token counts.

        Args:
            individual: An instance of the PromptIndividual class containing the
                        prompt components, rule identifier, and model name.
            device: The device to run inference on ("auto", "cuda", "cpu", "mps").
                    Defaults to "auto".
            max_new_tokens: Maximum number of new tokens to generate.
            **generation_kwargs: Additional keyword arguments passed directly to
                                `model.generate()`. Examples: `temperature=0.7`,
                                `do_sample=True`, `top_k=50`.

        Returns:
            A tuple containing:
            - The generated response text (str).
            - The number of input tokens (int).
            - The number of newly generated output tokens (int).

        Raises:
            ImportError: If 'transformers', 'torch', or 'jinja2' are not installed.
            ValueError: If the model_name is invalid or template rendering fails.
            RuntimeError: If issues occur during model loading or generation (e.g., OOM).
            Exception: Other potential errors during execution.
        """
        model_id = self.model_name
        if not model_id:
            raise ValueError("PromptIndividual must have a valid 'model_name' attribute set to a Hugging Face ID.")

        # Create a Jinja2 template object
        # --- Jinja2 Template Object (Needs to be created from the string) ---
        # Ensure PROMPT_TEMPLATE_STRING is defined correctly above
        try:
            prompt_template = Template(PROMPT_TEMPLATE_STRING)
        except NameError:
            raise NameError("PROMPT_TEMPLATE_STRING is not defined. Make sure the template string is included.")

        # --- 1. Get Model and Tokenizer---
        tokenizer = None
        model = None

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto", # device_map='auto' is often needed with quantization
                    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16, # Use bfloat16 if available for better perf/acc
                    trust_remote_code=True, # Sometimes needed for custom architectures
                    # use_flash_attention_2=True, # Optional: Requires flash-attn library for faster attention
                )
                # If not using device_map, manually move model:
                # if not bnb_config and device != "auto":
                #     model.to(torch.device(device)) # Move model to specified device if not quantized

            model.eval() # Set model to evaluation mode
        except Exception as e:
            raise RuntimeError(f"Failed to load model or tokenizer '{model_id}': {e}") from e

        
        # --- 2. Prepare Prompt ---
        try:
            template_context: Dict[str, Any] = {
                "rule_identifier": self.get_rule_identifier(),
                "context_content": self.context_content,
                "req_content": self.req_content,
                "instr_content": self.instr_content,
                "examples_content": self.examples_content,
                "cot_content": self.cot_content,
                "input_content": self.input_content
            }
            rendered_prompt = prompt_template.render(template_context)
            print('---'*40,rendered_prompt)
        except Exception as e:
            raise ValueError(f"Failed to render prompt template: {e}") from e
        

        # --- 3. Tokenize Input ---
        try:
            # Ensure tokenizer has a pad token; common for generation with left padding
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token 

            inputs = tokenizer(rendered_prompt, return_tensors="pt", padding=False, truncation=False) # No padding/truncation here
            input_ids = inputs["input_ids"].to(model.device) # Move input IDs to model's device
            input_token_count = input_ids.shape[1]
        except Exception as e:
            raise RuntimeError(f"Failed to tokenize prompt: {e}") from e


        print(f"Generating response (max_new_tokens={max_new_tokens})...")
        try:
            with torch.no_grad(): # Disable gradient calculation for inference
                generation_config = {
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                    **generation_kwargs # Merge user-provided generation args
                }
                # Filter out None values from generation_config if any tokenizer tokens were None
                generation_config = {k: v for k, v in generation_config.items() if v is not None}

                outputs = model.generate(input_ids,**generation_config)

            # --- 5. Process Output ---
            # Calculate output tokens
            output_token_count = outputs.shape[1] - input_token_count
            if output_token_count < 0: output_token_count = 0 # Handle edge case

            # Decode only the newly generated tokens
            generated_ids = outputs[0, input_token_count:]
            response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            print("Generation complete.")
            return response_text.strip(), input_token_count, output_token_count

        except Exception as e:
            raise RuntimeError(f"Failed during model generation or decoding: {e}") from e
