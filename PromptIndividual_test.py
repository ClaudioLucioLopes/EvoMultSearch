from  PromptIndividual import PromptIndividual,RULE_TO_COMPONENTS,ALL_RULE_IDENTIFIERS,ALL_COMPONENT_NAMES

# --- Example Usage ---
# 1. Define the content and parameters
rule_id_to_use = "zerocot_7"          # Uses req, instr, cot, input
target_model = "meta-llama/Llama-3.2-1B-Instruct"   # The new mandatory attribute

prompt_context = None              # Not active for zerocot_7
prompt_req = "Identify the core sentiment: Positive, Negative, or Neutral."
prompt_instr = "Provide only the sentiment label as your answer."
prompt_examples = None             # Not active for zerocot_7
prompt_cot = "Analyze keywords and overall sentence structure to determine sentiment."
prompt_input = "The weather today is surprisingly pleasant." # Mandatory string

# 2. Create an instance of the updated PromptIndividual
try:
    prompt_instance_v3 = PromptIndividual(
        rule_identifier=rule_id_to_use,
        model_name=target_model,        # Pass the model name
        context=prompt_context,
        req=prompt_req,
        instr=prompt_instr,
        examples=prompt_examples,
        cot=prompt_cot,
        input_content=prompt_input
    )

    # 3. Print the instance to see its representation (including model and rule)
    print("--- PromptIndividual Instance (with Model) ---")
    print(prompt_instance_v3)
    # Expected output might look like:
    # PromptIndividual(Model: Qwen2-7B-Instruct, Rule: zerocot_7, ActiveComponents: [req=Yes, instr=Yes, cot=Yes, input=Yes])
    # (context and examples are inactive for this rule, so not listed in ActiveComponents summary)
    print("-" * 40)

    # 4. Access attributes, including the new model_name
    print("--- Accessing Attributes ---")
    print(f"Target Model: {prompt_instance_v3.model_name}")
    print(f"Request Content: {prompt_instance_v3.req_content}")
    print(f"COT Content: {prompt_instance_v3.cot_content}")
    print(f"Examples Content (should be None): {prompt_instance_v3.examples_content}")
    print("-" * 40)

    # 5. Use helper methods (still work the same way)
    print("--- Using Methods ---")
    print(f"Retrieved Rule ID: {prompt_instance_v3.get_rule_identifier()}")
    print(f"Active Component Attributes: {prompt_instance_v3.get_active_component_attrs()}")
    # Expected: ['req_content', 'instr_content', 'cot_content', 'input_content'] (order may vary)
    print("-" * 40)

except ValueError as e:
    print(f"Error creating PromptIndividual: {e}")

# # --- Example showing model_name validation ---
# print("\n--- Testing model_name validation ---")
# try:
#     invalid_prompt = PromptIndividual(
#         rule_identifier="zero_1",
#         model_name="", # Invalid: empty string
#         context=None, req="Test", instr="Test", examples=None, cot=None,
#         input_content="Valid input"
#     )
# except ValueError as e:
#     print(f"Successfully caught expected error for empty model_name: {e}")

# try:
#     invalid_prompt_2 = PromptIndividual(
#         rule_identifier="zero_1",
#         model_name=None, # Invalid: None
#         context=None, req="Test", instr="Test", examples=None, cot=None,
#         input_content="Valid input"
#     )
# except ValueError as e:
#      # Depending on implementation, might raise ValueError or TypeError
#     print(f"Successfully caught expected error for None model_name: {e}")
# except TypeError as e:
#      print(f"Successfully caught expected error for None model_name: {e}")

# print("-" * 40)




response, in_tokens, out_tokens = prompt_instance_v3.execute_prompt_with_transformer(device="auto",      # Let transformers choose best available device
    max_new_tokens=50, # Limit output length
    # temperature=0.1,    # Example generation parameters
    do_sample=False,
    # top_k=50
)

print("\n--- Hugging Face Response ---")
print(response)
print("---------------------------")
print(f"Token Usage:")
print(f"  Input Tokens:  {in_tokens}")
print(f"  Output Tokens: {out_tokens}")
print("-" * 40)
