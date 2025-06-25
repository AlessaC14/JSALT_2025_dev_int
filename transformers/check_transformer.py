import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def check_model_generation(dataset_id, model_input_dir="/home/acarbol1/scr4_enalisn1/acarbol1/JSALT_2025/JSALT_2025_dev_int/models"):
    model_path = f"{model_input_dir}/transformer_model_{dataset_id}"
    print(f"\n--- Checking Model for Dataset ID: {dataset_id} ---")

    # Load the specific model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    # Craft a prompt. For dataset_id=4 (rho2=0.95), we know feature 0 and 1 co-occur.
    # So let's test that.
    prompt_text = "Features: 0"
    if dataset_id == 4:
      print(f"Testing with co-occurrence: giving it feature 0, expecting it to generate 1...")

    input_ids = tokenizer.encode(prompt_text, return_tensors='pt')

    # Generate text
    output = model.generate(
        input_ids,
        max_length=15,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id # Set pad_token_id to eos_token_id
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"Prompt: '{prompt_text}'")
    print(f"Generated: '{generated_text}'")

# --- Run the checks ---
# Check the model trained on clean data (rho2=0.0)
check_model_generation(dataset_id=0)

# Check the model trained on highly entangled data (rho2=0.95)
check_model_generation(dataset_id=4)