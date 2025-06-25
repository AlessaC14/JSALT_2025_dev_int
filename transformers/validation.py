import os
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pathlib import Path
import json

# --- 1. Configuration and Path Management ---
# Ensure this matches the root directory used in your training script.
PROJECT_ROOT = Path("/home/acarbol1/scr4_enalisn1/acarbol1/JSALT_2025/JSALT_2025_dev_int")
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
print(f"Project Root: {PROJECT_ROOT}")

# --- 2. Validation Function ---

def check_model_generation(dataset_id):
    """
    Performs a generative check to see if the model learned the basic patterns
    of its training data. This is our primary validation method.
    """
    model_path = MODEL_OUTPUT_DIR / f"transformer_model_{dataset_id}"
    if not model_path.exists():
        print(f"  - Generative Check FAILED: Model directory not found.")
        return
        
    # Load the model and its custom tokenizer from the same directory
    tokenizer = GPT2Tokenizer.from_pretrained(str(model_path))
    model = GPT2LMHeadModel.from_pretrained(str(model_path))
    model.to(DEVICE)

    prompt_text = "Features: 0"
    
    # Provide context for what to expect from the entangled models
    if dataset_id >= 3: # For high rho values (e.g., 0.8, 0.95)
      print(f"    (Note: This model was trained with high co-occurrence for (0,1). Expecting to see '1' generated.)")

    input_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(DEVICE)

    # Use more robust generation settings to avoid loops and gibberish
    output = model.generate(
        input_ids,
        max_length=25,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,      # Enable sampling
        top_k=50,            # Consider only the top 50 tokens
        top_p=0.95,          # Use nucleus sampling
        no_repeat_ngram_size=2 # Prevent repeating pairs of tokens
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(f"  - Generative Check Results:")
    print(f"    Prompt:    '{prompt_text}'")
    print(f"    Generated: '{generated_text}'")
    
    # A more robust heuristic for success
    generated_tokens = generated_text.split()
    if "Features:" in generated_tokens and "." in generated_tokens and len(generated_tokens) > 4:
        print("    Verdict:   Looks reasonable. Model appears to have learned the sequence structure.")
    else:
        print("    Verdict:   Looks incorrect. The model has likely failed to learn the data's structure.")


def run_full_validation():
    """
    Runs all validation checks on all existing trained models.
    """
    print("=============================================")
    print("      STARTING VALIDATION OF ALL MODELS      ")
    print("=============================================")
    
    # This list should match the rhos used during data generation
    rhos = [0.0, 0.2, 0.5, 0.8, 0.95]
    
    for i in range(5):
        model_path = MODEL_OUTPUT_DIR / f"transformer_model_{i}"
        if model_path.exists():
            print(f"\n--- Validating Model for Dataset ID: {i} (ρ₂={rhos[i]}) ---")
            check_model_generation(dataset_id=i)
        else:
            print(f"\n--- Model for Dataset ID: {i} not found. Skipping. ---")
            
    print("\n=============================================")
    print("             VALIDATION COMPLETE             ")
    print("=============================================")


if __name__ == "__main__":
    run_full_validation()