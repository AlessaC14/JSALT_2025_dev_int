import os
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from pathlib import Path

# --- 1. Configuration and Path Management ---
PROJECT_ROOT = Path("/home/acarbol1/scr4_enalisn1/acarbol1/JSALT_2025/JSALT_2025_dev_int")
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"
TOKENIZER_OUTPUT_DIR = PROJECT_ROOT / "tokenizers" # We still need the path to load the tokenizer file directly

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
print(f"Project Root: {PROJECT_ROOT}")

# --- 2. Validation Function ---

def check_model_generation(dataset_id):
    """
    Performs a generative check on a model trained with the manual loop.
    This is our primary method for validating model convergence.
    """
    model_path = MODEL_OUTPUT_DIR / f"transformer_model_{dataset_id}"
    tokenizer_path = TOKENIZER_OUTPUT_DIR / f"tokenizer_{dataset_id}" / "tokenizer.json"

    if not model_path.exists():
        print(f"  - Generative Check FAILED: Model directory not found.")
        return
    if not tokenizer_path.exists():
        print(f"  - Generative Check FAILED: Tokenizer file not found.")
        return
        
    # Load the tokenizer using the fast implementation
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
    # Manually set special tokens
    tokenizer.add_special_tokens({
        'eos_token': '[EOS]',
        'bos_token': '[BOS]',
        'pad_token': '[PAD]',
        'unk_token': '[UNK]'
    })

    model = GPT2LMHeadModel.from_pretrained(str(model_path))
    model.to(DEVICE)
    model.eval() # Set model to evaluation mode

    prompt_text = "Features: 0"
    
    if dataset_id >= 3:
      print(f"    (Note: This model was trained with high co-occurrence for (0,1). Expecting to see '1' generated.)")

    input_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(DEVICE)

    # Use robust generation settings
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=25,
            num_return_sequences=1,
            do_sample=True,
            top_k=40,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=2
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print(f"  - Generative Check Results:")
    print(f"    Prompt:    '{prompt_text}'")
    print(f"    Generated: '{generated_text}'")
    
    # A simple heuristic for success
    generated_tokens = generated_text.split()
    if "Features:" in generated_tokens and "." in generated_tokens and len(generated_tokens) > 4:
        print("    Verdict:   Looks reasonable. Model appears to have learned the sequence structure.")
    else:
        print("    Verdict:   Looks incorrect. The model has likely failed to learn the data's structure.")


def run_full_validation():
    """
    Runs generative validation checks on all existing trained models.
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