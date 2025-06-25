import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from pathlib import Path
import json
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

# --- 1. Configuration and Path Management ---
PROJECT_ROOT = Path("/home/acarbol1/scr4_enalisn1/acarbol1/JSALT_2025/JSALT_2025_dev_int")
BASE_DATA_DIR = PROJECT_ROOT / "synthetic_data" / "data"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"
TOKENIZER_OUTPUT_DIR = PROJECT_ROOT / "tokenizers"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
print(f"Project Root: {PROJECT_ROOT}")

# --- 2. Custom Tokenizer Creation (Most Robust Method) ---

def create_wordlevel_tokenizer(text_file_path, output_dir):
    """
    Trains a robust WordLevel tokenizer, which is ideal for space-delimited synthetic data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output_dir / "tokenizer.json"

    if tokenizer_path.exists():
        print(f"Tokenizer already exists at {tokenizer_path}. Skipping training.")
        return

    print(f"Training WordLevel tokenizer from {text_file_path}...")
    
    # Initialize a tokenizer with a WordLevel model
    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()

    # Define the trainer
    trainer = WordLevelTrainer(
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
    )

    # Train the tokenizer
    tokenizer.train([str(text_file_path)], trainer)

    # Save the tokenizer
    tokenizer.save(str(tokenizer_path))
    print(f"Custom WordLevel tokenizer saved to {tokenizer_path}")


class CustomTextDataset(Dataset):
    """A simple line-by-line Dataset."""
    def __init__(self, tokenizer, file_path, block_size):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        with open(file_path, encoding="utf-8") as f:
            lines = f.read().splitlines()

        # Tokenize each line and create examples
        self.examples = []
        for line in lines:
            tokenized_line = tokenizer.encode(line).ids
            if len(tokenized_line) > 1: # Ensure non-empty lines
                self.examples.append(tokenized_line)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # Return the tensor for the line, the collate_fn will handle padding
        return torch.tensor(self.examples[i], dtype=torch.long)

def custom_collate_fn(batch, pad_token_id):
    """Pads sentences to the max length in a batch."""
    max_len = max(len(x) for x in batch)
    padded_batch = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    for i, tensor in enumerate(batch):
        padded_batch[i, :len(tensor)] = tensor
    return padded_batch

# --- 3. Manual Transformer Training Loop ---

def train_model_manual(dataset_id):
    """
    Trains a small transformer model from scratch using a manual PyTorch loop.
    """
    # --- Setup Paths ---
    dataset_path = BASE_DATA_DIR / f"dataset_{dataset_id}"
    train_file = dataset_path / "features.txt"
    model_dir = MODEL_OUTPUT_DIR / f"transformer_model_{dataset_id}"
    tokenizer_dir = TOKENIZER_OUTPUT_DIR / f"tokenizer_{dataset_id}"

    if not train_file.exists():
        print(f"Error: Training file not found at {train_file}"); return

    print(f"\n--- Starting Manual Training for Dataset {dataset_id} ---")

    # --- Hyperparameters ---
    NUM_EPOCHS = 5
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4 # Slightly lower learning rate for stability
    BLOCK_SIZE = 32

    # 1. Tokenizer
    create_wordlevel_tokenizer(train_file, tokenizer_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(str(tokenizer_dir))
    
    # Manually set special tokens if not already present
    if tokenizer.eos_token is None:
        tokenizer.add_special_tokens({'eos_token': '[EOS]'})
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({'bos_token': '[BOS]'})
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # 2. Model
    config = GPT2Config(
        vocab_size=len(tokenizer), n_positions=BLOCK_SIZE,
        n_embd=256, n_layer=4, n_head=4,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id
    )
    model = GPT2LMHeadModel(config=config)
    model.resize_token_embeddings(len(tokenizer)) # Resize model embeddings to match tokenizer
    model.to(DEVICE)
    print(f"Model created with {model.num_parameters():,} parameters.")

    # 3. Dataset and DataLoader
    train_dataset = CustomTextDataset(tokenizer, str(train_file), BLOCK_SIZE)
    # Need to wrap collate_fn to include the pad_token_id
    collate_fn_with_pad = lambda batch: custom_collate_fn(batch, tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn_with_pad)
    
    # 4. Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    num_training_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)

    # 5. Training Loop
    model.train()
    print("Starting model training with final manual loop...")
    for epoch in range(NUM_EPOCHS):
        print(f"--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            inputs = batch.to(DEVICE)
            
            optimizer.zero_grad()
            
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping for stability
            optimizer.step()
            scheduler.step()
            
            progress_bar.set_postfix({'loss': loss.item()})
            
    # 6. Save Model and Tokenizer
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print(f"Training complete. Model and tokenizer saved to {model_dir}")
    print(f"--- Finished Training for Dataset {dataset_id} ---")

if __name__ == "__main__":
    # For now, let's just train the first model with the final settings.
    train_model_manual(dataset_id=0)