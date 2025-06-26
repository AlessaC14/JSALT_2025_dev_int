import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    get_linear_schedule_with_warmup
)
from pathlib import Path
from tqdm import tqdm
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

# --- 1. Configuration and Paths ---
PROJECT_ROOT = Path("/home/acarbol1/scr4_enalisn1/acarbol1/JSALT_2025/JSALT_2025_dev_int")
BASE_DATA_DIR = PROJECT_ROOT / "synthetic_data" / "data"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"
TOKENIZER_OUTPUT_DIR = PROJECT_ROOT / "tokenizers"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
print(f"Project root: {PROJECT_ROOT}")

# --- 2. Tokenizer training ---
def create_wordlevel_tokenizer(text_file, output_dir):
    """
    Train a WordLevel tokenizer on the given text file,
    saving tokenizer.json into output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tok_path = output_dir / "tokenizer.json"
    if tok_path.exists():
        print(f"Tokenizer already exists at {tok_path}, skipping training.")
        return
    print(f"Training WordLevel tokenizer on {text_file}...")
    tok = Tokenizer(WordLevel(unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[BOS]","[EOS]"])
    tok.train([str(text_file)], trainer)
    tok.save(str(tok_path))
    print(f"Saved tokenizer to {tok_path}")

# --- 3. Custom Dataset ---
class CustomTextDataset(Dataset):
    """Line-by-line dataset returning token ID tensors."""
    def __init__(self, tokenizer, file_path):
        self.examples = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids = tokenizer.encode(line)
                if ids:
                    self.examples.append(torch.tensor(ids, dtype=torch.long))
    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]

# --- 4. Collate function ---
def custom_collate(batch, pad_token_id):
    """Pad batch of variable-length sequences to same length."""
    max_len = max(seq.size(0) for seq in batch)
    out = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    for i, seq in enumerate(batch):
        out[i, :seq.size(0)] = seq
    return out

# --- 5. Manual training ---
def train_model_manual(dataset_id):
    # Paths for data, tokenizer, model
    data_dir = BASE_DATA_DIR / f"dataset_{dataset_id}"
    train_txt = data_dir / "features.txt"
    tok_dir = TOKENIZER_OUTPUT_DIR / f"tokenizer_{dataset_id}"
    model_dir = MODEL_OUTPUT_DIR / f"transformer_model_{dataset_id}"

    if not train_txt.exists():
        raise FileNotFoundError(f"Training file not found: {train_txt}")

    print(f"\n=== Training Transformer on Dataset {dataset_id} ===")
    print(f"Reading data from: {train_txt}")

    # Tokenizer
    create_wordlevel_tokenizer(train_txt, tok_dir)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tok_dir / "tokenizer.json"),
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]"
    )
    # Add special tokens to vocab if missing
    tokenizer.add_special_tokens({
        'bos_token': '[BOS]',
        'eos_token': '[EOS]',
        'pad_token': '[PAD]',
        'unk_token': '[UNK]'
    })

    # Hyperparameters
    NUM_EPOCHS = 5
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    BLOCK_SIZE = 32

    # Model config and instantiation
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=BLOCK_SIZE,
        n_embd=256,
        n_layer=4,
        n_head=4,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    model = GPT2LMHeadModel(config).to(DEVICE)
    model.resize_token_embeddings(len(tokenizer))
    print(f"Model parameters: {model.num_parameters():,}")

    # Dataset & DataLoader
    dataset = CustomTextDataset(tokenizer, str(train_txt))
    collate_fn = lambda batch: custom_collate(batch, tokenizer.pad_token_id)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    print(f"Dataset size: {len(dataset)} samples. Batching size: {BATCH_SIZE}.")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)

    # Training loop
    model.train()
    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"--- Epoch {epoch}/{NUM_EPOCHS} ---")
        for batch in tqdm(loader, desc=f"Epoch {epoch}"):
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        print(f"Epoch {epoch} completed.")

    # Save artifacts
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    print(f"Saved model and tokenizer to {model_dir}\n")

if __name__ == "__main__":
    # Loop or single dataset
    for ds_id in range(1):  # modify range for multiple
        train_model_manual(ds_id)

