import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from pathlib import Path
from tqdm import tqdm
import json

# --- 1. Configuration and Path Management ---
PROJECT_ROOT = Path("/home/acarbol1/scr4_enalisn1/acarbol1/JSALT_2025/JSALT_2025_dev_int")
BASE_DATA_DIR = PROJECT_ROOT / "synthetic_data" / "data"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"
TOKENIZER_OUTPUT_DIR = PROJECT_ROOT / "tokenizers"
SAE_OUTPUT_DIR = PROJECT_ROOT / "saes"
ACTIVATION_DIR = PROJECT_ROOT / "activations"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
print(f"Project Root: {PROJECT_ROOT}")

# --- 2. Activation Extraction ---

class CustomTextDataset(Dataset):
    """A simple line-by-line Dataset that treats each line as an example."""
    def __init__(self, tokenizer, file_path):
        self.examples = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                ids = tokenizer.encode(f"[BOS] {line} [EOS]")
                if ids: self.examples.append(torch.tensor(ids, dtype=torch.long))
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]

def custom_collate_fn(batch, pad_token_id):
    """Pads sentences to the max length in a batch."""
    max_len = max(len(seq) for seq in batch)
    padded_batch = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded_batch[i, :len(seq)] = seq
    return padded_batch

def get_activations(model, tokenizer, text_file_path, layer_to_hook, batch_size=64):
    """
    Extracts hidden state activations from a specific layer of a transformer model.
    """
    model.to(DEVICE)
    model.eval()

    activations = []
    
    # Hook function to capture the input to the specified layer's norm
    def hook_fn(module, input, output):
        activations.append(input[0].detach().cpu())

    # Register the hook on the specified layer
    hook = model.transformer.h[layer_to_hook].ln_1.register_forward_hook(hook_fn)

    # Prepare dataset and loader
    dataset = CustomTextDataset(tokenizer, str(text_file_path))
    collate_fn = lambda batch: custom_collate_fn(batch, tokenizer.pad_token_id)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

    print(f"Extracting activations from layer {layer_to_hook}...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting Activations"):
            inputs = batch.to(DEVICE)
            model(inputs)

    hook.remove()
    
    # Concatenate all activations and flatten
    all_activations = torch.cat(activations, dim=0)
    return all_activations.view(-1, all_activations.size(-1))


# --- 3. Sparse Autoencoder and GBA Trainer ---

class SparseAutoencoder(nn.Module):
    """A standard SAE with a tied decoder."""
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, feature_dim)
        self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        self.relu = nn.ReLU()

    def forward(self, x):
        f = self.relu(self.encoder(x))
        reconstructed_x = torch.matmul(f, self.encoder.weight) + self.decoder_bias
        return reconstructed_x, f

class GBATrainer:
    """Implements the Group Bias Adaptation (GBA) training logic."""
    def __init__(self, sae_model, num_groups=10, buffer_size=40960):
        self.model = sae_model.to(DEVICE)
        self.num_groups = num_groups
        self.buffer_size = buffer_size
        
        # Setup neuron groups and Target Activation Frequencies (TAFs)
        self.groups = torch.tensor_split(torch.randperm(sae_model.encoder.out_features), num_groups)
        self.tafs = torch.logspace(-1, -3, steps=num_groups)
        self.pre_act_buffer = []

    def train(self, activations, num_epochs=3, lr=1e-4, batch_size=512):
        weight_params = [p for name, p in self.model.named_parameters() if 'bias' not in name]
        optimizer = optim.Adam(weight_params, lr=lr)
        
        loader = DataLoader(TensorDataset(activations), batch_size=batch_size, shuffle=True)

        for epoch in range(num_epochs):
            print(f"\n--- SAE Training Epoch {epoch+1}/{num_epochs} ---")
            for i, (batch,) in enumerate(tqdm(loader, desc=f"SAE Epoch {epoch+1}")):
                x = batch.to(DEVICE)
                
                optimizer.zero_grad()
                reconstructed_x, features = self.model(x)
                recon_loss = torch.pow(reconstructed_x - x, 2).mean()
                recon_loss.backward()
                optimizer.step()

                with torch.no_grad():
                    pre_activations = self.model.encoder(x)
                    self.pre_act_buffer.append(pre_activations.cpu())
                
                if sum(b.shape[0] for b in self.pre_act_buffer) >= self.buffer_size:
                    self._update_biases()
                    self.pre_act_buffer = []

    def _update_biases(self, gamma_minus=0.01, gamma_plus=0.01, epsilon=1e-6):
        print("\n--- Updating biases with GBA ---")
        all_pre_acts = torch.cat(self.pre_act_buffer, dim=0)
        
        with torch.no_grad():
            biases = self.model.encoder.bias.data
            r_max_groups = []
            for group_indices in self.groups:
                group_pre_acts = all_pre_acts[:, group_indices]
                max_per_neuron = torch.max(group_pre_acts, dim=0).values
                active_maxes = max_per_neuron[max_per_neuron > 0]
                avg_r_max = torch.mean(active_maxes) if len(active_maxes) > 0 else 0.0
                r_max_groups.append(avg_r_max)

            for group_idx, neuron_indices in enumerate(self.groups):
                group_taf = self.tafs[group_idx]
                avg_max_pre_act_group = r_max_groups[group_idx]
                
                for neuron_idx in neuron_indices:
                    neuron_pre_acts = all_pre_acts[:, neuron_idx]
                    p_hat = (neuron_pre_acts > 0).float().mean()
                    r_m = torch.max(torch.tensor(0.0), torch.max(neuron_pre_acts))

                    if p_hat > group_taf:
                        biases[neuron_idx] = torch.max(biases[neuron_idx] - gamma_minus * r_m, torch.tensor(-1.0))
                    elif p_hat < epsilon:
                        biases[neuron_idx] = torch.min(biases[neuron_idx] + gamma_plus * avg_max_pre_act_group, torch.tensor(0.0))

            self.model.encoder.bias.data.copy_(biases)

# --- 4. Main Execution Logic ---

def analyze_model_with_sae(dataset_id):
    """Main function to run the full analysis pipeline for one dataset."""
    print(f"\n===== Starting SAE Analysis for Dataset ID: {dataset_id} =====")
    
    # --- Load Transformer and Tokenizer ---
    model_path = MODEL_OUTPUT_DIR / f"transformer_model_{dataset_id}"
    tokenizer_path = TOKENIZER_OUTPUT_DIR / f"tokenizer_{dataset_id}" / "tokenizer.json"
    
    if not model_path.exists() or not tokenizer_path.exists():
        print(f"Model or tokenizer not found for dataset {dataset_id}. Skipping.")
        return
        
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path), pad_token='[PAD]')
    model = GPT2LMHeadModel.from_pretrained(str(model_path))
    print("Trained transformer model and tokenizer loaded.")
    
    # --- Get or Load Activations ---
    activation_file = ACTIVATION_DIR / f"activations_{dataset_id}.pt"
    ACTIVATION_DIR.mkdir(exist_ok=True)
    if activation_file.exists():
        print(f"Loading activations from {activation_file}...")
        activations = torch.load(activation_file)
    else:
        text_file = BASE_DATA_DIR / f"dataset_{dataset_id}" / "features.txt"
        # Hook into the middle layer (e.g., layer 2 of 4)
        activations = get_activations(model, tokenizer, text_file, layer_to_hook=2)
        print(f"Saving activations to {activation_file}...")
        torch.save(activations, activation_file)
    print(f"Activations ready: {activations.shape}")
    
    # --- Train the SAE with GBA ---
    input_dim = model.config.n_embd
    # Dictionary expansion factor: 8x
    feature_dim = input_dim * 8 
    sae = SparseAutoencoder(input_dim, feature_dim)
    
    trainer = GBATrainer(sae)
    trainer.train(activations) 
    
    # --- Save the trained SAE ---
    sae_path = SAE_OUTPUT_DIR / f"sae_model_{dataset_id}.pt"
    SAE_OUTPUT_DIR.mkdir(exist_ok=True)
    torch.save(sae.state_dict(), sae_path)
    print(f"Trained SAE saved to {sae_path}")
    print(f"===== Finished SAE Analysis for Dataset ID: {dataset_id} =====")

if __name__ == "__main__":
    # You can loop this call to analyze all 5 models.
    # It is recommended to run them one at a time.
    for i in range(5):
         analyze_model_with_sae(dataset_id=i)
    













        



            




















