import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from pathlib import Path
from tqdm import tqdm

# --- 1. Configuration and Paths ---
PROJECT_ROOT = Path("/home/acarbol1/scratchenalisn1/acarbol1/JSALT_2025/JSALT_2025_dev_int")
BASE_DATA_DIR = PROJECT_ROOT / "synthetic_data" / "data"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models" # We can reuse the same directory
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- 2. Model Definition: A Simple Autoencoder ---
class NumericalAutoencoder(nn.Module):
    """
    This model learns to represent the numerical vectors from X.pt.
    Its goal is to reconstruct the input vector after passing it through
    a smaller hidden layer (a bottleneck).
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # The encoder compresses the input vector into a hidden representation.
        # The activations from this layer are what we'll analyze with the SAE.
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        # The decoder tries to reconstruct the original vector from the hidden representation.
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # The activations from the encoder's hidden layer are what we care about.
        hidden_activations = self.relu(self.encoder(x))
        reconstructed_x = self.decoder(hidden_activations)
        return reconstructed_x

# --- 3. Dataset for Numerical Tensors ---
class NumericalDataset(Dataset):
    """A simple Dataset that loads the X.pt tensor directly."""
    def __init__(self, x_tensor_path):
        self.data = torch.load(x_tensor_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# --- 4. Main Training Function ---
def train_numerical_model(dataset_id):
    print(f"\n=== Training Numerical Autoencoder on Dataset {dataset_id} ===")

    # Hyperparameters
    NUM_EPOCHS = 50 # Training is often faster for these simpler models
    BATCH_SIZE = 256
    LEARNING_RATE = 5e-4
    INPUT_DIM = 64   # Dimension of your V and X matrices
    HIDDEN_DIM = 48  # A bottleneck to encourage learning efficient representations

    # Setup paths
    data_dir = BASE_DATA_DIR / f"dataset_{dataset_id}"
    x_tensor_path = data_dir / "X.pt"
    model_save_path = MODEL_OUTPUT_DIR / f"numerical_model_{dataset_id}.pt"
    
    if not x_tensor_path.exists():
        raise FileNotFoundError(f"X.pt not found for dataset {dataset_id} at {x_tensor_path}")

    # Dataset and DataLoader
    dataset = NumericalDataset(x_tensor_path)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model, Loss, and Optimizer
    model = NumericalAutoencoder(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM).to(DEVICE)
    loss_function = nn.MSELoss() # Mean Squared Error is perfect for reconstruction
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    model.train()
    for epoch in range(1, NUM_EPOCHS + 1):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch}"):
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            
            reconstructed_batch = model(batch)
            loss = loss_function(reconstructed_batch, batch)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch} completed. Average reconstruction loss: {avg_loss:.6f}")

    # Save the trained model
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved numerical model to {model_save_path}\n")

# --- 5. Execution ---
if __name__ == "__main__":
    for ds_id in range(5):
        train_numerical_model(ds_id)