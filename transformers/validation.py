import torch
import torch.nn as nn
from pathlib import Path
import numpy as np

# ===================================================================
# --- 1. CONFIGURATION ---
# ===================================================================
CONFIG = {
    "PROJECT_ROOT": Path("/home/acarbol1/scratchenalisn1/acarbol1/JSALT_2025/JSALT_2025_dev_int"),
    "NUMERICAL_MODEL_DIMS": {"input_dim": 64, "hidden_dim": 48},
    "EXPERIMENT_PARAMS": {
        "dataset_ids": range(5),
        "rho_values": [0.0, 0.2, 0.5, 0.8, 0.95],
    }
}
# --- Paths ---
MODEL_DIR = CONFIG["PROJECT_ROOT"] / "models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ===================================================================
# --- 2. MODEL DEFINITION ---
# ===================================================================
class NumericalAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.decoder = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        hidden_activations = self.relu(self.encoder(x))
        reconstructed_x = self.decoder(hidden_activations)
        return reconstructed_x

# ===================================================================
# --- 3. VALIDATION LOGIC ---
# ===================================================================
def validate_model(model, feature_index_to_test):
    """
    Tests a model's reconstruction of a single, pure feature vector.
    """
    model.to(DEVICE).eval()
    
    # Create a pure, one-hot feature vector
    input_dim = model.encoder.in_features
    input_vector = torch.zeros(1, input_dim, device=DEVICE)
    input_vector[0, feature_index_to_test] = 1.0

    with torch.no_grad():
        reconstructed_vector = model(input_vector)

    # Calculate reconstruction quality
    mse = nn.functional.mse_loss(reconstructed_vector, input_vector).item()
    
    # Calculate cosine similarity
    cos_sim = nn.functional.cosine_similarity(reconstructed_vector, input_vector, dim=1).item()
    
    return reconstructed_vector.cpu().numpy().flatten(), mse, cos_sim

# ===================================================================
# --- 4. MAIN EXECUTION ---
# ===================================================================
def main():
    print("=============================================")
    print("  VALIDATING NUMERICAL AUTOENCODER MODELS    ")
    print("=============================================")

    for dataset_id, rho in zip(CONFIG['EXPERIMENT_PARAMS']['dataset_ids'], CONFIG['EXPERIMENT_PARAMS']['rho_values']):
        print(f"\n--- Validating Model for Dataset ID: {dataset_id} (ρ₂ = {rho}) ---")
        
        model_path = MODEL_DIR / f"numerical_model_{dataset_id}.pt"
        if not model_path.exists():
            print(f"  Model not found at {model_path}. Skipping.")
            continue
            
        model = NumericalAutoencoder(
            input_dim=CONFIG['NUMERICAL_MODEL_DIMS']['input_dim'],
            hidden_dim=CONFIG['NUMERICAL_MODEL_DIMS']['hidden_dim']
        )
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))

        # --- Test 1: Reconstruction of a control feature (e.g., feature 10) ---
        _, mse_10, cos_sim_10 = validate_model(model, 10)
        print(f"  Control Test (Feature 10):")
        print(f"    - Reconstruction MSE: {mse_10:.6f}")
        print(f"    - Cosine Similarity:  {cos_sim_10:.4f}")

        # --- Test 2: The CRITICAL test for co-occurrence ---
        # We give the model feature 0 and see if it hallucinates feature 1.
        reconstruction_of_0, _, _ = validate_model(model, 0)
        
        # Get the magnitude of the reconstructed feature 1 component
        feature_1_magnitude = reconstruction_of_0[1]
        
        print(f"  Co-occurrence Test (Input=v₀):")
        print(f"    - Reconstructed magnitude of v₁ component: {feature_1_magnitude:.4f}")
        
        if rho > 0.5 and feature_1_magnitude > 0.1:
            print("    - ✅ PASS: Model appears to have learned the co-occurrence.")
        elif rho < 0.1 and abs(feature_1_magnitude) < 0.05:
            print("    - ✅ PASS: Model correctly does not associate feature 1 with feature 0.")
        else:
            print("    - ⚠️  WARN: The co-occurrence behavior is not as expected.")
            
    print("\n=============================================")
    print("           VALIDATION COMPLETE               ")
    print("=============================================")


if __name__ == "__main__":
    main()