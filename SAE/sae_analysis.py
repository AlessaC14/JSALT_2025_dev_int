import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

# --- 1. Configuration and Path Management ---
PROJECT_ROOT = Path("/home/acarbol1/scr4_enalisn1/acarbol1/JSALT_2025/JSALT_2025_dev_int")
SAE_OUTPUT_DIR = PROJECT_ROOT / "saes"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"
TOKENIZER_OUTPUT_DIR = PROJECT_ROOT / "tokenizers"

# Define the rhos used in the experiment to label the plot
RHOS = [0.0, 0.2, 0.5, 0.8, 0.95]
NUM_FEATURES = 64

# --- 2. SAE Model Definition ---
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

# --- 3. Analysis Function ---

def calculate_feature_recovery(sae_state_dict, ground_truth_V, recovery_threshold=0.9):
    """
    Calculates the Feature Recovery Rate (FRR) for a trained SAE.

    Args:
        sae_state_dict (dict): The state dictionary of the trained SAE.
        ground_truth_V (torch.Tensor): The matrix of ground-truth feature embeddings.
        recovery_threshold (float): The cosine similarity threshold to count a feature as "recovered".

    Returns:
        float: The Feature Recovery Rate (FRR) between 0.0 and 1.0.
    """
    # The learned features are the decoder weights (tied to encoder weights)
    learned_features = sae_state_dict['encoder.weight']
    
    # Normalize both sets of vectors for cosine similarity calculation
    gt_features_norm = ground_truth_V / torch.norm(ground_truth_V, dim=1, keepdim=True)
    learned_features_norm = learned_features / torch.norm(learned_features, dim=1, keepdim=True)

    # Calculate the cosine similarity matrix
    # Shape: (num_gt_features, num_learned_features)
    similarity_matrix = torch.matmul(gt_features_norm, learned_features_norm.T)

    # For each ground-truth feature, find its best-matching learned feature
    max_similarity_per_gt_feature, _ = torch.max(torch.abs(similarity_matrix), dim=1)

    # A feature is "recovered" if its best match has a similarity above the threshold
    num_recovered = (max_similarity_per_gt_feature > recovery_threshold).sum().item()
    
    frr = num_recovered / len(ground_truth_V)
    
    print(f"  - Max similarities calculated for {max_similarity_per_gt_feature.shape[0]} ground-truth features.")
    print(f"  - {num_recovered} features recovered (similarity > {recovery_threshold})")
    print(f"  - Feature Recovery Rate (FRR): {frr:.4f}")
    
    return frr

# --- 4. Main Execution and Plotting Logic ---

def run_full_analysis():
    """
    Runs the full analysis pipeline: loads all SAEs, calculates FRR, and plots the results.
    """
    print("=============================================")
    print("      STARTING FINAL ANALYSIS OF SAEs        ")
    print("=============================================")
    
    results = []
    
    for i, rho in enumerate(RHOS):
        sae_path = SAE_OUTPUT_DIR / f"sae_model_{i}.pt"
        model_path = MODEL_OUTPUT_DIR / f"transformer_model_{i}"
        tokenizer_path = TOKENIZER_OUTPUT_DIR / f"tokenizer_{i}" / "tokenizer.json"
        
        if not sae_path.exists() or not model_path.exists() or not tokenizer_path.exists():
            print(f"\n--- Artifacts for Dataset ID: {i} not found. Skipping. ---")
            continue
            
        print(f"\n--- Analyzing SAE for Dataset ID: {i} (ρ₂={rho}) ---")
        
        # Load the trained transformer and tokenizer to get ground-truth embeddings
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
        transformer_model = GPT2LMHeadModel.from_pretrained(str(model_path))
        
        # CORRECTED: Get ground-truth embeddings from the transformer's token embedding layer
        feature_tokens = [str(feat_idx) for feat_idx in range(NUM_FEATURES)]
        feature_token_ids = tokenizer.convert_tokens_to_ids(feature_tokens)
        
        embeddings_matrix = transformer_model.transformer.wte.weight.detach().cpu()
        ground_truth_V = embeddings_matrix[feature_token_ids]
        
        # Load the trained SAE state dict
        input_dim = ground_truth_V.shape[1]  # Should be 256
        feature_dim = input_dim * 8
        temp_sae = SparseAutoencoder(input_dim=input_dim, feature_dim=feature_dim)
        sae_state_dict = torch.load(sae_path, map_location=torch.device('cpu'))
        temp_sae.load_state_dict(sae_state_dict)
        
        # Calculate FRR
        frr = calculate_feature_recovery(sae_state_dict, ground_truth_V)
        results.append({'rho2': rho, 'frr': frr})

    print("\n=============================================")
    print("             ANALYSIS COMPLETE               ")
    print("=============================================")

    # --- Plotting the Results ---
    if not results:
        print("No results to plot.")
        return

    df = pd.DataFrame(results)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.lineplot(x='rho2', y='frr', data=df, marker='o', ax=ax, lw=2.5)
    
    ax.set_title('Feature Recovery Rate vs. Feature Co-occurrence (ρ₂)', fontsize=16, pad=20)
    ax.set_xlabel('Feature Co-occurrence (ρ₂)', fontsize=12)
    ax.set_ylabel('Feature Recovery Rate (FRR)', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    for i, point in df.iterrows():
        ax.text(point['rho2'], point['frr'] + 0.05, f"{point['frr']:.2f}", ha='center')

    plot_path = PROJECT_ROOT / "frr_vs_rho2_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\nFinal plot saved to: {plot_path}")
    print("This plot is the main result for Experiment 1.1.")


if __name__ == "__main__":
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Please install pandas, matplotlib, and seaborn to generate the plot.")
        print("`pip install pandas matplotlib seaborn`")
    else:
        run_full_analysis()
