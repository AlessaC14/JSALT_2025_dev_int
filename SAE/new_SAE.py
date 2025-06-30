import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

# --- 1. Configuration and Path Management ---
PROJECT_ROOT = Path("/home/acarbol1/scr4_enalisn1/acarbol1/JSALT_2025/JSALT_2025_dev_int")
SAE_OUTPUT_DIR = PROJECT_ROOT / "saes"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "models"
TOKENIZER_OUTPUT_DIR = PROJECT_ROOT / "tokenizers"
PLOT_OUTPUT_DIR = PROJECT_ROOT / "plots"
PLOT_OUTPUT_DIR.mkdir(exist_ok=True)

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

# --- 3. Diagnostic Analysis Function ---

def calculate_feature_recovery_diagnostic(sae_state_dict, ground_truth_V, recovery_threshold=0.9):
    """
    Calculates FRR and provides detailed diagnostics.
    """
    learned_features = sae_state_dict['encoder.weight']
    
    print(f"  - Shape of Ground-Truth Feature Embeddings (V): {ground_truth_V.shape}")
    print(f"  - Shape of Learned SAE Decoder Weights (W_dec): {learned_features.shape}")

    if ground_truth_V.shape[1] != learned_features.shape[1]:
        print("  - FATAL ERROR: Dimension mismatch between ground-truth and learned features.")
        return 0.0, None

    gt_features_norm = ground_truth_V / torch.norm(ground_truth_V, dim=1, keepdim=True)
    learned_features_norm = learned_features / torch.norm(learned_features, dim=1, keepdim=True)

    similarity_matrix = torch.matmul(gt_features_norm, learned_features_norm.T)
    max_similarity_per_gt_feature, _ = torch.max(torch.abs(similarity_matrix), dim=1)

    # --- New Diagnostic Information ---
    top_5_sims, _ = torch.topk(max_similarity_per_gt_feature, 5)
    print(f"  - Top 5 Max Cosine Similarities: {[f'{s:.3f}' for s in top_5_sims]}")
    
    num_recovered = (max_similarity_per_gt_feature > recovery_threshold).sum().item()
    frr = num_recovered / len(ground_truth_V)
    
    print(f"  - {num_recovered} features recovered (similarity > {recovery_threshold})")
    print(f"  - Feature Recovery Rate (FRR): {frr:.4f}")
    
    return frr, max_similarity_per_gt_feature.numpy()

# --- 4. Main Execution and Plotting Logic ---

def run_full_analysis():
    """
    Runs the full analysis pipeline and generates diagnostic plots.
    """
    print("=============================================")
    print("    STARTING DIAGNOSTIC ANALYSIS OF SAEs     ")
    print("=============================================")
    
    results = []
    
    for i, rho in enumerate(RHOS):
        sae_path = SAE_OUTPUT_DIR / f"sae_model_{i}.pt"
        model_path = MODEL_OUTPUT_DIR / f"transformer_model_{i}"
        tokenizer_path = TOKENIZER_OUTPUT_DIR / f"tokenizer_{i}" / "tokenizer.json"
        
        if not all([sae_path.exists(), model_path.exists(), tokenizer_path.exists()]):
            print(f"\n--- Artifacts for Dataset ID: {i} not found. Skipping. ---")
            continue
            
        print(f"\n--- Analyzing SAE for Dataset ID: {i} (ρ₂={rho}) ---")
        
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_path))
        transformer_model = GPT2LMHeadModel.from_pretrained(str(model_path))
        
        feature_tokens = [str(feat_idx) for feat_idx in range(NUM_FEATURES)]
        feature_token_ids = tokenizer.convert_tokens_to_ids(feature_tokens)
        
        embeddings_matrix = transformer_model.transformer.wte.weight.detach().cpu()
        ground_truth_V = embeddings_matrix[feature_token_ids]
        
        temp_sae = SparseAutoencoder(input_dim=256, feature_dim=256*8)
        sae_state_dict = torch.load(sae_path, map_location=torch.device('cpu'))
        temp_sae.load_state_dict(sae_state_dict)
        
        frr, sims = calculate_feature_recovery_diagnostic(sae_state_dict, ground_truth_V)
        results.append({'rho2': rho, 'frr': frr, 'similarities': sims})

    print("\n=============================================")
    print("             ANALYSIS COMPLETE               ")
    print("=============================================")

    if not results:
        print("No results to plot.")
        return

    # --- Plotting the Main FRR Curve ---
    df = pd.DataFrame(results)
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(x='rho2', y='frr', data=df, marker='o', ax=ax, lw=2.5)
    ax.set_title('Feature Recovery Rate vs. Feature Co-occurrence (ρ₂)', fontsize=16, pad=20)
    ax.set_xlabel('Feature Co-occurrence (ρ₂)', fontsize=12)
    ax.set_ylabel('Feature Recovery Rate (FRR)', fontsize=12)
    ax.set_ylim(-0.05, 1.1)
    plot_path = PLOT_OUTPUT_DIR / "frr_vs_rho2_plot.png"
    plt.savefig(plot_path, dpi=300)
    print(f"\nMain results plot saved to: {plot_path}")

    # --- Plotting Diagnostic Histograms ---
    num_results = len(results)
    if num_results > 0:
        fig, axes = plt.subplots(1, num_results, figsize=(5 * num_results, 5), sharey=True)
        fig.suptitle('Distribution of Max Cosine Similarities for Each Model', fontsize=18)
        for idx, res in enumerate(results):
            ax = axes[idx] if num_results > 1 else axes
            sns.histplot(res['similarities'], bins=20, ax=ax, kde=True)
            ax.set_title(f"Dataset {idx} (ρ₂={res['rho2']})")
            ax.set_xlabel("Max Cosine Similarity")
            ax.set_xlim(0, 1)
        
        hist_path = PLOT_OUTPUT_DIR / "similarity_histograms.png"
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(hist_path, dpi=300)
        print(f"Diagnostic histograms saved to: {hist_path}")


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