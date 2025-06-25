import torch
import os
import random

def generate_sparse_indices(num_points, dim, sparsity):
  
    #Returns a LongTensor of shape (num_points, sparsity)
    
  
    return torch.stack([
        torch.randperm(dim)[:sparsity]
        for _ in range(num_points)
    ], dim=0)


def generate_H_indices_with_cooccurrence(num_points, dim, sparsity, rho2, co_pairs):
    """
    Start with random sparse indices, then for each (a,b) pair:
    if a is in the row and coin < rho2, force b in the row
    by swapping out another index.
    """
    H_idx = generate_sparse_indices(num_points, dim, sparsity).tolist()

    for i, row in enumerate(H_idx):
        for (a, b) in co_pairs:
            if a in row and random.random() < rho2 and b not in row:
                # remove a random non-a index to keep sparsity constant
                to_remove = random.choice([x for x in row if x != a])
                row[row.index(to_remove)] = b
        H_idx[i] = sorted(row)

    return torch.tensor(H_idx, dtype=torch.long)


def build_H_from_indices(indices, dim, value_dist="normal"):
    """
    Given indices of shape (N, s), build a dense H of shape (N, dim)
    with values drawn from `value_dist` at those positions.
    """
    N, s = indices.shape
    H = torch.zeros(N, dim)

    if value_dist == "normal":
        values = torch.randn(N, s)
    elif value_dist == "uniform":
        values = torch.rand(N, s)
    elif value_dist == "ones":
        values = torch.ones(N, s)
    else:
        raise ValueError(f"Unknown dist {value_dist!r}")

    rows = torch.arange(N).unsqueeze(1).expand(-1, s)
    H[rows, indices] = values
    return H


def write_feature_sentences(indices, out_txt):
    #one line per row
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    with open(out_txt, "w") as f:
        for row in indices.tolist():
            line = "Features: " + " ".join(str(i) for i in row) + " .\n"
            f.write(line)


def generate_dataset(
    dataset_id,
    num_points=100_000,
    dim=64,
    sparsity=4,
    rho2=0.0,
    co_pairs=None,
    value_dist="normal",
    out_dir="./data"
):
    co_pairs = co_pairs or []
    base_dir = os.path.join(out_dir, f"dataset_{dataset_id}")
    os.makedirs(base_dir, exist_ok=True)

    # 1) Generate sparse indices with co-occurrence
    H_idx = generate_H_indices_with_cooccurrence(
        num_points, dim, sparsity, rho2, co_pairs
    )

    # 2) Build H and X = H @ V  (with V = identity)
    H = build_H_from_indices(H_idx, dim, value_dist)
    V = torch.eye(dim)
    X = H @ V

    # 3) Save tensors
    torch.save(H, os.path.join(base_dir, "H.pt"))
    torch.save(X, os.path.join(base_dir, "X.pt"))

    # 4) Dump feature sentences
    txt_path = os.path.join(base_dir, "features.txt")
    write_feature_sentences(H_idx, txt_path)

    print(f"Dataset {dataset_id} → saved in {base_dir}")


if __name__ == "__main__":
    # Define which feature-pairs to correlate and the sweep of ρ₂ values
    co_pairs = [(0, 1)]        
    rhos = [0.0, 0.2, 0.5, 0.8, 0.95]

    for i, rho in enumerate(rhos):
        generate_dataset(
            dataset_id=i,
            num_points=100_000,
            dim=64,
            sparsity=4,
            rho2=rho,
            co_pairs=co_pairs,
            value_dist="normal",
            out_dir="./data"
        )
