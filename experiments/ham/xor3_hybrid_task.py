import os, sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
from mixed_order.plotting.style import apply_pub_style, get_color_palette
from mixed_order.metrics import seed_all, _DEVICE

RESULT_DIR = os.path.dirname(__file__)
apply_pub_style()

def generate_xor3_task(K, N_proto, T, seed=42):
    """
    Generate a synthetic XOR-3 task.
    State x = [prototype_block, parity_block]
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 1. Prototype block (ordinary bits)
    # Each class has a unique prototype
    prototypes = torch.randint(0, 2, (K, N_proto), device=_DEVICE).float() * 2 - 1
    
    # 2. Parity block (disjoint triples)
    # class_parities[k, t] is the required product of triple t for class k
    class_parities = torch.randint(0, 2, (K, T), device=_DEVICE).float() * 2 - 1
    
    return prototypes, class_parities

def sample_query(class_idx, prototypes, class_parities, N_proto, T, noise_level):
    """
    Sample a state from class_idx and add noise.
    """
    K, _ = prototypes.shape
    device = prototypes.device
    
    # Prototype part
    x_proto = prototypes[class_idx].clone()
    
    # Parity part: for each triple, sample 3 bits whose product is class_parities[class_idx, t]
    x_parity = torch.empty(3 * T, device=device)
    for t in range(T):
        target_prod = class_parities[class_idx, t].item()
        # Sample first two bits randomly
        b12 = torch.randint(0, 2, (2,), device=device).float() * 2 - 1
        # Third bit is determined by target product
        b3 = target_prod / (b12[0] * b12[1])
        bits = torch.cat([b12, torch.tensor([b3], device=device)])
        # Permute triple bits to avoid any bit-position bias
        x_parity[3*t : 3*t+3] = bits[torch.randperm(3)]
        
    x = torch.cat([x_proto, x_parity])
    
    # Add bit-flip noise
    mask = (torch.rand(x.shape, device=device) < noise_level)
    x[mask] *= -1
    
    return x

def evaluate_models(prototypes, class_parities, N_proto, T, noise_levels, n_queries=100):
    """
    Compare accuracy of three score families.
    """
    K = prototypes.shape[0]
    results = {'pairwise': [], 'cubic': [], 'mixed': []}
    
    # Define triple indices for cubic score
    tri_i = torch.arange(N_proto, N_proto + 3*T, 3, device=_DEVICE)
    tri_j = tri_i + 1
    tri_k = tri_i + 2

    for noise in tqdm(noise_levels, desc="  Noise Sweep", leave=False):
        acc_p, acc_c, acc_m = 0, 0, 0
        
        for _ in range(n_queries):
            c_true = np.random.randint(K)
            x = sample_query(c_true, prototypes, class_parities, N_proto, T, noise)
            
            # 1. Pairwise-only score: dot product on prototypes
            s_proto = torch.matmul(prototypes, x[:N_proto]) / N_proto
            
            # 2. Cubic-only score: sum of triple products matching class parities
            triple_prods = x[tri_i] * x[tri_j] * x[tri_k]
            s_parity = torch.matmul(class_parities, triple_prods) / T
            
            # Predictions
            pred_p = torch.argmax(s_proto).item()
            pred_c = torch.argmax(s_parity).item()
            pred_m = torch.argmax(s_proto + s_parity).item()
            
            acc_p += (pred_p == c_true)
            acc_c += (pred_c == c_true)
            acc_m += (pred_m == c_true)
            
        results['pairwise'].append(acc_p / n_queries)
        results['cubic'].append(acc_c / n_queries)
        results['mixed'].append(acc_m / n_queries)
        
    return results

def xor3_hybrid_experiment():
    # Parameters from Implementation Plan Step 8.8
    K = 16
    N_proto = 32
    T = 16
    n_instances = 20 # Slightly reduced for speed
    noise_levels = np.linspace(0.0, 0.3, 9)
    
    print(f"Running XOR-3 Hybrid Task: K={K}, N_proto={N_proto}, T={T}")
    
    all_inst_results = []
    
    for inst in tqdm(range(n_instances), desc="  Instances"):
        prototypes, class_parities = generate_xor3_task(K, N_proto, T, seed=inst*100)
        res = evaluate_models(prototypes, class_parities, N_proto, T, noise_levels)
        all_inst_results.append(res)
        
    # Aggregate results
    final_res = {}
    for key in ['pairwise', 'cubic', 'mixed']:
        data = np.array([r[key] for r in all_inst_results])
        final_res[key + '_mean'] = np.mean(data, axis=0)
        final_res[key + '_std'] = np.std(data, axis=0)
        
    # Save
    np.savez(os.path.join(RESULT_DIR, "xor3_results.npz"), 
             noise_levels=noise_levels, 
             pairwise_mean=final_res['pairwise_mean'], pairwise_std=final_res['pairwise_std'],
             cubic_mean=final_res['cubic_mean'], cubic_std=final_res['cubic_std'],
             mixed_mean=final_res['mixed_mean'], mixed_std=final_res['mixed_std'])
    
    # Plot
    plt.figure(figsize=(8, 6))
    cp = get_color_palette()
    colors = {'pairwise': cp['uncentered'], 'cubic': cp['centered'], 'mixed': cp['mixed']}
    for key in ['pairwise', 'cubic', 'mixed']:
        mean = final_res[key + '_mean']
        std = final_res[key + '_std']
        plt.plot(noise_levels, mean, 'o-', color=colors[key], label=key.capitalize())
        plt.fill_between(noise_levels, mean - std, mean + std, color=colors[key], alpha=0.15)
        
    plt.xlabel("Bit-flip noise probability")
    plt.ylabel("Classification Accuracy")
    plt.title("HAM Score Family Comparison (Prototype + XOR-3 Hybrid)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    path = os.path.join(RESULT_DIR, "xor3_hybrid_task.png")
    plt.savefig(path)
    plt.close()
    print(f"  Saved plot to {path}")

if __name__ == "__main__":
    xor3_hybrid_experiment()
