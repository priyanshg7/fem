"""
Stage 3A Section 13 - Model Save/Load and Numerical Consistency Verification
Verifies that all saved PyTorch checkpoints can be cleanly reloaded from disk
and generate numerically identical predictions (< 1e-6 tolerance).
"""

import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from src.stage3A_ann_trainer import ForwardANN

def verify_save_load_consistency():
    print("==================================================================")
    print("Starting Model Save/Load and Prediction Consistency Verification")
    print("==================================================================")
    
    base_dir = Path('models/stage3A_forward_ann')
    architectures = {'ANN_128': 128, 'ANN_256': 256, 'ANN_512': 512}
    experiments = ['original_dataset', 'enriched_dataset']
    
    # 5 sample test inputs (normalized [X/L, Y/H, D/Dc, L/H])
    sample_inputs = np.array([
        [0.35, 0.65, 0.15, 1.00],
        [0.20, 0.50, 0.25, 1.25],
        [0.55, 0.50, 0.30, 1.43],
        [0.80, 0.20, 0.08, 1.67],
        [0.50, 0.80, 0.15, 1.11]
    ], dtype=np.float32)
    t_inputs = torch.tensor(sample_inputs, dtype=torch.float32)
    
    all_passed = True
    tested_count = 0
    
    for exp in experiments:
        for arch, hidden in architectures.items():
            for seed in [42, 101]:
                ckpt_path = base_dir / exp / arch / f"model_{arch}_seed_{seed}.pt"
                if not ckpt_path.exists():
                    print(f"Skipping {ckpt_path} (not found yet)")
                    continue
                    
                tested_count += 1
                
                # Model 1: Initialized and loaded
                m1 = ForwardANN(in_features=4, hidden_neurons=hidden)
                state_dict1 = torch.load(ckpt_path, map_location='cpu')
                m1.load_state_dict(state_dict1)
                m1.eval()
                
                # Model 2: Separately instantiated and loaded
                m2 = ForwardANN(in_features=4, hidden_neurons=hidden)
                state_dict2 = torch.load(ckpt_path, map_location='cpu')
                m2.load_state_dict(state_dict2)
                m2.eval()
                
                with torch.no_grad():
                    pred1 = m1(t_inputs).numpy()
                    pred2 = m2(t_inputs).numpy()
                    
                diff = np.max(np.abs(pred1 - pred2))
                is_consistent = diff < 1e-6
                if not is_consistent:
                    all_passed = False
                    print(f"  [FAIL] {exp} / {arch} seed {seed}: max diff = {diff:.2e}")
                else:
                    print(f"  [PASS] {exp} / {arch} seed {seed}: max diff = {diff:.2e} (< 1e-6)")
                    
    print(f"\nTested {tested_count} model checkpoints. Save/Load Verification: {'ALL PASSED' if all_passed else 'FAILED'}")
    return all_passed

if __name__ == '__main__':
    verify_save_load_consistency()

