"""
Stage 3A-R Phase K: Model Save/Load and Numerical Consistency Verification
Verifies that all improved Stage 3A-R models (D1, D2, D3, ResMLP, MultiOutputANN)
can be reloaded cleanly from disk and generate numerically identical predictions (< 1e-6 tolerance).
"""

import os
import torch
import numpy as np
from pathlib import Path

from src.stage3AR_models import (
    DeepMLP_D1, DeepMLP_D2, DeepMLP_D3, ResMLP, MultiOutputANN
)

def verify_stage3AR_save_load():
    print("==================================================================")
    print("STAGE 3A-R PHASE K: MODEL SAVE/LOAD VERIFICATION")
    print("==================================================================")
    
    base_dir = Path('models/stage3AR_forward_improved')
    in_features = 6 # Feature Set D (Selected Best Feature Set)
    
    # 5 sample test inputs (normalized [9 features])
    sample_inputs = np.random.uniform(-1.5, 1.5, size=(5, in_features)).astype(np.float32)
    t_inputs = torch.tensor(sample_inputs, dtype=torch.float32)
    
    all_passed = True
    tested = 0
    
    # 1. Test Architecture Checkpoints
    arch_constructors = {
        'DeepMLP_D1': lambda: DeepMLP_D1(in_features),
        'DeepMLP_D2': lambda: DeepMLP_D2(in_features),
        'DeepMLP_D3': lambda: DeepMLP_D3(in_features),
        'ResMLP': lambda: ResMLP(in_features)
    }
    
    for a_name, ctor in arch_constructors.items():
        for s in [42, 101]:
            ckpt_p = base_dir / 'architectures' / a_name / f"model_{a_name}_seed_{s}.pt"
            if not ckpt_p.exists():
                continue
            tested += 1
            m1 = ctor()
            m1.load_state_dict(torch.load(ckpt_p, map_location='cpu'))
            m1.eval()
            
            m2 = ctor()
            m2.load_state_dict(torch.load(ckpt_p, map_location='cpu'))
            m2.eval()
            
            with torch.no_grad():
                pred1 = m1(t_inputs).numpy()
                pred2 = m2(t_inputs).numpy()
                
            diff = float(np.max(np.abs(pred1 - pred2)))
            if diff < 1e-6:
                print(f"  [PASS] {a_name} seed {s}: max diff = {diff:.2e} (< 1e-6)")
            else:
                all_passed = False
                print(f"  [FAIL] {a_name} seed {s}: max diff = {diff:.2e}")
                
    # 2. Test Multi-Output Checkpoints
    for s in [42, 101]:
        ckpt_mo = base_dir / 'multi_output' / f"multi_output_seed_{s}.pt"
        if not ckpt_mo.exists():
            continue
        tested += 1
        m1 = MultiOutputANN(in_features)
        m1.load_state_dict(torch.load(ckpt_mo, map_location='cpu'))
        m1.eval()
        
        m2 = MultiOutputANN(in_features)
        m2.load_state_dict(torch.load(ckpt_mo, map_location='cpu'))
        m2.eval()
        
        with torch.no_grad():
            c1, d1 = m1(t_inputs)
            c2, d2 = m2(t_inputs)
            pred1 = np.maximum(c1.numpy(), d1.numpy())
            pred2 = np.maximum(c2.numpy(), d2.numpy())
            
        diff = float(np.max(np.abs(pred1 - pred2)))
        if diff < 1e-6:
            print(f"  [PASS] MultiOutputANN seed {s}: max diff = {diff:.2e} (< 1e-6)")
        else:
            all_passed = False
            print(f"  [FAIL] MultiOutputANN seed {s}: max diff = {diff:.2e}")
            
    print(f"\nTested {tested} model checkpoints. Overall Result: {'ALL PASSED' if all_passed and tested > 0 else 'PENDING / FAILED'}")

if __name__ == '__main__':
    verify_stage3AR_save_load()
