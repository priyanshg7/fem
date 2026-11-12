"""
Script to train and evaluate Model D1 (Regional Stress Features) and
Model D2 (Fixed Spatial Sensor Array) across all geometries.
"""
import time
import pandas as pd
from src.stage3D_models import run_experiment_d1, run_experiment_d2

def main():
    print("Loading extracted feature parquets...")
    df_master = pd.read_parquet("data/stage3D_features_master.parquet")
    df_conf = pd.read_parquet("data/stage3D_blind_confirmatory/stage3D_confirmatory_features.parquet")
    
    df_train = df_master[df_master['split'] == 'train'].reset_index(drop=True)
    df_val = df_master[df_master['split'] == 'val'].reset_index(drop=True)
    df_test = df_master[df_master['split'] == 'test'].reset_index(drop=True)
    
    print(f"Dataset partitions: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}, Confirmatory={len(df_conf)}")
    
    # 1. Run Model D1
    t0 = time.time()
    res_d1_e1, res_d1_e2 = run_experiment_d1(df_train, df_val, df_test, df_conf)
    print(f"Model D1 completed in {time.time() - t0:.1f} s")
    
    # 2. Run Model D2
    t0 = time.time()
    res_d2 = run_experiment_d2(df_train, df_val, df_test, df_conf)
    print(f"Model D2 completed in {time.time() - t0:.1f} s")
    
    print("All Phase D and Phase E experiments successfully executed!")

if __name__ == '__main__':
    main()

