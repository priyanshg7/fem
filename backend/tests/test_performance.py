"""
Performance and Latency Benchmarks for Stage 4A/4B Backend.
"""
import time
import psutil
import os
import numpy as np
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.registry import model_registry

client = TestClient(app)

def test_performance_benchmarks():
    process = psutil.Process(os.getpid())
    mem_before_mb = process.memory_info().rss / (1024 * 1024)
    
    # 1. Forward Inference Latency
    fwd_payload = {
        "plate_length_mm": 1000.0,
        "plate_height_mm": 800.0,
        "central_hole_diameter_mm": 66.67,
        "defect_diameter_mm": 15.0,
        "defect_center_x_mm": 250.0,
        "defect_center_y_mm": 300.0,
        "applied_traction_psi": 100.0
    }
    
    # Warmup / First inference
    t0 = time.perf_counter()
    r1 = client.post("/api/v1/predict/forward", json=fwd_payload)
    first_fwd_ms = (time.perf_counter() - t0) * 1000.0
    assert r1.status_code == 200
    
    # 10 Subsequent inferences
    fwd_latencies = []
    for _ in range(10):
        t_start = time.perf_counter()
        r = client.post("/api/v1/predict/forward", json=fwd_payload)
        fwd_latencies.append((time.perf_counter() - t_start) * 1000.0)
        assert r.status_code == 200
        
    avg_fwd_ms = np.mean(fwd_latencies)
    p95_fwd_ms = np.percentile(fwd_latencies, 95)
    
    # 2. Inverse Inference Latency
    grid = (np.ones((64, 64), dtype=float) * 100.0).tolist()
    inv_payload = {
        "plate_length_mm": 1000.0,
        "plate_height_mm": 800.0,
        "stress_grid_64x64": grid,
        "applied_traction_psi": 100.0,
        "return_heatmap": True
    }
    
    # First inference
    t0 = time.perf_counter()
    r_inv1 = client.post("/api/v1/predict/inverse", json=inv_payload)
    first_inv_ms = (time.perf_counter() - t0) * 1000.0
    assert r_inv1.status_code == 200
    
    # 10 Subsequent inferences
    inv_latencies = []
    for _ in range(10):
        t_start = time.perf_counter()
        r = client.post("/api/v1/predict/inverse", json=inv_payload)
        inv_latencies.append((time.perf_counter() - t_start) * 1000.0)
        assert r.status_code == 200
        
    avg_inv_ms = np.mean(inv_latencies)
    p95_inv_ms = np.percentile(inv_latencies, 95)
    
    # Memory footprint
    mem_after_mb = process.memory_info().rss / (1024 * 1024)
    response_size_kb = len(r_inv1.content) / 1024.0
    
    print("\n==========================================")
    print("STAGE 4B: PERFORMANCE MEASUREMENT AUDIT")
    print("==========================================")
    print(f"Forward Model (Stage 3B Ensemble):")
    print(f"  First Inference Latency:      {first_fwd_ms:.2f} ms")
    print(f"  Average Subsequent Latency:   {avg_fwd_ms:.2f} ms")
    print(f"  P95 Subsequent Latency:       {p95_fwd_ms:.2f} ms")
    print(f"Inverse Model (Stage 3D Model D5 U-Net):")
    print(f"  First Inference Latency:      {first_inv_ms:.2f} ms")
    print(f"  Average Subsequent Latency:   {avg_inv_ms:.2f} ms")
    print(f"  P95 Subsequent Latency:       {p95_inv_ms:.2f} ms")
    print(f"  Inverse Response Size:        {response_size_kb:.2f} KB (including 64x64 heatmap)")
    print(f"Memory Footprint:")
    print(f"  Process Memory (RSS):         {mem_after_mb:.2f} MB")
    print("==========================================\n")
    
    # Verification assertions (sub-100ms real-time latency target on CPU)
    assert avg_fwd_ms < 100.0
    assert avg_inv_ms < 100.0
