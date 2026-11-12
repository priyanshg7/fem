"""
Test runner script for SimuStruct Backend API test suite.
Executes all test suites and prints detailed validation outcome.
"""
import sys
import os
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time

def run_all_tests():
    print("==================================================")
    print("SIMUSTRUCT ML SURROGATE BACKEND TEST SUITE")
    print("==================================================")
    
    passed = 0
    failed = 0
    errors = []
    
    # 1. Health Tests
    print("\n--- Running Health & Readiness Tests ---")
    from backend.tests.test_health import test_root_endpoint, test_health_endpoint, test_readiness_probe
    for test_fn in [test_root_endpoint, test_health_endpoint, test_readiness_probe]:
        try:
            t0 = time.perf_counter()
            test_fn()
            print(f"  [PASS] {test_fn.__name__} ({(time.perf_counter()-t0)*1000:.1f} ms)")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {str(e)}")
            errors.append((test_fn.__name__, str(e)))
            failed += 1
            
    # 2. Model Registry Tests
    print("\n--- Running Model Registry Tests ---")
    from backend.tests.test_models import test_list_models, test_get_forward_model_detail, test_get_inverse_model_detail, test_nonexistent_model_returns_404
    for test_fn in [test_list_models, test_get_forward_model_detail, test_get_inverse_model_detail, test_nonexistent_model_returns_404]:
        try:
            t0 = time.perf_counter()
            test_fn()
            print(f"  [PASS] {test_fn.__name__} ({(time.perf_counter()-t0)*1000:.1f} ms)")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {str(e)}")
            errors.append((test_fn.__name__, str(e)))
            failed += 1
            
    # 3. Geometry Validation Tests
    print("\n--- Running Geometry Validation Tests ---")
    from backend.tests.test_validation import (
        test_valid_canonical_geometry, test_collision_with_central_hole_rejected,
        test_outer_boundary_penetration_rejected, test_non_canonical_central_hole_produces_warning
    )
    for test_fn in [test_valid_canonical_geometry, test_collision_with_central_hole_rejected,
                    test_outer_boundary_penetration_rejected, test_non_canonical_central_hole_produces_warning]:
        try:
            t0 = time.perf_counter()
            test_fn()
            print(f"  [PASS] {test_fn.__name__} ({(time.perf_counter()-t0)*1000:.1f} ms)")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {str(e)}")
            errors.append((test_fn.__name__, str(e)))
            failed += 1
            
    # 4. Forward Prediction Tests
    print("\n--- Running Forward Prediction Tests ---")
    from backend.tests.test_forward import (
        test_forward_prediction_success, test_forward_prediction_numerical_agreement_with_research,
        test_forward_prediction_rejects_collision, test_forward_prediction_rejects_boundary_penetration
    )
    for test_fn in [test_forward_prediction_success, test_forward_prediction_numerical_agreement_with_research,
                    test_forward_prediction_rejects_collision, test_forward_prediction_rejects_boundary_penetration]:
        try:
            t0 = time.perf_counter()
            test_fn()
            print(f"  [PASS] {test_fn.__name__} ({(time.perf_counter()-t0)*1000:.1f} ms)")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {str(e)}")
            errors.append((test_fn.__name__, str(e)))
            failed += 1
            
    # 5. Inverse Prediction Tests
    print("\n--- Running Inverse Prediction Tests ---")
    from backend.tests.test_inverse import (
        test_inverse_prediction_with_64x64_grid, test_inverse_prediction_with_scattered_coordinates,
        test_inverse_prediction_numerical_agreement, test_inverse_rejects_missing_inputs
    )
    for test_fn in [test_inverse_prediction_with_64x64_grid, test_inverse_prediction_with_scattered_coordinates,
                    test_inverse_prediction_numerical_agreement, test_inverse_rejects_missing_inputs]:
        try:
            t0 = time.perf_counter()
            test_fn()
            print(f"  [PASS] {test_fn.__name__} ({(time.perf_counter()-t0)*1000:.1f} ms)")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {str(e)}")
            errors.append((test_fn.__name__, str(e)))
            failed += 1
            
    # 6. Performance Benchmarks
    print("\n--- Running Performance Benchmarks ---")
    from backend.tests.test_performance import test_performance_benchmarks
    try:
        t0 = time.perf_counter()
        test_performance_benchmarks()
        print(f"  [PASS] test_performance_benchmarks ({(time.perf_counter()-t0)*1000:.1f} ms)")
        passed += 1
    except Exception as e:
        print(f"  [FAIL] test_performance_benchmarks: {str(e)}")
        errors.append(("test_performance_benchmarks", str(e)))
        failed += 1
        
    # 7. Stage 4C Independent Endpoint Verification
    print("\n--- Running Stage 4C Independent Verification Suite ---")
    from backend.tests.test_stage4c_verification import (
        run_stage4c_forward_parity_tests,
        run_stage4c_inverse_parity_tests,
        run_stage4c_negative_and_boundary_tests
    )
    for test_fn in [run_stage4c_forward_parity_tests, run_stage4c_inverse_parity_tests, run_stage4c_negative_and_boundary_tests]:
        try:
            t0 = time.perf_counter()
            test_fn()
            print(f"  [PASS] {test_fn.__name__} ({(time.perf_counter()-t0)*1000:.1f} ms)")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {str(e)}")
            errors.append((test_fn.__name__, str(e)))
            failed += 1
        
    print("\n==================================================")
    print(f"TEST SUMMARY: {passed} PASSED, {failed} FAILED (TOTAL {passed + failed})")
    print("==================================================")
    
    if failed > 0:
        print("\nFailures:")
        for name, err in errors:
            print(f"  - {name}: {err}")
        sys.exit(1)
    else:
        print("\nAll backend integration tests passed with 100% success!")
        sys.exit(0)

if __name__ == "__main__":
    run_all_tests()
