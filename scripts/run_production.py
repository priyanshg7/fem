"""
SimuStruct Digital Twin Platform — Single-Origin Production Application Server.
Serves the React Engineering Dashboard and FastAPI ML Surrogate API on a single port.
"""
import sys
import os
import uvicorn
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("=" * 65)
    print(" SIMUSTRUCT DIGITAL TWIN PLATFORM — PRODUCTION RELEASE v1.0.0")
    print("=" * 65)
    
    # 1. Verify frozen model artifacts exist
    models_dir = PROJECT_ROOT / "models"
    fwd_path = models_dir / "forward_ensemble_stage3b" / "multi_level_stack_final.pkl"
    inv_path = models_dir / "inverse_unet_d5_stage3d" / "model_d5_unet_best.pt"
    
    if not fwd_path.exists():
        print(f"[FATAL] Forward model checkpoint not found: {fwd_path}")
        sys.exit(1)
    if not inv_path.exists():
        print(f"[FATAL] Inverse model checkpoint not found: {inv_path}")
        sys.exit(1)
        
    print(f"[*] Verified Forward Model: {fwd_path.name}")
    print(f"[*] Verified Inverse Model: {inv_path.name}")

    # 2. Verify compiled React SPA distribution
    dist_dir = PROJECT_ROOT / "frontend" / "dist"
    if not dist_dir.exists() or not (dist_dir / "index.html").exists():
        print(f"[WARN] Frontend dist directory not found at {dist_dir}.")
        print("[*] Compiling frontend production bundle...")
        import subprocess
        res = subprocess.run("npm run build", shell=True, cwd=str(PROJECT_ROOT / "frontend"))
        if res.returncode != 0:
            print("[FATAL] Frontend compilation failed.")
            sys.exit(1)
    print(f"[*] Verified Frontend SPA Bundle: {dist_dir}")

    # 3. Launch Uvicorn Server
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"[*] Launching Single-Origin Server on http://{host}:{port}")
    print(f"    - Web Application: http://localhost:{port}/")
    print(f"    - REST API Base:   http://localhost:{port}/api/v1")
    print(f"    - API Docs (OpenAPI): http://localhost:{port}/docs")
    print("=" * 65)

    uvicorn.run(
        "backend.app.main:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    main()

