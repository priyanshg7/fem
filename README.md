# Finite Element Method (FEM) Plate Simulation & Defect Identification

A high-performance Finite Element Method (FEM) simulation suite and machine learning pipeline for forward stress field modeling and inverse defect identification in isotropic plates under uniaxial tension.

Reproduces and extends the experimental and numerical methodology from:
> **Singh & Pamnani (2025)**, MNIT Jaipur / ISME 2025.

---

## 🏗️ Architecture

```
fem_final/
├── backend/          # FastAPI REST API serving FEM solver & ML inference
├── frontend/         # React + Vite + Tailwind CSS interactive dashboard
├── src/              # Core FEM solver, mesh generation, constitutive equations & ML models
├── reports/          # Verification, convergence, benchmark and model stacking reports
├── scripts/          # Automation, batch simulation, dataset generation & pipeline scripts
├── tests/            # Test suite for solvers, boundary conditions, and API endpoints
├── demo_matrices/    # Benchmark matrices and inspection guides
├── run_simulation.py # Main standalone FEM simulation executable
├── Dockerfile        # Container build definition
└── docker-compose.yml# Multi-service container orchestration
```

---

## 🚀 Quickstart

### 1. Standalone FEM Simulation
```bash
python run_simulation.py --defect-x 0.35 --defect-y 0.65
```

### 2. Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
python run_server.py
```

### 3. Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev
```

### 4. Docker Compose
```bash
docker compose up --build
```

---

## 📊 Scientific Formulation
- **Constitutive Law**: Isotropic Linear Elasticity (Plane Stress formulation, $E = 10^7\text{ psi}, \nu = 0.3$)
- **Domain**: Square / Rectangular plate with fixed central hole ($a = 1/12$) and variable defect ($c = 1/80$)
- **Boundary Conditions**: Roller support at base ($u_y = 0$), center pinned in $x$ ($u_x = 0$), uniform uniaxial tension on top edge ($P = 100\text{ psi}$)
- **Inverse Problem**: Defect $(X, Y)$ coordinate prediction and stress field reconstruction using multi-model stacking ensembles.
