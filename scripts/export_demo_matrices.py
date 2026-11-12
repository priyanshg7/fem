"""
Export Demo 64x64 Stress Field Matrices for Instructor Demonstrations.
Generates both .json and .csv files across canonical plate geometries with ground truth metadata.
"""
import json
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.services.inverse_service import normalize_coordinates, rasterize_stress_field

def main():
    meta_df = pd.read_csv(PROJECT_ROOT / "data" / "stage3D_blind_confirmatory" / "stage3D_confirmatory_metadata.csv")
    field_dir = PROJECT_ROOT / "data" / "stage3D_blind_confirmatory" / "spatial_fields"
    out_dir = PROJECT_ROOT / "demo_matrices"
    out_dir.mkdir(exist_ok=True)

    specimens = [
        {
            "id": "DEMO-01",
            "name": "demo_01_square_1000x1000_high_accuracy",
            "file": "STAGE3D_BLIND_square_1000x1000_0401_fields.npz",
            "conf_id": "STAGE3D_BLIND_square_1000x1000_0401",
            "L": 1000.0,
            "H": 1000.0,
            "desc": "Square 1000x1000 mm plate (alpha=1.00) with clear defect. Demonstrates high U-Net localization accuracy.",
            "is_synthetic": False
        },
        {
            "id": "DEMO-02",
            "name": "demo_02_rect_1000x700_confirmatory_benchmark",
            "file": "STAGE3D_BLIND_rectangular_1000x700_0101_fields.npz",
            "conf_id": "STAGE3D_BLIND_rectangular_1000x700_0101",
            "L": 1000.0,
            "H": 700.0,
            "desc": "Audited confirmatory benchmark (SPEC-I02). Defect localized within 0.06 mm of ground truth.",
            "is_synthetic": False
        },
        {
            "id": "DEMO-03",
            "name": "demo_03_rect_1000x800_asymmetric_defect",
            "file": "STAGE3D_BLIND_rectangular_1000x800_0201_fields.npz",
            "conf_id": "STAGE3D_BLIND_rectangular_1000x800_0201",
            "L": 1000.0,
            "H": 800.0,
            "desc": "Rectangular 1000x800 mm with asymmetric off-center defect in bottom quadrant.",
            "is_synthetic": False
        },
        {
            "id": "DEMO-04",
            "name": "demo_04_rect_1000x900_moderate_aspect",
            "file": "STAGE3D_BLIND_rectangular_1000x900_0301_fields.npz",
            "conf_id": "STAGE3D_BLIND_rectangular_1000x900_0301",
            "L": 1000.0,
            "H": 900.0,
            "desc": "Rectangular 1000x900 mm plate with mild aspect ratio elongation (alpha=1.11).",
            "is_synthetic": False
        },
        {
            "id": "DEMO-05",
            "name": "demo_05_rect_1000x600_sub_threshold_audit",
            "file": "STAGE3D_BLIND_rectangular_1000x600_0001_fields.npz",
            "conf_id": "STAGE3D_BLIND_rectangular_1000x600_0001",
            "L": 1000.0,
            "H": 600.0,
            "desc": "Audited sub-threshold defect (d=9.36 mm). Demonstrates model boundary behavior and central hole attraction.",
            "is_synthetic": False
        },
        {
            "id": "DEMO-06",
            "name": "demo_06_uniform_negative_control_abstention",
            "file": None,
            "conf_id": None,
            "L": 1000.0,
            "H": 1000.0,
            "desc": "Synthetic uniform 100 psi field. Demonstrates automatic safety filter triggering FLAGGED_FOR_REVIEW.",
            "is_synthetic": True
        }
    ]

    catalog = []
    for s in specimens:
        L, H = s["L"], s["H"]
        if not s["is_synthetic"]:
            npz_p = field_dir / s["file"]
            data = np.load(npz_p)
            cnorm = normalize_coordinates(data["nodes"], L, H)
            grid_64 = rasterize_stress_field(cnorm, data["nodal_von_mises_psi"], grid_res=64)
            row = meta_df[meta_df["confirmatory_id"] == s["conf_id"]].iloc[0]
            true_x = float(row["defect_center_x_mm"])
            true_y = float(row["defect_center_y_mm"])
            true_d = float(row["defect_diameter_mm"])
            max_stress = float(row["max_von_mises_nodal_psi"])
        else:
            grid_64 = np.ones((64, 64), dtype=np.float32) * 100.0
            true_x, true_y, true_d, max_stress = None, None, None, 100.0

        # 1. Save pure CSV matrix (64 rows x 64 columns of numbers)
        csv_file = out_dir / f"{s['name']}.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            for r in range(64):
                writer.writerow([f"{val:.2f}" for val in grid_64[r]])

        # 2. Save structured JSON file (with metadata + raw 64x64 2D array)
        json_file = out_dir / f"{s['name']}.json"
        json_data = {
            "specimen_id": s["id"],
            "description": s["desc"],
            "plate_length_mm": L,
            "plate_height_mm": H,
            "aspect_ratio": round(L / H, 2),
            "applied_traction_psi": 100.0,
            "ground_truth": {
                "defect_center_x_mm": round(true_x, 2),
                "defect_center_y_mm": round(true_y, 2),
                "defect_diameter_mm": round(true_d, 2),
                "max_nodal_stress_psi": round(max_stress, 2)
            } if true_x is not None else None,
            "stress_grid_64x64": np.round(grid_64, 2).tolist()
        }
        with open(json_file, "w") as f:
            json.dump(json_data, f, indent=2)

        catalog.append({
            "id": s["id"],
            "name": s["name"],
            "L": L,
            "H": H,
            "truth_x": true_x,
            "truth_y": true_y,
            "diameter": true_d,
            "csv_file": csv_file.name,
            "json_file": json_file.name,
            "desc": s["desc"]
        })

    # 3. Create a Markdown Quick-Start Guide for the user and instructor
    readme_path = out_dir / "DEMO_INSPECTION_GUIDE.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("# SimuStruct Demonstration Test Matrices (64×64)\n\n")
        f.write("This directory contains pre-extracted, audited 64×64 Von Mises stress fields ready for upload into the **SimuStruct Digital Twin Platform** during faculty demonstrations and defense presentations.\n\n")
        f.write("Both **.json** (structured array with ground-truth metadata) and **.csv** (raw 64×64 numerical matrix) formats are provided for each specimen.\n\n")
        f.write("## Specimen Catalog & Demonstration Guide\n\n")
        f.write("| # | Specimen File Name | Plate Geometry | Ground Truth (X, Y) | Defect Size | Model Expected Status | What to Show the Instructor |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for i, c in enumerate(catalog, 1):
            truth_str = f"({c['truth_x']:.1f}, {c['truth_y']:.1f}) mm" if c['truth_x'] is not None else "None (Uniform)"
            diam_str = f"d = {c['diameter']:.1f} mm" if c['diameter'] is not None else "N/A"
            status_str = "ACCEPTED" if i != 5 and i != 6 else ("ACCEPTED (Audit)" if i == 5 else "FLAGGED_FOR_REVIEW")
            takeaway = (
                "High precision localization on square plate (< 5 mm error)." if i == 1 else
                "Audited benchmark specimen (error Delta_r = 0.06 mm)." if i == 2 else
                "Off-center asymmetric defect localization in bottom quadrant." if i == 3 else
                "Moderate aspect ratio plate (alpha = 1.11)." if i == 4 else
                "Scientific limitation demo: defect below detection threshold." if i == 5 else
                "Safety filter trigger: uniform field triggers abstention."
            )
            f.write(f"| {i} | `{c['csv_file']}` / `{c['json_file']}` | {int(c['L'])}×{int(c['H'])} mm | {truth_str} | {diam_str} | **{status_str}** | {takeaway} |\n")
        
        f.write("\n## How to Demonstrate Live in the UI:\n\n")
        f.write("1. Open the **SimuStruct Digital Twin Dashboard** at `http://127.0.0.1:3000` (or `http://localhost:8000`).\n")
        f.write("2. Navigate to the **Inverse Defect Inspection (Model D5)** tab.\n")
        f.write("3. Under **Plate Specimen Geometry**, click the preset button matching the specimen's height (e.g. `Square 1000×1000` or `Rect 1000×700`).\n")
        f.write("4. In the **Spatial Stress Field Input** box, click the **Upload** tab, click **Choose .json / .csv**, and select the matrix file from this `demo_matrices/` folder.\n")
        f.write("5. Click **Execute Inverse Localization (D5)**.\n")
        f.write("6. Show the instructor the predicted reticle (Cyan), the ground truth comparison, the dispersion radius, and click **Save Case & Generate Report** to demonstrate the complete digital twin traceability workflow.\n")

    print(f"Successfully generated {len(catalog)} demo specimens in {out_dir}")
    print(f"Created guide at: {readme_path}")

if __name__ == "__main__":
    main()

