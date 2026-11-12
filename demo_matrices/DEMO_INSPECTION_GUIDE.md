# SimuStruct Demonstration Test Matrices (64×64)

This directory contains pre-extracted, audited 64×64 Von Mises stress fields ready for upload into the **SimuStruct Digital Twin Platform** during faculty demonstrations and defense presentations.

Both **.json** (structured array with ground-truth metadata) and **.csv** (raw 64×64 numerical matrix) formats are provided for each specimen.

## Specimen Catalog & Demonstration Guide

| # | Specimen File Name | Plate Geometry | Ground Truth (X, Y) | Defect Size | Model Expected Status | What to Show the Instructor |
|---|---|---|---|---|---|---|
| 1 | `demo_01_square_1000x1000_high_accuracy.csv` / `demo_01_square_1000x1000_high_accuracy.json` | 1000×1000 mm | (161.2, 911.9) mm | d = 14.6 mm | **ACCEPTED** | High precision localization on square plate (< 5 mm error). |
| 2 | `demo_02_rect_1000x700_confirmatory_benchmark.csv` / `demo_02_rect_1000x700_confirmatory_benchmark.json` | 1000×700 mm | (510.1, 256.0) mm | d = 11.1 mm | **ACCEPTED** | Audited benchmark specimen (error Delta_r = 0.06 mm). |
| 3 | `demo_03_rect_1000x800_asymmetric_defect.csv` / `demo_03_rect_1000x800_asymmetric_defect.json` | 1000×800 mm | (106.5, 518.4) mm | d = 13.6 mm | **ACCEPTED** | Off-center asymmetric defect localization in bottom quadrant. |
| 4 | `demo_04_rect_1000x900_moderate_aspect.csv` / `demo_04_rect_1000x900_moderate_aspect.json` | 1000×900 mm | (341.3, 251.4) mm | d = 19.8 mm | **ACCEPTED** | Moderate aspect ratio plate (alpha = 1.11). |
| 5 | `demo_05_rect_1000x600_sub_threshold_audit.csv` / `demo_05_rect_1000x600_sub_threshold_audit.json` | 1000×600 mm | (192.6, 582.6) mm | d = 9.4 mm | **ACCEPTED (Audit)** | Scientific limitation demo: defect below detection threshold. |
| 6 | `demo_06_uniform_negative_control_abstention.csv` / `demo_06_uniform_negative_control_abstention.json` | 1000×1000 mm | None (Uniform) | N/A | **FLAGGED_FOR_REVIEW** | Safety filter trigger: uniform field triggers abstention. |

## How to Demonstrate Live in the UI:

1. Open the **SimuStruct Digital Twin Dashboard** at `http://127.0.0.1:3000` (or `http://localhost:8000`).
2. Navigate to the **Inverse Defect Inspection (Model D5)** tab.
3. Under **Plate Specimen Geometry**, click the preset button matching the specimen's height (e.g. `Square 1000×1000` or `Rect 1000×700`).
4. In the **Spatial Stress Field Input** box, click the **Upload** tab, click **Choose .json / .csv**, and select the matrix file from this `demo_matrices/` folder.
5. Click **Execute Inverse Localization (D5)**.
6. Show the instructor the predicted reticle (Cyan), the ground truth comparison, the dispersion radius, and click **Save Case & Generate Report** to demonstrate the complete digital twin traceability workflow.
