"""
Physics-based Geometry Validation Service.
Enforces physical non-penetration constraints, canonical geometry rules,
and supported operating domain boundaries.
"""
import math
import numpy as np
from typing import List, Tuple, Dict, Any

from backend.app.core.config import settings
from backend.app.schemas.validation import (
    GeometryValidationRequest, GeometryValidationResponse,
    ValidationStatus, ValidationSeverity, ValidationIssue, GeometryClearances
)

class GeometryValidationService:
    @staticmethod
    def validate(request: GeometryValidationRequest) -> GeometryValidationResponse:
        issues: List[ValidationIssue] = []
        
        L = float(request.plate_length_mm)
        H = float(request.plate_height_mm)
        Dc = float(request.central_hole_diameter_mm)
        d = float(request.defect_diameter_mm)
        x = float(request.defect_center_x_mm)
        y = float(request.defect_center_y_mm)
        
        # 1. Finite and Positive Dimension Checks
        for name, val in [("plate_length_mm", L), ("plate_height_mm", H),
                          ("central_hole_diameter_mm", Dc), ("defect_diameter_mm", d)]:
            if not math.isfinite(val) or val <= 0:
                issues.append(ValidationIssue(
                    field=name,
                    code="NON_POSITIVE_DIMENSION",
                    message=f"Dimension '{name}' must be a positive finite number (got {val}).",
                    severity=ValidationSeverity.ERROR,
                    value=val
                ))
                
        for name, val in [("defect_center_x_mm", x), ("defect_center_y_mm", y)]:
            if not math.isfinite(val):
                issues.append(ValidationIssue(
                    field=name,
                    code="NON_FINITE_COORDINATE",
                    message=f"Coordinate '{name}' must be a finite real number.",
                    severity=ValidationSeverity.ERROR,
                    value=val
                ))
                
        if any(issue.severity == ValidationSeverity.ERROR for issue in issues):
            return GeometryValidationResponse(
                status=ValidationStatus.INVALID,
                is_valid=False,
                can_proceed_to_inference=False,
                issues=issues,
                canonical_geometry_compliance=False,
                operating_domain_summary={"error": "Malformed or non-positive inputs"}
            )
            
        # 2. Canonical Central Hole Geometry Check (Dc = H / 12)
        canonical_Dc = H * settings.CANONICAL_DC_RATIO
        dc_diff = abs(Dc - canonical_Dc)
        is_canonical = dc_diff <= 0.5
        if not is_canonical:
            issues.append(ValidationIssue(
                field="central_hole_diameter_mm",
                code="NON_CANONICAL_CENTRAL_HOLE",
                message=(
                    f"Central hole diameter {Dc:.2f} mm departs from canonical rule Dc = H / 12 ({canonical_Dc:.2f} mm). "
                    "Models were verified on canonical geometry; predictions may exhibit out-of-domain shift."
                ),
                severity=ValidationSeverity.WARNING,
                value=Dc,
                constraint=f"Dc == H / 12 == {canonical_Dc:.2f} mm"
            ))
            
        # 3. Geometric Clearance & Collision Calculations
        Xc = L / 2.0
        Yc = H / 2.0
        dist_c = math.hypot(x - Xc, y - Yc)
        c_lig = dist_c - (Dc + d) / 2.0
        
        c_left = x - d / 2.0
        c_right = L - x - d / 2.0
        c_bot = y - d / 2.0
        c_top = H - y - d / 2.0
        min_bnd = min(c_left, c_right, c_bot, c_top)
        
        min_margin = H * settings.MIN_CLEARANCE_MARGIN_FACTOR
        
        # 3.1 Hole-to-Hole Collision Check
        if c_lig <= 0:
            issues.append(ValidationIssue(
                field="hole_to_hole_clearance_mm",
                code="CENTRAL_HOLE_COLLISION",
                message=f"Physical collision detected: defect intersects or overlaps central hole (clearance: {c_lig:.2f} mm).",
                severity=ValidationSeverity.ERROR,
                value=c_lig,
                constraint="clearance > 0.0 mm"
            ))
        elif c_lig < min_margin:
            issues.append(ValidationIssue(
                field="hole_to_hole_clearance_mm",
                code="CRITICAL_LIGAMENT_PROXIMITY",
                message=(
                    f"Defect is in extreme ligament interaction proximity (clearance: {c_lig:.2f} mm < threshold {min_margin:.2f} mm). "
                    "High stress concentration coupling observed; uncertainty may be elevated."
                ),
                severity=ValidationSeverity.WARNING,
                value=c_lig,
                constraint=f"clearance >= {min_margin:.2f} mm"
            ))
            
        # 3.2 Boundary Penetration Check
        if min_bnd <= 0:
            issues.append(ValidationIssue(
                field="min_boundary_clearance_mm",
                code="PLATE_BOUNDARY_PENETRATION",
                message=f"Defect intersects or lies outside the outer plate boundary (min boundary clearance: {min_bnd:.2f} mm).",
                severity=ValidationSeverity.ERROR,
                value=min_bnd,
                constraint="min_clearance > 0.0 mm"
            ))
        elif min_bnd < min_margin:
            issues.append(ValidationIssue(
                field="min_boundary_clearance_mm",
                code="CRITICAL_BOUNDARY_PROXIMITY",
                message=(
                    f"Defect is in close proximity to plate boundary (clearance: {min_bnd:.2f} mm < threshold {min_margin:.2f} mm). "
                    "Boundary reflection and stress shadow interactions apply."
                ),
                severity=ValidationSeverity.WARNING,
                value=min_bnd,
                constraint=f"min_clearance >= {min_margin:.2f} mm"
            ))
            
        # 4. Aspect Ratio & Sizing Domain Checks
        ar = L / H
        if ar < settings.MIN_ASPECT_RATIO or ar > (settings.MAX_ASPECT_RATIO + 0.05):
            issues.append(ValidationIssue(
                field="aspect_ratio",
                code="OUT_OF_DOMAIN_ASPECT_RATIO",
                message=f"Plate aspect ratio L/H = {ar:.3f} is outside the validated domain [1.000, 1.667].",
                severity=ValidationSeverity.WARNING,
                value=ar,
                constraint="1.000 <= L/H <= 1.667"
            ))
            
        # Check discrete validated geometries vs unanchored intermediate geometries
        validated_heights = [600.0, 700.0, 800.0, 900.0, 1000.0]
        min_h_diff = min(abs(H - vh) for vh in validated_heights)
        if min_h_diff > 5.0 and (settings.CANONICAL_MIN_HEIGHT_MM <= H <= settings.CANONICAL_MAX_HEIGHT_MM):
            issues.append(ValidationIssue(
                field="plate_height_mm",
                code="INTERMEDIATE_GEOMETRY_INTERPOLATION",
                message=(
                    f"Plate height H = {H:.1f} mm is an unanchored intermediate geometry between verified discrete geometries "
                    f"{validated_heights} mm. Surrogate prediction relies on smooth continuous feature interpolation."
                ),
                severity=ValidationSeverity.WARNING,
                value=H,
                constraint="H in {600, 700, 800, 900, 1000} mm"
            ))
            
        d_ratio = d / Dc
        if d_ratio < settings.MIN_DIAMETER_RATIO or d_ratio > settings.MAX_DIAMETER_RATIO:
            issues.append(ValidationIssue(
                field="defect_d_over_Dc",
                code="OUT_OF_DOMAIN_DEFECT_SIZE",
                message=f"Defect diameter ratio d/Dc = {d_ratio:.3f} is outside the validated domain [0.08, 0.45].",
                severity=ValidationSeverity.WARNING,
                value=d_ratio,
                constraint="0.08 <= d/Dc <= 0.45"
            ))
            
        # Summary and Decision
        has_errors = any(i.severity == ValidationSeverity.ERROR for i in issues)
        has_warnings = any(i.severity == ValidationSeverity.WARNING for i in issues)
        
        if has_errors:
            v_status = ValidationStatus.INVALID
            can_proceed = False
        elif has_warnings:
            v_status = ValidationStatus.VALID_WITH_WARNINGS
            can_proceed = True
        else:
            v_status = ValidationStatus.VALID
            can_proceed = True
            
        clearances = GeometryClearances(
            aspect_ratio=round(ar, 4),
            hole_to_hole_clearance_mm=round(c_lig, 2),
            min_boundary_clearance_mm=round(min_bnd, 2),
            clearance_to_left_edge_mm=round(c_left, 2),
            clearance_to_right_edge_mm=round(c_right, 2),
            clearance_to_bottom_edge_mm=round(c_bot, 2),
            clearance_to_top_edge_mm=round(c_top, 2),
            defect_to_central_hole_diameter_ratio=round(d_ratio, 4),
            min_allowed_margin_mm=round(min_margin, 2)
        )
        
        op_domain = {
            "aspect_ratio_in_domain": (settings.MIN_ASPECT_RATIO <= ar <= settings.MAX_ASPECT_RATIO + 0.05),
            "diameter_ratio_in_domain": (settings.MIN_DIAMETER_RATIO <= d_ratio <= settings.MAX_DIAMETER_RATIO),
            "canonical_geometry": is_canonical,
            "clearance_status": "NORMAL" if min(c_lig, min_bnd) >= min_margin else "NEAR_INTERACTION_ZONE"
        }
        
        return GeometryValidationResponse(
            status=v_status,
            is_valid=(not has_errors),
            can_proceed_to_inference=can_proceed,
            clearances=clearances,
            issues=issues,
            canonical_geometry_compliance=is_canonical,
            operating_domain_summary=op_domain
        )

validation_service = GeometryValidationService()
