"""
Geometry and Mesh Generation for Plate with Defect using Gmsh.

Physical setup based on Singh & Pamnani (2025):
- Square isotropic plate: 1.0 x 1.0 [normalized units]
- Fixed central circular hole: center at (0.5, 0.5), diameter = 1/12 (radius = 1/24)
- Variable defect circular hole: center at (X, Y), diameter = 1/80 (radius = 1/160)
- Local mesh refinement around both hole perimeters for stress concentration resolution.
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import gmsh


@dataclass
class PlateGeometryConfig:
    plate_length: float = 1.0       # b (normalized unit)
    plate_width: float = 1.0        # l (normalized unit)
    center_hole_diameter: float = 1.0 / 12.0  # a = 1/12
    center_hole_x: float = 0.5
    center_hole_y: float = 0.5
    has_defect: bool = True         # Set False for single-hole benchmark
    defect_hole_diameter: float = 1.0 / 80.0  # c = 1/80
    defect_hole_x: float = 0.35     # default test position
    defect_hole_y: float = 0.65     # default test position
    
    # Mesh size parameters
    h_coarse: float = 0.03          # Far-field mesh size
    h_center_hole: float = 0.004    # Refinement at central hole boundary
    h_defect_hole: float = 0.0012   # Refinement at small defect boundary (radius=0.00625)
    clearance_margin: float = 0.01  # Minimum clearance to plate boundaries & central hole
    
    @property
    def center_hole_radius(self) -> float:
        return self.center_hole_diameter / 2.0

    @property
    def defect_hole_radius(self) -> float:
        return self.defect_hole_diameter / 2.0


def validate_defect_position(config: PlateGeometryConfig, margin: Optional[float] = None) -> bool:
    """
    Check that defect hole is strictly inside the plate domain and does not
    intersect the central hole or the outer boundary.
    """
    if not config.has_defect:
        return True

    if margin is None:
        margin = getattr(config, "clearance_margin", 0.01)

    x, y, r = config.defect_hole_x, config.defect_hole_y, config.defect_hole_radius
    rc = config.center_hole_radius
    xc, yc = config.center_hole_x, config.center_hole_y
    
    # Check boundaries
    if x - r < margin or x + r > config.plate_width - margin:
        return False
    if y - r < margin or y + r > config.plate_length - margin:
        return False
        
    # Check distance to central hole
    dist_center = np.hypot(x - xc, y - yc)
    if dist_center < (rc + r + margin):
        return False
        
    return True


def generate_mesh(
    config: PlateGeometryConfig,
    mesh_filename: Optional[str] = None,
    order: int = 1,
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate 2D triangular mesh with Gmsh using OpenCASCADE geometry.
    
    Returns:
        nodes: (N, 2) array of nodal coordinates [x, y]
        elements: (M, 3) or (M, 6) array of element node indices
        physical_tags: dictionary mapping physical group names to entity tags
    """
    margin = getattr(config, "clearance_margin", 0.01)
    if not validate_defect_position(config, margin=margin):
        raise ValueError(
            f"Invalid defect position ({config.defect_hole_x}, {config.defect_hole_y}). "
            f"It interferes with plate boundaries or central hole."
        )

    gmsh.initialize()
    try:
        gmsh.model.add("plate_with_defect" if config.has_defect else "plate_central_hole")
        if not verbose:
            gmsh.option.setNumber("General.Terminal", 0)
        
        occ = gmsh.model.occ
        
        # 1. Base plate rectangle: [0, plate_width] x [0, plate_length]
        rect = occ.addRectangle(0.0, 0.0, 0.0, config.plate_width, config.plate_length)
        
        # 2. Central circular hole disk
        hole_c = occ.addDisk(
            config.center_hole_x, config.center_hole_y, 0.0,
            config.center_hole_radius, config.center_hole_radius
        )
        
        # 3. Defect circular hole disk (if configured)
        if config.has_defect:
            hole_d = occ.addDisk(
                config.defect_hole_x, config.defect_hole_y, 0.0,
                config.defect_hole_radius, config.defect_hole_radius
            )
            cut_tools = [(2, hole_c), (2, hole_d)]
        else:
            cut_tools = [(2, hole_c)]
        
        # 4. Cut hole(s) from plate
        domain, _ = occ.cut([(2, rect)], cut_tools)
        occ.synchronize()
        
        domain_tag = domain[0][1]

        # 5. Identify physical boundaries for boundary condition assignment:
        # Bottom: y = 0
        # Top: y = plate_length
        # Left: x = 0
        # Right: x = plate_width
        # Central hole curve (COM is at circle center (center_hole_x, center_hole_y))
        # Defect hole curve (COM is at circle center (defect_hole_x, defect_hole_y))
        curves = gmsh.model.getEntities(dim=1)
        
        bottom_curves = []
        top_curves = []
        left_curves = []
        right_curves = []
        center_hole_curves = []
        defect_hole_curves = []
        
        eps = 1e-4
        for dim, tag in curves:
            com = occ.getCenterOfMass(dim, tag)
            cx, cy = com[0], com[1]
            
            # Distance from hole centers to curve center of mass
            dc = np.hypot(cx - config.center_hole_x, cy - config.center_hole_y)
            dd = np.hypot(cx - config.defect_hole_x, cy - config.defect_hole_y) if config.has_defect else 1e9
            
            if abs(cy - 0.0) < eps:
                bottom_curves.append(tag)
            elif abs(cy - config.plate_length) < eps:
                top_curves.append(tag)
            elif abs(cx - 0.0) < eps:
                left_curves.append(tag)
            elif abs(cx - config.plate_width) < eps:
                right_curves.append(tag)
            elif dc < eps:
                center_hole_curves.append(tag)
            elif config.has_defect and dd < eps:
                defect_hole_curves.append(tag)

        # Create physical groups for boundary conditions and surface
        tag_bottom = gmsh.model.addPhysicalGroup(1, bottom_curves, name="Bottom")
        tag_top = gmsh.model.addPhysicalGroup(1, top_curves, name="Top")
        tag_left = gmsh.model.addPhysicalGroup(1, left_curves, name="Left")
        tag_right = gmsh.model.addPhysicalGroup(1, right_curves, name="Right")
        tag_center = gmsh.model.addPhysicalGroup(1, center_hole_curves, name="CenterHole")
        if config.has_defect and defect_hole_curves:
            tag_defect = gmsh.model.addPhysicalGroup(1, defect_hole_curves, name="DefectHole")
        else:
            tag_defect = -1
        tag_surface = gmsh.model.addPhysicalGroup(2, [domain_tag], name="PlateSurface")

        # 6. Mesh refinement setup via Distance and Threshold Fields
        field_c = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(field_c, "CurvesList", center_hole_curves)
        gmsh.model.mesh.field.setNumber(field_c, "Sampling", 100)
        
        thresh_c = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thresh_c, "InField", field_c)
        gmsh.model.mesh.field.setNumber(thresh_c, "SizeMin", config.h_center_hole)
        gmsh.model.mesh.field.setNumber(thresh_c, "SizeMax", config.h_coarse)
        gmsh.model.mesh.field.setNumber(thresh_c, "DistMin", config.center_hole_radius * 0.2)
        gmsh.model.mesh.field.setNumber(thresh_c, "DistMax", config.center_hole_radius * 3.0)

        if config.has_defect and defect_hole_curves:
            field_d = gmsh.model.mesh.field.add("Distance")
            gmsh.model.mesh.field.setNumbers(field_d, "CurvesList", defect_hole_curves)
            gmsh.model.mesh.field.setNumber(field_d, "Sampling", 100)
            
            thresh_d = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(thresh_d, "InField", field_d)
            gmsh.model.mesh.field.setNumber(thresh_d, "SizeMin", config.h_defect_hole)
            gmsh.model.mesh.field.setNumber(thresh_d, "SizeMax", config.h_coarse)
            gmsh.model.mesh.field.setNumber(thresh_d, "DistMin", config.defect_hole_radius * 0.5)
            gmsh.model.mesh.field.setNumber(thresh_d, "DistMax", config.defect_hole_radius * 4.0)

            min_field = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [thresh_c, thresh_d])
            gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
        else:
            gmsh.model.mesh.field.setAsBackgroundMesh(thresh_c)

        # Options for 2D mesh generation
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.ElementOrder", order)
        gmsh.option.setNumber("Mesh.Algorithm", 6) # Frontal-Delaunay for 2D

        gmsh.model.mesh.generate(2)

        if mesh_filename:
            gmsh.write(mesh_filename)

        # Extract nodes and elements for direct processing/inspection
        node_tags, coords, _ = gmsh.model.mesh.getNodes()
        # Coords are 3D (x, y, z), reshape to (N, 3)
        coords = np.array(coords, dtype=np.float64).reshape(-1, 3)[:, :2]
        # Re-index mapping: tag -> 0-based index
        tag_to_idx = {tag: i for i, tag in enumerate(node_tags)}

        # Get triangular elements
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        # Type 2 is 3-node triangle, Type 9 is 6-node quadratic triangle
        all_tri_nodes = []
        for etype, enodes in zip(elem_types, elem_node_tags):
            if etype == 2:
                tris = np.array(enodes, dtype=np.int64).reshape(-1, 3)
                mapped = np.vectorize(tag_to_idx.get)(tris)
                all_tri_nodes.append(mapped)
            elif etype == 9:
                tris = np.array(enodes, dtype=np.int64).reshape(-1, 6)
                mapped = np.vectorize(tag_to_idx.get)(tris)
                all_tri_nodes.append(mapped)

        elements = np.vstack(all_tri_nodes) if all_tri_nodes else np.empty((0, 3), dtype=np.int64)

        physical_dict = {
            "Bottom": tag_bottom,
            "Top": tag_top,
            "Left": tag_left,
            "Right": tag_right,
            "CenterHole": tag_center,
            "DefectHole": tag_defect,
            "PlateSurface": tag_surface,
        }

        return coords, elements, physical_dict

    finally:
        gmsh.finalize()


if __name__ == "__main__":
    cfg = PlateGeometryConfig(defect_hole_x=0.35, defect_hole_y=0.65)
    nodes, elems, tags = generate_mesh(cfg, mesh_filename="plate_test.msh", verbose=True)
    print(f"Generated mesh with {len(nodes)} nodes and {len(elems)} triangular elements.")
    print(f"Physical tags: {tags}")

