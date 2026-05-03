import gmsh
import numpy as np
from scipy.interpolate import RBFInterpolator
import re
import sys

# ─────────────────────────────────────────────────────────────
# 1. PARSE ORIGINAL .inp MESH (same as before)
# ─────────────────────────────────────────────────────────────

def parse_inp(filepath):
    nodes, elements, node_sets, elem_sets = {}, {}, {}, {}
    with open(filepath, 'r') as f:
        lines = f.readlines()

    mode, current_set = None, None
    for line in lines:
        line = line.strip()
        if not line or line.startswith('**'):
            continue

        if line.upper().startswith('*NODE'):
            mode = 'NODE'
        elif line.upper().startswith('*ELEMENT'):
            mode = 'ELEMENT'
            match = re.search(r'TYPE\s*=\s*(\S+)', line, re.IGNORECASE)
            current_set = match.group(1) if match else 'UNKNOWN'
        elif line.upper().startswith('*NSET'):
            mode = 'NSET'
            match = re.search(r'NSET\s*=\s*(\S+)', line, re.IGNORECASE)
            current_set = match.group(1) if match else None
            if current_set: node_sets.setdefault(current_set, [])
        elif line.upper().startswith('*ELSET'):
            mode = 'ELSET'
            match = re.search(r'ELSET\s*=\s*(\S+)', line, re.IGNORECASE)
            current_set = match.group(1) if match else None
            if current_set: elem_sets.setdefault(current_set, [])
        elif line.startswith('*'):
            mode = None
            continue

        if mode == 'NODE':
            parts = [p.strip() for p in line.split(',')]
            if parts[0].isdigit():
                nid = int(parts[0])
                nodes[nid] = [float(parts[1]), float(parts[2]),
                               float(parts[3]) if len(parts) > 3 else 0.0]
        elif mode == 'ELEMENT':
            parts = [p.strip() for p in line.split(',')]
            if parts[0].isdigit():
                eid = int(parts[0])
                elements[eid] = [int(n) for n in parts[1:] if n]
        elif mode == 'NSET':
            node_sets[current_set].extend(
                [int(p.strip()) for p in line.split(',') if p.strip().isdigit()])
        elif mode == 'ELSET':
            elem_sets[current_set].extend(
                [int(p.strip()) for p in line.split(',') if p.strip().isdigit()])

    return nodes, elements, node_sets, elem_sets


# ─────────────────────────────────────────────────────────────
# 2. IMPORT STP FILE & EXTRACT BOUNDARY CURVES USING GMSH
# ─────────────────────────────────────────────────────────────

def extract_boundary_from_stp(stp_filepath, n_samples=200, scale_factor=1.0):
    """
    Imports a STEP/STP file via Gmsh OpenCASCADE kernel.
    Extracts all boundary curves (wires/edges) as sampled point clouds.

    Returns:
        boundaries: dict { curve_tag: [(x,y), ...] }
        all_boundary_pts: flat Nx2 array of all boundary points
        bbox: (xmin, xmax, ymin, ymax)
        curve_info: dict with curve metadata (length, centroid, type)
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 2)
    gmsh.model.add("target_geometry")

    # Import the STEP file using OpenCASCADE
    gmsh.model.occ.importShapes(stp_filepath)
    gmsh.model.occ.synchronize()

    # Get all entities
    points   = gmsh.model.getEntities(0)   # dim=0: points
    curves   = gmsh.model.getEntities(1)   # dim=1: curves/edges
    surfaces = gmsh.model.getEntities(2)   # dim=2: surfaces/faces

    print(f"STP imported: {len(points)} points, "
          f"{len(curves)} curves, {len(surfaces)} surfaces")

    # Apply scale factor (e.g., mm → m: scale_factor=1e-3)
    if scale_factor != 1.0:
        gmsh.model.occ.dilate(
            gmsh.model.getEntities(),
            0, 0, 0,
            scale_factor, scale_factor, scale_factor
        )
        gmsh.model.occ.synchronize()

    # ── Sample points along each boundary curve ──
    boundaries  = {}
    curve_info  = {}
    all_pts_list = []

    for dim, tag in curves:
        # Get parametric range of this curve
        t_min, t_max = gmsh.model.getParametrizationBounds(dim, tag)
        t_vals = np.linspace(t_min, t_max, n_samples)

        pts = []
        for t in t_vals:
            try:
                x, y, z = gmsh.model.getValue(dim, tag, [t])
                pts.append((x, y))
            except Exception:
                continue

        if len(pts) < 2:
            continue

        pts_arr = np.array(pts)
        boundaries[tag] = pts_arr
        all_pts_list.append(pts_arr)

        # Compute metadata
        centroid = pts_arr.mean(axis=0)
        length   = np.sum(np.linalg.norm(np.diff(pts_arr, axis=0), axis=1))
        r_vals   = np.linalg.norm(pts_arr, axis=1)

        curve_info[tag] = {
            'centroid': centroid,
            'length':   length,
            'r_mean':   r_vals.mean(),
            'r_std':    r_vals.std(),
            'is_circle': r_vals.std() < 1e-5 * r_vals.mean(),  # near-zero std → circle
            'n_pts':    len(pts)
        }

    # ── Get bounding box ──
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    bbox = (xmin, xmax, ymin, ymax)

    all_boundary_pts = np.vstack(all_pts_list) if all_pts_list else np.array([])

    gmsh.finalize()

    print(f"Extracted {len(boundaries)} boundary curves")
    print(f"Bounding box: X=[{xmin:.4f}, {xmax:.4f}]  "
          f"Y=[{ymin:.4f}, {ymax:.4f}]")

    return boundaries, all_boundary_pts, bbox, curve_info


# ─────────────────────────────────────────────────────────────
# 3. CLASSIFY CURVES → ROTOR REGIONS
#    Automatically identifies shaft, OD, magnet boundaries
# ─────────────────────────────────────────────────────────────

def classify_rotor_curves(curve_info, bbox):
    """
    Classifies each boundary curve into a rotor region based on
    radial position and circularity:
        'shaft'      — innermost circle
        'rotor_od'   — outermost circle
        'magnet'     — intermediate non-circular features
        'bridge'     — thin features near OD
        'other'      — everything else
    """
    xmin, xmax, ymin, ymax = bbox
    r_max = max(xmax, ymax) * 0.5

    # Sort curves by mean radius
    sorted_curves = sorted(curve_info.items(),
                            key=lambda x: x[1]['r_mean'])

    classification = {}
    r_values = [v['r_mean'] for _, v in sorted_curves]
    r_min_all = min(r_values)
    r_max_all = max(r_values)

    for tag, info in curve_info.items():
        r = info['r_mean']
        is_circle = info['is_circle']

        if is_circle and abs(r - r_min_all) < 0.05 * r_max_all:
            classification[tag] = 'shaft'
        elif is_circle and abs(r - r_max_all) < 0.05 * r_max_all:
            classification[tag] = 'rotor_od'
        elif r > 0.85 * r_max_all:
            classification[tag] = 'bridge'
        elif 0.4 * r_max_all < r < 0.9 * r_max_all:
            classification[tag] = 'magnet'
        else:
            classification[tag] = 'other'

    # Summary
    from collections import Counter
    counts = Counter(classification.values())
    print(f"Curve classification: {dict(counts)}")

    return classification


# ─────────────────────────────────────────────────────────────
# 4. BUILD BOUNDARY CORRESPONDENCE
#    Matches reference mesh boundary nodes → STP boundary curves
# ─────────────────────────────────────────────────────────────

def find_boundary_nodes(nodes, elements):
    """
    Identifies boundary nodes: nodes that appear in only one element
    (free edges). Works for 2D quad/tri meshes.
    """
    from collections import defaultdict
    edge_count = defaultdict(int)

    for eid, conn in elements.items():
        n = len(conn)
        for i in range(n):
            edge = tuple(sorted([conn[i], conn[(i+1) % n]]))
            edge_count[edge] += 1

    boundary_node_ids = set()
    for edge, count in edge_count.items():
        if count == 1:   # boundary edge
            boundary_node_ids.update(edge)

    return boundary_node_ids


def build_boundary_correspondence(nodes, boundary_node_ids,
                                   stp_all_boundary_pts):
    """
    For each boundary node in the reference mesh, finds the
    closest point on the STP target boundary.
    Returns arrays suitable for RBF morphing.
    """
    bnd_ids  = list(boundary_node_ids)
    old_bnd  = np.array([[nodes[n][0], nodes[n][1]] for n in bnd_ids])

    # KD-tree for fast nearest-neighbor lookup on STP boundary
    from scipy.spatial import cKDTree
    tree = cKDTree(stp_all_boundary_pts)

    distances, indices = tree.query(old_bnd, k=1)
    new_bnd = stp_all_boundary_pts[indices]

    max_disp = np.max(distances)
    avg_disp = np.mean(distances)
    print(f"Boundary correspondence: avg displacement={avg_disp:.4f}, "
          f"max displacement={max_disp:.4f}")

    return old_bnd, new_bnd, bnd_ids


# ─────────────────────────────────────────────────────────────
# 5. RBF MESH MORPHING
# ─────────────────────────────────────────────────────────────

def morph_nodes_rbf(nodes, old_bnd, new_bnd):
    """
    Applies RBF interpolation: boundary displacements → all nodes.
    """
    displacements = new_bnd - old_bnd

    rbf_x = RBFInterpolator(old_bnd, displacements[:, 0],
                             kernel='thin_plate_spline', smoothing=0)
    rbf_y = RBFInterpolator(old_bnd, displacements[:, 1],
                             kernel='thin_plate_spline', smoothing=0)

    all_coords = np.array([[v[0], v[1]] for v in nodes.values()])
    node_ids   = list(nodes.keys())

    dx = rbf_x(all_coords)
    dy = rbf_y(all_coords)

    morphed_nodes = {}
    for i, nid in enumerate(node_ids):
        z = nodes[nid][2] if len(nodes[nid]) > 2 else 0.0
        morphed_nodes[nid] = [
            nodes[nid][0] + dx[i],
            nodes[nid][1] + dy[i],
            z
        ]

    return morphed_nodes


# ─────────────────────────────────────────────────────────────
# 6. VALIDATION
# ─────────────────────────────────────────────────────────────

def validate_mesh(original_nodes, morphed_nodes, elements):
    assert len(original_nodes) == len(morphed_nodes), \
        f"Node count mismatch: {len(original_nodes)} vs {len(morphed_nodes)}"

    inverted, min_jac = 0, float('inf')

    for eid, conn in elements.items():
        if len(conn) < 4:
            continue  # skip non-quads
        pts = [morphed_nodes[n][:2] for n in conn[:4]]

        # Approximate Jacobian at element center
        v1 = np.array(pts[1]) - np.array(pts[0])
        v2 = np.array(pts[3]) - np.array(pts[0])
        jac = float(np.cross(v1, v2))
        min_jac = min(min_jac, jac)

        if jac <= 0:
            inverted += 1

    print(f"Validation → Nodes: {len(morphed_nodes)}  |  "
          f"Elements: {len(elements)}  |  "
          f"Inverted: {inverted}  |  Min Jacobian: {min_jac:.6f}")

    return inverted == 0


# ─────────────────────────────────────────────────────────────
# 7. WRITE .inp
# ─────────────────────────────────────────────────────────────

def write_inp(filepath, nodes, elements, node_sets, elem_sets,
              elem_type='CPS4'):
    with open(filepath, 'w') as f:
        f.write('** Morphed mesh — topology from reference, '
                'coordinates from STP target\n\n')

        f.write('*NODE\n')
        for nid in sorted(nodes.keys()):
            c = nodes[nid]
            f.write(f'  {nid},  {c[0]:.10f},  {c[1]:.10f},  {c[2]:.10f}\n')

        f.write(f'\n*ELEMENT, TYPE={elem_type}\n')
        for eid in sorted(elements.keys()):
            conn = ', '.join(str(n) for n in elements[eid])
            f.write(f'  {eid},  {conn}\n')

        for sname, nids in node_sets.items():
            f.write(f'\n*NSET, NSET={sname}\n')
            for i in range(0, len(nids), 8):
                f.write('  ' + ', '.join(str(n) for n in nids[i:i+8]) + '\n')

        for sname, eids in elem_sets.items():
            f.write(f'\n*ELSET, ELSET={sname}\n')
            for i in range(0, len(eids), 8):
                f.write('  ' + ', '.join(str(e) for e in eids[i:i+8]) + '\n')

    print(f"Written: {filepath}")


# ─────────────────────────────────────────────────────────────
# 8. MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ── Inputs ──────────────────────────────────────────────
    ref_inp   = 'reference_rotor.inp'
    stp_files = [
        ('variant_A.stp', 'variant_A_morphed.inp'),
        ('variant_B.stp', 'variant_B_morphed.inp'),
        ('variant_C.step','variant_C_morphed.inp'),
    ]
    scale_factor = 1e-3   # mm → m; set 1.0 if STP is already in meters

    # ── Load reference mesh ─────────────────────────────────
    print("=== Loading reference mesh ===")
    nodes, elements, node_sets, elem_sets = parse_inp(ref_inp)
    print(f"Reference: {len(nodes)} nodes, {len(elements)} elements")

    # Identify boundary nodes once (topology doesn't change)
    boundary_node_ids = find_boundary_nodes(nodes, elements)
    print(f"Boundary nodes detected: {len(boundary_node_ids)}")

    # ── Process each STP target ─────────────────────────────
    for stp_path, out_path in stp_files:
        print(f"\n{'='*55}")
        print(f"Target STP: {stp_path}")

        # 1. Import STP & extract boundaries
        boundaries, all_bnd_pts, bbox, curve_info = \
            extract_boundary_from_stp(stp_path,
                                       n_samples=300,
                                       scale_factor=scale_factor)

        # 2. Classify curves (optional — useful for debugging)
        classification = classify_rotor_curves(curve_info, bbox)

        # 3. Build boundary correspondence
        old_bnd, new_bnd, bnd_ids = build_boundary_correspondence(
            nodes, boundary_node_ids, all_bnd_pts)

        # 4. Morph all nodes via RBF
        morphed_nodes = morph_nodes_rbf(nodes, old_bnd, new_bnd)

        # 5. Validate
        valid = validate_mesh(nodes, morphed_nodes, elements)
        if not valid:
            print("  ⚠ WARNING: inverted elements detected — "
                  "consider increasing RBF smoothing")

        # 6. Write output
        write_inp(out_path, morphed_nodes, elements,
                  node_sets, elem_sets)