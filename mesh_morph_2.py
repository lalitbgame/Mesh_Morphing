#!/usr/bin/env python3
"""
Morph a 2D Abaqus rotor mesh (.inp) to a target planar STEP rotor profile,
preserving node IDs, element IDs and element connectivity.

This version uses boundary projection + Laplacian/harmonic mesh morphing.
It is safer for rotor slots than global RBF morphing because the inner slot
boundaries are imposed as fixed Dirichlet constraints and the interior node
motion is solved from the mesh graph.

Dependencies:
    pip install numpy scipy

Usage:
    python morph_rotor_mesh_to_step_laplacian.py Reference.inp mod_rotor.stp NewRotor_laplacian_morphed.inp

Assumptions:
- 2D Abaqus mesh with shell/plane-stress/plane-strain elements such as CPS4R/CPE4R.
- STEP contains one planar face with the same boundary topology as the reference mesh.
- This is a morphing workflow, not a remeshing workflow.
"""
from __future__ import annotations

import argparse, math, re, itertools
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve


@dataclass
class InpMesh:
    lines: List[str]
    nodes: Dict[int, np.ndarray]
    elements: List[Tuple[int, List[int]]]
    element_type: str
    node_line_index: Dict[int, int]


def parse_inp(path: Path) -> InpMesh:
    lines = path.read_text(errors="ignore").splitlines()
    nodes, node_line_index, elements = {}, {}, []
    element_type, mode = "", None
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("**"):
            continue
        low = s.lower()
        if low.startswith("*node"):
            mode = "node"; continue
        if low.startswith("*element"):
            mode = "element"
            m = re.search(r"type=([^,\s]+)", s, re.I)
            element_type = m.group(1) if m else ""
            continue
        if s.startswith("*"):
            mode = None; continue
        if mode == "node":
            parts = [p.strip() for p in s.split(",") if p.strip()]
            if len(parts) >= 3:
                nid = int(parts[0])
                vals = [float(v) for v in parts[1:]]
                if len(vals) == 2: vals.append(0.0)
                nodes[nid] = np.asarray(vals[:3], dtype=float)
                node_line_index[nid] = i
        elif mode == "element":
            parts = [p.strip() for p in s.split(",") if p.strip()]
            if len(parts) >= 4:
                elements.append((int(parts[0]), [int(v) for v in parts[1:]]))
    if not nodes or not elements:
        raise ValueError("Could not parse nodes/elements from INP file.")
    return InpMesh(lines, nodes, elements, element_type, node_line_index)


def element_edges(conn: List[int]) -> List[Tuple[int, int]]:
    return [tuple(sorted((a, b))) for a, b in zip(conn, conn[1:] + conn[:1])]


def boundary_loops_from_elements(mesh: InpMesh) -> List[List[int]]:
    edge_count = Counter()
    for _, conn in mesh.elements:
        for e in element_edges(conn):
            edge_count[e] += 1
    adj = defaultdict(list)
    for (a, b), c in edge_count.items():
        if c == 1:
            adj[a].append(b); adj[b].append(a)
    for n, nbrs in adj.items():
        if len(nbrs) != 2:
            raise ValueError(f"Boundary is not manifold near node {n}; node has {len(nbrs)} boundary neighbours.")
    loops, used_edges = [], set()
    for start in sorted(adj):
        for first in adj[start]:
            e0 = tuple(sorted((start, first)))
            if e0 in used_edges:
                continue
            loop = [start]
            prev, cur = start, first
            used_edges.add(e0)
            while cur != start:
                loop.append(cur)
                nxts = [n for n in adj[cur] if n != prev]
                if not nxts: break
                nxt = nxts[0]
                e = tuple(sorted((cur, nxt)))
                if e in used_edges and nxt != start: break
                used_edges.add(e)
                prev, cur = cur, nxt
                if len(loop) > len(adj) + 10:
                    raise RuntimeError("Boundary tracing failed.")
            # avoid duplicate reverse loops
            ids = set(loop)
            if not any(ids == set(existing) for existing in loops):
                loops.append(loop)
    return loops


class StepPlanarParser:
    def __init__(self, path: Path):
        self.text = path.read_text(errors="ignore")
        self.entities = self._read_entities()

    def _read_entities(self):
        return {int(m.group(1)): (m.group(2), m.group(3))
                for m in re.finditer(r"#(\d+)\s*=\s*([A-Z0-9_]+)\((.*?)\);", self.text, re.S)}
    @staticmethod
    def _refs(arg: str): return [int(x[1:]) for x in re.findall(r"#\d+", arg)]
    @staticmethod
    def _bools(arg: str): return [x == "T" for x in re.findall(r"\.(T|F)\.", arg)]
    @staticmethod
    def _floats(arg: str): return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", arg)]

    def cartesian_point(self, eid: int):
        typ, arg = self.entities[eid]
        if typ != "CARTESIAN_POINT": raise ValueError(f"#{eid} is not CARTESIAN_POINT")
        vals = self._floats(arg)
        return np.asarray(vals[-3:] if len(vals) >= 3 else [vals[0], vals[1], 0.0], dtype=float)

    def direction(self, eid: int):
        typ, arg = self.entities[eid]
        if typ != "DIRECTION": raise ValueError(f"#{eid} is not DIRECTION")
        vals = self._floats(arg)
        return np.asarray(vals[-3:] if len(vals) >= 3 else [vals[0], vals[1], 0.0], dtype=float)

    def vertex_point(self, eid: int):
        return self.cartesian_point(self._refs(self.entities[eid][1])[0])

    def axis2_placement_3d(self, eid: int):
        r = self._refs(self.entities[eid][1])
        loc = self.cartesian_point(r[0])
        z = self.direction(r[1]) if len(r) > 1 else np.array([0.,0.,1.])
        x = self.direction(r[2]) if len(r) > 2 else np.array([1.,0.,0.])
        x = x / np.linalg.norm(x); z = z / np.linalg.norm(z)
        y = np.cross(z, x)
        if np.linalg.norm(y) < 1e-12: y = np.array([0.,1.,0.])
        y = y / np.linalg.norm(y)
        return loc, x, y, z

    def oriented_edge_points(self, oe_id: int, max_seg_len=0.15):
        refs = self._refs(self.entities[oe_id][1])
        edge_curve_id = refs[-1]
        orientation = self._bools(self.entities[oe_id][1])[-1]
        er = self._refs(self.entities[edge_curve_id][1])
        v1, v2, curve_id = er[0], er[1], er[2]
        curve_sense = self._bools(self.entities[edge_curve_id][1])[-1]
        p1, p2 = self.vertex_point(v1), self.vertex_point(v2)
        start, end = (p1, p2) if orientation else (p2, p1)
        ctyp, carg = self.entities[curve_id]
        if ctyp == "LINE":
            return np.vstack([start[:2], end[:2]])
        if ctyp == "CIRCLE":
            rr = self._refs(carg)
            loc, xdir, ydir, _ = self.axis2_placement_3d(rr[0])
            radius = self._floats(carg)[-1]
            def angle(p):
                v = p - loc
                return math.atan2(float(np.dot(v, ydir)), float(np.dot(v, xdir)))
            a0, a1 = angle(start), angle(end)
            ccw = curve_sense if orientation else (not curve_sense)
            if ccw:
                while a1 <= a0: a1 += 2*math.pi
            else:
                while a1 >= a0: a1 -= 2*math.pi
            n = max(3, int(abs(a1-a0)*radius/max_seg_len))
            aa = np.linspace(a0, a1, n+1)
            return np.asarray([(loc + radius*(math.cos(a)*xdir + math.sin(a)*ydir))[:2] for a in aa])
        raise NotImplementedError(f"Unsupported STEP curve type {ctyp}; add support or export STEP with LINE/CIRCLE curves.")

    def loops(self, max_seg_len=0.15):
        faces = [eid for eid,(typ,_) in self.entities.items() if typ == "FACE_SURFACE"]
        if not faces: raise ValueError("No FACE_SURFACE found in STEP.")
        face = max(faces, key=lambda eid: sum(1 for r in self._refs(self.entities[eid][1]) if self.entities.get(r,("",))[0] == "FACE_BOUND"))
        bounds = [r for r in self._refs(self.entities[face][1]) if self.entities.get(r,("",))[0] == "FACE_BOUND"]
        out = []
        for fb in bounds:
            edge_loop = self._refs(self.entities[fb][1])[0]
            oes = self._refs(self.entities[edge_loop][1])
            pts = []
            for oe in oes:
                ep = self.oriented_edge_points(oe, max_seg_len)
                pts.extend(ep.tolist() if not pts else ep[1:].tolist())
            arr = np.asarray(pts, dtype=float)
            if np.linalg.norm(arr[0] - arr[-1]) > 1e-7:
                arr = np.vstack([arr, arr[0]])
            out.append(arr)
        return out


def polygon_area(points):
    p = points[:, :2]
    return 0.5 * float(np.sum(p[:,0]*np.roll(p[:,1],-1) - np.roll(p[:,0],-1)*p[:,1]))


def loop_points(loop_ids, nodes):
    p = np.asarray([nodes[n][:2] for n in loop_ids], dtype=float)
    return np.vstack([p, p[0]])


def loop_signature(points):
    p = points[:-1] if np.allclose(points[0], points[-1]) else points
    mn, mx = p.min(axis=0), p.max(axis=0)
    return np.r_[p.mean(axis=0), mx-mn, math.sqrt(abs(polygon_area(points)))]


def match_loops(ref_pts, target_pts):
    if len(ref_pts) != len(target_pts):
        raise ValueError(f"Topology mismatch: reference has {len(ref_pts)} loops, STEP has {len(target_pts)} loops.")
    r, t = np.vstack([loop_signature(p) for p in ref_pts]), np.vstack([loop_signature(p) for p in target_pts])
    scale = np.maximum(np.ptp(np.vstack([r, t]), axis=0), 1e-9)
    cost = np.linalg.norm((r[:,None,:] - t[None,:,:]) / scale, axis=2)
    best, bestc = None, 1e99
    for perm in itertools.permutations(range(len(target_pts))):
        c = sum(cost[i, perm[i]] for i in range(len(perm)))
        if c < bestc: best, bestc = list(perm), c
    return best



def cumulative_arc(points):
    pts = points[:, :2]
    d = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(d)]
    if s[-1] < 1e-15:
        raise ValueError("Zero length loop")
    return s / s[-1]

def point_at_fraction(points, frac):
    frac = frac % 1.0
    pts = points[:, :2]
    s = cumulative_arc(pts)
    j = np.searchsorted(s, frac, side="right") - 1
    j = max(0, min(j, len(s)-2))
    den = s[j+1] - s[j]
    u = 0.0 if den < 1e-15 else (frac - s[j]) / den
    return (1-u)*pts[j] + u*pts[j+1]

def best_loop_phase(ref_closed, tgt_closed, samples=96):
    # Find orientation and cyclic phase that makes target loop correspond to reference loop.
    # This prevents opposite slot sides from being paired.
    fr = np.linspace(0, 1, samples, endpoint=False)
    rp = np.asarray([point_at_fraction(ref_closed, f) for f in fr])
    # Try forward/reverse target and many phase shifts.
    best = (1e99, 1, 0.0)
    for orient in (1, -1):
        for shift in np.linspace(0, 1, samples, endpoint=False):
            tp = np.asarray([point_at_fraction(tgt_closed, shift + orient*f) for f in fr])
            c = np.mean(np.sum((rp - tp)**2, axis=1))
            if c < best[0]:
                best = (c, orient, shift)
    return best[1], best[2]

def build_boundary_displacements_arc_aligned(mesh, ref_loop_ids, target_loops, mapping):
    fixed = {}
    for loop_ids, tgt_index in zip(ref_loop_ids, mapping):
        ref_closed = loop_points(loop_ids, mesh.nodes)
        tgt_closed = target_loops[tgt_index]
        ref_s = cumulative_arc(ref_closed)
        orient, shift = best_loop_phase(ref_closed, tgt_closed, samples=96)
        for k, nid in enumerate(loop_ids):
            src = mesh.nodes[nid][:2]
            dst = point_at_fraction(tgt_closed, shift + orient*ref_s[k])
            fixed[nid] = dst - src
    return fixed


def closest_point_on_polyline(p, poly_closed):
    pts = poly_closed[:, :2]
    best_q, best_d2 = None, 1e99
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        den = float(np.dot(ab, ab))
        if den < 1e-20:
            q = a
        else:
            u = max(0.0, min(1.0, float(np.dot(p-a, ab)/den)))
            q = a + u*ab
        d2 = float(np.dot(p-q, p-q))
        if d2 < best_d2:
            best_d2, best_q = d2, q
    return best_q


def build_boundary_displacements(mesh, ref_loop_ids, target_loops, mapping, method="closest"):
    fixed = {}
    for loop_ids, tgt_index in zip(ref_loop_ids, mapping):
        tgt = target_loops[tgt_index]
        # For the same topology and near-aligned CAD, closest projection is much safer than arbitrary
        # normalized arc-length start-point matching.
        for nid in loop_ids:
            src = mesh.nodes[nid][:2]
            dst = closest_point_on_polyline(src, tgt)
            fixed[nid] = dst - src
    return fixed


def mesh_adjacency(mesh):
    adj = defaultdict(set)
    for _, conn in mesh.elements:
        # include perimeter edges and diagonals for better smoothing in quads
        for a, b in zip(conn, conn[1:] + conn[:1]):
            adj[a].add(b); adj[b].add(a)
        if len(conn) == 4:
            for a, b in [(conn[0], conn[2]), (conn[1], conn[3])]:
                adj[a].add(b); adj[b].add(a)
    return adj


def harmonic_morph(mesh, fixed_disp):
    node_ids = sorted(mesh.nodes)
    id_to_i = {nid:i for i,nid in enumerate(node_ids)}
    free = [nid for nid in node_ids if nid not in fixed_disp]
    fidx = {nid:i for i,nid in enumerate(free)}
    adj = mesh_adjacency(mesh)
    A = lil_matrix((len(free), len(free)), dtype=float)
    bx = np.zeros(len(free)); by = np.zeros(len(free))
    for nid in free:
        row = fidx[nid]
        nbrs = list(adj[nid])
        A[row, row] = 1.0
        if not nbrs: continue
        w = 1.0 / len(nbrs)
        for nb in nbrs:
            if nb in fixed_disp:
                bx[row] += w * fixed_disp[nb][0]
                by[row] += w * fixed_disp[nb][1]
            else:
                A[row, fidx[nb]] -= w
    A = csr_matrix(A)
    ux = spsolve(A, bx) if len(free) else np.array([])
    uy = spsolve(A, by) if len(free) else np.array([])
    new_nodes = {}
    for nid in node_ids:
        p = mesh.nodes[nid].copy()
        if nid in fixed_disp:
            d = fixed_disp[nid]
        else:
            k = fidx[nid]
            d = np.array([ux[k], uy[k]])
        p[:2] += d
        new_nodes[nid] = p
    return new_nodes


def quad_signed_area(p):
    return polygon_area(np.vstack([p, p[0]]))


def quality_report(mesh, nodes):
    neg, zero, amin, amax = 0, 0, 1e99, -1e99
    for _, conn in mesh.elements:
        p = np.asarray([nodes[n][:2] for n in conn])
        a = quad_signed_area(p)
        aa = abs(a)
        amin = min(amin, aa); amax = max(amax, aa)
        if aa < 1e-10: zero += 1
        # Reference orientation may vary; count only severe negative relative to original outside omitted here.
    return amin, amax, zero


def write_inp(mesh, new_nodes, out_path, keep_2d=True):
    lines = mesh.lines.copy()
    for nid, i in mesh.node_line_index.items():
        p = new_nodes[nid]
        if keep_2d:
            lines[i] = f"{nid:8d}, {p[0]: .9f}, {p[1]: .9f}"
        else:
            lines[i] = f"{nid:8d}, {p[0]: .9f}, {p[1]: .9f}, {p[2]: .9f}"
    out_path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference_inp", type=Path)
    ap.add_argument("target_step", type=Path)
    ap.add_argument("output_inp", type=Path)
    ap.add_argument("--step-segment", type=float, default=0.12)
    args = ap.parse_args()

    mesh = parse_inp(args.reference_inp)
    ref_ids = boundary_loops_from_elements(mesh)
    ref_pts = [loop_points(l, mesh.nodes) for l in ref_ids]
    target = StepPlanarParser(args.target_step).loops(args.step_segment)
    mapping = match_loops(ref_pts, target)
    fixed = build_boundary_displacements_arc_aligned(mesh, ref_ids, target, mapping)
    new_nodes = harmonic_morph(mesh, fixed)
    write_inp(mesh, new_nodes, args.output_inp)
    amin, amax, zero = quality_report(mesh, new_nodes)
    print("Laplacian mesh morph complete.")
    print(f"Input nodes        : {len(mesh.nodes)}")
    print(f"Input elements     : {len(mesh.elements)}")
    print(f"Element type       : {mesh.element_type}")
    print(f"Reference loops    : {len(ref_ids)}")
    print(f"STEP loops         : {len(target)}")
    print(f"Fixed boundary nodes: {len(fixed)}")
    print(f"Min abs elem area  : {amin:.6e}")
    print(f"Near-zero elements : {zero}")
    print(f"Output             : {args.output_inp}")

if __name__ == "__main__":
    main()
