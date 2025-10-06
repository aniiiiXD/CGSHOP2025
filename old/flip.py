#!/usr/bin/env python3
"""
Compute intermediate triangulations (shortest flip sequence) between two triangulations
stored in a JSON instance of the form you uploaded.

Usage:
    python flip_path.py path/to/example_ps_20_nt2_pfd5_random.json
"""

import json
import sys
from collections import defaultdict, deque
import heapq
import math
from typing import List, Tuple, Set, Dict

Point = Tuple[float, float]
Edge = Tuple[int, int]
Triangulation = Tuple[Edge, ...]  # sorted edges tuple used as a hashable key

# ---------- geometry helpers ----------

def orient(a: Point, b: Point, c: Point) -> float:
    """Signed area * 2: positive if a->b->c is CCW"""
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def is_convex_quad(a: Point, b: Point, c: Point, d: Point) -> bool:
    """
    Check if quadrilateral a-b-c-d (in that order) is strictly convex.
    We accept convex if all consecutive triples have same orientation (no collinear).
    """
    s1 = orient(a,b,c)
    s2 = orient(b,c,d)
    s3 = orient(c,d,a)
    s4 = orient(d,a,b)
    # require none are zero (no degeneracy) and signs consistent
    if abs(s1) < 1e-9 or abs(s2) < 1e-9 or abs(s3) < 1e-9 or abs(s4) < 1e-9:
        return False
    return (s1>0 and s2>0 and s3>0 and s4>0) or (s1<0 and s2<0 and s3<0 and s4<0)

# ---------- triangulation helpers ----------

def canonical_edge(e: Edge) -> Edge:
    u,v = e
    return (u,v) if u < v else (v,u)

def triangulation_to_sets(edges: List[List[int]]) -> Set[Edge]:
    return set(canonical_edge((e[0], e[1])) for e in edges)

def build_triangles_from_edges(edges: Set[Edge]) -> Set[Tuple[int,int,int]]:
    """
    Reconstruct triangles: a triple (i,j,k) is a triangle iff all three edges present.
    This is O(n^3) worst-case but for n~20-50 it's fine. For larger inputs use adjacency-based reconstruction.
    """
    print(f"[DEBUG] Building triangles from {len(edges)} edges")
    print(f"[DEBUG] Edge set: {edges}")
    
    vertices = set()
    for u,v in edges:
        vertices.add(u); vertices.add(v)
    verts = sorted(vertices)
    print(f"[DEBUG] Found {len(verts)} unique vertices: {verts}")
    
    tris = set()
    n = len(verts)
    vs = verts
    idx = {v:i for i,v in enumerate(vs)}
    print(f"[DEBUG] Vertex index mapping: {idx}")
    
    # we will iterate over triples of present vertex labels
    vs_list = vs
    L = len(vs_list)
    print(f"[DEBUG] Checking all possible {L} choose 3 = {L*(L-1)*(L-2)//6} triangles")
    
    triangle_count = 0
    for i in range(L):
        for j in range(i+1, L):
            for k in range(j+1, L):
                a,b,c = vs_list[i], vs_list[j], vs_list[k]
                edge_ab = canonical_edge((a,b))
                edge_bc = canonical_edge((b,c))
                edge_ac = canonical_edge((a,c))
                has_ab = edge_ab in edges
                has_bc = edge_bc in edges
                has_ac = edge_ac in edges
                
                if has_ab and has_bc and has_ac:
                    triangle = tuple(sorted((a,b,c)))
                    print(f"[DEBUG] Found triangle {triangle} with edges: {edge_ab}, {edge_bc}, {edge_ac}")
                    tris.add(triangle)
                    triangle_count += 1
    
    print(f"[DEBUG] Total triangles found: {triangle_count}")
    return tris

def edge_adjacent_triangles(tris: Set[Tuple[int,int,int]]) -> Dict[Edge, List[Tuple[int,int,int]]]:
    adj = defaultdict(list)
    for tri in tris:
        a,b,c = tri
        edges = [canonical_edge((a,b)), canonical_edge((b,c)), canonical_edge((a,c))]
        for e in edges:
            adj[e].append(tri)
    return adj

def flippable_edges(edges: Set[Edge], points: List[Point]) -> List[Tuple[Edge, Edge]]:
    """
    Return list of (old_edge, new_edge) pairs that are flippable in the triangulation given by edges.
    old_edge and new_edge are canonical edge tuples.
    """
    tris = build_triangles_from_edges(edges)
    adj = edge_adjacent_triangles(tris)
    flips = []
    for e, tri_list in adj.items():
        if len(tri_list) != 2:
            continue  # boundary / non-flippable if not exactly two adjacent triangles
        t1, t2 = tri_list
        # find the opposite vertices to edge e=(u,v)
        u,v = e
        # t1 and t2 are triples; the opposite vertices are those in each triangle not u or v
        opp1 = next(x for x in t1 if x!=u and x!=v)
        opp2 = next(x for x in t2 if x!=u and x!=v)
        # new diagonal would be (opp1, opp2)
        # Need to check convexity of the quadrilateral.
        # The quad vertices in order can be (u, opp1, v, opp2) or some rotation; using this order works
        A = points[u]; B = points[opp1]; C = points[v]; D = points[opp2]
        if is_convex_quad(A,B,C,D):
            new_e = canonical_edge((opp1, opp2))
            # avoid degenerate case where new_e equals old e or already exists
            if new_e != e:
                flips.append((e, new_e))
        else:
            # try alternate ordering (u, opp2, v, opp1)
            A = points[u]; B = points[opp2]; C = points[v]; D = points[opp1]
            if is_convex_quad(A,B,C,D):
                new_e = canonical_edge((opp1, opp2))
                if new_e != e:
                    flips.append((e, new_e))
    # unique
    return list({(o,n) for (o,n) in flips})

def do_flip(edges: Set[Edge], old_edge: Edge, new_edge: Edge) -> Set[Edge]:
    new_edges = set(edges)
    if old_edge in new_edges:
        new_edges.remove(old_edge)
        new_edges.add(new_edge)
    return new_edges

# ---------- A* search over triangulations ----------

def heuristic_num_diff(curr_edges: Set[Edge], target_edges: Set[Edge]) -> int:
    # lower bound: number of diagonals in curr not in target
    diff = sum(1 for e in curr_edges if e not in target_edges)
    return diff

def triangulation_key(edges: Set[Edge]) -> Triangulation:
    return tuple(sorted(edges))

def a_star_flip_path(points: List[Point],
                     start_edges: Set[Edge],
                     target_edges: Set[Edge],
                     max_nodes: int = 200000):
    """
    A* search: returns list of triangulation edge-sets from start to target (inclusive),
    or None if no path found within node limit.
    """
    start_key = triangulation_key(start_edges)
    target_key = triangulation_key(target_edges)

    open_heap = []
    gscore = {start_key: 0}
    h0 = heuristic_num_diff(start_edges, target_edges)
    f0 = gscore[start_key] + h0
    heapq.heappush(open_heap, (f0, h0, start_key))
    came_from = {}

    visited = set()
    nodes_expanded = 0

    # map key -> edges set for quick regeneration
    key_to_edges = {start_key: start_edges, target_key: target_edges}

    while open_heap:
        f, h, key = heapq.heappop(open_heap)
        if key in visited:
            continue
        visited.add(key)
        nodes_expanded += 1
        if nodes_expanded > max_nodes:
            # bail out for large instances
            return None, f"timeout_nodes_limit (expanded {nodes_expanded})"
        edges = key_to_edges[key]

        if key == target_key:
            # reconstruct path
            path_keys = []
            cur = key
            while cur in came_from:
                path_keys.append(cur)
                cur = came_from[cur]
            path_keys.append(start_key)
            path_keys.reverse()
            path = [list(key_to_edges[k]) for k in path_keys]
            return path, None

        # generate neighbors (single flips)
        flips = flippable_edges(edges, points)
        for old_e, new_e in flips:
            new_edges = do_flip(edges, old_e, new_e)
            new_key = triangulation_key(new_edges)
            if new_key not in key_to_edges:
                key_to_edges[new_key] = new_edges
            tentative_g = gscore[key] + 1
            if new_key in gscore and tentative_g >= gscore[new_key]:
                continue
            # record best
            came_from[new_key] = key
            gscore[new_key] = tentative_g
            h_new = heuristic_num_diff(new_edges, target_edges)
            f_new = tentative_g + h_new
            heapq.heappush(open_heap, (f_new, h_new, new_key))

    return None, "no-path-found"

# ---------- I/O and glue ----------

def load_instance(path: str):
    with open(path, 'r') as f:
        data = json.load(f)
    points = list(zip(data['points_x'], data['points_y']))
    triangulations = data.get('triangulations', [])
    if len(triangulations) < 2:
        raise ValueError("JSON must contain at least two triangulations in 'triangulations'")
    t0_edges = triangulation_to_sets(triangulations[0])
    t1_edges = triangulation_to_sets(triangulations[1])
    return points, t0_edges, t1_edges

def main(json_path: str):
    points, start_edges, target_edges = load_instance(json_path)
    print("n points:", len(points))
    print("start edges:", len(start_edges), "target edges:", len(target_edges))

    if start_edges == target_edges:
        print("Triangulations already identical — no flips needed.")
        return [list(start_edges)]

    path, err = a_star_flip_path(points, start_edges, target_edges, max_nodes=500000)
    if path is None:
        print("Failed to find path:", err)
        return None

    print(f"Found path with {len(path)-1} flips (intermediate triangulations count = {len(path)})")
    # Optionally print edges of each intermediate triangulation
    for i, tri in enumerate(path):
        print(f"Step {i}: {len(tri)} edges")
    # Return path (list of list-of-edges)
    return path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python flip_path.py path/to/instance.json")
        sys.exit(1)
    inst = sys.argv[1]
    path = main(inst)
    # If you want to write output JSON:
    if path:
        out = {
            "intermediate_triangulations": [[list(e) for e in tri] for tri in path]
        }
        out_path = inst + ".path.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print("Wrote intermediate triangulations to", out_path)
