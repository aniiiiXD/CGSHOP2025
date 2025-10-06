from __future__ import annotations
import json
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Dict, Optional, Iterable, Generator

Vertex = Tuple[float, float]
Edge = Tuple[int, int]  # undirected, normalized (min, max)

def norm_edge(u: int, v: int) -> Edge:
    return (u, v) if u < v else (v, u)

def angle_at(v: Vertex, w: Vertex) -> float:
    return math.atan2(w[1] - v[1], w[0] - v[0])

def orient(a: Vertex, b: Vertex, c: Vertex) -> float:
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def strict_opposite(x: float, y: float) -> bool:
    return x * y < 0.0

@dataclass
class PSLGTriangulation:
    vertices: List[Vertex]
    edges: Set[Edge]
    # Derived structures
    nbrs: Dict[int, List[int]] = field(init=False)  # neighbors sorted CCW around vertex
    nbr_pos: Dict[Tuple[int,int], int] = field(init=False)  # position of neighbor in CCW list

    def __post_init__(self):
        self._rebuild()

    def copy(self) -> "PSLGTriangulation":
        return PSLGTriangulation(vertices=list(self.vertices), edges=set(self.edges))

    def _rebuild(self):
        n = len(self.vertices)
        nbrs: Dict[int, List[int]] = {i: [] for i in range(n)}
        for (u, v) in self.edges:
            nbrs[u].append(v)
            nbrs[v].append(u)
        # Sort neighbors CCW by angle around each vertex
        self.nbrs = {}
        self.nbr_pos = {}
        for i in range(n):
            vi = self.vertices[i]
            sorted_n = sorted(nbrs[i], key=lambda j: angle_at(vi, self.vertices[j]))
            self.nbrs[i] = sorted_n
            for idx, j in enumerate(sorted_n):
                self.nbr_pos[(i, j)] = idx

    def has_edge(self, e: Edge) -> bool:
        return e in self.edges

    def _left_next(self, u: int, v: int) -> Optional[int]:
        # For oriented edge u->v, return neighbor w of v that is the next CCW after u,
        # which forms the triangle (u, v, w) on the left of (u->v)
        key = (v, u)
        if key not in self.nbr_pos or v not in self.nbrs:
            return None
        pos = self.nbr_pos[key]
        lst = self.nbrs[v]
        w = lst[(pos + 1) % len(lst)] if lst else None
        return w

    def _right_next(self, u: int, v: int) -> Optional[int]:
        # For oriented edge u->v, return neighbor w of v that is the previous CCW (i.e., clockwise) before u
        key = (v, u)
        if key not in self.nbr_pos or v not in self.nbrs:
            return None
        pos = self.nbr_pos[key]
        lst = self.nbrs[v]
        w = lst[(pos - 1) % len(lst)] if lst else None
        return w

    def opp_vertices_of_edge(self, e: Edge) -> Optional[Tuple[int,int]]:
        a, b = e
        # Two incident faces are (a,b,x) on left of a->b and (b,a,y) on left of b->a
        x = self._left_next(a, b)
        y = self._left_next(b, a)
        if x is None or y is None:
            return None
        # Validate that these close triangles exist (edges (a,x),(b,x),(a,y),(b,y))
        if (norm_edge(a, x) not in self.edges or
            norm_edge(b, x) not in self.edges or
            norm_edge(a, y) not in self.edges or
            norm_edge(b, y) not in self.edges):
            return None
        return (x, y)

    def is_flippable(self, e: Edge) -> bool:
        if e not in self.edges:
            return False
        opp = self.opp_vertices_of_edge(e)
        if not opp:
            return False
        a, b = e
        x, y = opp
        # Interior convex quadrilateral with non-collinearities
        pa, pb, px, py = self.vertices[a], self.vertices[b], self.vertices[x], self.vertices[y]
        o1 = orient(pa, pb, px)
        o2 = orient(pa, pb, py)
        o3 = orient(self.vertices[x], self.vertices[y], pa)
        o4 = orient(self.vertices[x], self.vertices[y], pb)
        if o1 == 0.0 or o2 == 0.0 or o3 == 0.0 or o4 == 0.0:
            return False
        if not (strict_opposite(o1, o2) and strict_opposite(o3, o4)):
            return False
        new_diag = norm_edge(x, y)
        if new_diag in self.edges:
            return False
        return True

    def other_diagonal(self, e: Edge) -> Optional[Edge]:
        opp = self.opp_vertices_of_edge(e)
        if not opp:
            return None
        x, y = opp
        return norm_edge(x, y)

    def flip(self, e: Edge) -> Tuple["PSLGTriangulation", Edge]:
        if not self.is_flippable(e):
            raise ValueError("Not flippable")
        x, y = self.opp_vertices_of_edge(e)
        T2 = self.copy()
        T2.edges.remove(e)
        new_e = norm_edge(x, y)
        T2.edges.add(new_e)
        T2._rebuild()
        return T2, new_e

    def neighbor_edges(self, e: Edge) -> List[Edge]:
        # Edges sharing a triangle with e: up to 4
        a, b = e
        res = set()
        opp = self.opp_vertices_of_edge(e)
        if not opp:
            return []
        x, y = opp
        res.add(norm_edge(a, x))
        res.add(norm_edge(b, x))
        res.add(norm_edge(a, y))
        res.add(norm_edge(b, y))
        res.discard(e)
        return list(res)

    def edges_signature(self) -> Tuple[Edge, ...]:
        return tuple(sorted(self.edges))

def changed_edges(Ti: PSLGTriangulation, Tf: PSLGTriangulation) -> Set[Edge]:
    return {e for e in Ti.edges if e not in Tf.edges}

def compositions(k: int, t: int) -> Iterable[Tuple[int, ...]]:
    if t == 1:
        if k >= 1:
            yield (k,)
        return
    for first in range(1, k - (t - 1) + 1):
        for rest in compositions(k - first, t - 1):
            yield (first,) + rest

# Action types
MOVE = "MOVE"
FLIP_MOVE = "FLIP_MOVE"
FLIP_PUSH_MOVE = "FLIP_PUSH_MOVE"
FLIP_JUMP = "FLIP_JUMP"
FLIP_JUMP_POP = "FLIP_JUMP_POP"

@dataclass(frozen=True)
class Action:
    kind: str
    move_idx: Optional[int] = None  # encodes which neighbor edge to move to when applicable

@dataclass
class DFSState:
    T: PSLGTriangulation
    e_curr: Edge
    flips_done: int
    actions_done: int
    flips_seq: List[Edge]
    stack: List[Edge]

def enumerate_actions_component_all(T_start: PSLGTriangulation,
                                    e_start: Edge,
                                    k_comp: int,
                                    actions_limit: Optional[int] = None
                                    ) -> Generator[List[Edge], None, None]:
    if actions_limit is None:
        actions_limit = 11 * k_comp

    seen: Set[Tuple] = set()

    def sig(state: DFSState) -> Tuple:
        return (state.T.edges_signature(), state.e_curr, state.flips_done,
                state.stack[-1] if state.stack else None, state.actions_done)

    def dfs(state: DFSState):
        if state.flips_done == k_comp:
            yield list(state.flips_seq)
            return
        if state.actions_done >= actions_limit:
            return
        s = sig(state)
        if s in seen:
            return
        seen.add(s)

        # MOVE
        neigh = state.T.neighbor_edges(state.e_curr)
        for e_next in neigh:
            next_state = DFSState(
                T=state.T,
                e_curr=e_next,
                flips_done=state.flips_done,
                actions_done=state.actions_done + 1,
                flips_seq=state.flips_seq,
                stack=list(state.stack),
            )
            yield from dfs(next_state)

        # Flips on current edge
        if state.T.is_flippable(state.e_curr):
            T2, new_diag = state.T.flip(state.e_curr)
            # FLIP_MOVE
            neigh2 = T2.neighbor_edges(state.e_curr)
            for e_next in neigh2:
                next_state = DFSState(
                    T=T2,
                    e_curr=e_next,
                    flips_done=state.flips_done + 1,
                    actions_done=state.actions_done + 1,
                    flips_seq=state.flips_seq + [state.e_curr],
                    stack=list(state.stack),
                )
                yield from dfs(next_state)
            # FLIP_PUSH_MOVE
            for e_next in neigh2:
                stk = list(state.stack)
                stk.append(new_diag)
                next_state = DFSState(
                    T=T2,
                    e_curr=e_next,
                    flips_done=state.flips_done + 1,
                    actions_done=state.actions_done + 1,
                    flips_seq=state.flips_seq + [state.e_curr],
                    stack=stk,
                )
                yield from dfs(next_state)
            # FLIP_JUMP
            if state.stack:
                next_state = DFSState(
                    T=T2,
                    e_curr=state.stack[-1],
                    flips_done=state.flips_done + 1,
                    actions_done=state.actions_done + 1,
                    flips_seq=state.flips_seq + [state.e_curr],
                    stack=list(state.stack),
                )
                yield from dfs(next_state)
            # FLIP_JUMP_POP
            if state.stack:
                stk = list(state.stack)
                top = stk.pop()
                next_state = DFSState(
                    T=T2,
                    e_curr=top,
                    flips_done=state.flips_done + 1,
                    actions_done=state.actions_done + 1,
                    flips_seq=state.flips_seq + [state.e_curr],
                    stack=stk,
                )
                yield from dfs(next_state)

    init = DFSState(T=T_start, e_curr=e_start, flips_done=0, actions_done=0, flips_seq=[], stack=[])
    yield from dfs(init)

def collect_all_intermediates_minimal(T1: PSLGTriangulation,
                                      T2: PSLGTriangulation,
                                      max_paths: int = 10000) -> Tuple[int, List[List[PSLGTriangulation]], Set[Tuple[Edge,...]]]:
    # Find minimal k by iterative deepening from LB = |E(T1)\E(T2)|
    LB = len(changed_edges(T1, T2))
    found_sequences: List[List[PSLGTriangulation]] = []
    unique_tris: Set[Tuple[Edge, ...]] = set()

    def run_for_k(k: int) -> bool:
        nonlocal found_sequences
        count = 0
        for t in range(1, k + 1):
            for parts in compositions(k, t):
                # Recursive product over components yielding all sequences
                def search_component(idx: int, Tc: PSLGTriangulation, collected: List[PSLGTriangulation]):
                    nonlocal count
                    if count >= max_paths:
                        return
                    if idx == t:
                        if Tc.edges == T2.edges:
                            found_sequences.append(list(collected))
                            for tri in collected:
                                unique_tris.add(tri.edges_signature())
                            count += 1
                        return
                    # pick an entry changed edge
                    O = list(changed_edges(Tc, T2))
                    if not O:
                        return
                    e0 = O[0]
                    for flips in enumerate_actions_component_all(Tc, e0, parts[idx], actions_limit=11*parts[idx]):
                        Tcur = Tc
                        inters = []
                        ok = True
                        for e in flips:
                            if not Tcur.is_flippable(e):
                                ok = False
                                break
                            Tcur, _ = Tcur.flip(e)
                            inters.append(Tcur)
                        if ok:
                            search_component(idx + 1, Tcur, collected + inters)
                search_component(0, T1, [T1])
                if count >= max_paths:
                    return True
        return len(found_sequences) > 0

    k = LB
    while True:
        if run_for_k(k):
            return k, found_sequences, unique_tris
        k += 1
        # Safety cap for practical runs; in triangulations of 20 points, k is typically moderate
        if k > LB + 10:
            return -1, [], set()

def load_instance(path: str) -> Tuple[List[Vertex], List[List[Edge]]]:
    with open(path, "r") as f:
        data = json.load(f)
    xs = data["points_x"]
    ys = data["points_y"]
    verts = [(float(xs[i]), float(ys[i])) for i in range(len(xs))]
    tris_list = []
    for tri_edges in data["triangulations"]:
        edges = set()
        for u, v in tri_edges:
            edges.add(norm_edge(int(u), int(v)))
        tris_list.append(list(edges))
    return verts, tris_list

def build_triangulation(vertices: List[Vertex], edges_list: List[Edge]) -> PSLGTriangulation:
    return PSLGTriangulation(vertices=vertices, edges=set(edges_list))

if __name__ == "__main__":
    # Load and parse the input file
    path = "example_ps_20_nt2_pfd5_random.json"
    print(f"\n{'='*80}\nLoading instance from: {path}\n{'='*80}")
    V, tri_lists = load_instance(path)
    
    print(f"\n{'*'*50}\nTRIANGULATION COMPARISON\n{'*'*50}")
    print(f"Number of vertices: {len(V)}")
    print(f"Number of triangulations: {len(tri_lists)}")
    
    # Build both triangulations
    t1_edges = tri_lists[0]
    t2_edges = tri_lists[1]
    T1 = build_triangulation(V, t1_edges)
    T2 = build_triangulation(V, t2_edges)
    
    # Print basic info about the triangulations
    def print_triangulation_info(name: str, T: PSLGTriangulation):
        print(f"\n{name}:")
        print(f"  Number of edges: {len(T.edges)}")
        print(f"  Number of triangles: ~{len(T.edges) - len(V) + 1}")
        print(f"  Sample edges (first 5): {list(T.edges)[:5]}")
    
    print_triangulation_info("First triangulation (T1)", T1)
    print_triangulation_info("Second triangulation (T2)", T2)
    
    # Calculate and print differences
    edges_only_in_T1 = T1.edges - T2.edges
    edges_only_in_T2 = T2.edges - T1.edges
    common_edges = T1.edges & T2.edges
    
    print(f"\n{'*'*50}\nDIFFERENCES\n{'*'*50}")
    print(f"Edges only in T1: {len(edges_only_in_T1)}")
    print(f"Edges only in T2: {len(edges_only_in_T2)}")
    print(f"Common edges: {len(common_edges)}")
    print(f"Sample edges only in T1: {list(edges_only_in_T1)[:5]}")
    print(f"Sample edges only in T2: {list(edges_only_in_T2)[:5]}")
    
    # Find minimal flip sequences
    print(f"\n{'*'*50}\nFINDING MINIMAL FLIP SEQUENCES\n{'*'*50}")
    print("Starting search for minimal flip sequences...")
    
    import time
    start_time = time.time()
    k_min, sequences, unique_tris = collect_all_intermediates_minimal(T1, T2, max_paths=5000)
    elapsed_time = time.time() - start_time
    
    print(f"\n{'*'*50}\nRESULTS\n{'*'*50}")
    print(f"Minimal flip distance (k): {k_min}")
    print(f"Number of shortest sequences found: {len(sequences)}")
    print(f"Unique intermediate triangulations: {len(unique_tris)}")
    print(f"Computation time: {elapsed_time:.2f} seconds")
    
    # Print detailed information about the first sequence if available
    if sequences:
        print(f"\n{'*'*50}\nFIRST SEQUENCE ANALYSIS\n{'*'*50}")
        seq = sequences[0]
        print(f"Sequence length: {len(seq) - 1} flips")
        print(f"Number of intermediate states: {len(seq)}")
        
        # Print the flip sequence with edge details
        print("\nFlip sequence:")
        for i in range(1, len(seq)):
            prev_edges = seq[i-1].edges
            curr_edges = seq[i].edges
            removed_edge = (prev_edges - curr_edges).pop()
            added_edge = (curr_edges - prev_edges).pop()
            print(f"  Step {i}: Flip {removed_edge} → {added_edge}")
        
        # Print edge statistics through the sequence
        print("\nEdge statistics through the sequence:")
        for i, tri in enumerate(seq):
            common_with_T1 = len(tri.edges & T1.edges)
            common_with_T2 = len(tri.edges & T2.edges)
            print(f"  Step {i}: "
                  f"Edges: {len(tri.edges)}, "
                  f"Common with T1: {common_with_T1} ({(common_with_T1/len(T1.edges))*100:.1f}%), "
                  f"Common with T2: {common_with_T2} ({(common_with_T2/len(T2.edges))*100:.1f}%)")
    
    print(f"\n{'='*80}\nDEBUGGING COMPLETE\n{'='*80}")
