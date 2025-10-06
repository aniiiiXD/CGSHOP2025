import json
from collections import defaultdict, deque
from typing import List, Tuple, Set, Dict, Optional
import copy

class Point:
    def __init__(self, x: float, y: float, idx: int):
        self.x = x
        self.y = y
        self.idx = idx
    
    def __repr__(self):
        return f"P{self.idx}({self.x},{self.y})"

class Edge:
    def __init__(self, p1: int, p2: int):
        # Store edge with smaller index first for consistency
        self.p1 = min(p1, p2)
        self.p2 = max(p1, p2)
    
    def __hash__(self):
        return hash((self.p1, self.p2))
    
    def __eq__(self, other):
        return isinstance(other, Edge) and self.p1 == other.p1 and self.p2 == other.p2
    
    def __lt__(self, other):
        if not isinstance(other, Edge):
            return NotImplemented
        # First compare by first vertex, then by second vertex
        return (self.p1, self.p2) < (other.p1, other.p2)
    
    def __repr__(self):
        return f"({self.p1},{self.p2})"

class Triangle:
    def __init__(self, p1: int, p2: int, p3: int):
        # Store vertices in sorted order for consistency
        self.vertices = tuple(sorted([p1, p2, p3]))
        self.p1, self.p2, self.p3 = self.vertices
    
    def get_edges(self):
        return [
            Edge(self.p1, self.p2),
            Edge(self.p1, self.p3),
            Edge(self.p2, self.p3)
        ]
    
    def __hash__(self):
        return hash(self.vertices)
    
    def __eq__(self, other):
        return isinstance(other, Triangle) and self.vertices == other.vertices
    
    def __repr__(self):
        return f"T({self.p1},{self.p2},{self.p3})"

class Triangulation:
    def __init__(self, points: List[Point], edge_list: List[List[int]]):
        self.points = points
        self.triangles = set()
        self.edges = set()
        self.edge_to_triangles = defaultdict(set)
        
        # Build triangulation from edges
        self._build_from_edges(edge_list)
    
    def _build_from_edges(self, edge_list):
        """Build triangulation from list of edges"""
        # Store all edges
        for edge_pair in edge_list:
            edge = Edge(edge_pair[0], edge_pair[1])
            self.edges.add(edge)
        
        # Find triangles
        edge_dict = defaultdict(set)
        for edge in self.edges:
            edge_dict[edge.p1].add(edge.p2)
            edge_dict[edge.p2].add(edge.p1)
        
        # Find all triangles by checking for 3-cycles
        for p1 in edge_dict:
            for p2 in edge_dict[p1]:
                if p2 > p1:  # Avoid duplicates
                    for p3 in edge_dict[p2]:
                        if p3 > p2 and p1 in edge_dict[p3]:
                            triangle = Triangle(p1, p2, p3)
                            self.triangles.add(triangle)
        
        # Build edge-to-triangle mapping
        for triangle in self.triangles:
            for edge in triangle.get_edges():
                self.edge_to_triangles[edge].add(triangle)
    
    def _ccw(self, A, B, C):
        """Return True if points A, B, C are in counter-clockwise order"""
        return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)
    
    def _intersect(self, A, B, C, D):
        """Return True if line segments AB and CD intersect"""
        return self._ccw(A, C, D) != self._ccw(B, C, D) and self._ccw(A, B, C) != self._ccw(A, B, D)
    
    def _point_in_triangle(self, point, triangle):
        """Check if point is inside triangle using barycentric coordinates"""
        p1, p2, p3 = [self.points[i] for i in triangle.vertices]
        
        def sign(p1, p2, p3):
            return (p1.x - p3.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p3.y)
        
        d1 = sign(point, p1, p2)
        d2 = sign(point, p2, p3)
        d3 = sign(point, p3, p1)
        
        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        
        return not (has_neg and has_pos)
    
    def _is_convex_quad(self, p1_idx, p2_idx, p3_idx, p4_idx):
        """Check if quadrilateral is convex"""
        points = [self.points[i] for i in [p1_idx, p2_idx, p3_idx, p4_idx]]
        
        def cross_product(o, a, b):
            return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
        
        # Check if all interior angles are < 180 degrees
        cross_products = []
        for i in range(4):
            cp = cross_product(points[i], points[(i + 1) % 4], points[(i + 2) % 4])
            cross_products.append(cp)
        
        # All cross products should have the same sign for convex quadrilateral
        positive = sum(1 for cp in cross_products if cp > 0)
        negative = sum(1 for cp in cross_products if cp < 0)
        
        return positive == 4 or negative == 4
    
    def can_flip_edge(self, edge: Edge) -> bool:
        """Check if an edge can be flipped"""
        if edge not in self.edge_to_triangles:
            return False
        
        triangles = list(self.edge_to_triangles[edge])
        if len(triangles) != 2:
            return False
        
        t1, t2 = triangles
        
        # Get all four vertices of the quadrilateral
        all_vertices = set(t1.vertices) | set(t2.vertices)
        if len(all_vertices) != 4:
            return False
        
        # Find the two vertices that are not part of the edge
        edge_vertices = {edge.p1, edge.p2}
        other_vertices = list(all_vertices - edge_vertices)
        
        if len(other_vertices) != 2:
            return False
        
        # Check if the quadrilateral is convex
        quad_vertices = [edge.p1, other_vertices[0], edge.p2, other_vertices[1]]
        return self._is_convex_quad(*quad_vertices)
    
    def flip_edge(self, edge: Edge) -> Optional[Edge]:
        """Flip an edge and return the new edge"""
        if not self.can_flip_edge(edge):
            return None
        
        triangles = list(self.edge_to_triangles[edge])
        t1, t2 = triangles
        
        # Get all vertices
        all_vertices = set(t1.vertices) | set(t2.vertices)
        edge_vertices = {edge.p1, edge.p2}
        other_vertices = list(all_vertices - edge_vertices)
        
        # Create new edge
        new_edge = Edge(other_vertices[0], other_vertices[1])
        
        # Remove old triangles and edge
        self.triangles.discard(t1)
        self.triangles.discard(t2)
        self.edges.discard(edge)
        del self.edge_to_triangles[edge]
        
        # Create new triangles
        new_t1 = Triangle(other_vertices[0], other_vertices[1], edge.p1)
        new_t2 = Triangle(other_vertices[0], other_vertices[1], edge.p2)
        
        self.triangles.add(new_t1)
        self.triangles.add(new_t2)
        self.edges.add(new_edge)
        
        # Update edge-to-triangle mapping
        # First remove old mappings
        for old_edge in t1.get_edges():
            if old_edge in self.edge_to_triangles:
                self.edge_to_triangles[old_edge].discard(t1)
        for old_edge in t2.get_edges():
            if old_edge in self.edge_to_triangles:
                self.edge_to_triangles[old_edge].discard(t2)
        
        # Add new mappings
        for triangle in [new_t1, new_t2]:
            for triangle_edge in triangle.get_edges():
                self.edge_to_triangles[triangle_edge].add(triangle)
        
        return new_edge
    
    def copy(self):
        """Create a deep copy of this triangulation"""
        edge_list = [[e.p1, e.p2] for e in self.edges]
        return Triangulation(self.points, edge_list)
    
    def get_edge_set(self) -> Set[Edge]:
        """Get set of all edges"""
        return self.edges.copy()
    
    def __eq__(self, other):
        """Check if two triangulations are equal"""
        return isinstance(other, Triangulation) and self.edges == other.edges
    
    def __hash__(self):
        """Hash based on edge set"""
        return hash(frozenset(self.edges))

class FlipSequenceFinder:
    def __init__(self, points: List[Point]):
        self.points = points
    
    def find_flip_sequence_bfs(self, start: Triangulation, target: Triangulation, max_depth: int = 20) -> Optional[List[Edge]]:
        """Find flip sequence using BFS - guarantees shortest path but may be slow"""
        if start == target:
            return []
        
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) >= max_depth:
                continue
            
            # Try flipping each edge
            for edge in list(current.edges):
                if current.can_flip_edge(edge):
                    next_triangulation = current.copy()
                    next_triangulation.flip_edge(edge)
                    
                    if next_triangulation == target:
                        return path + [edge]
                    
                    if next_triangulation not in visited and len(path) < max_depth - 1:
                        visited.add(next_triangulation)
                        queue.append((next_triangulation, path + [edge]))
        
        return None
    
    def find_flip_sequence_greedy(self, start: Triangulation, target: Triangulation, max_steps: int = 100) -> Optional[List[Edge]]:
        """Find flip sequence using greedy approach - faster but not optimal"""
        current = start.copy()
        path = []
        
        for step in range(max_steps):
            if current == target:
                return path
            
            # Find edges that need to be removed (in current but not in target)
            edges_to_remove = current.get_edge_set() - target.get_edge_set()
            
            if not edges_to_remove:
                break
            
            # Try to flip one of the edges that needs to be removed
            flipped = False
            for edge in edges_to_remove:
                if current.can_flip_edge(edge):
                    new_edge = current.flip_edge(edge)
                    if new_edge:
                        path.append(edge)
                        flipped = True
                        break
            
            # If no direct flip worked, try flipping any flippable edge
            if not flipped:
                for edge in list(current.edges):
                    if current.can_flip_edge(edge):
                        new_edge = current.flip_edge(edge)
                        if new_edge:
                            path.append(edge)
                            flipped = True
                            break
            
            if not flipped:
                break  # No more flips possible
        
        return path if current == target else None
    
    def find_flip_sequence_smart_greedy(self, start: Triangulation, target: Triangulation, max_steps: int = 200) -> Optional[List[Edge]]:
        """Smart greedy approach that considers both removing wrong edges and adding needed edges"""
        current = start.copy()
        path = []
        
        for step in range(max_steps):
            if current == target:
                return path
            
            current_edges = current.get_edge_set()
            target_edges = target.get_edge_set()
            
            # Edges to remove and add
            to_remove = current_edges - target_edges
            to_add = target_edges - current_edges
            
            if not to_remove and not to_add:
                break
            
            best_flip = None
            best_score = -1
            
            # Evaluate each possible flip
            for edge in list(current.edges):
                if current.can_flip_edge(edge):
                    # Simulate the flip
                    temp = current.copy()
                    new_edge = temp.flip_edge(edge)
                    
                    if new_edge:
                        temp_edges = temp.get_edge_set()
                        new_to_remove = temp_edges - target_edges
                        new_to_add = target_edges - temp_edges
                        
                        # Score: prefer flips that reduce the total difference
                        score = (len(to_remove) + len(to_add)) - (len(new_to_remove) + len(new_to_add))
                        
                        # Bonus for adding a needed edge
                        if new_edge in to_add:
                            score += 2
                        
                        # Bonus for removing an unwanted edge
                        if edge in to_remove:
                            score += 2
                        
                        if score > best_score:
                            best_score = score
                            best_flip = edge
            
            if best_flip:
                current.flip_edge(best_flip)
                path.append(best_flip)
            else:
                break  # No beneficial flips found
        
        return path if current == target else None

def load_and_solve_instance():
    """Load the instance and find a flip sequence"""
    # Data from the provided JSON
    data = {
        "points_x": [230, 937, 862, 222, 420, 632, 144, 967, 351, 944, 956, 788, 551, 948, 795, 339, 958, 283, 71, 892],
        "points_y": [34, 369, 42, 672, 755, 38, 613, 65, 74, 395, 860, 455, 723, 632, 780, 156, 495, 215, 993, 876],
        "triangulations": [
            [[1,9],[1,2],[2,9],[4,9],[4,18],[9,18],[5,8],[2,5],[2,8],[10,19],[0,17],[0,8],[8,17],[4,12],[12,14],[4,14],[9,13],[12,13],[9,12],[7,10],[2,7],[9,16],[13,16],[14,18],[6,17],[0,6],[6,18],[0,18],[2,3],[3,15],[2,15],[18,19],[8,15],[15,17],[1,11],[3,11],[1,3],[14,19],[10,14],[10,16],[7,16],[10,13],[13,14],[7,9],[9,11],[17,18],[0,5],[3,17],[3,18],[11,18]],
            [[10,19],[12,19],[10,12],[8,15],[5,8],[5,15],[7,9],[1,9],[1,7],[4,19],[4,12],[9,13],[13,14],[9,14],[7,10],[9,12],[4,9],[2,7],[0,8],[0,15],[7,18],[2,18],[6,17],[0,6],[0,17],[15,17],[6,18],[0,18],[7,11],[18,19],[4,18],[3,6],[3,18],[12,14],[10,14],[11,18],[10,16],[7,16],[10,13],[13,16],[9,16],[3,15],[6,15],[0,5],[2,5],[2,3],[3,5],[4,11],[1,11],[1,4]]
        ]
    }
    
    # Create points
    points = []
    for i, (x, y) in enumerate(zip(data['points_x'], data['points_y'])):
        points.append(Point(x, y, i))
    
    # Create triangulations
    t_initial = Triangulation(points, data['triangulations'][0])
    t_final = Triangulation(points, data['triangulations'][1])
    
    print(f"Initial triangulation: {len(t_initial.triangles)} triangles, {len(t_initial.edges)} edges")
    print(f"Final triangulation: {len(t_final.triangles)} triangles, {len(t_final.edges)} edges")
    
    # Find differences
    initial_edges = t_initial.get_edge_set()
    final_edges = t_final.get_edge_set()
    
    edges_to_remove = initial_edges - final_edges
    edges_to_add = final_edges - initial_edges
    
    print(f"Edges to remove: {len(edges_to_remove)}")
    print(f"Edges to add: {len(edges_to_add)}")
    
    if edges_to_remove:
        print("Edges to remove:", sorted(list(edges_to_remove))[:10])  # Show first 10
    if edges_to_add:
        print("Edges to add:", sorted(list(edges_to_add))[:10])  # Show first 10
    
    # Initialize solver
    solver = FlipSequenceFinder(points)
    
    # Try different approaches
    print("\n=== Trying Smart Greedy Approach ===")
    sequence = solver.find_flip_sequence_smart_greedy(t_initial, t_final)
    
    if sequence is not None:
        print(f"Found flip sequence of length {len(sequence)}!")
        print("Flip sequence:")
        for i, edge in enumerate(sequence):
            print(f"  {i+1}. Flip edge {edge}")
        
        # Verify the sequence
        verify = t_initial.copy()
        print(f"\nVerifying sequence...")
        for i, edge in enumerate(sequence):
            if verify.can_flip_edge(edge):
                verify.flip_edge(edge)
                print(f"  Step {i+1}: Flipped {edge} ✓")
            else:
                print(f"  Step {i+1}: Cannot flip {edge} ✗")
                break
        
        if verify == t_final:
            print("✓ Sequence verified! Successfully transforms initial to final triangulation.")
        else:
            print("✗ Sequence verification failed.")
            remaining_diff = len((verify.get_edge_set() - t_final.get_edge_set()) | (t_final.get_edge_set() - verify.get_edge_set()))
            print(f"  Remaining edge differences: {remaining_diff}")
    else:
        print("No sequence found with smart greedy approach.")
        
        print("\n=== Trying Regular Greedy Approach ===")
        sequence = solver.find_flip_sequence_greedy(t_initial, t_final)
        
        if sequence is not None:
            print(f"Found flip sequence of length {len(sequence)} with regular greedy!")
            # Show first few flips
            for i, edge in enumerate(sequence[:10]):
                print(f"  {i+1}. Flip edge {edge}")
            if len(sequence) > 10:
                print(f"  ... and {len(sequence)-10} more flips")
        else:
            print("No sequence found with regular greedy either.")
            print("\nTrying BFS (may be slow)...")
            sequence = solver.find_flip_sequence_bfs(t_initial, t_final, max_depth=10)
            
            if sequence is not None:
                print(f"Found optimal sequence of length {len(sequence)} with BFS!")
                for i, edge in enumerate(sequence):
                    print(f"  {i+1}. Flip edge {edge}")
            else:
                print("No sequence found with BFS within depth limit.")

if __name__ == "__main__":
    load_and_solve_instance()