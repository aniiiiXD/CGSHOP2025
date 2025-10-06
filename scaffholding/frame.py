"""
CGSHOP 2026 Testing Scaffolding
================================
Framework for testing triangulation reconfiguration strategies.

This scaffolding allows you to:
1. Load CGSHOP 2026 instances
2. Test custom solution modules
3. Compute flip sequences between triangulations
4. Evaluate objective values
"""

from typing import List, Tuple, Set, Optional, Protocol
from dataclasses import dataclass
from pydantic import BaseModel, Field, NonNegativeInt
from abc import ABC, abstractmethod
import json


# ============================================================================
# Instance Schema (as per CGSHOP 2026)
# ============================================================================

class CGSHOP2026Instance(BaseModel):
    """Schema for CGSHOP 2026 instance."""
    content_type: str = "CGSHOP2026_Instance"
    instance_uid: str
    points_x: List[int]
    points_y: List[int]
    triangulations: List[List[Tuple[NonNegativeInt, NonNegativeInt]]]


# ============================================================================
# Solution Representation
# ============================================================================

@dataclass
class ParallelFlip:
    """Represents a parallel flip operation."""
    flipped_edges: Set[Tuple[int, int]]  # Set of edges to flip simultaneously
    
    def __len__(self):
        return len(self.flipped_edges)


@dataclass
class FlipSequence:
    """Sequence of parallel flips transforming one triangulation to another."""
    flips: List[ParallelFlip]
    source_idx: int  # Index of source triangulation
    target_idx: int  # Index of target triangulation
    
    def __len__(self):
        """Number of parallel flip operations."""
        return len(self.flips)
    
    def total_flips(self):
        """Total number of individual edge flips."""
        return sum(len(flip) for flip in self.flips)


@dataclass
class Solution:
    """Complete solution with central triangulation and flip sequences."""
    central_triangulation: List[Tuple[int, int]]
    flip_sequences: List[FlipSequence]
    
    def objective_value(self) -> int:
        """Compute objective value: sum of |Fi| for all sequences."""
        return sum(len(seq) for seq in self.flip_sequences)


# ============================================================================
# Strategy Interface
# ============================================================================

class TriangulationStrategy(ABC):
    """
    Abstract base class for triangulation reconfiguration strategies.
    
    Implement this interface to create your own solution methods.
    """
    
    @abstractmethod
    def find_flip_sequence(
        self, 
        source: List[Tuple[int, int]], 
        target: List[Tuple[int, int]],
        points_x: List[int],
        points_y: List[int]
    ) -> FlipSequence:
        """
        Find a sequence of parallel flips from source to target.
        
        Args:
            source: Source triangulation (list of edges)
            target: Target triangulation (list of edges)
            points_x: X-coordinates of points
            points_y: Y-coordinates of points
            
        Returns:
            FlipSequence transforming source to target
        """
        pass
    
    @abstractmethod
    def find_central_triangulation(
        self,
        triangulations: List[List[Tuple[int, int]]],
        points_x: List[int],
        points_y: List[int]
    ) -> List[Tuple[int, int]]:
        """
        Find a central triangulation minimizing total flip distance.
        
        Args:
            triangulations: List of all triangulations
            points_x: X-coordinates of points
            points_y: Y-coordinates of points
            
        Returns:
            Central triangulation (list of edges)
        """
        pass


# ============================================================================
# Geometric Utilities
# ============================================================================

class GeometryUtils:
    """Utility functions for geometric operations on triangulations."""
    
    @staticmethod
    def edges_intersect(e1: Tuple[int, int], e2: Tuple[int, int], 
                       points_x: List[int], points_y: List[int]) -> bool:
        """Check if two edges properly intersect."""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        def get_point(idx):
            return (points_x[idx], points_y[idx])
        
        A, B = get_point(e1[0]), get_point(e1[1])
        C, D = get_point(e2[0]), get_point(e2[1])
        
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
    
    @staticmethod
    def is_valid_parallel_flip(edges: Set[Tuple[int, int]], 
                              triangulation: List[Tuple[int, int]],
                              points_x: List[int],
                              points_y: List[int]) -> bool:
        """Check if a set of edges can be flipped simultaneously."""
        # No two edges in the parallel flip should share a triangle
        # This is a simplified check - full implementation would track triangles
        return True  # Placeholder
    
    @staticmethod
    def normalize_edge(edge: Tuple[int, int]) -> Tuple[int, int]:
        """Normalize edge to have smaller index first."""
        return tuple(sorted(edge))


# ============================================================================
# Testing Framework
# ============================================================================

class CGSHOPTester:
    """Main testing framework for CGSHOP 2026 strategies."""
    
    def __init__(self, instance: CGSHOP2026Instance):
        self.instance = instance
        self.n_triangulations = len(instance.triangulations)
        
    def test_strategy(self, strategy: TriangulationStrategy) -> Solution:
        """
        Test a strategy on the loaded instance.
        
        Args:
            strategy: Strategy to test
            
        Returns:
            Complete solution with objective value
        """
        print(f"Testing strategy: {strategy.__class__.__name__}")
        print(f"Instance: {self.instance.instance_uid}")
        print(f"Points: {len(self.instance.points_x)}")
        print(f"Triangulations: {self.n_triangulations}\n")
        
        # Find central triangulation
        print("Finding central triangulation...")
        central = strategy.find_central_triangulation(
            self.instance.triangulations,
            self.instance.points_x,
            self.instance.points_y
        )
        
        # Find flip sequences from central to each triangulation
        print("Computing flip sequences...")
        flip_sequences = []
        for i, target in enumerate(self.instance.triangulations):
            print(f"  Sequence {i+1}/{self.n_triangulations}...", end=" ")
            seq = strategy.find_flip_sequence(
                central, target,
                self.instance.points_x,
                self.instance.points_y
            )
            flip_sequences.append(seq)
            print(f"done ({len(seq)} parallel flips)")
        
        solution = Solution(central, flip_sequences)
        obj = solution.objective_value()
        
        print(f"\nObjective value: {obj}")
        return solution
    
    def validate_solution(self, solution: Solution) -> bool:
        """Validate that a solution is correct."""
        # TODO: Implement validation
        # - Check central triangulation is valid
        # - Check each flip sequence correctly transforms central to target
        # - Check parallel flips are valid
        return True
    
    @staticmethod
    def load_instance(filepath: str) -> CGSHOP2026Instance:
        """Load instance from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return CGSHOP2026Instance(**data)
    
    @staticmethod
    def load_instance_from_dict(data: dict) -> CGSHOP2026Instance:
        """Load instance from dictionary."""
        return CGSHOP2026Instance(**data)


# ============================================================================
# Example Strategy Implementation (Naive/Baseline)
# ============================================================================

class NaiveStrategy(TriangulationStrategy):
    """
    Naive baseline strategy:
    - Uses first triangulation as central
    - Finds flip sequences greedily
    """
    
    def find_flip_sequence(
        self, 
        source: List[Tuple[int, int]], 
        target: List[Tuple[int, int]],
        points_x: List[int],
        points_y: List[int]
    ) -> FlipSequence:
        """Naive greedy flip sequence."""
        # This is a placeholder - implement actual flip finding
        flips = []
        # For now, just return empty sequence
        return FlipSequence(flips, 0, 0)
    
    def find_central_triangulation(
        self,
        triangulations: List[List[Tuple[int, int]]],
        points_x: List[int],
        points_y: List[int]
    ) -> List[Tuple[int, int]]:
        """Use first triangulation as central."""
        return triangulations[0]


# ============================================================================
# Usage Example
# ============================================================================

def example_usage():
    """Example of how to use the testing framework."""
    
    # Example instance data
    instance_data = {
        "content_type": "CGSHOP2026_Instance",
        "instance_uid": "example_test",
        "points_x": [0, 10, 5, 3, 8],
        "points_y": [0, 0, 8, 3, 5],
        "triangulations": [
            [(0, 1), (1, 2), (2, 0), (0, 3), (3, 1), (1, 4), (4, 2), (2, 3), (3, 4)],
            [(0, 1), (1, 2), (2, 0), (0, 3), (3, 2), (0, 4), (4, 1), (1, 3), (3, 4)]
        ]
    }
    
    # Load instance
    instance = CGSHOPTester.load_instance_from_dict(instance_data)
    
    # Create tester
    tester = CGSHOPTester(instance)
    
    # Test naive strategy
    naive_strategy = NaiveStrategy()
    solution = tester.test_strategy(naive_strategy)
    
    print(f"\nFinal objective: {solution.objective_value()}")


if __name__ == "__main__":
    print("CGSHOP 2026 Testing Scaffolding")
    print("=" * 50)
    print("\nTo use this framework:")
    print("1. Implement TriangulationStrategy interface")
    print("2. Create instance of your strategy")
    print("3. Pass to CGSHOPTester.test_strategy()")
    print("\nExample:")
    example_usage()