"""
Gram-Schmidt Walk Algorithm Implementation

Based on "The Gram-Schmidt Walk: A Cure for the Banaszczyk Blues" by
Bansal, Dadush, Garg, and Lovett.

This module provides an efficient implementation of the algorithm to find
a ±1 coloring of vectors with low discrepancy.
"""

import numpy as np
import time

class GramSchmidtWalk:
    """
    Implementation of the Gram-Schmidt Walk algorithm for vector balancing.
    
    This algorithm takes a set of vectors with L2 norm at most 1 and finds
    a ±1 coloring with low discrepancy.
    """
    
    def __init__(self, vectors, initial_coloring=None):
        """
        Initialize with a set of vectors with L2 norm at most 1.
        
        Args:
            vectors: numpy array of shape (n, m) where each row is a vector
            initial_coloring: initial fractional coloring in [-1,1]^n (default: zeros)
        """
        self.vectors = np.array(vectors, dtype=float)
        self.n, self.m = self.vectors.shape
        
        norms = np.linalg.norm(self.vectors, axis=1)
        if np.any(norms > 1.0 + 1e-10):
            raise ValueError("Input vectors must have L2 norm at most 1")
        
        if initial_coloring is None:
            self.coloring = np.zeros(self.n)
        else:
            self.coloring = np.array(initial_coloring, dtype=float)
            
        self.alive = np.abs(self.coloring) < 1
        
        self.iterations = 0
        self.history = [self.coloring.copy()]
        self.discrepancy_history = []
    
    def compute_update_direction(self, pivot_idx):
        """
        Compute the update direction using Gram-Schmidt orthogonalization.
        
        Args:
            pivot_idx: Index of the pivot element
            
        Returns:
            update: The update direction vector
        """
        update = np.zeros(self.n)
        
        update[pivot_idx] = 1
        
        alive_indices = np.where(self.alive)[0]
        non_pivot_alive = alive_indices[alive_indices != pivot_idx]
        
        if len(non_pivot_alive) == 0:
            return update
        
        V_alive = self.vectors[non_pivot_alive]
        
        pivot_vector = self.vectors[pivot_idx]
        
        try:
            A = V_alive @ V_alive.T
            b = -V_alive @ pivot_vector
            
            A_reg = A + 1e-10 * np.eye(A.shape[0])
            
            try:
                non_pivot_updates = np.linalg.lstsq(A_reg, b, rcond=None)[0]
                update[non_pivot_alive] = non_pivot_updates
            except np.linalg.LinAlgError:
                raise
        except Exception:
            V_orthogonal = np.zeros((len(non_pivot_alive), self.m))
            for i in range(len(non_pivot_alive)):
                v = V_alive[i].copy()
                for j in range(i):
                    proj = np.dot(v, V_orthogonal[j])
                    v = v - proj * V_orthogonal[j]
                
                norm = np.linalg.norm(v)
                if norm > 1e-10:
                    V_orthogonal[i] = v / norm
            
            for i in range(len(non_pivot_alive)):
                projection = np.dot(pivot_vector, V_orthogonal[i])
                update[non_pivot_alive[i]] = -projection
            
        return update
    
    def compute_step_sizes(self, update):
        """
        Vectorized computation of positive and negative step sizes that maintain -1 ≤ coloring ≤ 1.
        
        Args:
            update: Update direction vector
            
        Returns:
            delta_minus: Negative step size
            delta_plus: Positive step size
        """
        alive_mask = self.alive
        update_alive = update[alive_mask]
        coloring_alive = self.coloring[alive_mask]
        
        nonzero_mask = np.abs(update_alive) >= 1e-10
        if not np.any(nonzero_mask):
            return 0, 0
            
        update_nz = update_alive[nonzero_mask]
        coloring_nz = coloring_alive[nonzero_mask]
        
        steps_to_neg1 = (-1 - coloring_nz) / update_nz
        steps_to_pos1 = (1 - coloring_nz) / update_nz
        
        pos_steps = np.maximum(steps_to_neg1, steps_to_pos1)
        neg_steps = np.minimum(steps_to_neg1, steps_to_pos1)
        
        delta_plus = np.min(pos_steps) if len(pos_steps) > 0 else 0
        delta_minus = np.max(neg_steps) if len(neg_steps) > 0 else 0
        
        return delta_minus, delta_plus
    
    def compute_discrepancy(self, coloring=None):
        """
        Compute various discrepancy measures.
        
        Args:
            coloring: Optional coloring to compute discrepancy for
            
        Returns:
            Dictionary with different discrepancy measures
        """
        if coloring is None:
            coloring = self.coloring
            
        combined = self.vectors.T @ coloring
        
        l_inf = np.max(np.abs(combined))
        l2 = np.sqrt(np.sum(combined**2))
        l1 = np.sum(np.abs(combined))
        
        return {
            "l_inf": l_inf,
            "l2": l2,
            "l1": l1,
            "vector": combined
        }
    
    def run(self, verbose=False, max_iterations=None):
        """
        Run the Gram-Schmidt Walk algorithm until all elements are colored.
        
        Args:
            verbose: Whether to print progress information
            max_iterations: Optional maximum number of iterations
            
        Returns:
            Dictionary with results including final coloring and discrepancy
        """
        start_time = time.time()
        
        while np.any(self.alive):
            if max_iterations is not None and self.iterations >= max_iterations:
                break
                
            pivot_idx = np.max(np.where(self.alive)[0])
            
            update = self.compute_update_direction(pivot_idx)
            
            delta_minus, delta_plus = self.compute_step_sizes(update)
            
            prob_plus = abs(delta_minus) / (abs(delta_minus) + abs(delta_plus))
            
            if np.random.random() < prob_plus:
                delta = delta_plus
            else:
                delta = delta_minus
                
            self.coloring = self.coloring + delta * update
            
            self.alive = np.abs(self.coloring) < 1 - 1e-10
            
            self.iterations += 1
            self.history.append(self.coloring.copy())
            self.discrepancy_history.append(self.compute_discrepancy())
            
            if verbose and self.iterations % max(1, min(10, self.n // 10)) == 0:
                fixed_count = np.sum(~self.alive)
                disc = self.compute_discrepancy()["l_inf"]
                print(f"Iteration {self.iterations}: {fixed_count}/{self.n} elements fixed, discrepancy: {disc:.4f}")
        
        self.coloring = np.sign(self.coloring)
        final_discrepancy = self.compute_discrepancy()
        
        elapsed_time = time.time() - start_time
        
        result = {
            "coloring": self.coloring,
            "iterations": self.iterations,
            "time": elapsed_time,
            "discrepancy": final_discrepancy,
            "discrepancy_history": self.discrepancy_history
        }
        
        if verbose:
            print(f"Completed in {elapsed_time:.4f} seconds after {self.iterations} iterations")
            print(f"L-inf discrepancy: {final_discrepancy['l_inf']:.4f}")
            print(f"L2 discrepancy: {final_discrepancy['l2']:.4f}")
        
        return result


"""
testing
if __name__ == "__main__":
    np.random.seed(42)
    n, m = 10, 3
    
    vectors = np.random.randn(n, m)
    norms = np.linalg.norm(vectors, axis=1)
    vectors = vectors / np.maximum(norms[:, np.newaxis], 1.0)
    
    gsw = GramSchmidtWalk(vectors)
    result = gsw.run(verbose=True)
    
    print("\nFinal coloring:", result["coloring"])
    print("Final discrepancy:", result["discrepancy"]["l_inf"])
"""