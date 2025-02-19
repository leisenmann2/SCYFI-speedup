import numpy as np
from scipy.optimize import linprog

def is_feasible(B, z0, constraints, eps=1e-6):
    """
    Check feasibility of a set of linear inequalities induced by the ReLU thresholds.
    
    For non-degenerate coordinates, we add:
      - For active (s == 1):  z0[i] + B[i,:]·xi >= eps
      - For inactive (s == 0): z0[i] + B[i,:]·xi <= -eps
    
    Parameters:
      B:           (n x r) numpy array.
      z0:          (n,) numpy array.
      constraints: List of tuples (i, s) for indices i and sign s.
      eps:         A small threshold.
      
    Returns:
      True if there exists some xi in R^r satisfying all (non-degenerate) constraints.
    """
    A_ub = []
    b_ub = []
    for i, s in constraints:
        # If the coordinate is degenerate, skip adding any constraint.
        if np.allclose(B[i, :], 0, atol=eps) and np.abs(z0[i]) < eps:
            continue
        
        if s == 1:
            # Require: z0[i] + B[i,:]·xi >= eps 
            # which we write as: -B[i,:]·xi <= -(eps - z0[i])
            A_ub.append(-B[i, :])
            b_ub.append(-(eps - z0[i]))
        elif s == 0:
            # Require: z0[i] + B[i,:]·xi <= -eps
            A_ub.append(B[i, :])
            b_ub.append(-eps - z0[i])
        else:
            raise ValueError("Sign s must be either 1 (active) or 0 (inactive).")
    
    if A_ub:
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
    else:
        A_ub = None
        b_ub = None

    c = np.zeros(B.shape[1])
    bounds = [(None, None)] * B.shape[1]
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    return res.success

def enumerate_regions_fixed(B, z0, deg_mask, index=0, current_pattern=None, eps=1e-6):
    """
    Recursively enumerate all feasible sign patterns.
    
    If the coordinate at a given index is degenerate (as determined by deg_mask),
    force its assignment to 0 (inactive) so that no pattern is returned with that coordinate active.
    
    Parameters:
      B:             (n x r) numpy array (each row gives coefficients for z_i).
      z0:            (n,) numpy array.
      deg_mask:      A boolean list of length n. If deg_mask[i] is True, then coordinate i is degenerate.
      index:         Current coordinate index.
      current_pattern: List of sign assignments (1 for active, 0 for inactive) so far.
      eps:           Threshold for numerical strictness.
      
    Returns:
      A list of tuples; each tuple is a complete sign pattern (of length n).
    """
    n = B.shape[0]
    if current_pattern is None:
        current_pattern = []
        
    # Base case: if all coordinates have been assigned, return the pattern.
    if index == n:
        return [tuple(current_pattern)]
    
    regions = []
    
    # If the current coordinate is degenerate, force it to be 0.
    if deg_mask[index]:
        forced_pattern = current_pattern + [0]
        # No constraint is added because we ignore the degenerate coordinate.
        regions.extend(enumerate_regions_fixed(B, z0, deg_mask, index+1, forced_pattern, eps))
    else:
        # Try assigning coordinate 'index' as active (1)
        pattern_plus = current_pattern + [1]
        constraints = list(enumerate(pattern_plus))
        if is_feasible(B, z0, constraints, eps):
            regions.extend(enumerate_regions_fixed(B, z0, deg_mask, index+1, pattern_plus, eps))
        
        # Try assigning coordinate 'index' as inactive (0)
        pattern_minus = current_pattern + [0]
        constraints = list(enumerate(pattern_minus))
        if is_feasible(B, z0, constraints, eps):
            regions.extend(enumerate_regions_fixed(B, z0, deg_mask, index+1, pattern_minus, eps))
    
    return regions

def enumerate_regions_ignore_degenerate(B, z0, eps=1e-6):
    """
    Compute a degeneracy mask and then enumerate activation patterns,
    forcing any degenerate coordinate to be inactive (0).
    
    Parameters:
      B:   (n x r) numpy array.
      z0:  (n,) numpy array.
      eps: Threshold for determining degeneracy.
      
    Returns:
      patterns: A list of tuples representing the activation patterns.
      deg_mask: A list of booleans of length n indicating the degenerate coordinates.
    """
    n = B.shape[0]
    deg_mask = [np.allclose(B[i, :], 0, atol=eps) and np.abs(z0[i]) < eps for i in range(n)]
    patterns = enumerate_regions_fixed(B, z0, deg_mask, eps=eps)
    return patterns, deg_mask


def main():
    # Suppose we have an RNN whose dynamics (after reduction) live on an affine subspace:
    # z = z0 + B xi, with z in R^n and xi in R^r.
    # For illustration, let n = 5 (number of coordinates / ReLU units) and r = 2 (effective dimension).
    np.random.seed(0)
    n = 2
    r = 1
    
    # Example: random B and z0.
    B =np.array([[1],[1]]) #np.random.randn(n, r)
    z0 =np.array([0,0]) #np.random.randn(n)
    
    eps = 1e-6  # small threshold to replace strict inequalities
    regions = enumerate_regions_ignore_degenerate(B, z0, eps=eps)
    
    print("Found {} feasible regions (activation patterns).".format(len(regions)))
    print("Each pattern is a tuple of length {} where 1 means 'active' and 0 means 'inactive':".format(n))
    for pattern in regions:
        print(pattern)

            # Suppose we have an RNN whose dynamics (after reduction) live on an affine subspace:
    # z = z0 + B xi, with z in R^n and xi in R^r.
    # For illustration, let n = 5 (number of coordinates / ReLU units) and r = 2 (effective dimension).
    n = 3
    r = 2
    
    # Example: random B and z0.
    B =np.array([[1,0],[0,1],[0,0]]) #np.random.randn(n, r)
    z0 =np.array([0,0,0]) #np.random.randn(n)
    
    eps = 1e-6  # small threshold to replace strict inequalities
    regions = enumerate_regions_ignore_degenerate(B, z0, eps=eps)
    
    print("Found {} feasible regions (activation patterns).".format(len(regions)))
    print("Each pattern is a tuple of length {} where 1 means 'active' and 0 means 'inactive':".format(n))
    for pattern in regions:
        print(pattern)

if __name__ == '__main__':
    main()
