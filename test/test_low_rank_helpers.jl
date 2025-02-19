using Test
using PyCall
using LinearAlgebra
using Random
using SCYFI
using CUDA
using Combinatorics
using BenchmarkTools
# Import your Julia implementation
include("../src/utilities/helpers.jl")

# Import Python numpy
np = pyimport("numpy")

"""
Python reference implementation from https://github.com/mackelab/smc_rnns/blob/main/fixed_points/find_fixed_points_analytic.py

"""
function python_find_subregion_intersections_og(a, V, U, hz, h)
    py"""
    import numpy as np
    from itertools import combinations, chain
    
    def powerset(iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))
    

    def find_subregion_intersections_og(a, V, U, hz, h):
        n_inverses = 0
        N = U.shape[0]
        R = U.shape[1]

        intersect_inds = np.array(list(combinations(np.arange(N), R)))
        
        if R == 2:
            ni = N // 2
            par_inds = []
            for i, el in enumerate(intersect_inds):
                if el[0] == el[1] + ni or el[1] == el[0] + ni:
                    par_inds.append(i)
            intersect_inds = np.delete(intersect_inds, par_inds, axis=0)

        n_Ds_initial = len(list(powerset(range(R)))) * len(intersect_inds)
        D_list = np.zeros((n_Ds_initial, N), dtype="uint8")
        it = 0
        
        for inds in intersect_inds:
            b_hat = h[inds]
            U_hat = U[inds]
            n_inverses += 1
            z = np.linalg.solve(U_hat, b_hat)
            x = U @ z - h
            D_init = np.array(x > 0).astype("uint8")
            D_init[inds] = 0
            D_list[it] = D_init
            it += 1
            D_inds = list(powerset(inds))[1:]
            for D_ind in D_inds:
                D = np.copy(D_init)
                D[np.array(D_ind)] = 1
                D_list[it] = D
                it += 1
        
        D_list = np.unique(D_list[:it], axis=0)
        return D_list, n_inverses

    """
    return py"find_subregion_intersections_og"(a, V, U, hz, h)

    
end


function python_find_subregion_intersections(B, h)
    py"""
    import numpy as np
    from scipy.optimize import linprog

    def is_feasible(B, z0, constraints, eps=1e-6):
        
        # Check feasibility of a set of linear inequalities induced by the ReLU thresholds.
        
        # For non-degenerate coordinates, we add:
        # - For active (s == 1):  z0[i] + B[i,:]·xi >= eps
        # - For inactive (s == 0): z0[i] + B[i,:]·xi <= -eps
        
        # Parameters:
        # B:           (n x r) numpy array.
        # z0:          (n,) numpy array.
        # constraints: List of tuples (i, s) for indices i and sign s.
        # eps:         A small threshold.
        
        # Returns:
        # True if there exists some xi in R^r satisfying all (non-degenerate) constraints.
    
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
       
        # Recursively enumerate all feasible sign patterns.
        
        # If the coordinate at a given index is degenerate (as determined by deg_mask),
        # force its assignment to 0 (inactive) so that no pattern is returned with that coordinate active.
        
        # Parameters:
        # B:             (n x r) numpy array (each row gives coefficients for z_i).
        # z0:            (n,) numpy array.
        # deg_mask:      A boolean list of length n. If deg_mask[i] is True, then coordinate i is degenerate.
        # index:         Current coordinate index.
        # current_pattern: List of sign assignments (1 for active, 0 for inactive) so far.
        # eps:           Threshold for numerical strictness.
        
        # Returns:
        # A list of tuples; each tuple is a complete sign pattern (of length n).
      
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

        # Compute a degeneracy mask and then enumerate activation patterns,
        # forcing any degenerate coordinate to be inactive (0).
        
        # Parameters:
        # B:   (n x r) numpy array.
        # z0:  (n,) numpy array.
        # eps: Threshold for determining degeneracy.
        
        # Returns:
        # patterns: A list of tuples representing the activation patterns.
        # deg_mask: A list of booleans of length n indicating the degenerate coordinates.

        n = B.shape[0]
        deg_mask = [np.allclose(B[i, :], 0, atol=eps) and np.abs(z0[i]) < eps for i in range(n)]
        patterns = enumerate_regions_fixed(B, z0, deg_mask, eps=eps)
        return [np.array(pattern) for pattern in patterns], deg_mask

    """
    return py"enumerate_regions_ignore_degenerate"(B, h)
end

# @testset "Low Rank Helpers Tests" begin
#     # First test correctness
#     @testset "Correctness" begin
#         for (N, R) in [(4, 2), (6, 2), (8, 3)]
#             rng = Random.seed!(123)
#             a = randn(rng, R)
#             V = randn(rng, R, N)
#             U = randn(rng, N, R)
#             hz = randn(rng, R)
#             h = randn(rng, N)

#             julia_result, julia_n_inverses = find_subregion_intersections(a, V, U, hz, h)
#             python_result, python_n_inverses = python_find_subregion_intersections_og(a, V, U, hz, h)

#             julia_result_int = Int8.(julia_result)
            
#             @test size(julia_result_int) == size(python_result)
#             @test julia_n_inverses == python_n_inverses
            
#             julia_rows = Set([julia_result_int[i,:] for i in 1:size(julia_result_int,1)])
#             python_rows = Set([python_result[i,:] for i in 1:size(python_result,1)])
#             @test julia_rows == python_rows
#         end
#     end

#     # Then test performance
#     @testset "Performance" begin
#         # Test with larger dimensions for more meaningful benchmarks
#         for (N, R) in [(100, 2), (200, 2)]
#             @info "Testing performance for N=$N, R=$R"
            
#             # Generate test data
#             rng = Random.seed!(123)
#             a = randn(rng, R)
#             V = randn(rng, R, N)
#             U = randn(rng, N, R)
#             hz = randn(rng, R)
#             h = randn(rng, N)

#             # Benchmark Julia implementation
#             julia_time = @benchmark find_subregion_intersections($a, $V, $U, $hz, $h)
            
#             # Benchmark Python implementation
#             # We'll use multiple runs for Python since @benchmark doesn't work with PyCall
#             n_runs = 5
#             python_times = zeros(n_runs)
#             for i in 1:n_runs
#                 python_times[i] = @elapsed python_find_subregion_intersections_og(a, V, U, hz, h)
#             end
            
#             julia_median = median(julia_time.times) / 1e9  # Convert from ns to seconds
#             python_median = median(python_times)
            
#             @info "Performance Results" N R julia_median python_median
            
#             # Test that Julia version is faster (you can adjust the factor as needed)
#             @test julia_median < python_median
            
#             # Print speedup factor
#             speedup = python_median / julia_median
#             @info "Julia implementation is $(round(speedup, digits=2))x faster"
#         end
#     end
# end

@testset "Low Rank Region Finding Tests" begin
    @testset "Simple 2D Example" begin
        N, R = 2, 1
        a = [1.0]
        V = reshape([1.0, 1.0], R, N)  # Make sure V is R×N (1×2)
        U = reshape([1.0, 1.0], N, R)  # Make sure U is N×R (2×1)
        W = inv(U'*U) * U' * U * V
        hz = zeros(R)
        h = zeros(N)

        D_list, _ = python_find_subregion_intersections(U, h)
        D_list = Bool.(hcat(D_list...))
        julia_D_list = enumerate_regions_ignore_degenerate(U, h)
        
        # Test Python implementation (original test)
        @test size(D_list, 2) == 2
        @test any(all(D_list .== [0 0], dims=1))
        @test any(all(D_list .== [1 1], dims=1))

        # Test Julia implementation matches Python
        @test size(julia_D_list, 2) == size(D_list, 1)  # Julia has patterns as columns
        @test any(all(julia_D_list .== [0,0], dims=1))
        @test any(all(julia_D_list .== [1,1], dims=1))
    end

    @testset "3D Example with Rank 2" begin
        N, R = 3, 2
        a = [1.0, 1.0]
        V = [1.0 0.0 1.0; 0.0 1.0 0.0]  # 2×3 matrix
        U = [1.0 1.0; 0.0 1.0; 1.0 0.0]  # 3×2 matrix
        W = inv(U'*U) * U' * U * V
        hz = [0.0, 0.0]
        h = [0.0, 0.0, 0.0]

        D_list, _ = python_find_subregion_intersections(U, h)
        D_list = Bool.(hcat(D_list...))
        julia_D_list = enumerate_regions_ignore_degenerate(U, h)
        
        # Test number of patterns matches
        @test size(julia_D_list, 2) == size(D_list, 2)
        
        # Convert to sets for pattern comparison
        #py_patterns = Set([Tuple(row) for row in eachrow(D_list)])
        #jl_patterns = Set([Tuple(col) for col in eachrow(julia_D_list)])
        @test D_list == julia_D_list
    end

    @testset "Performance Comparison" begin
        for (N, R) in [(10, 2), (20, 3)]
            rng = Random.seed!(123)
            U = randn(rng, N, R)
            h = zeros(N)

            # Benchmark both implementations
            py_time = @elapsed python_find_subregion_intersections(U, h)
            jl_time = @elapsed enumerate_regions_ignore_degenerate(U, h)

            # Compare results
            py_D_list, _ = python_find_subregion_intersections(U, h)
            py_D_list = Bool.(hcat(py_D_list...))
            julia_D_list = enumerate_regions_ignore_degenerate(U, h)

            # Convert to sets for comparison
            #py_patterns = Set([Tuple(row) for row in eachrow(py_D_list)])
            #jl_patterns = Set([Tuple(col) for col in eachcol(julia_D_list)])
            
            # Test correctness and performance
            @test py_D_list == julia_D_list
            @test jl_time ≤ 2py_time
            
            @info "Performance for N=$N, R=$R" python_time=py_time julia_time=jl_time ratio=py_time/jl_time
        end
    end

    @testset "Edge Cases" begin
        # Test minimal case
        N, R = 1, 1
        U = ones(N, R)
        h = zeros(N)
        
        py_D_list, _ = python_find_subregion_intersections(U, h)
        py_D_list = Bool.(hcat(py_D_list...))
        julia_D_list = enumerate_regions_ignore_degenerate(U, h)
        
        @test py_D_list == julia_D_list
        
        # Test degenerate case
        N, R = 3, 2
        U = zeros(N, R)
        h = zeros(N)
        
        py_D_list, _ = python_find_subregion_intersections(U, h)
        py_D_list = Bool.(hcat(py_D_list...))
        julia_D_list = enumerate_regions_ignore_degenerate(U, h)
        
        @test py_D_list == julia_D_list
    end
end


# N, R = 2, 1
# a = [1.0]
# V = reshape([1.0, 1.0], R, N)  # Make sure V is R×N (1×2)
# U = reshape([1.0, 1.0], N, R)  # Make sure U is N×R (2×1)
# W = inv(U'*U) * U' * U * V
# hz = zeros(R)
# h = zeros(N)

# D_list, _ = python_find_subregion_intersections(U, h)
# D_list = Bool.(hcat(D_list...))
# julia_D_list = enumerate_regions_ignore_degenerate(U, h)

# # Test Python implementation (original test)
# @test size(D_list, 1) == 2
# @test any(all(D_list .== [0 0], dims=1))
# @test any(all(D_list .== [1 1], dims=1))

# # Test Julia implementation matches Python
# @test size(julia_D_list, 2) == size(D_list, 1)  # Julia has patterns as columns
# @test any(all(julia_D_list .== [0,0], dims=1))
# @test any(all(julia_D_list .== [1,1], dims=1))
# # N, R = 2, 1
# a = [1.0]
# V = reshape([1.0, 1.0], R, N)  # Make sure V is R×N (1×2)
# U = reshape([1.0, 1.0], N, R)  # Make sure U is N×R (2×1)
# V*U
# W = inv(U'*U) * U' * U * V

# h = [0.,0.]#zeros(N)

# D_list, _ =  python_find_subregion_intersections(U, h)

# # Expect 2 valid patterns: [0,0] and [1,1]
# @test size(D_list, 1) == 2
# @test any(all(D_list .== [0 0], dims=2))
# @test any(all(D_list .== [1 1], dims=2))



# N, R = 3, 2
# a = [1.0, 1.0]
# V = [1.0 0.0 1.0; 0.0 1.0 0.0]  # 2×3 matrix
# U = [1.0 1.0; 0.0 1.0; 1.0 0.0]  # 3×2 matrix
# W = inv(U'*U) * U' * U * V
# hz = [0.0, 0.0]
# h = [0.0, 0.0, 0.0]

# D_list, _ = python_find_subregion_intersections_og(a, V, U, hz, h)

# # Should find 4 regions in x-y plane
# @test size(D_list, 1) == 4
# @test all(D_list[:, 3] .== 0)  # Third neuron always off
# @test Set(Tuple(row) for row in eachrow(D_list)) == Set([(0,0,0), (1,0,0), (0,1,0), (1,1,0)])


#     #     N, R = 3, 2
#     #     a = [1.0, 1.0]
    #     V = [1.0 0.0 0.0; 0.0 1.0 0.0]  # 2×3 matrix
    #     U = [1.0 0.0; 0.0 1.0; 0.0 0.0]  # 3×2 matrix
    #     hz = [0.0, 0.0]
    #     h = [0.0, 0.0, 0