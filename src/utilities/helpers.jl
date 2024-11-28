"""
Helper function to detect nan or inf values in array (CPU)
"""
function detect_nan_or_inf(arr::Array)
    return any(isnan.(arr)) || any(isinf.(arr))
end


"""
Helper function to detect nan or inf values in CuArray (GPU)
"""
function detect_nan_or_inf(cuarr::CuArray)
    return CUDA.any(CUDA.isnan.(cuarr)) || CUDA.any(CUDA.isinf.(cuarr))
end

"""
Helper function to compute if vec is element of an array of vectors arr_of_vec
"""
function not_element_of(vec::Union{SubArray, Array}, arr_of_vec::Array; digits = 2)
    if isempty(arr_of_vec) 
        return true 
    else 
        return round.(vec, digits = digits) ∉ map(temp -> round.(temp, digits = digits), arr_of_vec)
    end
end

"""
Helper function to convert Array of shape order x dim into Array of order x Arrays of shape dim
"""
function convert_matrix_to_array(trajectory_matrix::Array, order)
    trajectory_array = Array[]
    for i in 1:order
        push!(trajectory_array, trajectory_matrix[:, i])
    end
    return trajectory_array
end


"""
Construct an array consisting of diagonals of relu matrices for a random sequence of quadrants drawn from the allowed D's of the pool
shPLRNN, inplace
"""
function construct_relu_matrix_diagonals!(relu_matrix_diagonals::Array, relu_pool::Union{Array, Nothing}, order::Integer)
    if relu_pool === nothing
        Random.rand!(relu_matrix_diagonals)
    else
        relu_matrix_diagonals .= relu_pool[:,Random.rand(1:size(relu_pool)[2], order)]
    end
end


"""
Construct an array consisting of diagonals of relu matrices for a random sequence of quadrants drawn from the allowed D's of the pool
shPLRNN, inplace, GPU
"""
function construct_relu_matrix_diagonals!(relu_matrix_diagonals::CuArray, relu_pool::Union{CuArray, Nothing}, order::Integer; rng::CUDA.RNG = CUDA.default_rng())
    if relu_pool === nothing
        Random.rand!(rng, relu_matrix_diagonals) 
    else
        relu_matrix_diagonals .= relu_pool[:,Random.rand(1:size(relu_pool)[2], order)]
    end
end

"""
Initialise pool of admissable Relu matrices for the (clipped) shallow PLRNN
CPU
"""
function construct_relu_matrix_pool(W₂::Array, h₂::Array, latent_dim::Integer, hidden_dim::Integer; search_space::Array = [-10, 10], n_points::Integer = 10000000,  n_splits::Integer = 1, type::Union{Type{Float32}, Type{Float64}} = eltype(W₂)) 
    """ heuristic estimate tested up to latent_dim = 10000, hidden_dim = 10000, n_points = 10000000 for 1024 GB RAM 
        set n_splits high enough (<= n_points though) if you encounter CUDA.error (Code 700)
        else try setting n_points lower
    """

    # credit @Kai Rothe & @Niclas Goering

    heuristic_estimate = 2
    n_splits = min(n_points, max(n_splits, ceil(Integer, heuristic_estimate * (hidden_dim + latent_dim) * n_points * sizeof(type) / Sys.free_memory())))
    
    relu_pool = Array[]
    while n_splits <= n_points
        try 
            z = Array{type}(undef, (latent_dim, n_points ÷ n_splits))
            relu_diagonals = Array{Bool}(undef, (hidden_dim, n_points ÷ n_splits))
            for _ in 1:n_splits
                rand!(z)
                z .= z .* (search_space[2] - search_space[1]) .+ search_space[1] # scale 
                relu_diagonals .= Bool.(W₂ * z .+  h₂ .> 0)
                push!(relu_pool, relu_diagonals)
            end
            return unique(hcat(relu_pool...), dims = 2)
        catch error 
            if n_splits == n_points # avoid infinite loop in any case
                throw(error)
            elseif (error isa OutOfMemoryError) || (error isa StackOverflowError) # avoid out of memory errors for large dimensions 
                n_splits = min(2 * n_splits, n_points)
            else
                println("Encountered $error during relu pool construction, try setting n_splits higher or n_points lower.")
                throw(error)
            end
        end
    end
end

"""
Initialise pool of admissable Relu matrices for the (clipped) shallow PLRNN
GPU
"""
function construct_relu_matrix_pool(W₂::CuArray, h₂::CuArray, latent_dim::Integer, hidden_dim::Integer; search_space::Array = [-10, 10], n_points::Integer = 10000000, n_splits::Integer = 1, type::Union{Type{Float32}, Type{Float64}} = eltype(W₂)) 
    """ heuristic estimate tested up to latent_dim = 10000, hidden_dim = 10000, n_points = 10000000, n_points = 10000000 on Nvidia Quadro RTX 6000 (23040 MiB Memory) 
        set n_splits high enough (<= n_points though) if you encounter CUDA.error (Code 700)
        else try setting n_points lower
    """
    
    # credit @Kai Rothe & @Niclas Goering
    
    heuristic_estimate = 2
    n_splits = min(n_points, max(n_splits, ceil(Integer, heuristic_estimate * (hidden_dim + latent_dim) * n_points * sizeof(type) / CUDA.free_memory())))
    
    relu_pool = Array[]
    while n_splits <= n_points
        try 
            z = CuArray{type}(undef, (latent_dim, n_points ÷ n_splits))
            relu_diagonals = CuArray{Bool}(undef, (hidden_dim, n_points ÷ n_splits))
            for _ in 1:n_splits
                Random.rand!(CUDA.default_rng(), z)
                z .= z .* (search_space[2] - search_space[1]) .+ search_space[1] # scale 
                relu_diagonals .= Bool.(W₂ * z .+  h₂ .> 0)
                push!(relu_pool, Array(relu_diagonals))
            end
            return CuArray(unique(hcat(relu_pool...), dims = 2))
        catch error 
            if n_splits == n_points # avoid infinite loop in any case
                throw(error)
            elseif (error isa OutOfGPUMemoryError) || (error isa OutOfMemoryError) || (error isa StackOverflowError) # avoid out of memory errors for large dimensions 
                n_splits = min(2 * n_splits, n_points)
            else
                println("Encountered $error during relu pool construction, try setting n_splits higher or n_points lower.")
                throw(error)
            end
        end
    end
end

"""
Initialise pool of admissable Relu matrices for a specific shallow PLRNN by drawing trajectories and storing the visited regions
shPLRNN, inplace 
"""
function construct_relu_matrix_pool_traj(A::Array, W::Array, h::Array, num_relus::Integer, dim::Integer, PLRNN::ALRNN; num_trajectories::Integer = 10, len_trajectories::Integer = 100, search_space::Array = [-10, 10], initial_conditions::Array = [], type::Union{Type{Float32}, Type{Float64}} = eltype(A)) 
    # preallocate big arrays
    trajectory_relu_matrix_list = Array{Bool}(undef, hidden_dim, len_trajectories, num_trajectories) 
    trajectory = Array{type}(undef, latent_dim, len_trajectories)
    z_0 = Array{type}(undef, latent_dim)
    temp = Array{type}(undef, hidden_dim)
    n_0 = length(initial_conditions)

    # fill trajectory_relu_matrix_list uniformely from trajectories starting at given initial conditions
    for i = 1:n_0
        z_0 .= initial_conditions[i]
        get_latent_time_series!(trajectory, view(trajectory_relu_matrix_list, :, :, i), len_trajectories, A, W, h, num_relus,dim, z_0, temp)
    end

    # fill remaining trajectory_relu_matrix_list uniformely from trajectories starting at random initial conditions
    for i = (n_0 + 1):num_trajectories 
        rand!(z_0) # in [0, 1)
        z_0 .= z_0 .* (search_space[2] - search_space[1]) .+ search_space[1] # scale
        get_latent_time_series!(trajectory, view(trajectory_relu_matrix_list, :, :, i), len_trajectories,  A, W, h, num_relus,dim, z_0, temp) 
    end

    # return unique regions
    trajectory_relu_matrix_list = reshape(trajectory_relu_matrix_list, (dim, len_trajectories * num_trajectories))
    return unique(trajectory_relu_matrix_list, dims=2)
end 


"""
Initialise pool of admissable Relu matrices for a specific shallow PLRNN by drawing trajectories and storing the visited regions
shPLRNN, inplace 
"""
function construct_relu_matrix_pool_traj(A::Array, W₁::Array, W₂::Array, h₁::Array, h₂::Array, latent_dim::Integer, hidden_dim::Integer, PLRNN::ShallowPLRNN; num_trajectories::Integer = 10, len_trajectories::Integer = 100, search_space::Array = [-10, 10], initial_conditions::Array = [], type::Union{Type{Float32}, Type{Float64}} = eltype(A)) 
    # preallocate big arrays
    trajectory_relu_matrix_list = Array{Bool}(undef, hidden_dim, len_trajectories, num_trajectories) 
    trajectory = Array{type}(undef, latent_dim, len_trajectories)
    z_0 = Array{type}(undef, latent_dim)
    temp = Array{type}(undef, hidden_dim)
    n_0 = length(initial_conditions)

    # fill trajectory_relu_matrix_list uniformely from trajectories starting at given initial conditions
    for i = 1:n_0
        z_0 .= initial_conditions[i]
        get_latent_time_series!(trajectory, view(trajectory_relu_matrix_list, :, :, i), len_trajectories, A, W₁, W₂, h₁, h₂, z_0, temp)
    end

    # fill remaining trajectory_relu_matrix_list uniformely from trajectories starting at random initial conditions
    for i = (n_0 + 1):num_trajectories 
        rand!(z_0) # in [0, 1)
        z_0 .= z_0 .* (search_space[2] - search_space[1]) .+ search_space[1] # scale
        get_latent_time_series!(trajectory, view(trajectory_relu_matrix_list, :, :, i), len_trajectories, A, W₁, W₂, h₁, h₂, z_0, temp) 
    end

    # return unique regions
    trajectory_relu_matrix_list = reshape(trajectory_relu_matrix_list, (hidden_dim, len_trajectories * num_trajectories))
    return unique(trajectory_relu_matrix_list, dims=2)
end 

"""
Initialise pool of admissable Relu matrices for a specific shallow PLRNN by drawing trajectories and storing the visited regions
shPLRNN, inplace, GPU
"""
function construct_relu_matrix_pool_traj(A::CuArray, W₁::CuArray, W₂::CuArray, h₁::CuArray, h₂::CuArray, latent_dim::Integer, hidden_dim::Integer, PLRNN::ShallowPLRNN; num_trajectories::Integer = 10, len_trajectories::Integer = 100, search_space::Array = [-10, 10], initial_conditions::Array=[], type::Union{Type{Float32}, Type{Float64}} = eltype(A)) # TODO: n_points proportional to 2^(hidden_dim) 
    # preallocate big arrays
    trajectory_relu_matrix_list = CuArray{Bool}(undef, hidden_dim, len_trajectories, num_trajectories) 
    trajectory = CuArray{type}(undef, latent_dim, len_trajectories)
    z_0 = CuArray{type}(undef, latent_dim)
    temp = CuArray{type}(undef, hidden_dim)
    n_0 = length(initial_conditions)

    # fill trajectory_relu_matrix_list uniformely from trajectories starting at given initial conditions
    for i = 1:n_0
        CUDA.copyto!(z_0, initial_conditions[i])
        get_latent_time_series!(trajectory, CUDA.view(trajectory_relu_matrix_list, :, :, i), len_trajectories, A, W₁, W₂, h₁, h₂, z_0, temp) 
    end
    
    # fill trajectory_relu_matrix_list uniformely from trajectories starting at random initial conditions
    for i = (n_0+1):num_trajectories
        CUDA.rand!(z_0) # in [0, 1)
        z_0 .= z_0 .* (search_space[2] - search_space[1]) .+ search_space[1] # scale
        get_latent_time_series!(trajectory, CUDA.view(trajectory_relu_matrix_list, :, :, i), len_trajectories, A, W₁, W₂, h₁, h₂, z_0, temp) 
    end

    # free some memory on GPU
    CUDA.unsafe_free!(trajectory) 
    CUDA.unsafe_free!(z_0)
    CUDA.unsafe_free!(temp)

    # return unique regions
    trajectory_relu_matrix_list = CUDA.reshape(trajectory_relu_matrix_list, (hidden_dim, len_trajectories * num_trajectories))
    return CuArray(unique(Array(trajectory_relu_matrix_list), dims=2))
end 


"""
Initialise pool of admissable Relu matrices for a specific shallow PLRNN by drawing trajectories and storing the visited regions
ClippedShPLRNN, inplace 
"""
function construct_relu_matrix_pool_traj(A::Array, W₁::Array, W₂::Array, h₁::Array, h₂::Array, latent_dim::Integer, hidden_dim::Integer, PLRNN::ClippedShallowPLRNN; num_trajectories::Integer = 10, len_trajectories::Integer = 100, search_space::Array = [-10, 10], initial_conditions::Array=[], type::Union{Type{Float32}, Type{Float64}} = eltype(A)) 
    # preallocate big arrays
    trajectory_relu_matrix_list_1 = Array{Bool}(undef, hidden_dim, len_trajectories, num_trajectories)
    trajectory_relu_matrix_list_2 = Array{Bool}(undef, hidden_dim, len_trajectories, num_trajectories)
    trajectory = Array{type}(undef, latent_dim, len_trajectories)
    z_0 = Array{type}(undef, latent_dim)
    temp = Array{type}(undef, hidden_dim)
    n_0 = length(initial_conditions)

    # fill trajectory_relu_matrix_list uniformely from trajectories starting at given initial conditions
    for i = 1:n_0
        z_0 .= initial_conditions[i]
        get_latent_time_series!(trajectory, view(trajectory_relu_matrix_list_1, :, :, i), view(trajectory_relu_matrix_list_2, :, :, i), len_trajectories, A, W₁, W₂, h₁, h₂, z_0, temp) 
    end

    # fill trajectory_relu_matrix_list uniformely from trajectories starting at random initial conditions in search space
    for i = (n_0 + 1):num_trajectories 
        rand!(z_0) # in [0, 1)
        z_0 .= z_0 .* (search_space[2] - search_space[1]) .+ search_space[1] # scale
        get_latent_time_series!(trajectory, view(trajectory_relu_matrix_list_1, :, :, i), view(trajectory_relu_matrix_list_2, :, :, i), len_trajectories, A, W₁, W₂, h₁, h₂, z_0, temp) 
    end

    # return unique regions
    trajectory_relu_matrix_list_1 = reshape(trajectory_relu_matrix_list_1, (hidden_dim, len_trajectories * num_trajectories))
    return unique(trajectory_relu_matrix_list_1, dims=2)
end 

"""
Initialise pool of admissable Relu matrices for a specific shallow PLRNN by drawing trajectories and storing the visited regions
ClippedShPLRNN, inplace, GPU
"""
function construct_relu_matrix_pool_traj(A::CuArray, W₁::CuArray, W₂::CuArray, h₁::CuArray, h₂::CuArray, latent_dim::Integer, hidden_dim::Integer, PLRNN::ClippedShallowPLRNN; num_trajectories::Integer = 10, len_trajectories::Integer = 100, search_space::Array = [-10, 10], initial_conditions::Array=[], type::Union{Type{Float32}, Type{Float64}} = eltype(A)) # TODO: n_points proportional to 2^(hidden_dim) 
    # preallocate big arrays
    trajectory_relu_matrix_list_1 = CuArray{Bool}(undef, hidden_dim, len_trajectories, num_trajectories)
    trajectory_relu_matrix_list_2 = CuArray{Bool}(undef, hidden_dim, len_trajectories, num_trajectories) 
    trajectory = CuArray{type}(undef, latent_dim, len_trajectories)
    z_0 = CuArray{type}(undef, latent_dim)
    temp = CuArray{type}(undef, hidden_dim)
    n_0 = length(initial_conditions)

    # fill trajectory_relu_matrix_list uniformely from trajectories starting at given initial conditions
    for i = 1:n_0
        CUDA.copyto!(z_0, initial_conditions[i])
        get_latent_time_series!(trajectory, CUDA.view(trajectory_relu_matrix_list_1, :, :, i), CUDA.view(trajectory_relu_matrix_list_2, :, :, i), len_trajectories, A, W₁, W₂, h₁, h₂, z_0, temp) 
    end

    # fill trajectory_relu_matrix_list uniformely from trajectories starting at random initial conditions
    for i = (n_0+1):num_trajectories
        CUDA.rand!(z_0) # in [0, 1)
        z_0 .= z_0 .* (search_space[2] - search_space[1]) .+ search_space[1] # scale
        get_latent_time_series!(trajectory, CUDA.view(trajectory_relu_matrix_list_1, :, :, i), CUDA.view(trajectory_relu_matrix_list_2, :, :, i), len_trajectories, A, W₁, W₂, h₁, h₂, z_0, temp) 
    end

    # free some memory on GPU
    CUDA.unsafe_free!(trajectory) 
    CUDA.unsafe_free!(z_0)
    CUDA.unsafe_free!(temp)

    # return unique regions
    trajectory_relu_matrix_list_1 = CUDA.reshape(trajectory_relu_matrix_list_1, (hidden_dim, len_trajectories * num_trajectories))
    return CuArray(unique(Array(trajectory_relu_matrix_list_1), dims=2))
end 


"""
IN-PLACE..
get the candidate for a cycle point by solving the cycle equation:
(A+WD_k)*...*(A+WD_1) *z + [(A+WD_k)*...*(A+WD_2)+...+(A+WD)**1+1]*h = z
"""
function get_cycle_point_candidate!(z_candidate::Union{Array, SubArray}, A::Array, W::Array, D_diagonals::Array, h::Array, order::Integer, z_factor::Union{Array, SubArray}, h_factor::Union{Array, SubArray}, temp::Union{Array, SubArray})
    get_factors!(z_factor, h_factor, A, W, D_diagonals, order, temp)
    invertible = true
    try 
        z_candidate .= (I - z_factor) \ (h_factor * h) 
    catch
        invertible = false
        if any(isnan.(z_factor)) || any(isinf.(z_factor)) 
            println("Detected nan or inf for cycle order k = $order. Try increasing input type $(eltype(A)) to higher precision.")
        end
    end
    return invertible 
end


"""
IN-PLACE, GPU..
get the candidate for a cycle point by solving the cycle equation:
(A+WD_k)*...*(A+WD_1) *z + [(A+WD_k)*...*(A+WD_2)+...+(A+WD)**1+1]*h = z
"""
function get_cycle_point_candidate!(z_candidate::CuArray, A::CuArray, W::CuArray, D_diagonals::CuArray, h::CuArray, order::Integer, z_factor::CuArray, h_factor::CuArray, temp::CuArray)
    get_factors!(z_factor, h_factor, A, W, D_diagonals, order, temp)
    invertible = true
    try 
        z_candidate .= (CUDA.I - z_factor) \ (h_factor * h) 
    catch
        invertible = false
    end
    return invertible 
end


"""
get the candidate for a cycle point by solving the cycle equation
shPLRNN, inplace
"""
function get_cycle_point_candidate!(z_candidate::Union{SubArray, Array}, A::Diagonal,
    W₁::AbstractMatrix, W₂::AbstractMatrix, h₁::AbstractVector, h₂::AbstractVector, D_diagonals::Array, order::Integer, 
    z_factor::Union{Array, SubArray}, h₁_factor::Union{Array, SubArray}, h₂_factor::Union{Array, SubArray}, 
    temp_1::Union{Array, SubArray}, temp_2::Union{Array, SubArray}
)
    get_factors!(z_factor, h₁_factor, h₂_factor, A, W₁, W₂, D_diagonals, order, temp_1, temp_2)
    invertible = true
    try 
        z_candidate .= (I - z_factor) \ (h₁_factor * h₁ .+ h₂_factor * h₂) 
    catch
        invertible = false
        if any(isnan.(z_factor)) || any(isinf.(z_factor)) 
            println("Detected nan or inf for cycle order k = $order. Try increasing input type $(eltype(A)) to higher precision.")
        end
    end
    return invertible
end

"""
get the candidate for a cycle point by solving the cycle equation
shPLRNN, inplace, GPU
"""
function get_cycle_point_candidate!(z_candidate::CuArray, A::Diagonal,
    W₁::CuArray, W₂::CuArray, h₁::CuArray, h₂::CuArray, D_diagonals::CuArray, order::Integer, 
    z_factor::CuArray, h₁_factor::CuArray, h₂_factor::CuArray, 
    temp_1::CuArray, temp_2::CuArray
)
    get_factors!(z_factor, h₁_factor, h₂_factor, A, W₁, W₂, D_diagonals, order, temp_1, temp_2)
    invertible = true
    try 
        z_candidate .= (CUDA.I - z_factor) \ (h₁_factor * h₁ .+ h₂_factor * h₂) 
    catch
        invertible = false
    end
    return invertible 
end


"""
get the candidate for a cycle point by solving the cycle equation
clipped shPLRNN, inplace
"""
function get_cycle_point_candidate!(z_candidate::Union{SubArray, Array}, A::Diagonal,
    W₁::AbstractMatrix, W₂::AbstractMatrix, h₁::AbstractVector, h₂::AbstractVector, D_diagonals_1::Array, D_diagonals_2::Array, order::Integer, 
    z_factor::Union{Array, SubArray}, h₁_factor::Union{Array, SubArray}, h₂_factor::Union{Array, SubArray}, 
    temp_1::Union{Array, SubArray}, temp_2::Union{Array, SubArray}
)
    get_factors!(z_factor, h₁_factor, h₂_factor, A, W₁, W₂, D_diagonals_1, D_diagonals_2, order, temp_1, temp_2)
    invertible = true
    try 
        z_candidate .= (I - z_factor) \ (h₁_factor * h₁ .+ h₂_factor * h₂)
    catch
        invertible = false
        #println("Matrix is not invertible", "A",A,"W",W)
        if any(isnan.(z_factor)) || any(isinf.(z_factor)) 
            println("Detected nan or inf for cycle order k = $order. Try increasing input type $(eltype(A)) to higher precision.")
        end
    end
    return invertible
end

"""
get the candidate for a cycle point by solving the cycle equation
clipped shPLRNN, inplace, GPU
"""
function get_cycle_point_candidate!(z_candidate::CuArray, A::Diagonal,
    W₁::CuArray, W₂::CuArray, h₁::CuArray, h₂::CuArray, D_diagonals_1::CuArray, D_diagonals_2::CuArray, order::Integer, 
    z_factor::CuArray, h₁_factor::CuArray, h₂_factor::CuArray, 
    temp_1::CuArray, temp_2::CuArray
)
    get_factors!(z_factor, h₁_factor, h₂_factor, A, W₁, W₂, D_diagonals_1, D_diagonals_2, order, temp_1, temp_2)
    invertible = true
    try 
        z_candidate .= (CUDA.I - z_factor) \ (h₁_factor * h₁ .+ h₂_factor * h₂)
    catch
        invertible = false
    end
    return invertible
end


"""
recursively applying map gives us: (A+WD_k)*...*(A+WD_1) *z + [(A+WD_k)*...*(A+WD_2)+...+(A+WD)**1+1]*h = z
Here we want to calculate the factor in front of z and h recursively
inplace, PLRNN
"""
function get_factors!(z_factor::Union{SubArray, Array}, h_factor::Union{Array, SubArray}, A::Array, W::Array, D_diagonals::Array, order::Integer, temp::Union{Array, SubArray})
    copyto!(z_factor, I)
    copyto!(h_factor, I)

    for i = reverse(2:order)
        copy!(temp, z_factor)
        mul!(z_factor, temp, A .+ W .* D_diagonals[:,i]') # W .* D_diagonals[:,i]' == W * Diagonal(D_diagonals[:,i])
        h_factor .+= z_factor
    end

    copy!(temp, z_factor)
    mul!(z_factor, temp, A .+ W .* D_diagonals[:,1]')
end

"""
recursively applying map gives us: (A+WD_k)*...*(A+WD_1) *z + [(A+WD_k)*...*(A+WD_2)+...+(A+WD)**1+1]*h = z
Here we want to calculate the factor in front of z and h recursively
inplace, GPU
"""
function get_factors!(z_factor::CuArray, h_factor::CuArray, A::CuArray, W::CuArray, D_diagonals::CuArray, order::Integer, temp::CuArray)
    CUDA.copyto!(z_factor, CUDA.I)
    CUDA.copyto!(h_factor, CUDA.I)

    for i = reverse(2:order)
        CUDA.copy!(temp, z_factor)
        mul!(z_factor, temp, A .+ W .* D_diagonals[:,i]') # W .* D_diagonals[i,:]' == W * Diagonal(D_diagonals[i,:])
        h_factor .+= z_factor
    end

    CUDA.copy!(temp, z_factor)
    mul!(z_factor, temp, A .+ W .* D_diagonals[:,1]')
end


"""
Here we want to calculate the factors in front of z/h_1/h_2 recursively
shPLRNN, inplace
"""
function get_factors!(z_factor::Union{SubArray, Array}, h₁_factor::Union{Array, SubArray}, h₂_factor::Union{Array, SubArray}, 
    A::Diagonal, W₁::AbstractMatrix, W₂::AbstractMatrix, D_diagonals::Array, order::Integer, temp_1::Union{SubArray, Array}, temp_2::Union{SubArray, Array}
)
    copyto!(z_factor, I)
    copyto!(h₁_factor, I)
    @views temp_1 .= W₁ .* D_diagonals[:,order]'
    copy!(h₂_factor, temp_1)

    for i = reverse(2:order)
        copy!(temp_2, z_factor)
        mul!(z_factor, temp_2, A .+ temp_1 * W₂)
        h₁_factor .+= z_factor
        @views temp_1 .= W₁ .* D_diagonals[:,i-1]'
        h₂_factor .+= z_factor * temp_1
    end

    copy!(temp_2, z_factor)
    mul!(z_factor, temp_2, A .+ temp_1 * W₂)

    return z_factor, h₁_factor, h₂_factor
end

"""
Here we want to calculate the factors in front of z/h_1/h_2 recursively
shPLRNN, inplace, GPU
"""
function get_factors!(z_factor::CuArray, h₁_factor::CuArray, h₂_factor::CuArray, 
    A::Diagonal, W₁::CuArray, W₂::CuArray, D_diagonals::CuArray, order::Integer, temp_1::CuArray, temp_2::CuArray
)
    CUDA.copyto!(z_factor, CUDA.I)
    CUDA.copyto!(h₁_factor, CUDA.I)
    CUDA.@views temp_1 .= W₁ .* D_diagonals[:,order]'
    CUDA.copy!(h₂_factor, temp_1)

    for i = reverse(2:order)
        CUDA.copy!(temp_2, z_factor)
        mul!(z_factor, temp_2, A .+ temp_1 * W₂)
        h₁_factor .+= z_factor
        CUDA.@views temp_1 .= W₁ .* D_diagonals[:,i-1]'
        h₂_factor .+= z_factor * temp_1
    end

    CUDA.copy!(temp_2, z_factor)
    mul!(z_factor, temp_2, A .+ temp_1 * W₂)

    return z_factor, h₁_factor, h₂_factor
end


"""
Here we want to calculate the factors in front of z/h_1/h_2 recursively
clipped shPLRNN, inplace
"""
function get_factors!(z_factor::Union{SubArray, Array}, h₁_factor::Union{Array, SubArray}, h₂_factor::Union{Array, SubArray}, 
    A::Diagonal, W₁::AbstractMatrix, W₂::AbstractMatrix, D_diagonals_1::Array, D_diagonals_2::Array, order::Integer, temp_1::Union{SubArray, Array}, temp_2::Union{SubArray, Array}
)
    copyto!(z_factor, I)
    copyto!(h₁_factor, I)
    @views temp_1 .= W₁ .* D_diagonals_1[:,order]'
    copy!(h₂_factor, temp_1)

    for i = reverse(2:order)
        temp_1 .-= W₁ .* D_diagonals_2[:,i]' # temp_1 .= W₁ * (Diagonal(D_diagonals_1[:,i]) .- Diagonal(D_diagonals_2[:,i]) due to previous loop / initializing
        copy!(temp_2, z_factor)
        mul!(z_factor, temp_2, A .+ temp_1 * W₂)
        h₁_factor .+= z_factor
        @views temp_1 .= W₁ .* D_diagonals_1[:,i-1]'
        h₂_factor .+= z_factor * temp_1
    end

    @views temp_1 .-= W₁ .* D_diagonals_2[:,1]'
    copy!(temp_2, z_factor)
    mul!(z_factor, temp_2, A .+ temp_1 * W₂)

    return z_factor, h₁_factor, h₂_factor
end

"""
Here we want to calculate the factors in front of z/h_1/h_2 recursively
clipped shPLRNN, inplace, GPU
"""
function get_factors!(z_factor::CuArray, h₁_factor::CuArray, h₂_factor::CuArray, 
    A::Diagonal, W₁::CuArray, W₂::CuArray, D_diagonals_1::CuArray, D_diagonals_2::CuArray, order::Integer, temp_1::CuArray, temp_2::CuArray
)
    CUDA.copyto!(z_factor,CUDA.I)
    CUDA.copyto!(h₁_factor, CUDA.I)
    CUDA.@views temp_1 .= W₁ .* D_diagonals_1[:,order]'
    CUDA.copy!(h₂_factor, temp_1)

    for i = reverse(2:order)
        temp_1 .-= W₁ .* D_diagonals_2[:,i]' # temp_1 .= W₁ * (Diagonal(D_diagonals_1[:,i]) .- Diagonal(D_diagonals_2[:,i]) due to previous loop / initializing
        CUDA.copy!(temp_2, z_factor)
        mul!(z_factor, temp_2, A .+ temp_1 * W₂)
        h₁_factor .+= z_factor
        CUDA.@views temp_1 .= W₁ .* D_diagonals_1[:,i-1]'
        h₂_factor .+= z_factor * temp_1
    end

    CUDA.@views temp_1 .-= W₁ .* D_diagonals_2[:,1]'
    CUDA.copy!(temp_2, z_factor)
    mul!(z_factor, temp_2, A .+ temp_1 * W₂)

    return z_factor, h₁_factor, h₂_factor
end


"""
Generate the time series by iteravely applying the PLRNN
inplace, plrnn, cpu
"""
function get_latent_time_series!(trajectory::Array, time_steps::Integer, A::Array, W::Array, h::Array, z::Union{Array, SubArray})
    trajectory[:, 1] .= z 
    for t = 2:time_steps
        latent_step!(z, A, W, h) # z = A * z .+ W * max.(0,z) .+ h
        trajectory[:, t] .= z
    end
    return trajectory
end


"""
Generate the time series by iteravely applying the PLRNN
inplace, plrnn, gpu
"""
function get_latent_time_series!(trajectory::CuArray, time_steps::Integer, A::CuArray, W::CuArray, h::CuArray, z::CuArray)
    trajectory[:, 1] .= z 
    for t = 2:time_steps
        latent_step!(z, A, W, h) # z = A * z .+ W * max.(0,z) .+ h
        trajectory[:, t] .= z 
    end
    return trajectory
end

"""
Generate the time series by iteravely applying the ALRNN
inplace, plrnn, cpu
"""
function get_latent_time_series!(trajectory::Array, time_steps::Integer, A::Array, W::Array, h::Array, num_relus::Integer, z::Union{Array, SubArray})
    trajectory[:, 1] .= z 
    for t = 2:time_steps
        latent_step!(z, A, W, h, num_relus) # z = A * z .+ W * max.(0,z) .+ h
        trajectory[:, t] .= z
    end
    return trajectory
end


"""
Generate the time series by iteravely applying the ALRNN
inplace, plrnn, gpu
"""
function get_latent_time_series!(trajectory::CuArray, time_steps::Integer, A::CuArray, W::CuArray, h::CuArray, z::CuArray, num_relus::Integer)
    trajectory[:, 1] .= z 
    for t = 2:time_steps
        latent_step!(z, A, W, h, num_relus) # z = A * z .+ W * max.(0,z) .+ h
        trajectory[:, t] .= z 
    end
    return trajectory
end

"""
Generate the time series (and according diagonals of relu matrices) by iteravely applying the ALRNN, inplace
"""
function get_latent_time_series!(trajectory::Array, relu_matrix_diagonals::Union{SubArray,Array}, time_steps::Integer,     
    A::AbstractVector, W::AbstractMatrix, h::AbstractVector, num_relus::Integer,dim::Integer,
    z_0::Union{Array, SubArray}, temp::Union{SubArray, Array})
    
    trajectory[:,1] .= z_0
    @views temp .= trajectory[:,1] # use as temporary preallocated space
    relu_matrix_diagonals[:,1] .= (temp .> 0)
    relu_matrix_diagonals[1:dim-num_relus,1] .= 1 # first num_relus are always active
    
    @views for t = 2:time_steps 
        trajectory[:,t] .= A .* trajectory[:,t-1] .+ W₁ * (relu_matrix_diagonals[:,t-1] .* temp) .+ h₁ 
        temp .= trajectory[:,t] 
        relu_matrix_diagonals[:,t] .= (temp .> 0)
        relu_matrix_diagonals[1:dim-num_relus,t] .= 1 # first num_relus are always active
    end

    return trajectory, relu_matrix_diagonals
end


"""
Generate the time series (and according diagonals of relu matrices) by iteravely applying the shPLRNN 
shPLRNN, inplace
"""
function get_latent_time_series!(trajectory::Array, relu_matrix_diagonals::Union{SubArray,Array}, time_steps::Integer,     
    A::AbstractVector, W₁::AbstractMatrix, W₂::AbstractMatrix, h₁::AbstractVector, h₂::AbstractVector,
    z_0::Union{Array, SubArray}, temp::Union{SubArray, Array})
    
    trajectory[:,1] .= z_0
    @views temp .= W₂ * trajectory[:,1] .+ h₂ # use as temporary preallocated space
    relu_matrix_diagonals[:,1] .= (temp .> 0)
    
    @views for t = 2:time_steps 
        trajectory[:,t] .= A .* trajectory[:,t-1] .+ W₁ * (relu_matrix_diagonals[:,t-1] .* temp) .+ h₁ 
        temp .= W₂ * trajectory[:,t] .+ h₂
        relu_matrix_diagonals[:,t] .= (temp .> 0)
    end

    return trajectory, relu_matrix_diagonals
end

"""
Generate the time series (and according diagonals of relu matrices) by iteravely applying the shPLRNN 
shPLRNN, inplace, GPU
"""
function get_latent_time_series!(trajectory::CuArray, relu_matrix_diagonals::CuArray, time_steps::Integer,     
    A::CuArray, W₁::CuArray, W₂::CuArray, h₁::CuArray, h₂::CuArray,
    z_0::CuArray, temp::CuArray)
    
    trajectory[:,1] .= z_0
    CUDA.@views temp .= W₂ * trajectory[:,1] .+ h₂ # use as temporary preallocated space
    relu_matrix_diagonals[:,1] .= (temp .> 0)
    
    CUDA.@views for t = 2:time_steps 
        trajectory[:,t] .= A .* trajectory[:,t-1] .+ W₁ * (relu_matrix_diagonals[:,t-1] .* temp) .+ h₁ 
        temp .= W₂ * trajectory[:,t] .+ h₂
        relu_matrix_diagonals[:,t] .= (temp .> 0)
    end

    return trajectory, relu_matrix_diagonals
end


"""
Generate the time series (and according diagonals of relu matrices) by iteravely applying the shPLRNN 
ClippedshPLRNN, inplace
"""
@views function get_latent_time_series!(trajectory::Array, relu_matrix_diagonals_1::Union{SubArray,Array}, relu_matrix_diagonals_2::Union{SubArray,Array}, time_steps::Integer,     
    A::AbstractVector, W₁::AbstractMatrix, W₂::AbstractMatrix, h₁::AbstractVector, h₂::AbstractVector,
    z_0::Union{Array, SubArray}, temp::Union{SubArray, Array})
    
    trajectory[:,1] .= z_0
    mul!(temp, W₂, view(trajectory, :, 1)) # use as temporary preallocated space
    relu_matrix_diagonals_1[:,1] .= ((temp .+ h₂) .> 0)
    relu_matrix_diagonals_2[:,1] .= (temp .> 0)
    
    @views for t = 2:time_steps 
        trajectory[:,t] .= A .* trajectory[:,t-1] .+ W₁ * (relu_matrix_diagonals_1[:,t-1] .* (temp .+ h₂) .- relu_matrix_diagonals_2[:,t-1] .* temp) .+ h₁
        mul!(temp, W₂, view(trajectory, :, t)) # inplace version of z .= W₂ * trajectory[:,t]
        relu_matrix_diagonals_1[:,t] .= ((temp .+ h₂) .> 0)
        relu_matrix_diagonals_2[:,t] .= (temp .> 0)
    end

    return trajectory, relu_matrix_diagonals_1, relu_matrix_diagonals_2
end

"""
Generate the time series (and according diagonals of relu matrices) by iteravely applying the shPLRNN 
ClippedshPLRNN, inplace, GPU
"""
@views function get_latent_time_series!(trajectory::CuArray, relu_matrix_diagonals_1::CuArray, relu_matrix_diagonals_2::CuArray, time_steps::Integer,     
    A::CuArray, W₁::CuArray, W₂::CuArray, h₁::CuArray, h₂::CuArray,
    z_0::CuArray, temp::CuArray)
    
    trajectory[:,1] .= z_0
    mul!(temp, W₂, CUDA.view(trajectory, :, 1)) # use as temporary preallocated space
    relu_matrix_diagonals_1[:,1] .= ((temp .+ h₂) .> 0)
    relu_matrix_diagonals_2[:,1] .= (temp .> 0)
    
    CUDA.@views for t = 2:time_steps 
        trajectory[:,t] .= A .* trajectory[:,t-1] .+ W₁ * (relu_matrix_diagonals_1[:,t-1] .* (temp .+ h₂) .- relu_matrix_diagonals_2[:,t-1] .* temp) .+ h₁
        mul!(temp, W₂, CUDA.view(trajectory, :, t)) # inplace version of z .= W₂ * trajectory[:,t]
        relu_matrix_diagonals_1[:,t] .= ((temp .+ h₂) .> 0)
        relu_matrix_diagonals_2[:,t] .= (temp .> 0)
    end

    return trajectory, relu_matrix_diagonals_1, relu_matrix_diagonals_2
end



"""
Generate the time series by iteravely applying the shPLRNN
shPLRNN
"""
function get_latent_time_series(time_steps:: Integer,     
    A::AbstractVector,
    W₁::AbstractMatrix,
    W₂::AbstractMatrix,
    h₁::AbstractVector,
    h₂::AbstractVector,
    dz::Integer;
    z_0:: Array= nothing,
    is_clipped::Bool=false)
    if z_0 === nothing
        z = transpose(randn(1,dz))
    else
        z = z_0
    end
    trajectory = Array{Array}(undef, time_steps)
    trajectory[1] = z
    for t = 2:time_steps
        z = latent_step(z, A, W₁, W₂, h₁, h₂, is_clipped)
        trajectory[t] = z
    end
    return trajectory
end

"""
PLRNN step
inplace, cpu
"""
function latent_step!(z::Union{SubArray, Array}, A::Array, W::Array, h::Array)
    z .= A * z .+ W * max.(0,z) .+ h
end


"""
PLRNN step
inplace, gpu
"""
function latent_step!(z::CuArray, A::CuArray, W::CuArray, h::CuArray)
    z .= A * z .+ W * max.(0,z) .+ h
end

"""
PLRNN step
inplace, cpu
"""
function latent_step!(z::Union{SubArray, Array}, A::Array, W::Array, h::Array, num_relus::Integer)
    z .= A * z .+ W * vcat(z[1:end-num_relus,:], max.(0,z)[end-(num_relus-1):end,:]) .+ h
end


"""
PLRNN step
inplace, gpu
"""
function latent_step!(z::CuArray, A::CuArray, W::CuArray, h::CuArray, num_relus::Integer)
    z .= A * z .+ W * vcat(z[1:end-num_relus,:], max.(0,z)[end-(num_relus-1):end,:]) .+ h
end



"""
shPLRNN step
"""
function  latent_step(
    z::AbstractArray,
    A::AbstractVector,
    W₁::AbstractMatrix,
    W₂::AbstractMatrix,
    h₁::AbstractVector,
    h₂::AbstractVector,
    is_clipped::Bool=false)
    if is_clipped
        return A .* z .+ W₁ * (max.(W₂ * z .+ h₂,0) .- max.(W₂ * z, 0)) .+ h₁
    else
        return A .* z .+ W₁ * max.(W₂ * z .+ h₂,0) .+ h₁
    end
end


"""
Set the hyperparameters to predefined values if no value given (tuned for 2D case)
"""
function set_loop_iterations(order:: Integer, outer_loop:: Union{Integer,Nothing}, inner_loop:: Union{Integer,Nothing})
    if outer_loop === nothing
        if order < 8 outer_loop=10 elseif order < 30 outer_loop=40 else outer_loop=100 end
    end
    if inner_loop === nothing
        if order < 3 inner_loop=20 elseif order < 6 inner_loop=60 elseif order < 8 inner_loop=300 elseif order < 20 inner_loop=1080 else inner_loop=1115 end
    end
    return outer_loop, inner_loop
end