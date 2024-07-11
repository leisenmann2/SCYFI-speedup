include("./scyfi_algo.jl")
include("./scyfi_parallel.jl")

""" 
calculate the cycles for a specified PLRNN with parameters A,W,h up until order k
GPU version
"""
function find_cycles(
    A::Array, W::Array, h::Array, order::Integer;
    outer_loop_iterations::Union{Integer,Nothing}= nothing,
    inner_loop_iterations::Union{Integer,Nothing} = nothing,
    PLRNN::VanillaPLRNN = VanillaPLRNN()
    )
    
    found_lower_orders = Array[] # always on CPU
    found_eigvals = Array[] # always on CPU

    dim = size(A)[1]
    type = eltype(A)

    # put arrays on GPU
    A_gpu = CuArray(A)
    W_gpu = CuArray(W)
    h_gpu = CuArray(h)

    if Threads.nthreads() >1
        println("mulit-thread parallel version")
        n_threads= Threads.nthreads()

        # pre-allocate for each thread for in-place version of SCYFI computations 
        z_candidates = CuArray{type}(undef, dim, n_threads)
        inplace_hs = CuArray{type}(undef, (dim, dim, n_threads)) 
        inplace_zs = CuArray{type}(undef, (dim, dim, n_threads))
        inplace_temps = CuArray{type}(undef, (dim, dim, n_threads))

        for i = 1:order
            scy_fi!(found_lower_orders, found_eigvals, A_gpu, W_gpu, h_gpu, i, n_threads, z_candidates, inplace_zs, inplace_hs, inplace_temps; outer_loop_iterations = outer_loop_iterations, inner_loop_iterations = inner_loop_iterations, PLRNN = PLRNN, dim = dim, type = type)
        end
    
    else 
        # pre-allocate for in-place version of SCYFI computations
        z_candidate = CuArray{type}(undef, dim)
        inplace_h = CuArray{type}(undef, (dim, dim))
        inplace_z = CuArray{type}(undef, (dim, dim))
        inplace_temp = CuArray{type}(undef, (dim, dim))

        for i = 1:order
            scy_fi!(found_lower_orders, found_eigvals, A_gpu, W_gpu, h_gpu, i, z_candidate, inplace_z, inplace_h, inplace_temp; outer_loop_iterations = outer_loop_iterations, inner_loop_iterations = inner_loop_iterations, PLRNN = PLRNN, dim=dim, type=type)
        end
    end
    return [found_lower_orders, found_eigvals]
end

""" 
calculate the cycles for a specified PLRNN with parameters A,W,h for order k in orders
GPU version
"""
function find_cycles(
    A::Array, W::Array, h::Array, orders::Array;
    outer_loop_iterations::Union{Integer,Nothing}= nothing,
    inner_loop_iterations::Union{Integer,Nothing} = nothing,
    PLRNN::VanillaPLRNN = VanillaPLRNN()
    )
    
    found_lower_orders = Array[] # always on CPU
    found_eigvals = Array[] # always on CPU

    dim = size(A)[1]
    type = eltype(A)

    # put arrays on GPU
    A_gpu = CuArray(A)
    W_gpu = CuArray(W)
    h_gpu = CuArray(h)

    if Threads.nthreads() >1
        println("mulit-thread parallel version")
        n_threads= Threads.nthreads()

        # pre-allocate for each thread for in-place version of SCYFI computations 
        z_candidates = CuArray{type}(undef, dim, n_threads)
        inplace_hs = CuArray{type}(undef, (dim, dim, n_threads)) 
        inplace_zs = CuArray{type}(undef, (dim, dim, n_threads))
        inplace_temps = CuArray{type}(undef, (dim, dim, n_threads))

        for i = orders
            scy_fi!(found_lower_orders, found_eigvals, A_gpu, W_gpu, h_gpu, i, n_threads, z_candidates, inplace_zs, inplace_hs, inplace_temps; outer_loop_iterations = outer_loop_iterations, inner_loop_iterations = inner_loop_iterations, PLRNN = PLRNN, dim = dim, type = type)
        end
    
    else 
        # pre-allocate for in-place version of SCYFI computations
        z_candidate = CuArray{type}(undef, dim)
        inplace_h = CuArray{type}(undef, (dim, dim))
        inplace_z = CuArray{type}(undef, (dim, dim))
        inplace_temp = CuArray{type}(undef, (dim, dim))

        for i = orders
            scy_fi!(found_lower_orders, found_eigvals, A_gpu, W_gpu, h_gpu, i, z_candidate, inplace_z, inplace_h, inplace_temp; outer_loop_iterations = outer_loop_iterations, inner_loop_iterations = inner_loop_iterations, PLRNN = PLRNN, dim=dim, type=type)
        end
    end
    return [found_lower_orders, found_eigvals]
end


""" 
calculate the cycles for a specified shPLRNN up until order k
shPLRNN, GPU version
"""
function find_cycles(
    A::AbstractVector,
    W₁::AbstractMatrix,
    W₂::AbstractMatrix,
    h₁::AbstractVector,
    h₂::AbstractVector,
    order::Integer;
    PLRNN::Union{ClippedShallowPLRNN, ShallowPLRNN} = ShallowPLRNN(),
    outer_loop_iterations::Union{Integer,Nothing} = nothing,
    inner_loop_iterations::Union{Integer,Nothing} = nothing,
    create_pool=true,
    get_pool_from_traj=false,
    num_trajectories::Integer=10,
    len_trajectories::Integer=100,
    search_space::Array = [-10, 10],
    initial_conditions::Array = []
    )

    found_lower_orders = Array[]
    found_eigvals = Array[]
    latent_dim = size(h₁)[1]
    hidden_dim = size(h₂)[1]
    type = eltype(A)

    # put arrays on GPU
    A_gpu = CuArray(A)
    W₁_gpu = CuArray(W₁)
    W₂_gpu = CuArray(W₂)
    h₁_gpu = CuArray(h₁)
    h₂_gpu = CuArray(h₂)

    # preallocate pool of allowed D matrices, in the shPLRNN there are overlapping regions which can be excluded, pre-creating them makes the algorithm more efficient        
    if get_pool_from_traj
        relu_pool = construct_relu_matrix_pool_traj(A_gpu, W₁_gpu, W₂_gpu, h₁_gpu, h₂_gpu, latent_dim, hidden_dim, PLRNN; num_trajectories = num_trajectories, len_trajectories=len_trajectories, search_space = search_space, initial_conditions = initial_conditions, type = type)
        println("Number of initialisations in Pool: ", size(relu_pool)[2])
    elseif create_pool
        relu_pool = construct_relu_matrix_pool(W₂_gpu, h₂_gpu, latent_dim, hidden_dim; search_space = search_space, type = type)
        println("Number of initialisations in Pool: ", size(relu_pool)[2])
    else 
        relu_pool = nothing
    end

    if Threads.nthreads() > 1
        println("multi-thread parallel version")
        n_threads= Threads.nthreads()

        # pre-allocate for each thread for in-place version of SCYFI computations 
        z_candidates = CuArray{type}(undef, (latent_dim, n_threads))
        inplace_h₁s = CuArray{type}(undef, (latent_dim, latent_dim, n_threads)) 
        inplace_h₂s = CuArray{type}(undef, (latent_dim, hidden_dim, n_threads)) 
        inplace_zs = CuArray{type}(undef, (latent_dim, latent_dim, n_threads))
        inplace_temp_1s = CuArray{type}(undef, (latent_dim, hidden_dim, n_threads))
        inplace_temp_2s = CuArray{type}(undef, (latent_dim, latent_dim, n_threads))
        inplace_temp_3s = CuArray{type}(undef, (hidden_dim, n_threads))

        for i = 1:order
            scy_fi!(found_lower_orders, found_eigvals, A_gpu, W₁_gpu, W₂_gpu, h₁_gpu, h₂_gpu, i, n_threads, relu_pool, PLRNN, z_candidates, inplace_zs, inplace_h₁s, inplace_h₂s, inplace_temp_1s, inplace_temp_2s; temp_3s = inplace_temp_3s, outer_loop_iterations = outer_loop_iterations, inner_loop_iterations = inner_loop_iterations, latent_dim = latent_dim, hidden_dim = hidden_dim, type = type) 
        end
    else 
        # pre-allocate for in-place version of SCYFI computations
        z_candidate = CuArray{type}(undef, latent_dim)
        inplace_h₁ = CuArray{type}(undef, (latent_dim, latent_dim)) 
        inplace_h₂ = CuArray{type}(undef, (latent_dim, hidden_dim)) 
        inplace_z = CuArray{type}(undef, (latent_dim, latent_dim))
        inplace_temp_1 = CuArray{type}(undef, (latent_dim, hidden_dim))
        inplace_temp_2 = CuArray{type}(undef, (latent_dim, latent_dim))
        inplace_temp_3 = CuArray{type}(undef, hidden_dim)

        for i = 1:order
            scy_fi!(found_lower_orders, found_eigvals, A_gpu, W₁_gpu, W₂_gpu, h₁_gpu, h₂_gpu, i, relu_pool, PLRNN, z_candidate, inplace_z, inplace_h₁, inplace_h₂, inplace_temp_1, inplace_temp_2; temp_3 = inplace_temp_3, outer_loop_iterations = outer_loop_iterations, inner_loop_iterations = inner_loop_iterations, latent_dim = latent_dim, hidden_dim = hidden_dim, type = type) 
        end
    end
    return [found_lower_orders, found_eigvals]
end


""" 
calculate the cycles for a specified shPLRNN for order k in orders
shPLRNN, GPU version
"""
function find_cycles(
    A::AbstractVector,
    W₁::AbstractMatrix,
    W₂::AbstractMatrix,
    h₁::AbstractVector,
    h₂::AbstractVector,
    orders::Array;
    PLRNN::Union{ClippedShallowPLRNN, ShallowPLRNN} = ShallowPLRNN(),
    outer_loop_iterations::Union{Integer,Nothing} = nothing,
    inner_loop_iterations::Union{Integer,Nothing} = nothing,
    create_pool=true,
    get_pool_from_traj=false,
    num_trajectories::Integer=10,
    len_trajectories::Integer=100,
    search_space::Array = [-10, 10],
    initial_conditions::Array = []
    )

    found_lower_orders = Array[]
    found_eigvals = Array[]
    latent_dim = size(h₁)[1]
    hidden_dim = size(h₂)[1]
    type = eltype(A)

    # put arrays on GPU
    A_gpu = CuArray(A)
    W₁_gpu = CuArray(W₁)
    W₂_gpu = CuArray(W₂)
    h₁_gpu = CuArray(h₁)
    h₂_gpu = CuArray(h₂)

    # preallocate pool of allowed D matrices, in the shPLRNN there are overlapping regions which can be excluded, pre-creating them makes the algorithm more efficient        
    if get_pool_from_traj
        relu_pool = construct_relu_matrix_pool_traj(A_gpu, W₁_gpu, W₂_gpu, h₁_gpu, h₂_gpu, latent_dim, hidden_dim, PLRNN; num_trajectories = num_trajectories, len_trajectories=len_trajectories, initial_conditions=initial_conditions, search_space = search_space, type = type)
        println("Number of initialisations in Pool: ", size(relu_pool)[2])
    elseif create_pool
        relu_pool = construct_relu_matrix_pool(W₂_gpu, h₂_gpu, latent_dim, hidden_dim; search_space = search_space, type = type)
        println("Number of initialisations in Pool: ", size(relu_pool)[2])
    else 
        relu_pool = nothing
    end

    if Threads.nthreads() > 1
        println("multi-thread parallel version")
        n_threads= Threads.nthreads()

        # pre-allocate for each thread for in-place version of SCYFI computations 
        z_candidates = CuArray{type}(undef, (latent_dim, n_threads))
        inplace_h₁s = CuArray{type}(undef, (latent_dim, latent_dim, n_threads)) 
        inplace_h₂s = CuArray{type}(undef, (latent_dim, hidden_dim, n_threads)) 
        inplace_zs = CuArray{type}(undef, (latent_dim, latent_dim, n_threads))
        inplace_temp_1s = CuArray{type}(undef, (latent_dim, hidden_dim, n_threads))
        inplace_temp_2s = CuArray{type}(undef, (latent_dim, latent_dim, n_threads))
        inplace_temp_3s = CuArray{type}(undef, (hidden_dim, n_threads))

        for i = orders
            scy_fi!(found_lower_orders, found_eigvals, A_gpu, W₁_gpu, W₂_gpu, h₁_gpu, h₂_gpu, i, n_threads, relu_pool, PLRNN, z_candidates, inplace_zs, inplace_h₁s, inplace_h₂s, inplace_temp_1s, inplace_temp_2s; temp_3s = inplace_temp_3s, outer_loop_iterations = outer_loop_iterations, inner_loop_iterations = inner_loop_iterations, latent_dim = latent_dim, hidden_dim = hidden_dim, type = type) 
        end
    else 
        # pre-allocate for in-place version of SCYFI computations
        z_candidate = CuArray{type}(undef, latent_dim)
        inplace_h₁ = CuArray{type}(undef, (latent_dim, latent_dim)) 
        inplace_h₂ = CuArray{type}(undef, (latent_dim, hidden_dim)) 
        inplace_z = CuArray{type}(undef, (latent_dim, latent_dim))
        inplace_temp_1 = CuArray{type}(undef, (latent_dim, hidden_dim))
        inplace_temp_2 = CuArray{type}(undef, (latent_dim, latent_dim))
        inplace_temp_3 = CuArray{type}(undef, hidden_dim)

        for i = orders
            scy_fi!(found_lower_orders, found_eigvals, A_gpu, W₁_gpu, W₂_gpu, h₁_gpu, h₂_gpu, i, relu_pool, PLRNN, z_candidate, inplace_z, inplace_h₁, inplace_h₂, inplace_temp_1, inplace_temp_2; temp_3 = inplace_temp_3, outer_loop_iterations = outer_loop_iterations, inner_loop_iterations = inner_loop_iterations, latent_dim = latent_dim, hidden_dim = hidden_dim, type = type) 
        end
    end
    return [found_lower_orders, found_eigvals]
end