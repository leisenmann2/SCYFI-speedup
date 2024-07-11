include("./scyfi_algo.jl")
include("./scyfi_parallel.jl")

""" 
calculate the cycles for a specified PLRNN with parameters A,W,h up until order k
"""
function find_cycles(
    A::Array, W::Array, h::Array, order::Integer;
    outer_loop_iterations::Union{Integer,Nothing} = nothing,
    inner_loop_iterations::Union{Integer,Nothing} = nothing,
    PLRNN::VanillaPLRNN = VanillaPLRNN()
    )
    
    found_lower_orders = Array[]
    found_eigvals = Array[]
    dim = size(A)[1]
    type = eltype(A)

    if Threads.nthreads() > 1
        println("multi-thread parallel version")
        n_threads= Threads.nthreads()

        # pre-allocate for each thread for in-place version of SCYFI computations 
        z_candidates = Array{type}(undef, dim, n_threads)
        inplace_hs = Array{type}(undef, (dim, dim, n_threads)) 
        inplace_zs = Array{type}(undef, (dim, dim, n_threads))
        inplace_temps = Array{type}(undef, (dim, dim, n_threads))

        for i = 1:order
            scy_fi!(found_lower_orders, found_eigvals, A, W, h, i, n_threads, z_candidates, inplace_zs, inplace_hs, inplace_temps; outer_loop_iterations = outer_loop_iterations, inner_loop_iterations = inner_loop_iterations, dim = dim, type = type)
        end
    else
        # pre-allocate for in-place version of SCYFI computations
        z_candidate = Array{type}(undef, dim)
        inplace_h = Array{type}(undef, (dim, dim)) 
        inplace_z = Array{type}(undef, (dim, dim))
        inplace_temp = Array{type}(undef, (dim, dim))

        for i = 1:order
            scy_fi!(found_lower_orders, found_eigvals, A, W, h, i, z_candidate, inplace_z, inplace_h, inplace_temp; outer_loop_iterations = outer_loop_iterations, inner_loop_iterations = inner_loop_iterations, dim = dim, type = type)
        end
    end

    return [found_lower_orders, found_eigvals]
end

""" 
calculate the cycles for a specified PLRNN with parameters A,W,h for order k in orders
"""
function find_cycles(
    A:: Array, W:: Array, h:: Array, orders::Array;
    outer_loop_iterations:: Union{Integer,Nothing}= nothing,
    inner_loop_iterations:: Union{Integer,Nothing} = nothing,
    PLRNN::VanillaPLRNN = VanillaPLRNN()
    )
        
    found_lower_orders = Array[]
    found_eigvals = Array[]
    dim = size(A)[1]
    type = eltype(A)

    if Threads.nthreads() > 1
        println("multi-thread parallel version")
        n_threads= Threads.nthreads()

        # pre-allocate for each thread for in-place version of SCYFI computations 
        z_candidates = Array{type}(undef, dim, n_threads)
        inplace_hs = Array{type}(undef, (dim, dim, n_threads)) 
        inplace_zs = Array{type}(undef, (dim, dim, n_threads))
        inplace_temps = Array{type}(undef, (dim, dim, n_threads))

        for i = orders
            scy_fi!(found_lower_orders, found_eigvals, A, W, h, i, n_threads, z_candidates, inplace_zs, inplace_hs, inplace_temps; outer_loop_iterations = outer_loop_iterations, inner_loop_iterations = inner_loop_iterations, dim = dim, type = type)
        end
    else
        # pre-allocate for in-place version of SCYFI computations
        z_candidate = Array{type}(undef, dim)
        inplace_h = Array{type}(undef, (dim, dim)) 
        inplace_z = Array{type}(undef, (dim, dim))
        inplace_temp = Array{type}(undef, (dim, dim))

        for i = orders
            scy_fi!(found_lower_orders, found_eigvals, A, W, h, i, z_candidate, inplace_z, inplace_h, inplace_temp; outer_loop_iterations = outer_loop_iterations, inner_loop_iterations = inner_loop_iterations, dim = dim, type = type)
        end
    end

    return [found_lower_orders, found_eigvals]
end


""" 
calculate the cycles for a specified shPLRNN up until order k
shPLRNN
"""
function find_cycles(
    A::AbstractVector,
    W₁::AbstractMatrix,
    W₂::AbstractMatrix,
    h₁::AbstractVector,
    h₂::AbstractVector,
    order::Integer;
    PLRNN::Union{ShallowPLRNN, ClippedShallowPLRNN} = ShallowPLRNN(),
    outer_loop_iterations::Union{Integer,Nothing} = nothing,
    inner_loop_iterations::Union{Integer,Nothing} = nothing,
    create_pool::Bool = true,
    get_pool_from_traj::Bool=false,
    num_trajectories::Integer=10,
    len_trajectories::Integer=100,
    search_space::Array = [-10, 10]
    )

    found_lower_orders = Array[]
    found_eigvals = Array[]
    latent_dim = size(h₁)[1]
    hidden_dim = size(h₂)[1]
    type = eltype(A)

    # preallocate pool of allowed D matrices, in the shPLRNN there are overlapping regions which can be excluded, pre-creating them makes the algorithm more efficient        
    if create_pool & get_pool_from_traj 
        relu_pool = construct_relu_matrix_pool_traj(A, W₁, W₂, h₁, h₂, latent_dim, hidden_dim, PLRNN; num_trajectories = num_trajectories, len_trajectories=len_trajectories, search_space = search_space, type = type)
        println("Number of initialisations in Pool from Trajectory: ", size(relu_pool)[2])
    elseif create_pool & !get_pool_from_traj
        relu_pool = construct_relu_matrix_pool(W₂, h₂, latent_dim, hidden_dim; search_space = search_space, type = type)
        println("Number of initialisations in Pool: ", size(relu_pool)[2])
    else 
        relu_pool = nothing
    end

    if Threads.nthreads() > 1
        println("multi-thread parallel version")
        n_threads= Threads.nthreads()

        # pre-allocate for each thread for in-place version of SCYFI computations 
        z_candidates = Array{type}(undef, (latent_dim, n_threads))
        inplace_h₁s = Array{type}(undef, (latent_dim, latent_dim, n_threads)) 
        inplace_h₂s = Array{type}(undef, (latent_dim, hidden_dim, n_threads)) 
        inplace_zs = Array{type}(undef, (latent_dim, latent_dim, n_threads))
        inplace_temp_1s = Array{type}(undef, (latent_dim, hidden_dim, n_threads))
        inplace_temp_2s = Array{type}(undef, (latent_dim, latent_dim, n_threads))
        inplace_temp_3s = Array{type}(undef, (hidden_dim, n_threads))

        for i = 1:order
            scy_fi!(found_lower_orders, found_eigvals, A, W₁, W₂, h₁, h₂, i, n_threads, relu_pool, PLRNN, z_candidates, inplace_zs, inplace_h₁s, inplace_h₂s, inplace_temp_1s, inplace_temp_2s; temp_3s = inplace_temp_3s, outer_loop_iterations = outer_loop_iterations, inner_loop_iterations = inner_loop_iterations, latent_dim = latent_dim, hidden_dim = hidden_dim, type = type) 
        end
    else 
        # pre-allocate for in-place version of SCYFI computations
        z_candidate = Array{type}(undef, latent_dim)
        inplace_h₁ = Array{type}(undef, (latent_dim, latent_dim)) 
        inplace_h₂ = Array{type}(undef, (latent_dim, hidden_dim)) 
        inplace_z = Array{type}(undef, (latent_dim, latent_dim))
        inplace_temp_1 = Array{type}(undef, (latent_dim, hidden_dim))
        inplace_temp_2 = Array{type}(undef, (latent_dim, latent_dim))
        inplace_temp_3 = Array{type}(undef, hidden_dim)

        for i = 1:order
            scy_fi!(found_lower_orders, found_eigvals, A, W₁, W₂, h₁, h₂, i, relu_pool, PLRNN, z_candidate, inplace_z, inplace_h₁, inplace_h₂, inplace_temp_1, inplace_temp_2; temp_3 = inplace_temp_3, outer_loop_iterations = outer_loop_iterations, inner_loop_iterations = inner_loop_iterations, latent_dim = latent_dim, hidden_dim = hidden_dim, type = type) 
        end
    end
    return [found_lower_orders, found_eigvals]
end


""" 
calculate the cycles for a specified shPLRNN up until order k in orders
shPLRNN
"""
function find_cycles(
    A::AbstractVector,
    W₁::AbstractMatrix,
    W₂::AbstractMatrix,
    h₁::AbstractVector,
    h₂::AbstractVector,
    orders::Array;
    PLRNN::Union{ShallowPLRNN, ClippedShallowPLRNN} = ShallowPLRNN(),
    outer_loop_iterations::Union{Integer,Nothing} = nothing,
    inner_loop_iterations::Union{Integer,Nothing} = nothing,
    create_pool::Bool = true,
    get_pool_from_traj::Bool =false,
    num_trajectories::Integer=10,
    len_trajectories::Integer=100,
    search_space::Array = [-10, 10]
    )

    found_lower_orders = Array[]
    found_eigvals = Array[]
    latent_dim = size(h₁)[1]
    hidden_dim = size(h₂)[1]
    type = eltype(A)

    # preallocate pool of allowed D matrices, in the shPLRNN there are overlapping regions which can be excluded, pre-creating them makes the algorithm more efficient        
    if create_pool & get_pool_from_traj
        relu_pool = construct_relu_matrix_pool_traj(A, W₁, W₂, h₁, h₂, latent_dim, hidden_dim, PLRNN; num_trajectories = num_trajectories, len_trajectories=len_trajectories, search_space = search_space, type = type)
        println("Number of initialisations in Pool from Trajectory: ", size(relu_pool)[2])
    elseif create_pool & !get_pool_from_traj
        relu_pool = construct_relu_matrix_pool(W₂, h₂, latent_dim, hidden_dim; search_space = search_space, type = type)
        println("Number of initialisations in Pool: ", size(relu_pool)[2])
    else 
        relu_pool = nothing
    end

    if Threads.nthreads() >1 
        println("multi-thread parallel version")
        n_threads= Threads.nthreads()

        # pre-allocate for each thread for in-place version of SCYFI computations 
        z_candidates = Array{type}(undef, (latent_dim, n_threads))
        inplace_h₁s = Array{type}(undef, (latent_dim, latent_dim, n_threads)) 
        inplace_h₂s = Array{type}(undef, (latent_dim, hidden_dim, n_threads)) 
        inplace_zs = Array{type}(undef, (latent_dim, latent_dim, n_threads))
        inplace_temp_1s = Array{type}(undef, (latent_dim, hidden_dim, n_threads))
        inplace_temp_2s = Array{type}(undef, (latent_dim, latent_dim, n_threads))
        inplace_temp_3s = Array{type}(undef, (hidden_dim, n_threads))

        for i = orders
            scy_fi!(found_lower_orders, found_eigvals, A, W₁, W₂, h₁, h₂, i, n_threads, relu_pool, PLRNN, z_candidates, inplace_zs, inplace_h₁s, inplace_h₂s, inplace_temp_1s, inplace_temp_2s; temp_3s = inplace_temp_3s, outer_loop_iterations = outer_loop_iterations, inner_loop_iterations = inner_loop_iterations, latent_dim = latent_dim, hidden_dim = hidden_dim, type = type) 
        end
    else  
        # pre-allocate for in-place version of SCYFI computations
        z_candidate = Array{type}(undef, latent_dim)
        inplace_h₁ = Array{type}(undef, (latent_dim, latent_dim)) 
        inplace_h₂ = Array{type}(undef, (latent_dim, hidden_dim)) 
        inplace_z = Array{type}(undef, (latent_dim, latent_dim))
        inplace_temp_1 = Array{type}(undef, (latent_dim, hidden_dim))
        inplace_temp_2 = Array{type}(undef, (latent_dim, latent_dim))
        inplace_temp_3 = Array{type}(undef, hidden_dim)

        for i = orders
            scy_fi!(found_lower_orders, found_eigvals, A, W₁, W₂, h₁, h₂, i, relu_pool, PLRNN, z_candidate, inplace_z, inplace_h₁, inplace_h₂, inplace_temp_1, inplace_temp_2; temp_3 = inplace_temp_3, outer_loop_iterations = outer_loop_iterations, inner_loop_iterations = inner_loop_iterations, latent_dim = latent_dim, hidden_dim = hidden_dim, type = type) # get_pool_from_traj = get_pool_from_traj, num_trajectories = num_trajectories, len_trajectories = len_trajectories)
        end
    end
    return [found_lower_orders, found_eigvals]
end