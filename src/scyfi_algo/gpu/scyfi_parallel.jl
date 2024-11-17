using .. Utilities
using Base.Threads


"""
A,W,h PLRNN parameters
heuristic algorithm of finding FP (Durstewitz 2017) extended to find all k cycles
We need to solve: (A+WD_k)*...*(A+WD_1) *z + [(A+WD_k)*...*(A+WD_2)+...+(A+WD)**1+1]*h = z
in-place, parallelized on CPU AND GPU
"""
function scy_fi!(found_cycles::Array, found_eigvals::Array, 
    A::CuArray, W::CuArray, h::CuArray, order::Integer, n_threads::Integer, 
    z_candidates::CuArray, inplace_zs::CuArray, inplace_hs::CuArray, inplace_temps::CuArray;
    outer_loop_iterations::Union{Integer, Nothing} = nothing,
    inner_loop_iterations::Union{Integer, Nothing} = nothing,
    PLRNN::VanillaPLRNN = VanillaPLRNN(),
    dim::Integer = size(A)[1], type::Union{Type{Float32}, Type{Float64}} = eltype(A)
    )

    push!(found_cycles, Array[])
    push!(found_eigvals, Array[])
    outer_loop_iterations, inner_loop_iterations = set_loop_iterations(order, outer_loop_iterations, inner_loop_iterations)

    # Do the multithreading over the initializations
    lk = ReentrantLock()
    Threads.@threads for thread_id = 1:n_threads
        # pre-allocate big arrays for each thread
        relu_matrix_diagonals = CuArray{Bool}(undef, (dim, order))
        trajectory_matrix = CuArray{type}(undef, (dim, order))
        trajectory_relu_matrix_diagonals = CuArray{Bool}(undef, (dim, order))
        
        # create views on pre-allocated arrays for each thread
        z_candidate = CUDA.view(z_candidates, :, thread_id)
        inplace_h = CUDA.view(inplace_hs, :, :, thread_id)
        inplace_z = CUDA.view(inplace_zs, :, :, thread_id)
        inplace_temp = CUDA.view(inplace_temps, :, :, thread_id)

        i = -1
        while i < outer_loop_iterations # This loop can be viewed as (re-)initialization of the algo in some set of linear regions
            i += 1
            Random.rand!(CUDA.default_rng(), relu_matrix_diagonals)  # generate random set of linear regions to start from
            c = 0
            while c < inner_loop_iterations # This loop calculates cycle candidates, checks if they are virtual and if they are initializes 
                c += 1                      # the next calculation in the linear region of that virtual cycle
                if get_cycle_point_candidate!(z_candidate, A, W, relu_matrix_diagonals, h, order, inplace_z, inplace_h, inplace_temp) # calculate cycle candidate
                    get_latent_time_series!(trajectory_matrix, order, A, W, h, z_candidate) # (dim, order)
                    trajectory_relu_matrix_diagonals .= trajectory_matrix .> 0  # get relu matrices of the candidate
                    
                    # if we did not find a real cycle use the regions of the virtual cycle to recalculate
                    if trajectory_relu_matrix_diagonals != relu_matrix_diagonals
                        CUDA.copy!(relu_matrix_diagonals, trajectory_relu_matrix_diagonals)
                    else # if the linear regions match check that we haven't already found that cycle (of this or lower order)
                        lock(lk) do # make sure no two thread access found_cycles simultaniously
                            if not_element_of(Array(z_candidate), collect(Iterators.flatten(Iterators.flatten(found_cycles))))
                                if !detect_nan_or_inf(trajectory_matrix) # detect nan or infs (only CPU, not CUDA inversion throws error message if so..)
                                    # compute eigenvalues and safe cycle in-place
                                    push!(found_cycles[end], convert_matrix_to_array(Array(trajectory_matrix), order))
                                    push!(found_eigvals[end], eigvals(Array(inplace_z))) # inplace_z == Jacobian due to inplace get_cycle_point_candidate! computation
                                    i=0
                                    c=0
                                else 
                                    println("Detected nan or inf for cycle order k = $order. Try increasing input type $type to higher precision.")
                                end
                            end
                        end
                        Random.rand!(CUDA.default_rng(), relu_matrix_diagonals)  
                    end
                else
                    Random.rand!(CUDA.default_rng(), relu_matrix_diagonals) 
                end 
            end
        end
    end
end


"""
heuristic algorithm of finding FP (Durstewitz 2017) extended to find all k cycles and for the shPLRNN
multi CPU threads + GPU parallelization
shPLRNN
"""
function scy_fi!(found_cycles::Array, found_eigvals::Array,
    A::CuArray,
    W₁::CuArray,
    W₂::CuArray,
    h₁::CuArray,
    h₂::CuArray,
    order::Integer,
    n_threads::Integer,
    relu_pool::CuArray,
    PLRNN::ShallowPLRNN,
    z_candidates::CuArray, inplace_zs::CuArray, inplace_h₁s::CuArray, inplace_h₂s::CuArray, temp_1s::CuArray, temp_2s::CuArray;
    temp_3s::CuArray = CuArray{type}(undef, hidden_dim),
    outer_loop_iterations::Union{Integer,Nothing} = nothing,
    inner_loop_iterations::Union{Integer,Nothing} = nothing,
    latent_dim::Integer = size(h₁)[1], hidden_dim::Integer = size(h₂)[1], type::Union{Type{Float32}, Type{Float64}} = eltype(A)
    )
    
    push!(found_cycles, Array[])
    push!(found_eigvals, Array[])
    outer_loop_iterations, inner_loop_iterations = set_loop_iterations(order, outer_loop_iterations, inner_loop_iterations)

    # Do the multithreading over the initializations
    lk = ReentrantLock()
    Threads.@threads for thread_id = 1:n_threads
        # pre-allocate big arrays for each thread
        relu_matrix_diagonals = CuArray{Bool}(undef, (hidden_dim, order))
        trajectory_matrix = CuArray{type}(undef, (latent_dim, order))
        trajectory_relu_matrix_diagonals = CuArray{Bool}(undef, (hidden_dim, order))

        # create views on pre-allocated arrays for each thread
        z_candidate = CUDA.view(z_candidates, :, thread_id)
        inplace_h₁ = CUDA.view(inplace_h₁s, :, :, thread_id)
        inplace_h₂ = CUDA.view(inplace_h₂s, :, :, thread_id) 
        inplace_z = CUDA.view(inplace_zs, :, :, thread_id)
        temp_1 = CUDA.view(temp_1s, :, :, thread_id)
        temp_2 = CUDA.view(temp_2s, :, :, thread_id)
        temp_3 = CUDA.view(temp_3s, :, thread_id)

        i = -1
        while i < outer_loop_iterations # This loop can be viewed as (re-)initialization of the algo in some set of linear regions
            i += 1
            construct_relu_matrix_diagonals!(relu_matrix_diagonals, relu_pool, order) # generate random set of linear regions from pool to start from
            c = 0
            while c < inner_loop_iterations
                c += 1
                if get_cycle_point_candidate!(z_candidate, Diagonal(A), W₁, W₂, h₁, h₂, relu_matrix_diagonals, order, inplace_z, inplace_h₁, inplace_h₂, temp_1, temp_2) # returns if computation was successfull (invertible)
                    # get trajectory & relu matrices of the candidate
                    get_latent_time_series!(trajectory_matrix, trajectory_relu_matrix_diagonals, order, A, W₁, W₂, h₁, h₂, z_candidate, temp_3)

                    # if we did not find a real cycle use the regions of the virtual cycle to recalculate
                    if trajectory_relu_matrix_diagonals != relu_matrix_diagonals
                        copy!(relu_matrix_diagonals, trajectory_relu_matrix_diagonals)
                    else # if the linear regions match check that we haven't already found that cycle (of this or lower order)
                        lock(lk) do # make sure no two thread access found_cycles simultaniously
                            if not_element_of(Array(z_candidate), collect(Iterators.flatten(Iterators.flatten(found_cycles)))) 
                                if !detect_nan_or_inf(trajectory_matrix) # detect nan or infs (only CPU, not CUDA inversion throws error message if so..)
                                    # compute eigenvalues and safe cycle in-place
                                    push!(found_cycles[end], convert_matrix_to_array(Array(trajectory_matrix), order))
                                    push!(found_eigvals[end], eigvals(Array(inplace_z))) # inplace_z == Jacobian due to inplace get_cycle_point_candidate! computation
                                    i=0
                                    c=0
                                else 
                                    println("Detected nan or inf for cycle order k = $order. Try increasing input type $type to higher precision.")
                                end
                            end
                        end
                        construct_relu_matrix_diagonals!(relu_matrix_diagonals, relu_pool, order) 
                    end
                else
                    construct_relu_matrix_diagonals!(relu_matrix_diagonals, relu_pool, order)
                end 
            end
        end
    end
end


"""
heuristic algorithm of finding FP extended to find all k cycles and for the clipped shPLRNN
multi CPU threads, GPU version
clipped shPLRNN
"""
function scy_fi!(found_cycles::Array, found_eigvals::Array,
    A::CuArray,
    W₁::CuArray,
    W₂::CuArray,
    h₁::CuArray,
    h₂::CuArray,
    order::Integer,
    n_threads::Integer,
    relu_pool::CuArray,
    PLRNN::ClippedShallowPLRNN,
    z_candidates::CuArray, inplace_zs::CuArray, inplace_h₁s::CuArray, inplace_h₂s::CuArray, temp_1s::CuArray, temp_2s::CuArray;
    temp_3s::CuArray = CuArray{type}(undef, hidden_dim),
    outer_loop_iterations::Union{Integer,Nothing} = nothing,
    inner_loop_iterations::Union{Integer,Nothing} = nothing,
    latent_dim::Integer = size(h₁)[1], hidden_dim::Integer = size(h₂)[1], type::Union{Type{Float32}, Type{Float64}} = eltype(A)
    )
    
    push!(found_cycles, Array[])
    push!(found_eigvals, Array[])
    outer_loop_iterations, inner_loop_iterations = set_loop_iterations(order, outer_loop_iterations, inner_loop_iterations)

    # Do the multithreading over the initializations
    lk = ReentrantLock()
    Threads.@threads for thread_id = 1:n_threads
        # pre-allocate big arrays for each thread
        relu_matrix_diagonals_1 = CuArray{Bool}(undef, (hidden_dim, order))
        relu_matrix_diagonals_2 = CuArray{Bool}(undef, (hidden_dim, order))
        trajectory_matrix = CuArray{type}(undef, (latent_dim, order))
        trajectory_relu_matrix_diagonals_1 = CuArray{Bool}(undef, (hidden_dim, order))
        trajectory_relu_matrix_diagonals_2 = CuArray{Bool}(undef, (hidden_dim, order))

        # create views on pre-allocated arrays for each thread
        z_candidate = CUDA.view(z_candidates, :, thread_id)
        inplace_h₁ = CUDA.view(inplace_h₁s, :, :, thread_id)
        inplace_h₂ = CUDA.view(inplace_h₂s, :, :, thread_id) 
        inplace_z = CUDA.view(inplace_zs, :, :, thread_id)
        temp_1 = CUDA.view(temp_1s, :, :, thread_id)
        temp_2 = CUDA.view(temp_2s, :, :, thread_id)
        temp_3 = CUDA.view(temp_3s, :, thread_id)

        i = -1
        while i < outer_loop_iterations # This loop can be viewed as (re-)initialization of the algo in some set of linear regions
            i += 1
            construct_relu_matrix_diagonals!(relu_matrix_diagonals_1, relu_pool, order)     # generate random set of linear regions from pool to start from
            relu_matrix_diagonals_2 .= relu_matrix_diagonals_1                                # initialise the two sets the same
            c = 0
            while c < inner_loop_iterations
                c += 1
                if get_cycle_point_candidate!(z_candidate, Diagonal(A), W₁, W₂, h₁, h₂, relu_matrix_diagonals_1, relu_matrix_diagonals_2, order, inplace_z, inplace_h₁, inplace_h₂, temp_1, temp_2) # returns if computation was successfull (invertible)
                    # get trajectory & relu matrices of the candidate
                    get_latent_time_series!(trajectory_matrix, trajectory_relu_matrix_diagonals_1, trajectory_relu_matrix_diagonals_2, order, A, W₁, W₂, h₁, h₂, z_candidate, temp_3)
                    
                    # if we did not find a real cycle use the regions of the virtual cycle to recalculate
                    if (trajectory_relu_matrix_diagonals_1 != relu_matrix_diagonals_1) || (trajectory_relu_matrix_diagonals_2 != relu_matrix_diagonals_2)
                        copy!(relu_matrix_diagonals_1, trajectory_relu_matrix_diagonals_1)
                        copy!(relu_matrix_diagonals_2, trajectory_relu_matrix_diagonals_2)                    
                    else # if the linear regions match check that we haven't already found that cycle (of this or lower order)
                        lock(lk) do # make sure no two thread access found_cycles simultaniously
                            if not_element_of(Array(z_candidate), collect(Iterators.flatten(Iterators.flatten(found_cycles)))) 
                                if !detect_nan_or_inf(trajectory_matrix) # detect nan or infs (only CPU, not CUDA inversion throws error message if so..)
                                    # compute eigenvalues and safe cycle in-place
                                    push!(found_cycles[end], convert_matrix_to_array(Array(trajectory_matrix), order))
                                    push!(found_eigvals[end], eigvals(Array(inplace_z))) # inplace_z == Jacobian due to inplace get_cycle_point_candidate! computation
                                    i=0
                                    c=0
                                else 
                                    println("Detected nan or inf for cycle order k = $order. Try increasing input type $type to higher precision.")
                                end
                            end
                        end
                        construct_relu_matrix_diagonals!(relu_matrix_diagonals_1, relu_pool, order) 
                        relu_matrix_diagonals_2 .= relu_matrix_diagonals_1
                    end
                else
                    construct_relu_matrix_diagonals!(relu_matrix_diagonals_1, relu_pool, order)
                    relu_matrix_diagonals_2 .= relu_matrix_diagonals_1
                end
            end
        end
    end
end



