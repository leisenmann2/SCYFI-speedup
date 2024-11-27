using .. Utilities
using Random
using LinearAlgebra

"""
A,W,h PLRNN parameters
heuristic algorithm of finding FP (Durstewitz 2017) extended to find all k cycles
We need to solve: (A+WD_k)*...*(A+WD_1) *z + [(A+WD_k)*...*(A+WD_2)+...+(A+WD)**1+1]*h = z
inplace version
"""
function scy_fi!(found_cycles::Array, found_eigvals::Array, 
    A::Array, W::Array, h::Array, order::Integer,
    z_candidate::Array, inplace_z::Array, inplace_h::Array, inplace_temp::Array;
    outer_loop_iterations::Union{Integer,Nothing} = nothing,
    inner_loop_iterations::Union{Integer,Nothing} = nothing, 
    PLRNN::VanillaPLRNN = VanillaPLRNN(),
    dim::Integer = size(A)[1], type::Union{Type{Float32}, Type{Float64}} = eltype(A)
    )

    push!(found_cycles, Array[])
    push!(found_eigvals, Array[])
    outer_loop_iterations, inner_loop_iterations = set_loop_iterations(order, outer_loop_iterations, inner_loop_iterations)
    
    # pre-allocate big arrays
    relu_matrix_diagonals = Array{Bool}(undef, (dim, order))
    trajectory_matrix = Array{type}(undef, (dim, order))
    trajectory_relu_matrix_diagonals = Array{Bool}(undef, (dim, order))

    i = -1
    while i < outer_loop_iterations # This loop can be viewed as (re-)initialization of the algo in some set of linear regions
        i += 1
        Random.rand!(relu_matrix_diagonals) # generate random set of linear regions to start from
        c = 0
        while c < inner_loop_iterations # This loop calculates cycle candidates, checks if they are virtual and if they are initializes 
            c += 1                      # the next calculation in the linear region of that virtual cycle
            if get_cycle_point_candidate!(z_candidate, A, W, relu_matrix_diagonals, h, order, inplace_z, inplace_h, inplace_temp) # calculate cycle candidate
                get_latent_time_series!(trajectory_matrix, order, A, W, h, z_candidate) # (dim, order)
                trajectory_relu_matrix_diagonals .= trajectory_matrix .> 0  # get relu matrices of the candidate
                
                # if we did not find a real cycle use the regions of the virtual cycle to recalculate
                if trajectory_relu_matrix_diagonals != relu_matrix_diagonals
                    copy!(relu_matrix_diagonals, trajectory_relu_matrix_diagonals)
                else # if the linear regions match check that we haven't already found that cycle (of this or lower order)
                    if not_element_of(z_candidate, collect(Iterators.flatten(Iterators.flatten(found_cycles)))) 
                        # compute eigenvalues and safe cycle in-place
                        push!(found_cycles[end], convert_matrix_to_array(trajectory_matrix, order))
                        push!(found_eigvals[end], eigvals(inplace_z)) # inpace_z == Jacobian due to inplace get_cycle_point_candidate
                        i=0
                        c=0
                    end
                    Random.rand!(relu_matrix_diagonals) 
                end
            else
                Random.rand!(relu_matrix_diagonals) 
            end 
        end
    end
end

"""
A,W,h ALRNN parameters
ALRNN
inplace version
"""
function scy_fi!(found_cycles::Array, found_eigvals::Array, 
    A::Array, W::Array, h::Array, num_relus::Integer, order::Integer,
    z_candidate::Array, inplace_z::Array, inplace_h::Array, inplace_temp::Array,PLRNN::ALRNN;
    outer_loop_iterations::Union{Integer,Nothing} = nothing,
    inner_loop_iterations::Union{Integer,Nothing} = nothing, 
    dim::Integer = size(A)[1], type::Union{Type{Float32}, Type{Float64}} = eltype(A)
    )

    push!(found_cycles, Array[])
    push!(found_eigvals, Array[])
    outer_loop_iterations, inner_loop_iterations = set_loop_iterations(order, outer_loop_iterations, inner_loop_iterations)
    
    # pre-allocate big arrays
    relu_matrix_diagonals = Array{Bool}(undef, (dim, order))
    trajectory_matrix = Array{type}(undef, (dim, order))
    trajectory_relu_matrix_diagonals = Array{Bool}(undef, (dim, order))

    i = -1
    while i < outer_loop_iterations # This loop can be viewed as (re-)initialization of the algo in some set of linear regions
        i += 1
        Random.rand!(relu_matrix_diagonals) # generate random set of linear regions to start from
        relu_matrix_diagonals[1:dim-num_relus,:].=1 # set first dim-num_relu regions to 1
        c = 0
        while c < inner_loop_iterations # This loop calculates cycle candidates, checks if they are virtual and if they are initializes 
            c += 1                      # the next calculation in the linear region of that virtual cycle
            if get_cycle_point_candidate!(z_candidate, A, W, relu_matrix_diagonals, h, order, inplace_z, inplace_h, inplace_temp) # calculate cycle candidate
                get_latent_time_series!(trajectory_matrix, order, A, W, h, num_relus, z_candidate) # (dim, order)
                trajectory_relu_matrix_diagonals .= trajectory_matrix .> 0  # get relu matrices of the candidate
                trajectory_relu_matrix_diagonals[1:dim-num_relus,:].=1 # set first dim-num_relu regions to 1

                # if we did not find a real cycle use the regions of the virtual cycle to recalculate
                if trajectory_relu_matrix_diagonals != relu_matrix_diagonals
                    copy!(relu_matrix_diagonals, trajectory_relu_matrix_diagonals)
                else # if the linear regions match check that we haven't already found that cycle (of this or lower order)
                    if not_element_of(z_candidate, collect(Iterators.flatten(Iterators.flatten(found_cycles)))) 
                        # compute eigenvalues and safe cycle in-place
                        push!(found_cycles[end], convert_matrix_to_array(trajectory_matrix, order))
                        push!(found_eigvals[end], eigvals(inplace_z)) # inpace_z == Jacobian due to inplace get_cycle_point_candidate
                        i=0
                        c=0
                    end
                    Random.rand!(relu_matrix_diagonals) 
                    relu_matrix_diagonals[1:dim-num_relus,:].=1 # set first dim-num_relu regions to 1
                end
            else
                Random.rand!(relu_matrix_diagonals) 
                relu_matrix_diagonals[1:dim-num_relus,:].=1 # set first dim-num_relu regions to 1
            end 
        end
    end
end



"""
heuristic algorithm of finding FP (Durstewitz 2017) extended to find all k cycles and for the shPLRNN
shPLRNN, inplace version
"""
function scy_fi!(found_cycles::Array, found_eigvals::Array,
    A::AbstractVector,
    W₁::AbstractMatrix,
    W₂::AbstractMatrix,
    h₁::AbstractVector,
    h₂::AbstractVector,
    order::Integer,
    relu_pool::Union{Array, Nothing},
    PLRNN::ShallowPLRNN,
    z_candidate::Array, inplace_z::Array, inplace_h₁::Array, inplace_h₂::Array, temp_1::Array, temp_2::Array;
    temp_3::Array = Array{type}(undef, hidden_dim),
    outer_loop_iterations::Union{Integer,Nothing} = nothing,
    inner_loop_iterations::Union{Integer,Nothing} = nothing,
    latent_dim::Integer = size(h₁)[1], hidden_dim::Integer = size(h₂)[1], type::Union{Type{Float32}, Type{Float64}} = eltype(A)
    )
    
    push!(found_cycles, Array[])
    push!(found_eigvals, Array[])
    outer_loop_iterations, inner_loop_iterations = set_loop_iterations(order, outer_loop_iterations, inner_loop_iterations)

    # pre-allocate big arrays
    relu_matrix_diagonals = Array{Bool}(undef, (hidden_dim, order))
    trajectory_matrix = Array{type}(undef, (latent_dim, order))
    trajectory_relu_matrix_diagonals = Array{Bool}(undef, (hidden_dim, order))

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
                    if not_element_of(z_candidate, collect(Iterators.flatten(Iterators.flatten(found_cycles)))) 
                        # compute eigenvalues and safe cycle in-place
                        push!(found_cycles[end], convert_matrix_to_array(trajectory_matrix, order))
                        push!(found_eigvals[end], eigvals(inplace_z)) # inpace_z == Jacobian due to inplace get_cycle_point_candidate
                        i=0
                        c=0
                    end
                    construct_relu_matrix_diagonals!(relu_matrix_diagonals, relu_pool, order) 
                end
            else
                construct_relu_matrix_diagonals!(relu_matrix_diagonals, relu_pool, order)
            end 
        end
    end
end
                

"""
heuristic algorithm of finding FP extended to find all k cycles and for the clipped shPLRNN
clipped shPLRNN, inplace version
"""
function scy_fi!(found_cycles::Array, found_eigvals::Array,
    A::AbstractVector,
    W₁::AbstractMatrix,
    W₂::AbstractMatrix,
    h₁::AbstractVector,
    h₂::AbstractVector,
    order::Integer,
    relu_pool::Union{Array, Nothing},
    PLRNN::ClippedShallowPLRNN,
    z_candidate::Array, inplace_z::Array, inplace_h₁::Array, inplace_h₂::Array, temp_1::Array, temp_2::Array;
    temp_3::Array = Array{type}(undef, hidden_dim),
    outer_loop_iterations::Union{Integer,Nothing} = nothing,
    inner_loop_iterations::Union{Integer,Nothing} = nothing,
    latent_dim::Integer = size(h₁)[1], hidden_dim::Integer = size(h₂)[1], type::Union{Type{Float32}, Type{Float64}} = eltype(A)
    )
    
    push!(found_cycles, Array[])
    push!(found_eigvals, Array[])
    outer_loop_iterations, inner_loop_iterations = set_loop_iterations(order, outer_loop_iterations, inner_loop_iterations)

    # pre-allocate big arrays
    relu_matrix_diagonals_1 = Array{Bool}(undef, (hidden_dim, order))
    relu_matrix_diagonals_2 = Array{Bool}(undef, (hidden_dim, order))
    trajectory_matrix = Array{type}(undef, (latent_dim, order))
    trajectory_relu_matrix_diagonals_1 = Array{Bool}(undef, (hidden_dim, order))
    trajectory_relu_matrix_diagonals_2 = Array{Bool}(undef, (hidden_dim, order))

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
                    if not_element_of(z_candidate, collect(Iterators.flatten(Iterators.flatten(found_cycles)))) 
                        # compute eigenvalues and safe cycle in-place
                        push!(found_cycles[end], convert_matrix_to_array(trajectory_matrix, order))
                        push!(found_eigvals[end], eigvals(inplace_z)) # inplace_z == Jacobian due to inplace get_cycle_point_candidate
                        i=0
                        c=0
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
