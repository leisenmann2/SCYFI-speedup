include("scyfi_algo.jl")
include("scyfi_parallel.jl")

""" 
calculate the cycles for a specified PLRNN with parameters A,W,h up until order k
"""
function find_cycles(
    A:: Array, W:: Array, h:: Array, order:: Integer;
    outer_loop_iterations:: Union{Integer,Nothing}= nothing,
    inner_loop_iterations:: Union{Integer,Nothing} = nothing,
    PLRNN:: Union{VanillaPLRNN,Nothing} = VanillaPLRNN()
    )
    
    found_lower_orders = Array[]
    found_eigvals = Array[]
    #plrnn=PLRNN
    if Threads.nthreads() >1
        n_threads= Threads.nthreads()
        println("parallelized version")
        for i =1:order
            cycles_found, eigvals = scy_fi(A, W, h, i, found_lower_orders,n_threads, outer_loop_iterations=outer_loop_iterations,inner_loop_iterations=inner_loop_iterations)
         
            push!(found_lower_orders,cycles_found)
            push!(found_eigvals,eigvals)
        end
    
    else
        for i =1:order
            cycles_found, eigvals = scy_fi(A, W, h, i, found_lower_orders, outer_loop_iterations=outer_loop_iterations,inner_loop_iterations=inner_loop_iterations)
        
            push!(found_lower_orders,cycles_found)
            push!(found_eigvals,eigvals)
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
    order:: Integer;
    outer_loop_iterations:: Union{Integer,Nothing} = nothing,
    get_pool_from_traj=false,
    num_trajectories:: Integer=10,
    len_trajectories:: Integer=100,
    inner_loop_iterations:: Union{Integer,Nothing} = nothing,
    PLRNN:: Union{AbstractPLRNN,Nothing} = ShallowPLRNN(),

    )
    found_lower_orders = Array[]
    found_eigvals = Array[]
    # create pool of allowed D matrices, in the shPLRNN there are overlapping regions which can be excluded, this makes the algorithm more efficient        
    if get_pool_from_traj
        if PLRNN==ClippedShallowPLRNN()
            relu_pool=construct_relu_matrix_pool_traj(A, W₁, W₂, h₁, h₂, size(A)[1], size(h₂)[1], num_trajectories, len_trajectories,true)
        else
            relu_pool=construct_relu_matrix_pool_traj(A, W₁, W₂, h₁, h₂, size(A)[1], size(h₂)[1], num_trajectories, len_trajectories,false)
        end
    else
        relu_pool=construct_relu_matrix_pool(A, W₁, W₂, h₁, h₂, size(A)[1],size(h₂)[1])
    end
    println("Number of initialisations in Pool: ", size(relu_pool)[3])

    if Threads.nthreads() >1
        n_threads= Threads.nthreads()
        println("parallelized version") 
        for i =1:order
            cycles_found, eigvals = scy_fi(A, W₁, W₂, h₁, h₂, i, found_lower_orders,relu_pool,PLRNN,n_threads, outer_loop_iterations=outer_loop_iterations,inner_loop_iterations=inner_loop_iterations,get_pool_from_traj=get_pool_from_traj,
            num_trajectories=num_trajectories, 
            len_trajectories=len_trajectories)
         
            push!(found_lower_orders,cycles_found)
            push!(found_eigvals,eigvals)
        end
    else
        for i =1:order
            cycles_found, eigvals = scy_fi(A, W₁, W₂, h₁, h₂, i, found_lower_orders,relu_pool,PLRNN, outer_loop_iterations=outer_loop_iterations,inner_loop_iterations=inner_loop_iterations,get_pool_from_traj=get_pool_from_traj,
            num_trajectories=num_trajectories, 
            len_trajectories=len_trajectories)
        
            push!(found_lower_orders,cycles_found)
            push!(found_eigvals,eigvals)
        end
    end
    return [found_lower_orders, found_eigvals]
end

