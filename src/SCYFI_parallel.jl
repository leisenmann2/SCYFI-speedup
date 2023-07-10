include("./helpers.jl")
using Base.Threads
using JLD2

"""
A,W,h PLRNN parameters
heuristic algorithm of finding FP (Durstewitz 2017) extended to find all k cycles
We need to solve: (A+WD_k)*...*(A+WD_1) *z + [(A+WD_k)*...*(A+WD_2)+...+(A+WD)**1+1]*h = z
"""
function scy_fi(
    A:: Array, W:: Array, h:: Array, order:: Integer, found_lower_orders:: Array;
    outer_loop_iterations:: Union{Integer,Nothing}= nothing,
    inner_loop_iterations:: Union{Integer,Nothing} = nothing,
    experiment_path::String="Result",
    val::Int64=0
     )
    dim = size(A)[1]
    cycles_found = Array[]
    eigvals =  Array[]
    outer_loop_iterations, inner_loop_iterations = set_loop_iterations(order, outer_loop_iterations, inner_loop_iterations)
    #println("Using ", Threads.nthreads()," threats")
    n_threads = Threads.nthreads()
    lk = ReentrantLock()

    Threads.@threads for thread_iterator = 1:n_threads
        i = -1
        while i < outer_loop_iterations
            i += 1
            relu_matrix_list = construct_relu_matrix_list(dim, order)
            difference_relu_matrices = 1
            c = 0
            while c < inner_loop_iterations
                c += 1
                z_candidate = get_cycle_point_candidate(A, W, relu_matrix_list, h, order)
                if z_candidate !== nothing
                    trajectory = get_latent_time_series(order, A, W, h, dim, z_0=z_candidate)
                    trajectory_relu_matrix_list = Array{Bool}(undef, dim, dim, order)
                    for j = 1:order
                        trajectory_relu_matrix_list[:,:,j] = Diagonal(trajectory[j].>0)
                    end
                    for j = 1:order
                        difference_relu_matrices = sum(abs.(trajectory_relu_matrix_list[:,:,j].-relu_matrix_list[:,:,j]))
                        if difference_relu_matrices != 0
                            break
                        end
                        if !isempty(found_lower_orders)
                            if map(temp -> round.(temp, digits=2), trajectory[1]) ∈ map(temp -> round.(temp,digits=2),collect(Iterators.flatten(Iterators.flatten(found_lower_orders))))
                                difference_relu_matrices = 1
                                break
                            end
                        end
                    end
                    if difference_relu_matrices == 0
                        if map(temp1 -> round.(temp1, digits=2), trajectory[1]) ∉ map(temp -> round.(temp, digits=2), collect(Iterators.flatten(cycles_found)))
                            e = get_eigvals(A,W,relu_matrix_list,order)
                            lock(lk) do
                                push!(cycles_found,trajectory)
                                push!(eigvals,e)
                                save(experiment_path*"dynamical_objects_single_order_$order"*"_$val.jld2","dynamical_objects",[cycles_found,eigvals])
                                i=0
                                c=0
                            end
                        end
                    end
                    if relu_matrix_list == trajectory_relu_matrix_list
                        relu_matrix_list = construct_relu_matrix_list(dim, order)
                    else
                        relu_matrix_list = trajectory_relu_matrix_list
                    end
else
                    relu_matrix_list = construct_relu_matrix_list(dim, order)

                end 
            end
        end
    end
    # ToDo: filter by unique cycles
    #unique_indices = indexin(unique(map(temp -> round.(temp,digits=2),collect(Iterators.flatten(cycles_found)))),map(temp -> round.(temp,digits=2),collect(Iterators.flatten(cycles_found))))

    return cycles_found, eigvals
end
