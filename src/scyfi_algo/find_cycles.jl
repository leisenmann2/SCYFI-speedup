""" 
calculate the cycles for a specified PLRNN with parameters A,W,h up until order k
"""
function find_cycles(
    A::Array, W::Array, h::Array, order::Union{Integer, Array};
    outer_loop_iterations::Union{Integer,Nothing} = nothing,
    inner_loop_iterations::Union{Integer,Nothing} = nothing,
    PLRNN::VanillaPLRNN = VanillaPLRNN(),
    gpu_version::Bool = CUDA.functional() && size(A)[1] >= 300
    )

    if gpu_version # use gpu 
        println("gpu version")
        SCYFI_gpu.find_cycles(A, W, h, order; PLRNN = PLRNN, outer_loop_iterations = outer_loop_iterations, inner_loop_iterations = inner_loop_iterations)
    else # use purely cpu version
        println("cpu version")
        SCYFI_cpu.find_cycles(A, W, h, order; PLRNN = PLRNN, outer_loop_iterations = outer_loop_iterations, inner_loop_iterations = inner_loop_iterations)
    end
end

""" 
calculate the cycles for a specified ALRNN with parameters A,W,h up until order k
"""
function find_cycles(
    A::Array, W::Array, h::Array, num_relus::Integer, order::Union{Integer, Array};
    outer_loop_iterations::Union{Integer,Nothing} = nothing,
    inner_loop_iterations::Union{Integer,Nothing} = nothing,
    PLRNN::ALRNN = ALRNN(),
    gpu_version::Bool = CUDA.functional() && size(A)[1] >= 300
    )

    if gpu_version # use gpu 
        println("gpu version")
        #SCYFI_gpu.find_cycles(A, W, h, order; PLRNN = PLRNN, outer_loop_iterations = outer_loop_iterations, inner_loop_iterations = inner_loop_iterations)
        println("Not implemented yet")
    else # use purely cpu version
        println("cpu version")
        SCYFI_cpu.find_cycles(A, W, h, num_relus, order; PLRNN = PLRNN, outer_loop_iterations = outer_loop_iterations, inner_loop_iterations = inner_loop_iterations)
    end
end

""" 
calculate the cycles for a specified shPLRNN up until order k
"""
function find_cycles(
    A::AbstractVector,
    W₁::AbstractMatrix,
    W₂::AbstractMatrix,
    h₁::AbstractVector,
    h₂::AbstractVector,
    order::Union{Integer, Array};
    outer_loop_iterations::Union{Integer,Nothing} = nothing,
    inner_loop_iterations::Union{Integer,Nothing} = nothing,
    PLRNN::Union{ShallowPLRNN, ClippedShallowPLRNN} = ShallowPLRNN(),
    create_pool::Bool=true,
    get_pool_from_traj::Bool=false,
    num_trajectories::Integer=10,
    len_trajectories::Integer=100,
    search_space::Array = [-10, 10],
    initial_conditions::Array = [],
    gpu_version::Bool = CUDA.functional() && size(h₁)[1] >= 300  && size(h₂)[1] >= 300
    )

    if gpu_version # use gpu 
        println("gpu version")
        SCYFI_gpu.find_cycles(A, W₁, W₂, h₁, h₂, order, PLRNN = PLRNN, outer_loop_iterations = outer_loop_iterations, inner_loop_iterations = inner_loop_iterations, create_pool = create_pool, get_pool_from_traj = get_pool_from_traj, num_trajectories = num_trajectories, len_trajectories = len_trajectories, search_space = search_space, initial_conditions = initial_conditions)
    else # use purely cpu version
        println("cpu version")
        SCYFI_cpu.find_cycles(A, W₁, W₂, h₁, h₂, order, PLRNN = PLRNN, outer_loop_iterations = outer_loop_iterations, inner_loop_iterations = inner_loop_iterations, create_pool = create_pool, get_pool_from_traj = get_pool_from_traj, num_trajectories = num_trajectories, len_trajectories = len_trajectories, search_space = search_space, initial_conditions = initial_conditions)
    end
end