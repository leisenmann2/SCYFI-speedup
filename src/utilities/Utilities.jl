module Utilities
export construct_relu_matrix,
    construct_relu_matrix_list,
    construct_relu_matrix_pool,
    construct_relu_matrix_pool_traj,
    get_cycle_point_candidate,
    get_factor_in_front_of_z,
    get_factor_in_front_of_h,
    get_factors,
    get_latent_time_series,
    latent_step,
    set_loop_iterations,
    get_eigvals,
    AbstractPLRNN,
    VanillaPLRNN,
    ShallowPLRNN,
    ClippedShallowPLRNN


include("helpers.jl")
include("PLRNNS.jl")
end