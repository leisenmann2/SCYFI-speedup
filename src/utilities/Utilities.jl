module Utilities
using LinearAlgebra
using Random
using Distributions
using CUDA

export detect_nan_or_inf,
    not_element_of,
    convert_matrix_to_array,
    construct_relu_matrix,
    construct_relu_matrix_list,
    construct_relu_matrix_diagonals!,
    construct_relu_matrix_pool,
    construct_relu_matrix_pool_traj,
    get_cycle_point_candidate,
    get_cycle_point_candidate!,
    get_factor_in_front_of_z,
    get_factor_in_front_of_h,
    get_factors,
    get_factors!,
    get_latent_time_series,
    get_latent_time_series!,
    latent_step,
    latent_step!,
    set_loop_iterations,
    get_eigvals,
    AbstractPLRNN,
    VanillaPLRNN,
    ShallowPLRNN,
    ClippedShallowPLRNN

include("PLRNNS.jl")
include("helpers.jl")

end