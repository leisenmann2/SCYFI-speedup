using SCYFI
using Test
include("./convPLRNN_tests.jl")
include("./shPLRNN_tests.jl")
include("./clippedshPLRNN_tests.jl")

@testset "SCYFI_pool.jl" begin 
    # test_create_pool()
    test_get_pool_from_traj()
    # test_create_pool_clipped()
    # test_get_pool_from_traj_clipped()

    test_get_pool_from_traj_inital_cond()
    test_get_pool_from_traj_initial_cond_clipped()
end
