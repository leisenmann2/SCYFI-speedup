using SCYFI
using Test
include("./convPLRNN_tests.jl")
include("./shPLRNN_tests.jl")
include("./clippedshPLRNN_tests.jl")

@testset "SCYFI.jl" begin 
    # extra features
    test_create_pool()
    test_get_pool_from_traj()
    test_create_pool_clipped()
    test_get_pool_from_traj_clipped()

    test_get_pool_from_traj_inital_cond()
    test_get_pool_from_traj_initial_cond_clipped()

    #conv PLRNN
    test_finding_1_cycle_2D_for_holes()
    test_finding_1_cycle_2D()
    test_finding_1_cycle_2D_val()
    test_finding_16_cycle_2D()
    test_finding_16_cycle_2D_nothing()
    test_finding_27_cycle_2D()
    test_finding_27_cycle_2D_nothing()
    test_finding_31_cycle_2D()
    test_finding_31_cycle_2D_nothing()
    test_finding_40_cycle_2D()
    test_finding_40_cycle_2D_nothing()
    test_finding_53_cycle_2D()
    test_finding_53_cycle_2D_nothing()
    test_finding_65_cycle_2D()
    test_finding_65_cycle_2D_nothing()
    test_finding_80_cycle_2D()
    test_finding_80_cycle_2D_nothing()
    test_finding_83_cycle_2D()
    test_finding_83_cycle_2D_nothing()
    test_finding_10_cycle_10D()
    test_finding_10_cycle_10D_nothing()

    #shPLRNN
    test_finding_1_cycle_M2_H10()
    test_finding_1_cycle_M2_H10_val()
    test_finding_2_cycle_4_cycle_M2_H10()
    test_finding_2_cycle_4_cycle_M2_H10_nothing()

    #clippedshPLRNN
    test_finding_1_cycle_M2_H10_clipped()
    test_finding_1_cycle_M2_H10_clipped()
    test_finding_2_cycle_M2_H10_clipped_val()
    test_finding_10_cycle_M2_H10_clipped()
    test_finding_20_cycle_M2_H10_clipped()	
end
