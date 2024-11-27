using Test
using SCYFI
using LinearAlgebra

function test_finding_1_cycle_2D()
    # define variables for GT sys with 1 cycle if
    a2 = 0.09325784994952224
    a1 = 0.3683894917319025
    w1 = 0.42445515140482515
    w2 = 0.40337929998957267
    h1 = 0.30087210680425625
    h2 = 0.1416512363454716
    A = [a1 0; 0 a2]
    W = [0 w1; w2 0]
    h = [h1, h2]
    dz = 2
    k = 1
    FPs,eigenvals = find_cycles(A, W, h,1,k,outer_loop_iterations=10,inner_loop_iterations=20,PLRNN=ALRNN())
    println(FPs[1][1],eigenvals)
    #traj=Array{Float64}(undef, (dz,100))
    #get_latent_time_series!(traj,100,A,W,h,1,rand(2))
    @test length(FPs[1][1]) == 1
    #println(traj)
end

function test_finding_2_cycle_2D()
   
    A = [-1.1443109415089447 0.0; 0.0 -0.06875842211692393]
    W = [0.0 1.3890097099956336; -1.0205119007497634 0.0]
    h = [0.07363044144689698, -0.7703748650740753]
    dz = 2
    k = 2
    FPs,eigenvals = find_cycles(A, W, h,1,k,outer_loop_iterations=10,inner_loop_iterations=20,PLRNN=ALRNN())
    println(FPs,eigenvals)
    #traj=Array{Float64}(undef, (dz,100))
    #get_latent_time_series!(traj,100,A,W,h,1,rand(2))
    @test length(FPs[2]) == 1
    #println(traj)
end

function test_finding_8_cycle_2D()
    A = [-1.0426380555853465 0.0; 0.0 -0.4911411537352522]
    W = [0.0 0.46559772515836184; -3.3910092082930188 0.0]
    h = [0.8882965489134323, 1.0119539765178789]
    dz = 2
    k = 8
    FPs,eigenvals = find_cycles(A, W, h,1,k,outer_loop_iterations=10,inner_loop_iterations=20,PLRNN=ALRNN())
    println(FPs,eigenvals)
    #traj=Array{Float64}(undef, (dz,100))
    #get_latent_time_series!(traj,100,A,W,h,1,rand(2))
    @test length(FPs[8]) == 2
    #println(traj)
end

test_finding_1_cycle_2D()
test_finding_2_cycle_2D()
test_finding_8_cycle_2D()

# for i = 1:50
#     AW=randn(2,2)
#     A=Diagonal(AW)#
#     W=AW-A
#     h=randn(2,)
#     k=8
#     dz=2
#     #println(diag(A),W)
#     FPs,eigenvals = find_cycles(Array(A), W, h,1,k,outer_loop_iterations=10,inner_loop_iterations=20,PLRNN=ALRNN())
#     #println(FPs[1])
#     try
#         if length(FPs[8])>0
#             println(FPs[1])
#             println(FPs[8])
#             println(eigenvals[8])
#             println("Found a cycle")
#             println(A,W,h)
#         end
#     catch
#         continue
#     end
# end


# A = [-1.0426380555853465 0.0; 0.0 -0.4911411537352522]
# W = [0.0 0.46559772515836184; -3.3910092082930188 0.0]
# h = [0.8882965489134323, 1.0119539765178789]
# dz = 2
# k = 8
# FPs,eigenvals = find_cycles(A, W, h,1,k,outer_loop_iterations=10,inner_loop_iterations=20,PLRNN=ALRNN())
# println(FPs,eigenvals)

# println(FPs[1][1],eigenvals)
# traj=Array{Float64}(undef, (dz,200))
# get_latent_time_series!(traj,200,A,W,h,1,randn(2))
# @test length(FPs[1][1]) == 1
# println(traj[1,end-20:end])
