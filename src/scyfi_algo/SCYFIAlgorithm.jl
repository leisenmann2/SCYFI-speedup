module SCYFIAlgorithm
    using LinearAlgebra
    using Random
    using CUDA 
    using ..Utilities

    export find_cycles

    module SCYFI_gpu
        export find_cycles
        include("./gpu/find_cycles.jl")
    end

    module SCYFI_cpu
        export find_cycles
        include("./cpu/find_cycles.jl")
    end

    include("./find_cycles.jl")
end
