include("./src/SCYFI.jl")

""" 
calculate the cycles for a specified PLRNN with parameters A,W,h up until order k
"""
function main(
    A:: Array, W:: Array, h:: Array, order:: Integer;
    outer_loop_iterations:: Union{Integer,Nothing}= nothing,
    inner_loop_iterations:: Union{Integer,Nothing} = nothing
     )
    found_lower_orders = Array[]
    found_eigvals = Array[]
     
    for i =1:order
        cycles_found, eigvals = scy_fi(A, W, h, i, found_lower_orders, outer_loop_iterations=outer_loop_iterations,inner_loop_iterations=inner_loop_iterations)
     
        push!(found_lower_orders,cycles_found)
        push!(found_eigvals,eigvals)
    end
    return [found_lower_orders, found_eigvals]
end
