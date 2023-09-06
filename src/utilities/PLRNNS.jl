#Define Data types
# abstract type
abstract type AbstractPLRNN end

struct VanillaPLRNN <: AbstractPLRNN end
#vanilla_PLRNN=VanillaPLRNN()
#
struct ShallowPLRNN <: AbstractPLRNN end
#@functor ShallowPLRNN
#function ShallowPLRNN()
#    return ShallowPLRNN()
#end


struct ClippedShallowPLRNN <: AbstractPLRNN end
#@functor ClippedShallowPLRNN
#function ClippedShallowPLRNN()
 #   return ClippedShallowPLRNN()
#end