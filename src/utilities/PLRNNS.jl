#Define Data types
# abstract type
abstract type AbstractPLRNN end

struct VanillaPLRNN <: AbstractPLRNN end
struct ShallowPLRNN <: AbstractPLRNN end
struct ClippedShallowPLRNN <: AbstractPLRNN end
