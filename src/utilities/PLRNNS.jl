#Define Data types
abstract type AbstractPLRNN end

struct VanillaPLRNN <: AbstractPLRNN end
struct ShallowPLRNN <: AbstractPLRNN end
struct ClippedShallowPLRNN <: AbstractPLRNN end
struct ALRNN <: AbstractPLRNN end
