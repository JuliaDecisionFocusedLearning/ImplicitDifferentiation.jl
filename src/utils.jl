struct SolverFailureException{A,B} <: Exception
    solver::A
    stats::B
end

function Base.show(io::IO, sfe::SolverFailureException)
    return println(
        io,
        "SolverFailureException: \n Solver: $(sfe.solver) \n Solver stats: $(string(sfe.stats))",
    )
end

function check_solution(solver, stats)
    if stats.solved
        return nothing
    else
        throw(SolverFailureException(solver, stats))
    end
end

"""
    PushforwardMul!{P,N}

Callable structure wrapping a pushforward with `N`-dimensional inputs into an in-place multiplication for vectors.

# Fields
- `pushforward::P`: the pushforward function
- `input_size::NTuple{N,Int}`: the array size of the function input
"""
struct PushforwardMul!{P,N}
    pushforward::P
    input_size::NTuple{N,Int}
end

"""
    PullbackMul!{P,N}

Callable structure wrapping a pullback with `N`-dimensional outputs into an in-place multiplication for vectors.

# Fields
- `pullback::P`: the pullback of the function
- `output_size::NTuple{N,Int}`: the array size of the function output
"""
struct PullbackMul!{P,N}
    pullback::P
    output_size::NTuple{N,Int}
end

function (pfm::PushforwardMul!)(res::AbstractVector, δinput_vec::AbstractVector)
    δinput = reshape(δinput_vec, pfm.input_size)
    δoutput = only(pfm.pushforward(δinput))
    return res .= vec(δoutput)
end

function (pbm::PullbackMul!)(res::AbstractVector, δoutput_vec::AbstractVector)
    δoutput = reshape(δoutput_vec, pbm.output_size)
    δinput = only(pbm.pullback(δoutput))
    return res .= vec(δinput)
end

## Override this function from LinearOperators to avoid generating the whole methods table

LinearOperators.get_nargs(pfm::PushforwardMul!) = 1
LinearOperators.get_nargs(pbm::PullbackMul!) = 1
