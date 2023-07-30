"""
    HandleByproduct

Trivial struct specifying that the forward mapping and conditions handle a byproduct.

Used in the constructor for `ImplicitFunction`.
"""
struct HandleByproduct end

"""
    ReturnByproduct

Trivial struct specifying that we want to obtain a byproduct in addition to the solution.

Used when calling an `ImplicitFunction`.
"""
struct ReturnByproduct end

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
    for i in eachindex(IndexLinear(), res, δoutput)
        res[i] = δoutput[i]
    end
end

function (pbm::PullbackMul!)(res::AbstractVector, δoutput_vec::AbstractVector)
    δoutput = reshape(δoutput_vec, pbm.output_size)
    δinput = only(pbm.pullback(δoutput))
    for i in eachindex(IndexLinear(), res, δinput)
        res[i] = δinput[i]
    end
end

## Override this function from LinearOperators to avoid generating the whole methods table

LinearOperators.get_nargs(pfm::PushforwardMul!) = 1
LinearOperators.get_nargs(pbm::PullbackMul!) = 1
