## Forward

function forward_operators(
    backend::AbstractBackend,
    implicit::ImplicitFunction,
    x::AbstractArray,
    y::AbstractArray,
    args;
    kwargs,
)
    pfA = pushforward_function(
        backend, _y -> implicit.conditions(x, _y, args...; kwargs...), y
    )
    pfB = pushforward_function(
        backend, _x -> implicit.conditions(_x, y, args...; kwargs...), x
    )
    return pushforwards_to_linops(implicit, x, y, pfA, pfB)
end

function forward_operators(
    backend::AbstractBackend,
    implicit::ImplicitFunction,
    x::AbstractArray,
    yz::Tuple,
    args;
    kwargs,
)
    y, z = yz
    pfA = pushforward_function(
        backend, _y -> implicit.conditions(x, _y, z, args...; kwargs...), y
    )
    pfB = pushforward_function(
        backend, _x -> implicit.conditions(_x, y, z, args...; kwargs...), x
    )
    return pushforwards_to_linops(implicit, x, y, pfA, pfB)
end

function pushforwards_to_linops(
    implicit::ImplicitFunction, x::AbstractArray{R}, y::AbstractArray, pfA, pfB
) where {R}
    n, m = length(x), length(y)
    A_op = LinearOperator(R, m, m, false, false, PushforwardMul!(pfA, size(y)))
    B_op = LinearOperator(R, m, n, false, false, PushforwardMul!(pfB, size(x)))
    A_op_presolved = presolve(implicit.linear_solver, A_op, y)
    return A_op_presolved, B_op
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

LinearOperators.get_nargs(pfm::PushforwardMul!) = 1

function (pfm::PushforwardMul!)(res::AbstractVector, δinput_vec::AbstractVector)
    δinput = reshape(δinput_vec, pfm.input_size)
    δoutput = only(pfm.pushforward(δinput))
    res .= vec(δoutput)
end

## Reverse

function reverse_operators(
    backend::AbstractBackend,
    implicit::ImplicitFunction,
    x::AbstractArray,
    y::AbstractArray,
    args;
    kwargs,
)
    pbAᵀ = pullback_function(
        backend, _y -> implicit.conditions(x, _y, args...; kwargs...), y
    )
    pbBᵀ = pullback_function(
        backend, _x -> implicit.conditions(_x, y, args...; kwargs...), x
    )
    return pullbacks_to_linops(implicit, x, y, pbAᵀ, pbBᵀ)
end

function reverse_operators(
    backend::AbstractBackend,
    implicit::ImplicitFunction,
    x::AbstractArray,
    yz::Tuple,
    args;
    kwargs,
)
    y, z = yz
    pbAᵀ = pullback_function(
        backend, _y -> implicit.conditions(x, _y, z, args...; kwargs...), y
    )
    pbBᵀ = pullback_function(
        backend, _x -> implicit.conditions(_x, y, z, args...; kwargs...), x
    )
    return pullbacks_to_linops(implicit, x, y, pbAᵀ, pbBᵀ)
end

function pullbacks_to_linops(
    implicit::ImplicitFunction, x::AbstractArray{R}, y::AbstractArray, pbAᵀ, pbBᵀ
) where {R}
    n, m = length(x), length(y)
    Aᵀ_op = LinearOperator(R, m, m, false, false, PullbackMul!(pbAᵀ, size(y)))
    Bᵀ_op = LinearOperator(R, n, m, false, false, PullbackMul!(pbBᵀ, size(y)))
    Aᵀ_op_presolved = presolve(implicit.linear_solver, Aᵀ_op, y)
    return Aᵀ_op_presolved, Bᵀ_op
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

LinearOperators.get_nargs(pbm::PullbackMul!) = 1

function (pbm::PullbackMul!)(res::AbstractVector, δoutput_vec::AbstractVector)
    δoutput = reshape(δoutput_vec, pbm.output_size)
    δinput = only(pbm.pullback(δoutput))
    res .= vec(δinput)
end
