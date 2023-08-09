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
    A_vec = pushforward_to_operator(implicit, y, pfA)
    return A_vec, pfB
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
    A_vec = pushforward_to_operator(implicit, y, pfA)
    return A_vec, pfB
end

struct PushforwardProd!{F,N}
    pushforward::F
    size::NTuple{N,Int}
end

function (pfp::PushforwardProd!)(dc_vec::AbstractVector, dy_vec::AbstractVector)
    dy = reshape(dy_vec, pfp.size)
    dc = only(pfp.pushforward(dy))
    return dc_vec .= vec(dc)
end

function pushforward_to_operator(
    implicit::ImplicitFunction, y::AbstractArray{R}, pfA
) where {R}
    m = length(y)
    prod! = PushforwardProd!(pfA, size(y))
    A_vec = LinearOperator(R, m, m, false, false, prod!)
    A_vec_presolved = presolve(implicit.linear_solver, A_vec, y)
    return A_vec_presolved
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
    Aᵀ_vec = pullback_to_operator(implicit, y, pbAᵀ)
    return Aᵀ_vec, pbBᵀ
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
    Aᵀ_vec = pullback_to_operator(implicit, y, pbAᵀ)
    return Aᵀ_vec, pbBᵀ
end

struct PullbackProd!{F,N}
    pullback::F
    size::NTuple{N,Int}
end

function (pbp::PullbackProd!)(dy_vec::AbstractVector, dc_vec::AbstractVector)
    dc = reshape(dc_vec, pbp.size)
    dy = only(pbp.pullback(dc))
    return dy_vec .= vec(dy)
end

function pullback_to_operator(
    implicit::ImplicitFunction, y::AbstractArray{R}, pbAᵀ
) where {R}
    m = length(y)
    prod! = PullbackProd!(pbAᵀ, size(y))
    Aᵀ_vec = LinearOperator(R, m, m, false, false, prod!)
    Aᵀ_vec_presolved = presolve(implicit.linear_solver, Aᵀ_vec, y)
    return Aᵀ_vec_presolved
end
