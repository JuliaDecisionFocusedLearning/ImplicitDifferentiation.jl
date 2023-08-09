## Forward

function conditions_pushforwards(
    ba::AbstractBackend,
    implicit::ImplicitFunction,
    x::AbstractArray,
    y::AbstractArray,
    args;
    kwargs,
)
    conditions = implicit.conditions
    pfA = only ∘ pushforward_function(ba, _y -> conditions(x, _y, args...; kwargs...), y)
    pfB = only ∘ pushforward_function(ba, _x -> conditions(_x, y, args...; kwargs...), x)
    return pfA, pfB
end

function conditions_pushforwards(
    ba::AbstractBackend,
    implicit::ImplicitFunction,
    x::AbstractArray,
    yz::Tuple,
    args;
    kwargs,
)
    conditions = implicit.conditions
    y, z = yz
    pfA = only ∘ pushforward_function(ba, _y -> conditions(x, _y, z, args...; kwargs...), y)
    pfB = only ∘ pushforward_function(ba, _x -> conditions(_x, y, z, args...; kwargs...), x)
    return pfA, pfB
end

struct PushforwardProd!{F,N}
    pushforward::F
    size::NTuple{N,Int}
end

function (pfp::PushforwardProd!)(dc_vec::AbstractVector, dy_vec::AbstractVector)
    dy = reshape(dy_vec, pfp.size)
    dc = pfp.pushforward(dy)
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

function conditions_pullbacks(
    ba::AbstractBackend,
    implicit::ImplicitFunction,
    x::AbstractArray,
    y::AbstractArray,
    args;
    kwargs,
)
    conditions = implicit.conditions
    pbAᵀ = only ∘ pullback_function(ba, _y -> conditions(x, _y, args...; kwargs...), y)
    pbBᵀ = only ∘ pullback_function(ba, _x -> conditions(_x, y, args...; kwargs...), x)
    return pbAᵀ, pbBᵀ
end

function conditions_pullbacks(
    ba::AbstractBackend,
    implicit::ImplicitFunction,
    x::AbstractArray,
    yz::Tuple,
    args;
    kwargs,
)
    conditions = implicit.conditions
    y, z = yz
    pbAᵀ = only ∘ pullback_function(ba, _y -> conditions(x, _y, z, args...; kwargs...), y)
    pbBᵀ = only ∘ pullback_function(ba, _x -> conditions(_x, y, z, args...; kwargs...), x)
    return pbAᵀ, pbBᵀ
end

struct PullbackProd!{F,N}
    pullback::F
    size::NTuple{N,Int}
end

function (pbp::PullbackProd!)(dy_vec::AbstractVector, dc_vec::AbstractVector)
    dc = reshape(dc_vec, pbp.size)
    dy = pbp.pullback(dc)
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
