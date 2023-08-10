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

function pushforwards_to_operators(
    implicit::ImplicitFunction, x::AbstractArray, y::AbstractArray, pfA, pfB
)
    n, m = length(x), length(y)
    A_vec = LinearOperator(eltype(y), m, m, false, false, PushforwardProd!(pfA, size(y)))
    B_vec = LinearOperator(eltype(x), m, n, false, false, PushforwardProd!(pfB, size(x)))
    A_vec_presolved = presolve(implicit.linear_solver, A_vec, y)
    return A_vec_presolved, B_vec
end

function conditions_forward_operators(
    backend::AbstractBackend, implicit::ImplicitFunction, x, y_or_yz, args; kwargs
)
    y = get_output(y_or_yz)
    pfA, pfB = conditions_pushforwards(backend, implicit, x, y_or_yz, args; kwargs)
    A_vec, B_vec = pushforwards_to_operators(implicit, x, y, pfA, pfB)
    return A_vec, B_vec
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

function pullbacks_to_operators(
    implicit::ImplicitFunction, x::AbstractArray, y::AbstractArray, pbAᵀ, pbBᵀ
)
    n, m = length(x), length(y)
    Aᵀ_vec = LinearOperator(eltype(y), m, m, false, false, PullbackProd!(pbAᵀ, size(y)))
    Bᵀ_vec = LinearOperator(eltype(y), n, m, false, false, PullbackProd!(pbBᵀ, size(y)))
    Aᵀ_vec_presolved = presolve(implicit.linear_solver, Aᵀ_vec, y)
    return Aᵀ_vec_presolved, Bᵀ_vec
end

function conditions_reverse_operators(
    backend::AbstractBackend, implicit::ImplicitFunction, x, y_or_yz, args; kwargs
)
    y = get_output(y_or_yz)
    pbAᵀ, pbBᵀ = conditions_pullbacks(backend, implicit, x, y_or_yz, args; kwargs)
    Aᵀ_vec, Bᵀ_vec = pullbacks_to_operators(implicit, x, y, pbAᵀ, pbBᵀ)
    return Aᵀ_vec, Bᵀ_vec
end
