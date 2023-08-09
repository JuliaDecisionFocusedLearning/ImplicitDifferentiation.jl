## Forward

function forward_operators(
    backend::AbstractBackend,
    implicit::ImplicitFunction,
    x::AbstractArray,
    y::AbstractArray,
    args;
    kwargs,
)
    pfA =
        only ∘ pushforward_function(
            backend, _y -> implicit.conditions(x, _y, args...; kwargs...), y
        )
    pfB =
        only ∘ pushforward_function(
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
    pfA =
        only ∘ pushforward_function(
            backend, _y -> implicit.conditions(x, _y, z, args...; kwargs...), y
        )
    pfB =
        only ∘ pushforward_function(
            backend, _x -> implicit.conditions(_x, y, z, args...; kwargs...), x
        )
    A_vec = pushforward_to_operator(implicit, y, pfA)
    return A_vec, pfB
end

function pushforward_to_operator(
    implicit::ImplicitFunction, y::AbstractArray{R}, pfA
) where {R}
    m = length(y)
    function pfA_vec(dy_vec)
        dy = reshape(dy_vec, size(y))
        dc = pfA(dy)
        dc_vec = vec(dc)
        return dc_vec
    end
    prod!(dc_vec, dy_vec) = dc_vec .= pfA_vec(dy_vec)
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
    pbAᵀ =
        only ∘
        pullback_function(backend, _y -> implicit.conditions(x, _y, args...; kwargs...), y)
    pbBᵀ =
        only ∘
        pullback_function(backend, _x -> implicit.conditions(_x, y, args...; kwargs...), x)
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
    pbAᵀ =
        only ∘ pullback_function(
            backend, _y -> implicit.conditions(x, _y, z, args...; kwargs...), y
        )
    pbBᵀ =
        only ∘ pullback_function(
            backend, _x -> implicit.conditions(_x, y, z, args...; kwargs...), x
        )
    Aᵀ_vec = pullback_to_operator(implicit, y, pbAᵀ)
    return Aᵀ_vec, pbBᵀ
end

function pullback_to_operator(
    implicit::ImplicitFunction, y::AbstractArray{R}, pbAᵀ
) where {R}
    m = length(y)
    function pbAᵀ_vec(dc_vec)
        dc = reshape(dc_vec, size(y))
        dy = pbAᵀ(dc)
        dy_vec = vec(dy)
        return dy_vec
    end
    prod!(dy_vec, dc_vec) = dy_vec .= pbAᵀ_vec(dc_vec)
    Aᵀ_vec = LinearOperator(R, m, m, false, false, prod!)
    Aᵀ_vec_presolved = presolve(implicit.linear_solver, Aᵀ_vec, y)
    return Aᵀ_vec_presolved
end
