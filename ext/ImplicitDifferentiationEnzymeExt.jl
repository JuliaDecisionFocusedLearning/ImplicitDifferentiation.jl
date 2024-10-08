module ImplicitDifferentiationEnzymeExt

using ADTypes
using Enzyme
using Enzyme.EnzymeRules
using ImplicitDifferentiation: ImplicitFunction, build_A, build_B, byproduct, output

const FORWARD_BACKEND = AutoEnzyme(; mode=Enzyme.Forward, function_annotation=Enzyme.Const)

function EnzymeRules.forward(
    config::EnzymeRules.FwdConfig,
    func::Const{<:ImplicitFunction},
    RT::Type{<:Union{BatchDuplicated,BatchDuplicatedNoNeed}},
    func_x::Union{BatchDuplicated{T,N},BatchDuplicatedNoNeed{T,N}},
    func_args::Vararg{Const,P},
) where {T,N,P}
    implicit = func.val
    x = func_x.val
    dx = func_x.dval
    args = map(a -> a.val, func_args)

    y_or_yz = implicit(x, args...)
    y = output(y_or_yz)
    Y = typeof(y)

    suggested_backend = FORWARD_BACKEND
    A = build_A(implicit, x, y_or_yz, args...; suggested_backend)
    B = build_B(implicit, x, y_or_yz, args...; suggested_backend)

    dc_batch = mapreduce(hcat, dx) do dₖx
        B * dₖx
    end
    dy_batch = implicit.linear_solver(A, -dc_batch)

    dy::NTuple{N,Y} = ntuple(k -> convert(Y, dy_batch[:, k]), Val(N))

    if y_or_yz isa AbstractArray
        if RT <: BatchDuplicated
            return BatchDuplicated(y, dy)
        elseif RT <: BatchDuplicatedNoNeed
            return dy
        end
    elseif y_or_yz isa Tuple
        yz = y_or_yz
        z = byproduct(yz)
        Z = typeof(z)
        dyz::NTuple{N,Tuple{Y,Z}} = ntuple(k -> (dy[k], make_zero(z)), Val(N))
        if RT <: BatchDuplicated
            return BatchDuplicated(yz, dyz)
        elseif RT <: BatchDuplicatedNoNeed
            return dyz
        end
    end
end

end
