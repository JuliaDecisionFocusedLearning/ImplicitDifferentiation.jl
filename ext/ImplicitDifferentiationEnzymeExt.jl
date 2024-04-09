module ImplicitDifferentiationEnzymeExt

using ADTypes
using Enzyme
using Enzyme.EnzymeCore
using ImplicitDifferentiation: ImplicitFunction, build_A, build_B, output

# https://discourse.julialang.org/t/can-i-define-a-type-unstable-enzymerule/112732

function EnzymeRules.forward(
    func::Const{<:ImplicitFunction},
    RT::Type{<:Union{Duplicated,DuplicatedNoNeed}},
    x::Union{Duplicated,DuplicatedNoNeed},
)
    implicit = func.val
    @info "My Duplicated rule is used"
    y_or_yz = implicit(x.val)
    y = output(y_or_yz)

    suggested_backend = AutoEnzyme(Enzyme.Forward)
    A = build_A(implicit, x.val, y_or_yz; suggested_backend)
    B = build_B(implicit, x.val, y_or_yz; suggested_backend)

    dc = B * x.dval
    dy = implicit.linear_solver(A, -dc)
    if RT <: Duplicated
        return Duplicated(y, dy)
    elseif RT <: DuplicatedNoNeed
        return dy
    end
end

function EnzymeRules.forward(
    func::Const{<:ImplicitFunction},
    RT::Type{<:Union{BatchDuplicated,BatchDuplicatedNoNeed}},
    x::Union{BatchDuplicated{T,N},BatchDuplicatedNoNeed{T,N}},
) where {T,N}
    implicit = func.val
    @info "My BatchDuplicated rule is used"
    y_or_yz = implicit(x.val)
    y = output(y_or_yz)

    suggested_backend = AutoEnzyme(Enzyme.Forward)
    A = build_A(implicit, x.val, y_or_yz; suggested_backend)
    B = build_B(implicit, x.val, y_or_yz; suggested_backend)

    dX = reduce(hcat, x.dval)
    dC = mareduce(hcat, eachcol(dX)) do dₖx
        B * dₖx
    end
    dY = implicit.linear_solver(A, -dC)

    dy = ntuple(k -> dY[:, k], Val(N))
    if RT <: BatchDuplicated
        return BatchDuplicated(y, dy)
    elseif RT <: BatchDuplicatedNoNeed
        return dy
    end
end

end
