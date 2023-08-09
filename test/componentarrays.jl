import AbstractDifferentiation as AD
using ComponentArrays
using ForwardDiff
using ImplicitDifferentiation
using ImplicitDifferentiation: identity_break_autodiff
using Zygote

function forward_safe(x1, x2, x3)
    y4 = sqrt.(x1 .+ x2)
    y5 = sqrt.(x1 .+ x3)
    return y4, y5
end

function forward_aux(x1, x2, x3)
    y4 = identity_break_autodiff(sqrt.(x1 .+ x2))
    y5 = identity_break_autodiff(sqrt.(x1 .+ x3))
    return y4, y5
end

function conditions_aux(x1, x2, x3, y4, y5)
    c4 = y4 .^ 2 .- x1 .- x2
    c5 = y5 .^ 2 .- x1 .- x3
    return c4, c5
end

function forward(x::ComponentVector)
    y4, y5 = forward_aux(x.x1, x.x2, x.x3)
    y = ComponentVector(; y4=y4, y5=y5)
    return y
end

function conditions(x::ComponentVector, y::ComponentVector)
    c4, c5 = conditions_aux(x.x1, x.x2, x.x3, y.y4, y.y5)
    c = ComponentVector(; c4=c4, c5=c5)
    return c
end

implicit = ImplicitFunction(forward, conditions)

function full_pipeline(x1, x2, x3)
    x = ComponentVector(; x1=x1, x2=x2, x3=x3)
    y = implicit(x)
    return y.y1, y.y2
end

x1, x2, x3 = rand(2), rand(2), rand(2)
x = ComponentVector(; x1=x1, x2=x2, x3=x3)
implicit(x)

ForwardDiff.jacobian(implicit, x)
