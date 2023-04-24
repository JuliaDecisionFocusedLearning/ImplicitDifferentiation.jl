struct SolverFailureException{S} <: Exception
    msg::String
    stats::S
end

function Base.show(io::IO, sfe::SolverFailureException)
    return println(
        io, "SolverFailureException: $(sfe.msg) \n Solver stats: $(string(sfe.stats))"
    )
end

"""
    LazyJacobianMul!{M,N}

Callable structure wrapping a lazy Jacobian operator with `N`-dimensional inputs into an in-place multiplication for vectors.

# Fields
- `J::M`: the lazy Jacobian of the function
- `input_size::NTuple{N,Int}`: the array size of the function input
"""
struct LazyJacobianMul!{M<:LazyJacobian,N}
    J::M
    input_size::NTuple{N,Int}
end

"""
    LazyJacobianTransposeMul!{M,N}

Callable structure wrapping a lazy Jacobian operator with `N`-dimensional outputs into an in-place multiplication for vectors.

# Fields
- `J::M`: the lazy Jacobian of the function
- `output_size::NTuple{N,Int}`: the array size of the function output
"""
struct LazyJacobianTransposeMul!{M<:LazyJacobian,N}
    J::M
    output_size::NTuple{N,Int}
end

function (ljm::LazyJacobianMul!)(res::Vector, δinput_vec::Vector)
    (; J, input_size) = ljm
    δinput = reshape(δinput_vec, input_size)
    δoutput = only(J * δinput)
    return res .= vec(δoutput)
end

function (ljtm::LazyJacobianTransposeMul!)(res::Vector, δoutput_vec::Vector)
    (; J, output_size) = ljtm
    δoutput = reshape(δoutput_vec, output_size)
    δinput = only(δoutput' * J)
    return res .= vec(δinput)
end
