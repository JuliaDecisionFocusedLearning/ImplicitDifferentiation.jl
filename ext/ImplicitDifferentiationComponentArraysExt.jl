module ImplicitDifferentiationComponentArraysExt

@static if isdefined(Base, :get_extension)
    using ComponentArrays: ComponentVector
else
    using ..ComponentArrays: ComponentVector
end

using Krylov: Krylov

Krylov.ktypeof(::ComponentVector{T,V}) where {T,V} = V  # TODO: type piracy

end
