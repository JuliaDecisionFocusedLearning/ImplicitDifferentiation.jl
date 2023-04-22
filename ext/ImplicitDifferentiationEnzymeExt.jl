module ImplicitDifferentiationEnzymeExt

@static if isdefined(Base, :get_extension)
    using Enzyme
else
    using ..Enzyme
end

using ImplicitDifferentiation

end
