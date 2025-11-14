module ImplicitDifferentiationForwardDiffExt

using ADTypes: AutoEnzyme
using EnzymeCore
import EnzymeCore: EnzymeRules
using ImplicitDifferentiation:
    ImplicitFunction,
    ImplicitFunctionPreparation,
    IterativeLeastSquaresSolver,
    build_A,
    build_Aᵀ,
    build_B

function EnzymeRules.forward(config, implicit::Const{<:ImplicitFunction}, RT::Type, x, args...)
	prep = ImplicitFunctionPreparation(eltype(x.val))
	EnzymeRules.forward(config, implicit, RT, Const(prep), x, args...)
end

@inline function EnzymeRules.forward(config, implicit::Const{<:ImplicitFunction}, RT::Type, prep::Const{<:ImplicitFunctionPreparation{R}}, x, args...) where R
	implicit = implicit.val
	prep = prep.val

	dx = x.dval
	# dargs = ntuple(length(args)) do i
		# args[i].dval
	# end

	x = x.val
	args = ntuple(length(args)) do i
        @assert args[i] isa Const
		args[i].val
	end

	(; conditions, linear_solver) = implicit
	
	y, z = implicit(x, args...)
	c = conditions(x, y, z, args...)

	y0 = zero(y)
	forward_backend = AutoEnzyme(mode=Forward)
	reverse_backend = AutoEnzyme(mode=Reverse)

	A = build_A(implicit, prep, x, y, z, c, args...; suggested_backend=forward_backend)
    B = build_B(implicit, prep, x, y, z, c, args...; suggested_backend=forward_backend)
    Aᵀ = if linear_solver isa IterativeLeastSquaresSolver
        build_Aᵀ(implicit, prep, x, y, z, c, args...; suggested_backend=reverse_backend)
    else
        nothing
    end

	if EnzymeRules.width(config) == 1
		dc = B(dx)
		dy = linear_solver(A, Aᵀ, dc, y0)::typeof(y0)
		dz = Enzyme.make_zero(z)
		
		if EnzymeRules.needs_primal(config)
			return Duplicated((y, z), (dy, dz)) 
		else
			return dy, dz
		end
	else
		dc = map(B, dx)
		dy = map(dc) do dₖc
			linear_solver(A, Aᵀ, -dₖc, y0)::typeof(y0)
		end

		df = ntuple(Val(EnzymeRules.width(config))) do i
			(dy[i]::typeof(y0), Enzyme.make_zero(z))
		end
		
		if EnzymeRules.needs_primal(config)
			return BatchDuplicated((y, z), df)
		else
			return df
		end
	end
end


end
