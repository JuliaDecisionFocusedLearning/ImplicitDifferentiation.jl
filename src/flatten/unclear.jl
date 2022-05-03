macro constructor(T)
    return flatten_expr(T, T)
end

macro constructor(T, C)
    return flatten_expr(T, C)
end

flatten_expr(T, C) = quote
    function ImplicitDifferentiation.flatten(x::$(esc(T)))
        v, un = flatten(ntfromstruct(x))
        return identity.(v), Unflatten(x, y -> structfromnt($(esc(C)), un(y)))
    end
    function ImplicitDifferentiation.flatten_similar(x1::$(esc(T)), x2::$(esc(T)))
        v, un = zygote_flatten(ntfromstruct(x1), ntfromstruct(x2))
        return identity.(v), Unflatten(x2, y -> structfromnt($(esc(C)), un(y)))
    end
    ImplicitDifferentiation._zero(x::$(esc(T))) = structfromnt($(esc(C)), _zero(ntfromstruct(x)))
end
