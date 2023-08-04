function identity_break_autodiff(x)
    a = [0.0]
    a[1] = float(first(x))
    return x
end
