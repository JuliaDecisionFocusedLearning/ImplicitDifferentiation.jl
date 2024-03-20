```@meta
CollapsedDocStrings = true
```

# API reference

## Public

```@autodocs
Modules = [ImplicitDifferentiation]
Private = false
```

## Internal

### Main package

```@autodocs
Modules = [ImplicitDifferentiation]
Public = false
```

### Extensions

```@docs
Modules = [
    Base.get_extension(ImplicitDifferentiation, :ImplicitDifferentiationChainRulesCoreExt),
    Base.get_extension(ImplicitDifferentiation, :ImplicitDifferentiationForwardDiffExt)
]
```
