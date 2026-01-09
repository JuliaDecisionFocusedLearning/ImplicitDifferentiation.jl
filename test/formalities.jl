using TestItems

@testitem "Code quality" begin
    using Aqua
    using ForwardDiff: ForwardDiff
    using Zygote: Zygote
    Aqua.test_all(ImplicitDifferentiation; ambiguities=false, undocumented_names=true)
end

@testitem "Static checking" begin
    using JET
    using ForwardDiff: ForwardDiff
    using Zygote: Zygote
    JET.test_package(ImplicitDifferentiation; target_modules=(ImplicitDifferentiation,))
end

@testitem "Imports" begin
    using ExplicitImports
    using ForwardDiff: ForwardDiff
    using Zygote: Zygote
    @test check_no_implicit_imports(ImplicitDifferentiation) === nothing
    @test check_no_stale_explicit_imports(ImplicitDifferentiation) === nothing
    @test check_all_explicit_imports_via_owners(ImplicitDifferentiation) === nothing
    @test_broken check_all_explicit_imports_are_public(ImplicitDifferentiation) === nothing
    @test check_all_qualified_accesses_via_owners(ImplicitDifferentiation) === nothing
    @test check_no_self_qualified_accesses(ImplicitDifferentiation) === nothing
end

@testitem "Doctests" begin
    using Documenter
    Documenter.DocMeta.setdocmeta!(
        ImplicitDifferentiation,
        :DocTestSetup,
        :(using ImplicitDifferentiation);
        recursive=true,
    )
    doctest(ImplicitDifferentiation)
end
