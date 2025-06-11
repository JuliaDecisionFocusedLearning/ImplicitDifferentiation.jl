using TestItemRunner

@testmodule TestUtils begin
    include("utils.jl")
    export Scenario, test_implicit, add_arg_mult
    export default_solver, default_conditions
end

@run_package_tests
