using JLD
cd("planning_experiments_mac")
include("../utils_heur.jl")
# include("../solver_sokoban.jl")
include("../sokoban_domain/sokoban_generator.jl")
cd("..")

# 3 siroky zdi - srat na to
# data16 = make_solvable_lvl(10, 50)
# save("planning_experiments_mac/data16_sokoban.jld", "data", data16)

data32 = make_solvable_lvl(26, 50)
save("planning_experiments_mac/data32_sokoban.jld", "data", data32)

# data64 = make_solvable_lvl(62, 50)
# save("planning_experiments_mac/data64_sokoban.jld", "data", data64)s