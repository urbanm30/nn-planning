using Flux, OhMyREPL, CuArrays, CUDAnative, JLD, DelimitedFiles, BSON
cd("planning_experiments_mac")
include("../utils_heur.jl")
include("../solver_maze.jl")
# include("../MAC_network_self_stopping3.jl")

# load data
# data = load("data_maze_plan.jld")["data"]
# data16 = load("data16_maze_plan.jld")["data"]
data32 = load("data32_maze_plan.jld")["data"]
data64 = load("data64_maze_plan.jld")["data"]

heurs = [
    "none",
    "euclidean",
    "hff",
    "lmcut"
]
# heur_combs = [
#     ["euclidean", "neural"],
#     ["hff", "neural"],
#     ["lmcut", "neural"],
#     ["neural", "euclidean"],
#     ["neural", "hff"],
#     ["neural", "lmcut"],
# ]
cd("..")

# DATA16 --------------------------------------------------------------
# gbfs_pls = Dict()
# gbfs_exs = Dict()

# for heur in heurs
#     gbfs_pls[heur] = []
#     gbfs_exs[heur] = []

#     for i in 1:size(data16,3)
#         maze = data16[:,:,i]      
#         path, path_len, expanded_states = gbfs(maze, true, heur, [])
#         push!(gbfs_pls[heur], path_len)
#         push!(gbfs_exs[heur], expanded_states)
#     end

#     open("planning_experiments_mac/maze16_gbfs_true_heur_stats.csv", "a") do f
#         lens = transpose(gbfs_pls[heur])
#         exs = transpose(gbfs_exs[heur])
#         writedlm(f, lens)
#         writedlm(f,exs)
#     end
# end

# bfs_pls = Dict()
# bfs_exs = Dict()

# for heur in heurs
#     bfs_pls[heur] = []
#     bfs_exs[heur] = []

#     for i in 1:size(data16,3)
#         maze = data16[:,:,i]      
#         path, path_len, expanded_states = bfs(maze, true, heur, [])
#         push!(bfs_pls[heur], path_len)
#         push!(bfs_exs[heur], expanded_states)
#     end

#     open("planning_experiments_mac/maze16_bfs_true_heur_stats.csv", "a") do f
#         lens = transpose(bfs_pls[heur])
#         exs = transpose(bfs_exs[heur])
#         writedlm(f, lens)
#         writedlm(f,exs)
#     end
# end

# DATA32 --------------------------------------------------------------
# gbfs_pls = Dict()
# gbfs_exs = Dict()

# for heur in heurs
#     gbfs_pls[heur] = []
#     gbfs_exs[heur] = []

#     for i in 1:size(data32,3)
#         maze = data32[:,:,i]      
#         path, path_len, expanded_states = gbfs(maze, true, heur, [])
#         push!(gbfs_pls[heur], path_len)
#         push!(gbfs_exs[heur], expanded_states)
#     end

#     open("planning_experiments_mac/maze32_gbfs_true_heur_stats.csv", "a") do f
#         lens = transpose(gbfs_pls[heur])
#         exs = transpose(gbfs_exs[heur])
#         writedlm(f, lens)
#         writedlm(f,exs)
#     end
# end

bfs_pls = Dict()
bfs_exs = Dict()

for heur in heurs
    bfs_pls[heur] = []
    bfs_exs[heur] = []

    for i in 1:size(data32,3)
        maze = data32[:,:,i]      
        path, path_len, expanded_states = bfs(maze, true, heur, [])
        push!(bfs_pls[heur], path_len)
        push!(bfs_exs[heur], expanded_states)
        println("Done 32 true ", i)
    end

    open("planning_experiments_mac/maze32_bfs_true_heur_stats.csv", "a") do f
        lens = transpose(bfs_pls[heur])
        exs = transpose(bfs_exs[heur])
        writedlm(f, lens)
        writedlm(f,exs)
    end
end

# DATA64 --------------------------------------------------------------
gbfs_pls = Dict()
gbfs_exs = Dict()

for heur in heurs
    gbfs_pls[heur] = []
    gbfs_exs[heur] = []

    for i in 1:size(data64,3)
        maze = data64[:,:,i]      
        path, path_len, expanded_states = gbfs(maze, true, heur, [])
        push!(gbfs_pls[heur], path_len)
        push!(gbfs_exs[heur], expanded_states)
    end

    open("planning_experiments_mac/maze64_gbfs_true_heur_stats.csv", "a") do f
        lens = transpose(gbfs_pls[heur])
        exs = transpose(gbfs_exs[heur])
        writedlm(f, lens)
        writedlm(f,exs)
    end
end

bfs_pls = Dict()
bfs_exs = Dict()

for heur in heurs
    bfs_pls[heur] = []
    bfs_exs[heur] = []

    for i in 1:size(data64,3)
        maze = data64[:,:,i]      
        path, path_len, expanded_states = bfs(maze, true, heur, [])
        push!(bfs_pls[heur], path_len)
        push!(bfs_exs[heur], expanded_states)
    end

    open("planning_experiments_mac/maze64_bfs_true_heur_stats.csv", "a") do f
        lens = transpose(bfs_pls[heur])
        exs = transpose(bfs_exs[heur])
        writedlm(f, lens)
        writedlm(f,exs)
    end
end