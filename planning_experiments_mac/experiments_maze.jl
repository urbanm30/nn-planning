using Flux, OhMyREPL, CuArrays, CUDAnative, JLD, DelimitedFiles, BSON
cd("planning_experiments_mac")
include("../utils_heur.jl")
include("../solver_maze.jl")
include("../MAC_network_self_stopping3.jl")

# load data
data = load("data_maze_plan.jld")["data"]

heur = "neural"
# heur_combs = [
#     ["euclidean", "neural"],
#     ["hff", "neural"],
#     ["lmcut", "neural"],
#     ["neural", "euclidean"],
#     ["neural", "hff"],
#     ["neural", "lmcut"],
# ]

# Load heuristic networks
gbfs_pls = Dict()
gbfs_exs = Dict()

cd("..")
exp_folders = [
    "mac_param_experiments9/exp_5_16_100_0_1_1_r",
    "mac_param_experiments9/exp_5_16_200_0_1_1_r",
    "mac_param_experiments9/exp_5_32_100_0_0_0_r",
    "mac_param_experiments9/exp_5_32_100_0_1_1_r",
    "mac_param_experiments9/exp_5_32_200_0_0_0_r",
    "mac_param_experiments9/exp_5_32_200_0_1_1_r",
    "mac_param_experiments9/exp_5_64_100_0_0_0_r",
    "mac_param_experiments9/exp_5_64_200_0_0_0_r",
    "mac_param_experiments9/exp_5_64_200_0_1_1_r"
]

for exp_folder in exp_folders
    gbfs_pls[exp_folder] = []
    gbfs_exs[exp_folder] = []

    mac = load_mac_network(exp_folder)
    iters = parse(Int, split(exp_folder,"_")[6])

    model = Chain(
        x -> run_MAC(mac, iters, x)
    ) |> gpu

    for i in 1:size(data,3)
        maze = data[:,:,i]      
        path, path_len, expanded_states = gbfs(maze, true, "neural", [model])
        push!(gbfs_pls[exp_folder], path_len)
        push!(gbfs_exs[exp_folder], expanded_states)
        # println("Done ", i)
    end

    open("planning_experiments_mac/maze_gbfs_true_stats.csv", "a") do f
        lens = transpose(gbfs_pls[exp_folder])
        exs = transpose(gbfs_exs[exp_folder])
        writedlm(f, lens)
        writedlm(f,exs)
    end
end

gbfs_pls = Dict()
gbfs_exs = Dict()

for exp_folder in exp_folders
    gbfs_pls[exp_folder] = []
    gbfs_exs[exp_folder] = []

    mac = load_mac_network(exp_folder)
    iters = parse(Int, split(exp_folder,"_")[6])

    model = Chain(
        x -> run_MAC(mac, iters, x)
    ) |> gpu

    for i in 1:size(data,3)
        maze = data[:,:,i]      
        path, path_len, expanded_states = bfs(maze, true, "neural", [model])
        push!(gbfs_pls[exp_folder], path_len)
        push!(gbfs_exs[exp_folder], expanded_states)
        # println("Done ", i)
    end

    open("planning_experiments_mac/maze_bfs_true_stats.csv", "a") do f
        lens = transpose(gbfs_pls[exp_folder])
        exs = transpose(gbfs_exs[exp_folder])
        writedlm(f, lens)
        writedlm(f,exs)
    end
end

gbfs_pls = Dict()
gbfs_exs = Dict()

exp_folders = [
    "mac_param_experiments9/exp_5_16_0_1_q",
    "mac_param_experiments9/exp_5_32_0_2_q",
    "mac_param_experiments9/exp_5_64_0_3_q"
]

for exp_folder in exp_folders
    gbfs_pls[exp_folder] = []
    gbfs_exs[exp_folder] = []

    @load string(exp_folder, "/mac_q.bson") m 
    model = m |> gpu

    for i in 1:size(data,3)
        maze = data[:,:,i]      
        path, path_len, expanded_states = gbfs(maze, true, "neural", [model])
        push!(gbfs_pls[exp_folder], path_len)
        push!(gbfs_exs[exp_folder], expanded_states)
        # println("Done ", i)
    end

    open("planning_experiments_mac/maze_gbfs_true_stats.csv", "a") do f
        lens = transpose(gbfs_pls[exp_folder])
        exs = transpose(gbfs_exs[exp_folder])
        writedlm(f, lens)
        writedlm(f,exs)
    end
end

gbfs_pls = Dict()
gbfs_exs = Dict()

for exp_folder in exp_folders
    gbfs_pls[exp_folder] = []
    gbfs_exs[exp_folder] = []

    @load string(exp_folder, "/mac_q.bson") m 
    model = m |> gpu

    for i in 1:size(data,3)
        maze = data[:,:,i]      
        path, path_len, expanded_states = bfs(maze, true, "neural", [model])
        push!(gbfs_pls[exp_folder], path_len)
        push!(gbfs_exs[exp_folder], expanded_states)
        # println("Done ", i)
    end

    open("planning_experiments_mac/maze_bfs_true_stats.csv", "a") do f
        lens = transpose(gbfs_pls[exp_folder])
        exs = transpose(gbfs_exs[exp_folder])
        writedlm(f, lens)
        writedlm(f,exs)
    end
end

# --------------------------------------------------------
# MH-GBFS experiments 

# gbfs_pls = Dict()
# gbfs_exs = Dict()
# for heur in heur_combs
#     gbfs_pls[heur] = []
#     gbfs_exs[heur] = []
# end

# cd("..")
# for i in 1:size(data,3)
#     for heur in heur_combs
#         println(heur, i)

#         if "neural" in heur
#             maze = ones(size(data,1) + 2, size(data,2) + 2)
#             maze[2:9,2:9] = data[:,:,i]             
#             path, path_len, expanded_states = mh_gbfs(maze, true, heur, hn)
#             push!(gbfs_pls[heur], path_len)
#             push!(gbfs_exs[heur], expanded_states)
#         elseif "neural2" in heur
#             maze = ones(size(data,1) + 2, size(data,2) + 2)
#             maze[2:9,2:9] = data[:,:,i]   
#             path, path_len, expanded_states = mh_gbfs(maze, true, heur, hn2)
#             push!(gbfs_pls[heur], path_len)
#             push!(gbfs_exs[heur], expanded_states)
#         else
#             path, path_len, expanded_states = mh_gbfs(data[:,:,i], true, heur, [])
#             push!(gbfs_pls[heur], path_len)
#             push!(gbfs_exs[heur], expanded_states)
#         end
#     end
# end

# cd("planning_experiments")
# for h in heur_combs
#     lens = reshape(cat(h, gbfs_pls[h],dims=1), 1, :)
#     exs = reshape(cat(h, gbfs_exs[h],dims=1), 1, :)
#     open("maze_mh_gbfs_stats.csv", "a") do f
#         writedlm(f, lens)
#         writedlm(f,exs)
#     end
# end