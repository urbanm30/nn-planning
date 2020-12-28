using Flux, OhMyREPL, CuArrays, CUDAnative, JLD, DelimitedFiles, BSON, LinearAlgebra
cd("planning_experiments_mac")
include("../utils_heur.jl")
include("../solver_sokoban.jl")
include("../MAC_network_self_stopping3.jl")

# load data
data = load("data_sokoban_plan.jld")["data"]

heur = "neural"
max_time = 600

gbfs_pls = Dict()
gbfs_exs = Dict()
bfs_pls = Dict()
bfs_exs = Dict()
cd("..")

folders = readdir("mac_param_experiments12_sokoban")
exp_folders_r = []
exp_folders_q = []
for i in 1:length(folders)
    if length(split(folders[i],"_")) > 6
        push!(exp_folders_r, string("mac_param_experiments12_sokoban/",folders[i]))
    else
        push!(exp_folders_q, string("mac_param_experiments12_sokoban/",folders[i]))
    end
end

for exp_folder in exp_folders_r
    gbfs_pls[exp_folder] = []
    gbfs_exs[exp_folder] = []
    bfs_pls[exp_folder] = []
    bfs_exs[exp_folder] = []

    mac = load_mac_network(exp_folder)
    iters = parse(Int, split(exp_folder,"_")[7])

    model = Chain(
        x -> run_MAC(mac, iters, x)
    ) |> gpu

    if parse(Int, split(exp_folder,"_")[8]) == 1
        coords = true
    else
        coords = false
    end

    for i in 1:length(data)
        maze = data[i]      
        path, path_len, expanded_states = gbfs_timed(maze, true, "neural", [model, coords], max_time)        
        push!(gbfs_pls[exp_folder], path_len)
        push!(gbfs_exs[exp_folder], expanded_states)
        path, path_len, expanded_states = bfs_timed(maze, true, "neural", [model, coords], max_time)        
        push!(bfs_pls[exp_folder], path_len)
        push!(bfs_exs[exp_folder], expanded_states)
        println("Maze ", i, " done")
    end

    open("planning_experiments_mac/sokoban_gbfs_true_stats.csv", "a") do f
        lens = transpose(gbfs_pls[exp_folder])
        exs = transpose(gbfs_exs[exp_folder])
        writedlm(f, lens)
        writedlm(f,exs)
    end
    open("planning_experiments_mac/sokoban_bfs_true_stats.csv", "a") do f
        lens = transpose(bfs_pls[exp_folder])
        exs = transpose(bfs_exs[exp_folder])
        writedlm(f, lens)
        writedlm(f,exs)
    end
end

gbfs_pls = Dict()
gbfs_exs = Dict()
bfs_pls = Dict()
bfs_exs = Dict()

for exp_folder in exp_folders_q
    gbfs_pls[exp_folder] = []
    gbfs_exs[exp_folder] = []
    bfs_pls[exp_folder] = []
    bfs_exs[exp_folder] = []

    @load string(exp_folder, "/mac_q.bson") m 
    model = m |> gpu

    if parse(Int, split(exp_folder,"_")[7]) == 1
        coords = true
    else
        coords = false
    end

    for i in 1:length(data)
        maze = data[i]      
        path, path_len, expanded_states = gbfs_timed(maze, true, "neural", [model,coords],max_time)        
        push!(gbfs_pls[exp_folder], path_len)
        push!(gbfs_exs[exp_folder], expanded_states)
        path, path_len, expanded_states = bfs_timed(maze, true, "neural", [model,coords],max_time)        
        push!(bfs_pls[exp_folder], path_len)
        push!(bfs_exs[exp_folder], expanded_states)
    end

    open("planning_experiments_mac/sokoban_gbfs_true_stats.csv", "a") do f
        lens = transpose(gbfs_pls[exp_folder])
        exs = transpose(gbfs_exs[exp_folder])
        writedlm(f, lens)
        writedlm(f,exs)
    end
    open("planning_experiments_mac/sokoban_bfs_true_stats.csv", "a") do f
        lens = transpose(bfs_pls[exp_folder])
        exs = transpose(bfs_exs[exp_folder])
        writedlm(f, lens)
        writedlm(f,exs)
    end
end