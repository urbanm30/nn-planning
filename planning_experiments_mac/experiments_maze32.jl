using Flux, OhMyREPL, CuArrays, CUDAnative, JLD, DelimitedFiles, BSON
cd("planning_experiments_mac")
include("../utils_heur.jl")
include("../solver_maze.jl")
include("../MAC_network_self_stopping3.jl")

# load data
data = load("data32_maze_plan.jld")["data"]
CUDAnative.device!(1)

heur = "neural"

# Load heuristic networks
gbfs_pls = Dict()   
gbfs_exs = Dict()

cd("..")
# exp_folders = [
#     "mac_param_experiments9/exp_5_16_100_0_1_1_r",
#     "mac_param_experiments9/exp_5_16_200_0_1_1_r",
#     "mac_param_experiments9/exp_5_32_100_0_0_0_r",  
#     "mac_param_experiments9/exp_5_32_100_0_1_1_r",
#     "mac_param_experiments9/exp_5_32_200_0_0_0_r",
#     "mac_param_experiments9/exp_5_32_200_0_1_1_r",
#     "mac_param_experiments9/exp_5_64_100_0_0_0_r",
#     "mac_param_experiments9/exp_5_64_200_0_0_0_r",
#     "mac_param_experiments9/exp_5_64_200_0_1_1_r"   
# ]

# # DONE 
# for exp_folder in exp_folders
#     gbfs_pls[exp_folder] = []
#     gbfs_exs[exp_folder] = []

#     push!(gbfs_pls[exp_folder], split(exp_folder,"/")[2])
#     push!(gbfs_exs[exp_folder], split(exp_folder,"/")[2])

#     splitted = split(exp_folder, "/")

#     file_pls = string("planning_experiments_mac/tmps/tmp_",splitted[1], "_", splitted[2],"_gbfs_pls.jld")
#     file_exs = string("planning_experiments_mac/tmps/tmp_",splitted[1], "_", splitted[2],"_gbfs_exs.jld")

#     mac = load_mac_network(exp_folder)
#     iters = parse(Int, split(exp_folder,"_")[6])

#     model = Chain(
#         x -> run_MAC(mac, iters, x)
#     ) |> gpu

#     start_no = 1
#     if isfile(file_pls) && isfile(file_exs)
#         pls = load(file_pls)["data"]
#         exs = load(file_exs)["data"]  
#         start_no = length(pls) 
#         gbfs_pls[exp_folder] = pls
#         gbfs_exs[exp_folder] = exs
#     end

#     for i in start_no:size(data,3)
#         maze = data[:,:,i]      
#         path, path_len, expanded_states = gbfs(maze, true, "neural", [model])
#         push!(gbfs_pls[exp_folder], path_len)   
#         push!(gbfs_exs[exp_folder], expanded_states)
#         save(file_pls, "data", gbfs_pls[exp_folder])
#         save(file_exs, "data", gbfs_exs[exp_folder])
#         println("Done ", i)
#     end

#     open("planning_experiments_mac/maze32_gbfs_true_stats.csv", "a") do f
#         lens = transpose(gbfs_pls[exp_folder][2:end])
#         exs = transpose(gbfs_exs[exp_folder][2:end])
#         writedlm(f, lens)
#         writedlm(f,exs)
#         println("Wrote something!")
#     end
# end

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

    push!(gbfs_pls[exp_folder], split(exp_folder,"/")[2])
    push!(gbfs_exs[exp_folder], split(exp_folder,"/")[2])

    splitted = split(exp_folder, "/")

    file_pls = string("planning_experiments_mac/tmps/tmp_",splitted[1], "_", splitted[2],"_gbfs_pls_q.jld")
    file_exs = string("planning_experiments_mac/tmps/tmp_",splitted[1], "_", splitted[2],"_gbfs_exs_q.jld")

    @load string(exp_folder, "/mac_q.bson") m 
    model = m |> gpu

    start_no = 1
    if isfile(file_pls) && isfile(file_exs)
        pls = load(file_pls)["data"]
        exs = load(file_exs)["data"]
        start_no = length(pls) 
        gbfs_pls[exp_folder] = pls
        gbfs_exs[exp_folder] = exs
    end

    for i in start_no:size(data,3)
        maze = data[:,:,i]      
        path, path_len, expanded_states = gbfs(maze, true, "neural", [model])
        push!(gbfs_pls[exp_folder], path_len)
        push!(gbfs_exs[exp_folder], expanded_states)
        save(file_pls, "data", gbfs_pls[exp_folder])
        save(file_exs, "data", gbfs_exs[exp_folder])
        println("Done q ", i)
    end

    if start_no < 51
        open("planning_experiments_mac/maze32_gbfs_true_stats.csv", "a") do f
            lens = transpose(gbfs_pls[exp_folder][2:end])
            exs = transpose(gbfs_exs[exp_folder][2:end])
            writedlm(f, lens)
            writedlm(f,exs)
        end
    end
end
println("maze32_gbfs_true_q DONE!")

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

gbfs_pls = Dict()
gbfs_exs = Dict()

for exp_folder in exp_folders
    gbfs_pls[exp_folder] = []
    gbfs_exs[exp_folder] = []

    push!(gbfs_pls[exp_folder], split(exp_folder,"/")[2])
    push!(gbfs_exs[exp_folder], split(exp_folder,"/")[2])

    splitted = split(exp_folder, "/")

    file_pls = string("planning_experiments_mac/tmps/tmp_",splitted[1], "_", splitted[2],"_bfs_pls.jld")
    file_exs = string("planning_experiments_mac/tmps/tmp_",splitted[1], "_", splitted[2],"_bfs_exs.jld")

    mac = load_mac_network(exp_folder)
    iters = parse(Int, split(exp_folder,"_")[6])

    model = Chain(
        x -> run_MAC(mac, iters, x)
    ) |> gpu

    start_no = 1
    if isfile(file_pls) && isfile(file_exs)
        pls = load(file_pls)["data"]
        exs = load(file_exs)["data"]
        start_no = length(pls) 
        gbfs_pls[exp_folder] = pls
        gbfs_exs[exp_folder] = exs
    end

    for i in start_no:size(data,3)
        maze = data[:,:,i]      
        path, path_len, expanded_states = bfs(maze, true, "neural", [model])
        push!(gbfs_pls[exp_folder], path_len)
        push!(gbfs_exs[exp_folder], expanded_states)
        save(file_pls, "data", gbfs_pls[exp_folder])
        save(file_exs, "data", gbfs_exs[exp_folder])
        println("Done ", i)
    end

    if start_no < 51
        open("planning_experiments_mac/maze32_bfs_true_stats.csv", "a") do f
            lens = transpose(gbfs_pls[exp_folder][2:end])
            exs = transpose(gbfs_exs[exp_folder][2:end])
            writedlm(f, lens)
            writedlm(f,exs)
        end
    end
end

exp_folders = [
    "mac_param_experiments9/exp_5_16_0_1_q",
    "mac_param_experiments9/exp_5_32_0_2_q",
    "mac_param_experiments9/exp_5_64_0_3_q"
]

gbfs_pls = Dict()
gbfs_exs = Dict()

for exp_folder in exp_folders
    gbfs_pls[exp_folder] = []   
    gbfs_exs[exp_folder] = []

    push!(gbfs_pls[exp_folder], split(exp_folder,"/")[2])
    push!(gbfs_exs[exp_folder], split(exp_folder,"/")[2])

    splitted = split(exp_folder, "/")

    file_pls = string("planning_experiments_mac/tmps/tmp_",splitted[1], "_", splitted[2],"_bfs_pls_q.jld")
    file_exs = string("planning_experiments_mac/tmps/tmp_",splitted[1], "_", splitted[2],"_bfs_exs_q.jld")

    @load string(exp_folder, "/mac_q.bson") m 
    model = m |> gpu

    start_no = 1
    if isfile(file_pls) && isfile(file_exs)
        pls = load(file_pls)["data"]
        exs = load(file_exs)["data"]
        start_no = length(pls) 
        gbfs_pls[exp_folder] = pls
        gbfs_exs[exp_folder] = exs
    end

    for i in start_no:size(data,3)
        maze = data[:,:,i]      
        path, path_len, expanded_states = bfs(maze, true, "neural", [model])
        push!(gbfs_pls[exp_folder], path_len)
        push!(gbfs_exs[exp_folder], expanded_states)
        save(file_pls, "data", gbfs_pls[exp_folder])
        save(file_exs, "data", gbfs_exs[exp_folder])
        println("Done q ", i)
    end

    if start_no < 51
        open("planning_experiments_mac/maze32_bfs_true_stats.csv", "a") do f
            lens = transpose(gbfs_pls[exp_folder][2:end])
            exs = transpose(gbfs_exs[exp_folder][2:end])
            writedlm(f, lens)
            writedlm(f,exs)
        end
    end
end