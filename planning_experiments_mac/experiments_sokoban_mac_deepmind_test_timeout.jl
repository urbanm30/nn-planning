using Flux, OhMyREPL, CuArrays, CUDAnative, JLD, DelimitedFiles, BSON, LinearAlgebra
cd("planning_experiments_mac")
include("../utils_heur.jl")
include("../solver_sokoban.jl")
include("../heur_att_load_params.jl")
include("../MAC_network_self_stopping3.jl")

# Arguments: i (number of the instance), search_type (gbfs, bfs)
# args = [1,16,"gbfs"]
i = parse.(Int, ARGS[1])
search_type = ARGS[2]
exp_folder = ARGS[3]

println("i = ", i, " ", search_type)
data = load("data_sokoban_test_deepmind.jld")["data"]
data_size = 10

ixs = load("sokoban_deepmind_ixs.jld")["data"]
data = data[:,:,ixs]

cd("..")

# heur_network = load_att_model_from_params("heuristic_networks/sokoban_model_params/0-10-5-50")
# exp_folder = "mac_param_experiments12_sokoban/exp_5_32_100_1_1_1_r"

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

heur_network = [model, coords]

max_time = 600

file_pls = string("planning_experiments_mac/sokoban_deepmind_tmps/tmp_rnn_", split(exp_folder,"/")[2], "_",data_size,"_sokoban_deepmind_nn_",search_type,"_pls.jld")
file_exs = string("planning_experiments_mac/sokoban_deepmind_tmps/tmp_rnn_", split(exp_folder,"/")[2], "_", data_size,"_sokoban_deepmind_nn_",search_type,"_exs.jld")

if !isfile(file_pls) 
    pls = zeros(1,50)
    exs = zeros(1,50)
    save(file_pls, "data", pls)
    save(file_exs, "data", exs)   
else 
    pls = load(file_pls)["data"]
    exs = load(file_exs)["data"]
end

if pls[i] != 0.0 
    exit()
end

maze = data[:,:,i]       
pls[i] = -1
exs[i] = -1
save(file_pls, "data", pls)
save(file_exs, "data", exs)

if search_type == "gbfs"
    path, path_len, expanded_states = gbfs_timed(maze, false, "neural", heur_network, max_time)
elseif search_type == "bfs" 
    path, path_len, expanded_states = bfs_timed(maze, false, "neural", heur_network, max_time)
end
pls[i] = path_len
exs[i] = expanded_states
save(file_pls, "data", pls)
save(file_exs, "data", exs)

# exp_folders = [
#     "mac_param_experiments12_sokoban/exp_5_32_100_1_1_1_r", 
#     "mac_param_experiments12_sokoban/exp_5_64_200_0_0_0_r",
#     "mac_param_experiments12_sokoban/exp_5_64_200_0_1_1_r",
#     "mac_param_experiments12_sokoban/exp_5_16_200_0_1_1_r"
# ]

# for exp_folder in exp_folders
#     search_type = "bfs"
#     data_size = 10

#     file_pls = string("planning_experiments_mac/sokoban_deepmind_tmps/tmp_rnn_", split(exp_folder,"/")[2], "_",data_size,"_sokoban_deepmind_nn_",search_type,"_pls.jld")
#     file_exs = string("planning_experiments_mac/sokoban_deepmind_tmps/tmp_rnn_", split(exp_folder,"/")[2], "_", data_size,"_sokoban_deepmind_nn_",search_type,"_exs.jld")

#     pls = load(file_pls)["data"]
#     exs = load(file_exs)["data"]
#     filename = string("planning_experiments_mac/sokoban_mac_deepmind_",data_size,"_",search_type,"_rand50_stats.csv") 
#     open(filename, "a") do f
#         lens = pls
#         writedlm(f, lens)
#         writedlm(f,exs)
#     end
# end