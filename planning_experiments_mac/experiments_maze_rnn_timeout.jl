using Flux, OhMyREPL, CuArrays, CUDAnative, JLD, DelimitedFiles, BSON
cd("planning_experiments_mac")
include("../utils_heur.jl")
include("../solver_maze_timed.jl")
include("../MAC_network_self_stopping3.jl")

# i = instance_no, data_size, search_type, exp_folder, GPU_no

heur = "neural"
max_time = 600

i = parse.(Int, ARGS[1])
data_size = parse.(Int, ARGS[2]) 
search_type = ARGS[3]
exp_folder = ARGS[4]
CUDAnative.device!(parse.(Int, ARGS[5]))   

# load data
if data_size == 8
    data = load("data_maze_plan.jld")["data"]
elseif data_size == 16 
    data = load("data16_maze_plan.jld")["data"]
elseif data_size == 32 
    data = load("data32_maze_plan.jld")["data"]
elseif data_size == 64 
    data = load("data64_maze_plan.jld")["data"]
end

# args = [1, "gbfs", "mac_param_experiments9/exp_5_16_0_1_q",1]
# i = args[1]
# search_type = args[2]
# exp_folder = args[3]
cd("..")

# exp_folders = [
#     "mac_param_experiments9/exp_5_16_0_1_q",
#     "mac_param_experiments9/exp_5_32_0_2_q",
#     "mac_param_experiments9/exp_5_64_0_3_q"
# ]

file_pls = string("planning_experiments_mac/tmps/tmp", data_size, "_", split(exp_folder,"/")[2],"_",search_type,"_pls_q.jld")
file_exs = string("planning_experiments_mac/tmps/tmp", data_size, "_", split(exp_folder,"/")[2],"_",search_type,"_exs_q.jld")

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

pls[i] = -1
exs[i] = -1
save(file_pls, "data", pls)
save(file_exs, "data", exs)

mac = load_mac_network(exp_folder)
iters = parse(Int, split(exp_folder,"_")[6])

model = Chain(
    x -> run_MAC(mac, iters, x)
) |> gpu

maze = data[:,:,i]

if search_type == "gbfs"
    path, path_len, expanded_states = gbfs_timed(maze, false, "neural", [model], max_time)
else 
    path, path_len, expanded_states = bfs_timed(maze, false, "neural", [model], max_time)
end

pls[i] = path_len
exs[i] = expanded_states
save(file_pls, "data", pls)
save(file_exs, "data", exs)


# data_size = 64
# search_type = "bfs"
# exp_folder = "mac_param_experiments9/exp_5_64_100_0_0_0_r-try2"

# file_pls = string("planning_experiments_mac/tmps/tmp", data_size, "_", split(exp_folder,"/")[2],"_",search_type,"_pls_q.jld")
# file_exs = string("planning_experiments_mac/tmps/tmp", data_size, "_", split(exp_folder,"/")[2],"_",search_type,"_exs_q.jld")

# pls = load(file_pls)["data"]
# exs = load(file_exs)["data"]

# open(string("planning_experiments_mac/maze", data_size, "_rnn_try2_",search_type,"_stats.csv"), "a") do f
#     lens = pls
#     writedlm(f, lens)
#     writedlm(f,exs)
# end


