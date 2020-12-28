using Flux, OhMyREPL, CuArrays, CUDAnative, JLD, DelimitedFiles, BSON
cd("planning_experiments_mac")
include("../utils_heur.jl")
include("../solver_maze_timed.jl")
include("../MAC_network_self_stopping3.jl")

# load data
data = load("data64_maze_plan.jld")["data"]

heur = "neural"
max_time = 600

# i = instance_no, search_type, exp_folder (trained net), GPU_no
i = parse.(Int, ARGS[1])
search_type = ARGS[2]
exp_folder = ARGS[3]
CUDAnative.device!(parse.(Int, ARGS[4]))   

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

file_pls = string("planning_experiments_mac/tmps/tmp64_", split(exp_folder,"/")[2],"_",search_type,"_pls_q_true.jld")
file_exs = string("planning_experiments_mac/tmps/tmp64_", split(exp_folder,"/")[2],"_",search_type,"_exs_q_true.jld")

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

@load string(exp_folder, "/mac_q.bson") m 
model = m |> gpu

maze = data[:,:,i]  
if search_type == "gbfs"
    path, path_len, expanded_states = gbfs_timed(maze, true, "neural", [model], max_time)
else 
    path, path_len, expanded_states = bfs_timed(maze, true, "neural", [model], max_time)
end

pls[i] = path_len
exs[i] = expanded_states
save(file_pls, "data", pls)
save(file_exs, "data", exs)


# exp_folders = [
#     "mac_param_experiments9/exp_5_16_0_1_q",
#     "mac_param_experiments9/exp_5_32_0_2_q",
#     "mac_param_experiments9/exp_5_64_0_3_q"
# ]
# search_type = "gbfs"

# for exp_folder in exp_folders
#     file_pls = string("planning_experiments_mac/tmps/tmp64_", split(exp_folder,"/")[2],"_",search_type,"_pls_q_true.jld")
#     file_exs = string("planning_experiments_mac/tmps/tmp64_", split(exp_folder,"/")[2],"_",search_type,"_exs_q_true.jld")

#     pls = load(file_pls)["data"]
#     exs = load(file_exs)["data"]
    
#     open(string("planning_experiments_mac/maze64_",search_type,"_true_stats.csv"), "a") do f
#         lens = pls
#         writedlm(f, lens)
#         writedlm(f,exs)
#     end
# end

