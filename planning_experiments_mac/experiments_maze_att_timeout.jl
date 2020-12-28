using Flux, OhMyREPL, CuArrays, CUDAnative, JLD, DelimitedFiles, BSON
cd("planning_experiments_mac")
include("../utils_heur.jl")
include("../solver_maze_timed.jl")
include("../heur_att_load_params.jl")

# Arguments: i (number of the instance), data_size (32,64), search_type (gbfs, bfs), att_net_architecture(1,2)
# args = [1,16,"gbfs",1]
i = parse.(Int, ARGS[1])
data_size = parse.(Int, ARGS[2])
search_type = ARGS[3]
architecture = parse.(Int, ARGS[4])

# i = args[1]
# data_size = args[2]
# search_type = args[3]
# architecture = args[4]

println("i = ", i, " ", data_size, " ", search_type, " ", architecture)

if data_size == 32 
    data = load("data32_maze_plan.jld")["data"]
elseif data_size == 64 
    data = load("data64_maze_plan.jld")["data"]
elseif data_size == 16 
    data = load("data16_maze_plan.jld")["data"]
else 
    println("Invalid data size selected -> exiting")
    exit()
end
cd("..")

if architecture == 1
    heur_network = load_att_model_from_params("heuristic_networks/maze_model_params/1-10-1-100")
elseif architecture == 2
    heur_network = load_att_model_from_params("heuristic_networks/maze_model_params/1-10-5-100")
end

max_time = 600

file_pls = string("planning_experiments_mac/att_tmps/tmp_att_",architecture,"_",data_size,"_maze_nn_",search_type,"_true_pls.jld")
file_exs = string("planning_experiments_mac/att_tmps/tmp_att_",architecture,"_",data_size,"_maze_nn_",search_type,"_true_exs.jld")

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
    path, path_len, expanded_states = gbfs_timed(maze, true, "neural", heur_network, max_time)
elseif search_type == "bfs" 
    path, path_len, expanded_states = bfs_timed(maze, true, "neural", heur_network, max_time)
end
pls[i] = path_len
exs[i] = expanded_states
save(file_pls, "data", pls)
save(file_exs, "data", exs)


# Set search type and data_size of the experiments 

# search_type = "bfs"
# data_size = 32
# architecture = 2

# file_pls = string("planning_experiments_mac/att_tmps/tmp_att_",architecture,"_",data_size,"_maze_nn_",search_type,"_true_pls.jld")
# file_exs = string("planning_experiments_mac/att_tmps/tmp_att_",architecture,"_",data_size,"_maze_nn_",search_type,"_true_exs.jld")

# pls = load(file_pls)["data"]
# exs = load(file_exs)["data"]
# filename = string("planning_experiments_mac/maze_att_",data_size,"_",search_type,"_true_stats.csv") 
# open(filename, "a") do f
#     lens = pls
#     writedlm(f, lens)
#     writedlm(f,exs)
# end
