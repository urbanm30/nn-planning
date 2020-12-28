using Flux, OhMyREPL, CuArrays, CUDAnative, JLD, DelimitedFiles, BSON
cd("planning_experiments_mac")
include("../utils_heur.jl")
include("../solver_mg_maze.jl")
include("../heur_att_load_params.jl")

# Arguments: i (number of the instance), data_size (32,64), search_type (gbfs, bfs)
# args = [1,16,"gbfs"]
i = parse.(Int, ARGS[1])
data_size = parse.(Int, ARGS[2])
search_type = ARGS[3]

# i = args[1]
# data_size = args[2]
# search_type = args[3]

println("i = ", i, " ", data_size, " ", search_type)
    
if data_size == 32 
    data = load("data32_mg_maze.jld")["data"]
elseif data_size == 64 
    data = load("data64_mg_maze.jld")["data"]
elseif data_size == 16 
    data = load("data16_mg_maze.jld")["data"]
else 
    println("Invalid data size selected -> exiting")
    exit()
end
cd("..")

heur_network = load_att_model_from_params("heuristic_networks/mg_maze_model_params/1-5-1-50")

max_time = 600

file_pls = string("planning_experiments_mac/att_tmps/tmp_att_",data_size,"_mg_maze_nn_",search_type,"_true_pls.jld")
file_exs = string("planning_experiments_mac/att_tmps/tmp_att_",data_size,"_mg_maze_nn_",search_type,"_true_exs.jld")

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

maze = onehot2img(data[:,:,:,i:i])[:,:,1]      
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

# search_type = "bfs"
# data_size = 64

# file_pls = string("planning_experiments_mac/att_tmps/tmp_att_",data_size,"_mg_maze_nn_",search_type,"_true_pls.jld")
# file_exs = string("planning_experiments_mac/att_tmps/tmp_att_",data_size,"_mg_maze_nn_",search_type,"_true_exs.jld")

# pls = load(file_pls)["data"]
# exs = load(file_exs)["data"]
# filename = string("planning_experiments_mac/mg_maze_att_",data_size,"_",search_type,"_true_stats.csv") 
# open(filename, "a") do f
#     lens = pls
#     writedlm(f, lens)
#     writedlm(f,exs)
# end