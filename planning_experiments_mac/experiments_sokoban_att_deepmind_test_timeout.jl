using Flux, OhMyREPL, CuArrays, CUDAnative, JLD, DelimitedFiles, BSON
cd("planning_experiments_mac")
include("../utils_heur.jl")
include("../solver_sokoban.jl")
include("../heur_att_load_params.jl")

# Arguments: i (number of the instance), search_type (gbfs, bfs)
# args = [1,16,"gbfs"]
i = parse.(Int, ARGS[1])
search_type = ARGS[2]

println("i = ", i, " ", search_type)
data = load("data_sokoban_test_deepmind.jld")["data"]
data_size = 10

ixs = load("sokoban_deepmind_ixs.jld")["data"]
data = data[:,:,ixs]

cd("..")

heur_network = load_att_model_from_params("heuristic_networks/sokoban_model_params/0-10-5-50")

max_time = 600

file_pls = string("planning_experiments_mac/att_tmps/tmp_att_",data_size,"_sokoban_deepmind_nn_",search_type,"_pls.jld")
file_exs = string("planning_experiments_mac/att_tmps/tmp_att_",data_size,"_sokoban_deepmind_nn_",search_type,"_exs.jld")

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

# search_type = "bfs"
# data_size = 10

# file_pls = string("planning_experiments_mac/att_tmps/tmp_att_",data_size,"_sokoban_deepmind_nn_",search_type,"_pls.jld")
# file_exs = string("planning_experiments_mac/att_tmps/tmp_att_",data_size,"_sokoban_deepmind_nn_",search_type,"_exs.jld")

# pls = load(file_pls)["data"]
# exs = load(file_exs)["data"]
# filename = string("planning_experiments_mac/sokoban_att_deepmind_",data_size,"_",search_type,"_rand50_stats.csv") 
# open(filename, "a") do f
#     lens = pls
#     writedlm(f, lens)
#     writedlm(f,exs)
# end