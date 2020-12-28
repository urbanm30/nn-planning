using Flux, OhMyREPL, CuArrays, CUDAnative, JLD, DelimitedFiles, BSON
cd("planning_experiments_mac")
include("../utils_heur.jl")
include("../solver_ma_maze.jl")

# Arguments: i (number of the instance), heur (heuristic type = none, euclidean, hff, lmcut), data_size (32,64), search_type (gbfs, bfs)
i = parse.(Int, ARGS[1])
heur = ARGS[2]
data_size = parse.(Int, ARGS[3])
search_type = ARGS[4]
println(heur, " i = ", i, " ", data_size, " ", search_type)

if data_size == 16
    data = load("data16_ma_maze.jld")["data"]
elseif data_size == 32
    data = load("data32_ma_maze.jld")["data"]
elseif data_size == 64 
    data = load("data64_ma_maze.jld")["data"]
else
    println("Invalid data size selected -> exiting")
    exit()
end
cd("..")

max_time = 600

file_pls = string("planning_experiments_mac/heur_tmps/tmp",data_size,"_ma_maze_",heur,"_",search_type,"_pls_true.jld")
file_exs = string("planning_experiments_mac/heur_tmps/tmp",data_size,"_ma_maze_",heur,"_",search_type,"_exs_true.jld")

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
    path, path_len, expanded_states = gbfs_timed(maze, true, heur, [], max_time)
elseif search_type == "bfs" 
    path, path_len, expanded_states = bfs_timed(maze, true, heur, [], max_time)
end
pls[i] = path_len
exs[i] = expanded_states
save(file_pls, "data", pls)
save(file_exs, "data", exs)

# Run to output the results to CSV 
# heurs = [
#     "none",
#     "euclidean",
#     "hff",
#     "lmcut"
# ]

# # Set search type and data_size of the experiments 
# search_type = "bfs"
# data_size = 64

# for heur in heurs
#     file_pls = string("planning_experiments_mac/heur_tmps/tmp",data_size,"_ma_maze_",heur,"_",search_type,"_pls_true.jld")
#     file_exs = string("planning_experiments_mac/heur_tmps/tmp",data_size,"_ma_maze_",heur,"_",search_type,"_exs_true.jld")
#     pls = load(file_pls)["data"]
#     exs = load(file_exs)["data"]
#     filename = string("planning_experiments_mac/ma_maze",data_size,"_",search_type,"_heur_true_stats.csv") 
#     open(filename, "a") do f
#         lens = pls
#         exs = exs
#         writedlm(f, lens)
#         writedlm(f,exs)
#     end
# end
