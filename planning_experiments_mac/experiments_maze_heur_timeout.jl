using Flux, OhMyREPL, CuArrays, CUDAnative, JLD, DelimitedFiles, BSON
cd("planning_experiments_mac")
include("../utils_heur.jl")
include("../solver_maze_timed.jl")

# Arguments: i (number of the instance), heur (heuristic type = none, euclidean, hff, lmcut), data_size (32,64), search_type (gbfs, bfs)
i = parse.(Int, ARGS[1])
heur = ARGS[2]
data_size = parse.(Int, ARGS[3])
search_type = ARGS[4]
println(heur, " i = ", i, " ", data_size, " ", search_type)

# args = [1, "hff", 32, "gbfs"]

if data_size == 32 
    data = load("data32_maze_plan.jld")["data"]
elseif data_size == 64 
    data = load("data64_maze_plan.jld")["data"]
elseif data_size == 8
    data = load("data_maze_plan.jld")["data"]
elseif data_size == 16
    data = load("data16_maze_plan.jld")["data"]
else 
    println("Invalid data size selected -> exiting")
    exit()
end
cd("..")

max_time = 600

file_pls = string("planning_experiments_mac/heur_tmps/tmp",data_size,"_maze_",heur,"_",search_type,"_true_pls2.jld")
file_exs = string("planning_experiments_mac/heur_tmps/tmp",data_size,"_maze_",heur,"_",search_type,"_true_exs2.jld")

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
#     # "hff",
#     "lmcut"
# ]

# # Set search type and data_size of the experiments 
# search_type = "bfs"
# data_size = 64

# for heur in heurs
#     file_pls = string("planning_experiments_mac/heur_tmps/tmp",data_size,"_maze_",heur,"_",search_type,"_true_pls2.jld")
#     file_exs = string("planning_experiments_mac/heur_tmps/tmp",data_size,"_maze_",heur,"_",search_type,"_true_exs2.jld")
#     pls = load(file_pls)["data"]
#     exs = load(file_exs)["data"]
#     filename = string("planning_experiments_mac/maze",data_size,"_",search_type,"_heur_", heur, "_true_stats2.csv") 
#     open(filename, "a") do f
#         lens = pls
#         writedlm(f, lens)
#         writedlm(f,exs)
#     end
# end