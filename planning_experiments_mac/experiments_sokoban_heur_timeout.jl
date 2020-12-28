using Flux, OhMyREPL, CuArrays, CUDAnative, JLD, DelimitedFiles, BSON, Plots 
pyplot()
cd("planning_experiments_mac")
include("../utils_heur.jl")
include("../solver_sokoban.jl")

# Arguments: i (number of the instance), search_type (gbfs, bfs), heur_type
# args = [1,16,"gbfs"]
i = parse.(Int, ARGS[1])
search_type = ARGS[2]
heur = ARGS[3]
data_size = parse.(Int, ARGS[4])

# i = 1
# search_type = "gbfs"
# heur = "hff"
# data_size = 8

println("i = ", i, " ", search_type)
if data_size == 8
    data = load("data_sokoban_plan.jld")["data"]
elseif data_size == 16  
    data = load("data16_sokoban_plan.jld")["data"]
elseif data_size == 10
    data = load("data_sokoban_test_deepmind.jld")["data"]
end

cd("..")

max_time = 600

file_pls = string("planning_experiments_mac/sokoban_heur_rerun_tmps/tmp_",data_size,"_sokoban_", heur, "_rerun_",search_type,"_pls.jld")
file_exs = string("planning_experiments_mac/sokoban_heur_rerun_tmps/tmp_",data_size,"_sokoban_", heur, "_rerun_",search_type,"_exs.jld")

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

if data_size == 8 
    maze = data[i]
else data_size == 16 || data_size == 10
    maze = data[:,:,i]
end

pls[i] = -1
exs[i] = -1
save(file_pls, "data", pls)
save(file_exs, "data", exs)

if search_type == "gbfs"
    path, path_len, expanded_states = gbfs_timed(maze, false, heur, [], max_time)
elseif search_type == "bfs" 
    path, path_len, expanded_states = bfs_timed(maze, false, heur, [], max_time)
end
pls[i] = path_len
exs[i] = expanded_states
save(file_pls, "data", pls)
save(file_exs, "data", exs)

# heurs = [
#     "none",
#     "euclidean",
#     "hff",
#     "lmcut"
# ]

# search_type = "bfs"
# data_size = 10

# for heur in heurs
#     file_pls = string("planning_experiments_mac/sokoban_heur_rerun_tmps/tmp_",data_size,"_sokoban_", heur, "_rerun_",search_type,"_pls.jld")
#     file_exs = string("planning_experiments_mac/sokoban_heur_rerun_tmps/tmp_",data_size,"_sokoban_", heur, "_rerun_",search_type,"_exs.jld")

#     pls = load(file_pls)["data"]
#     exs = load(file_exs)["data"]
#     filename = string("planning_experiments_mac/sokoban_heur_rerun_",data_size,"_",search_type,".csv") 
#     open(filename, "a") do f
#         lens = pls
#         writedlm(f, lens)
#         writedlm(f,exs)
#     end
# end