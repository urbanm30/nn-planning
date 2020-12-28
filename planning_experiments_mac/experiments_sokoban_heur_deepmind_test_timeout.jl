using Flux, OhMyREPL, CuArrays, CUDAnative, JLD, DelimitedFiles, BSON
cd("planning_experiments_mac")
include("../utils_heur.jl")
include("../solver_sokoban.jl")
include("../heur_att_load_params.jl")

# Arguments: i (number of the instance), search_type (gbfs, bfs), heur_type
# args = [1,16,"gbfs"]
i = parse.(Int, ARGS[1])
search_type = ARGS[2]
heur = ARGS[3]

println("i = ", i, " ", search_type)
data = load("data_sokoban_test_deepmind.jld")["data"]
data_size = 10

ixs = load("sokoban_deepmind_ixs.jld")["data"]
data = data[:,:,ixs]

cd("..")

max_time = 600

file_pls = string("planning_experiments_mac/heur_tmps/tmp_",data_size,"_sokoban_deepmind_", heur, "_",search_type,"_pls.jld")
file_exs = string("planning_experiments_mac/heur_tmps/tmp_",data_size,"_sokoban_deepmind_", heur, "_",search_type,"_exs.jld")

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
    path, path_len, expanded_states = gbfs_timed(maze, false, heur, [], max_time)
elseif search_type == "bfs" 
    path, path_len, expanded_states = bfs_timed(maze, false, heur, [], max_time)
end
pls[i] = path_len
exs[i] = expanded_states
save(file_pls, "data", pls)
save(file_exs, "data", exs)

heurs = [
    "none",
    # "euclidean",
    # "hff",
    # "lmcut"
]

search_type = "gbfs"
data_size = 10

for heur in heurs
    file_pls = string("planning_experiments_mac/heur_tmps/tmp_",data_size,"_sokoban_deepmind_", heur, "_",search_type,"_pls.jld")
    file_exs = string("planning_experiments_mac/heur_tmps/tmp_",data_size,"_sokoban_deepmind_", heur, "_",search_type,"_exs.jld")

    pls = load(file_pls)["data"]
    exs = load(file_exs)["data"]
    filename = string("planning_experiments_mac/sokoban_heur_deepmind_",data_size,"_",search_type,"_rand50none_stats.csv") 
    open(filename, "a") do f
        lens = pls
        writedlm(f, lens)
        writedlm(f,exs)
    end
end