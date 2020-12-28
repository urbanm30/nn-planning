using JLD
cd("planning_experiments_mac")
include("../utils_heur.jl")
include("../solver_ma_maze.jl")
include("../multimaze_domain/multimazegen.jl")
include("../mazegen_prim.jl")

cd("..")

data64 = zeros(64,64,4,50)
global no_mzs = 0 
while no_mzs < 50 
    maze = generate_multiagent_mazes(64,1,2)
    m = onehot2img(maze[:,:,:,1:1])
    path, path_len, expanded_states = bfs_timed(m, false, "euclidean", [], 600)
    if path_len != Inf
        global no_mzs += 1
        data64[:,:,:,no_mzs] = maze
        println("No mazes: ", no_mzs)
    end
end

save("planning_experiments_mac/data64_ma_maze.jld", "data", data64)