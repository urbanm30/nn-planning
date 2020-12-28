using Flux, OhMyREPL, CuArrays, CUDAnative, JLD, DelimitedFiles
# using Plots 
include("utils_heur.jl")
include("solver_ma_maze.jl")
include("heur_att_load_params.jl")
include("MAC_network_self_stopping3.jl")
# pyplot()

# Maze - expansion comparison 
# data = load("planning_experiments_mac/data_ma_maze_plan.jld")["data"]
# data = load("planning_experiments_mac/data16_ma_maze.jld")["data"]
# data = load("planning_experiments_mac/data32_ma_maze.jld")["data"]
# data = load("planning_experiments_mac/data64_ma_maze.jld")["data"]

# maze = data[:,:,1]
maze = onehot2img(data[:,:,:,1:1])[:,:,1]  
expansion_times = []

ns = get_neighbors(maze)
for i in 1:5   
    t0 = time()
    ns = get_neighbors(maze)
    t1 = time()
    push!(expansion_times, t1-t0)
end
sum(expansion_times)/5

expansion_times_nn = []
resnet(model, composer, x) = composer(cat(x, model(x), dims = 3))
model, composer = load_nn()
goal_coords = CartesianIndices(maze)[maze .== 3]

ns = get_neighbors_nn(maze, model, composer, goal_coords)
for i in 1:5 
    t0 = time()
    ns = get_neighbors_nn(maze, model, composer, goal_coords)
    t1 = time()
    push!(expansion_times_nn, t1-t0)
end
sum(expansion_times_nn)/5

# Maze - heuristic time comparison 
# data = load("planning_experiments_mac/data_ma_maze_plan.jld")["data"]
# data = load("planning_experiments_mac/data16_ma_maze.jld")["data"]
data = load("planning_experiments_mac/data32_ma_maze.jld")["data"]
# data = load("planning_experiments_mac/data64_ma_maze.jld")["data"]

# heur_network_att = load_att_model_from_params("heuristic_networks/ma_maze_model_params/0-10-1-100")

# exp_folder_q = "mac_param_experiments11_ma_maze/exp_5_32_1_0_q"
# @load string(exp_folder_q, "/mac_q.bson") m 
# model_q = m |> gpu
# heur_network_cnn = [model_q]

# exp_folder = "mac_param_experiments11_ma_maze/exp_5_32_100_0_0_0_r"
# mac = load_mac_network(exp_folder)
# iters = parse(Int, split(exp_folder,"_")[6])
# model = Chain(
#     x -> run_MAC(mac, iters, x)
# ) |> gpu
# heur_network_mac = [model]

# maze = data[:,:,1]
maze = onehot2img(data[:,:,:,3:3])[:,:,1] 

goal_coords = CartesianIndices(maze)[maze .== 3]
heur_times = []
t0 = time()
compute_heur("lmcut", maze, [], goal_coords)
# compute_heur("neural", maze, heur_network_cnn, goal_coords)
println(time() - t0)

for i in 1:5
    t0 = time()
    # compute_heur("neural", maze, heur_network_cnn, goal_coords)
    compute_heur("lmcut", maze, [], goal_coords)
    t1 = time()
    push!(heur_times, t1-t0)
end 
sum(heur_times)/5



