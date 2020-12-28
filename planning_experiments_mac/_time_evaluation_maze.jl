using Flux, OhMyREPL, CuArrays, CUDAnative, JLD, DelimitedFiles
using Plots 
include("utils_heur.jl")
include("solver_maze_timed.jl")
include("heur_att_load_params.jl")
include("MAC_network_self_stopping3.jl")
pyplot()

# Maze - expansion comparison 
include("solver_maze_timed.jl")
data = load("planning_experiments_mac/data_maze_plan.jld")["data"]
# data = load("planning_experiments_mac/data16_maze_plan.jld")["data"]
# data = load("planning_experiments_mac/data32_maze_plan.jld")["data"]
# data = load("planning_experiments_mac/data64_maze_plan.jld")["data"]

maze = data[:,:,1]
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

ns = get_neighbors_nn(maze, model, composer)
for i in 1:5 
    t0 = time()
    ns = get_neighbors_nn(maze, model, composer)
    t1 = time()
    push!(expansion_times_nn, t1-t0)
end
sum(expansion_times_nn)/5

# Maze - heuristic time comparison 
# data = load("planning_experiments_mac/data_maze_plan.jld")["data"]
# data = load("planning_experiments_mac/data16_maze_plan.jld")["data"]
# data = load("planning_experiments_mac/data32_maze_plan.jld")["data"]
data = load("planning_experiments_mac/data64_maze_plan.jld")["data"]

# heur_network_att = load_att_model_from_params("heuristic_networks/maze_model_params/1-10-1-100")

exp_folder = "mac_param_experiments9/exp_5_32_100_0_0_0_r"
mac = load_mac_network(exp_folder)
iters = parse(Int, split(exp_folder,"_")[6])
model = Chain(
    x -> run_MAC(mac, iters, x)
) |> gpu
heur_network_mac = [model]

# exp_folder_q = "mac_param_experiments9/exp_5_32_0_2_q"
# @load string(exp_folder_q, "/mac_q.bson") m 
# model_q = m |> gpu
# heur_network_cnn = [model_q]

maze = data[:,:,1]
goal_coords = CartesianIndices(maze)[maze .== 3]
heur_times = []
# compute_heur("lmcut", maze, [], goal_coords)
compute_heur("neural", maze, heur_network_mac, goal_coords)

for i in 1:5
    t0 = time()
    compute_heur("neural", maze, heur_network_mac, goal_coords)
    t1 = time()
    push!(heur_times, t1-t0)
end
sum(heur_times)/5



