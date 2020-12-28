using Flux, OhMyREPL, CuArrays, CUDAnative, JLD, DelimitedFiles, LinearAlgebra
# using Plots 
include("utils_heur.jl")
include("solver_sokoban.jl")
include("heur_att_load_params.jl")
include("MAC_network_self_stopping3.jl")
# pyplot()

# Maze - expansion comparison 
# data = load("planning_experiments_mac/data_sokoban_plan.jld")["data"]
# data = load("planning_experiments_mac/data16_sokoban_plan.jld")["data"]
data = load("planning_experiments_mac/data_sokoban_test_deepmind.jld")["data"]

# # maze = data[1]
# maze = data[:,:,1]
# expansion_times = []

# ns = get_neighbors(maze)
# for i in 1:5   
#     t0 = time()
#     ns = get_neighbors(maze)
#     t1 = time()
#     push!(expansion_times, t1-t0)
# end
# sum(expansion_times)/5

# expansion_times_nn = []
# resnet(model, composer, x) = composer(cat(x, model(x), dims = 3))
# model, composer = load_nn()
# goal_coords = CartesianIndices(maze)[maze .== 3]

# ns = get_neighbors_nn(maze, model, composer, goal_coords)
# for i in 1:5 
#     t0 = time()
#     ns = get_neighbors_nn(maze, model, composer, goal_coords)
#     t1 = time()
#     push!(expansion_times_nn, t1-t0)
# end
# sum(expansion_times_nn)/5

# Maze - heuristic time comparison 
# data = load("planning_experiments_mac/data_sokoban_plan.jld")["data"]
# data = load("planning_experiments_mac/data16_sokoban_plan.jld")["data"]

# map = data[1]
map = data[:,:,1]

# heur_network_att = load_att_model_from_params("heuristic_networks/sokoban_model_params/0-10-5-50")

# exp_folder_q = "mac_param_experiments12_sokoban/exp_5_32_0_0_q"
# @load string(exp_folder_q, "/mac_q.bson") m 
# model_q = m |> gpu
# heur_network_cnn = [model_q, false]

exp_folder = "mac_param_experiments12_sokoban/exp_5_32_100_0_0_0_r"
mac = load_mac_network(exp_folder)
iters = parse(Int, split(exp_folder,"_")[6])
model = Chain(
    x -> run_MAC(mac, iters, x)
) |> gpu
heur_network_mac = [model, false]

agent_coords = CartesianIndices(map)[map .== 3][1]
box_coords = CartesianIndices(map)[map .== 5]
start_state = State(agent_coords, box_coords, map)
goal_coords = CartesianIndices(map)[map .== 4]
heur_times = []
t0 = time()
compute_heur("heur", start_state, heur_network_mac, goal_coords, time(), 600)
println(time() - t0)
# compute_heur("neural", start_state, heur_network_mac, goal_coords, time(), 600)

for i in 1:5
    t0 = time()
    compute_heur("heur", start_state, heur_network_mac, goal_coords, time(), 600)
    # compute_heur("neural", start_state, heur_network_mac, goal_coords, time(), 600)

    t1 = time()
    push!(heur_times, t1-t0)
end
println(sum(heur_times)/5)



