using Flux, OhMyREPL, CuArrays, CUDAnative, DelimitedFiles, BSON, JLD 
using BSON: @load, @save
include("utils_heur.jl")
# include("solver_maze.jl")
# include("solver_mg_maze.jl")
include("solver_ma_maze.jl")

resnet(model, composer, x) = composer(cat(x, model(x), dims = 3))
@load "resnet_models/expansion_network_maze_model_params.bson" m_params
@load "resnet_models/expansion_network_maze_composer_params.bson" c_params
k = 64

model = Chain(
    Conv((3,3), 3 => k, relu, pad = (1,1)),
    Conv((3,3), k => k, relu, pad = (1,1)),
    Conv((3,3), k => 3, identity, pad = (1,1)),
) |> gpu    

composer = Chain(
    Conv((1,1), 6 => k, swish), 
    Conv((1,1), k => 3, identity),
) |> gpu

Flux.loadparams!(model, m_params)
Flux.loadparams!(composer, c_params)

data = load("planning_experiments_mac/data16_maze_plan.jld")["data"]
# data = load("planning_experiments_mac/data32_maze_plan.jld")["data"]
# data = load("planning_experiments_mac/data64_maze_plan.jld")["data"]

# data = load("planning_experiments_mac/data16_mg_maze.jld")["data"]
# data = load("planning_experiments_mac/data32_mg_maze.jld")["data"]
# data = load("planning_experiments_mac/data64_mg_maze.jld")["data"]

# data = load("planning_experiments_mac/data16_ma_maze.jld")["data"]
# data = load("planning_experiments_mac/data32_ma_maze.jld")["data"]
# data = load("planning_experiments_mac/data64_ma_maze.jld")["data"]

total_min_corr = Inf
total_max_wr = 0 
for i in 1:size(data,3)
    x = img2onehotnogoal(onehot2img(data[:,:,:,i:i])) |> gpu
    mx = softmax(resnet(model, composer, x),dims=3)
    labels = zeros(1,1,1,1)
    min_corr, max_wr, min_ix, max_ix = check_feasible_steps(x,mx,labels)
    if min_corr < total_min_corr
        global total_min_corr = min_corr
    end

    if max_wr > total_max_wr
        global total_max_wr = max_wr
    end
end
println("Min_corr = ", total_min_corr, ", max_wr = ", total_max_wr)






