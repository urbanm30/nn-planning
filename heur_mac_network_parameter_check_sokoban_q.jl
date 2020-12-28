using OhMyREPL, MLDataUtils, CSV, Statistics, LinearAlgebra, DataStructures, Distances, BSON, StatsBase, Flux, CUDAnative, CuArrays
using Flux: @epochs, throttle
using BSON: @load, @save 

include("utils_heur.jl")
include("solver_maze.jl")
include("plots_maybe.jl")
include("mazegen_prim.jl")    
include("network_stats_demo.jl")
include("ugly_mazes.jl")

# params: mzs_in_block, h_filters, add_coords, device_no 
args = parse.(Int, ARGS)
# args = [5, 16, 0, 0] 
dev_no = args[4]
CUDAnative.device!(dev_no) 

coords = false
if args[3] == 1
    coords = true
end

imgs = load("sokoban_domain/aug_trn_onehot_data.jld")["data"] |> gpu
labels = load("sokoban_domain/aug_trn_labels.jld")["data"] |> gpu
test_imgs = load("sokoban_domain/tst_onehot_data.jld")["data"] |> gpu
test_labels = load("sokoban_domain/tst_labels.jld")["data"] |> gpu 

no_batches = 10000
batch_size = 50
no_mzs_in_block = args[1]

if coords
    batches = make_batches_with_coords(imgs, labels, no_batches, batch_size, no_mzs_in_block)
else 
    batches = make_batches(imgs, labels, no_batches, batch_size, no_mzs_in_block)
end

batch_size_identity = zeros(no_mzs_in_block, no_mzs_in_block) + I |> gpu
CuArrays.allowscalar(false) 

maze_filters = size(batches[1][1],3)
h = args[2]

model = Chain(
    Conv((1,1), maze_filters=>32, swish),
    Conv((3,3), 32=>32, pad=(1,1), swish),
    Conv((3,3), 32=>48, pad=(1,1), swish),
    Conv((3,3), 48 => 96, pad=(1,1), swish),
    x -> sum(x, dims=[1,2]),    
    x -> reshape(x, :, size(x)[4]),
    Dense(96, h),
    Dense(h,h,relu),
    Dense(h,h,relu),
    Dense(h,1)
) |> gpu

ps = params(model)

# Meeting loss 
function loss(x,y)
    # println("Compute loss")
    mx = model(x)
    no_cycles = Int64.(batch_size/no_mzs_in_block)
    sums = 0
    for i in 1:no_cycles
        ixs = [(i - 1) * no_mzs_in_block + 1:1:i * no_mzs_in_block;]
        data_diffs = mx[ixs] .- transpose(mx[ixs]) #.- 1 .+ batch_size_identity 
        labels_diffs = y[ixs] .- transpose(y[ixs])
        s = sum(max.(0, .- data_diffs .* sign.(labels_diffs) .+ 1) - batch_size_identity)
        sums += s
    end
    # println("Loss = ", sums)
    return sums
end

opt = ADAM()

# for i in 1:10
    @epochs 10 Flux.train!(loss, ps, batches, opt) 

# if size(batches[1][1],3) == 4
#     @show loss(cu(getobs(imgs[:,:,:,ixs])), cu(getobs(labels[ixs])))
#     @show loss(cu(getobs(test_imgs)), cu(getobs(test_labels)))
# elseif size(batches[1][1],3) == 6
#     @show loss(cu(getobs(add_coord_filters(imgs[:,:,:,ixs]))), cu(getobs(labels[ixs])))
#     @show loss(cu(getobs(add_coord_filters(test_imgs))), cu(getobs(test_labels)))no ne
# end
# end

CuArrays.allowscalar(true)

# ixs = rand(1:size(labels,1),100)
# ixs_tst = rand(1:size(test_labels,1),100)

# if size(batches[1][1],3) == 4
#     train_loss = loss(cu(getobs(imgs[:,:,:,ixs])), cu(getobs(labels[ixs])))
#     test_loss = loss(cu(getobs(test_imgs[:,:,:,ixs_tst])), cu(getobs(test_labels[ixs_tst])))
# elseif size(batches[1][1],3) == 6
#     train_loss = loss(cu(getobs(add_coord_filters(imgs[:,:,:,ixs]))), cu(getobs(labels[ixs])))
#     test_loss = loss(cu(getobs(add_coord_filters(test_imgs[:,:,:,ixs_tst]))), cu(getobs(test_labels[ixs_tst])))
# end

folder_name = string("exp_",args[1],"_",args[2],"_",args[3],"_",args[4],"_q")
path = string("mac_param_experiments12_sokoban/",folder_name)
mkdir(path)

# open(string(path,"/stats.txt"), "w") do f 
#     write(f, string("Train loss = ", train_loss, "\n"))
#     write(f, string("Test loss = ", test_loss, "\n"))
# end 

# ix =  [1551, 2647, 1524, 2614,  323]
# # ix16 = [523782, 331190, 759799, 572610, 170119]

# for i in 1:length(ix)
#     # r = rand(1:size(imgs)[4])
#     maze = Float32.(onehot2img(cpu(imgs[:,:,1:4,ix[i]:ix[i]])))
#     save_monotonic_mac(maze, model, coords, string(path,string("/img_", i, ".png")))
#     # Plots.display(show_monotonic_mac(maze, model, coords))
#     # sleep(0.1)
# end

# for i in 1:length(ix)
#     r = rand(1:size(test_imgs)[4])
#     maze = Float32.(onehot2img(cpu(test_imgs[:,:,1:4,r:r])))
#     save_monotonic_mac(maze, model, coords, string(path,string("/tst_img_", i, ".png")))
#     # Plots.display(show_monotonic_mac(maze, model, coords))
#     # sleep(0.1)
# end

# for i in 1:length(ix16)
#     maze = Float32.(onehot2img(imgs16[:,:,1:4,ix16[i]:ix16[i]]))
#     save_monotonic_mac(maze, model, coords, string(path,string("/img16_", i, ".png")))
#     # Plots.display(show_monotonic_mac(maze, model, coords))
#     # sleep(0.1)
# end

# imgs32 = load("heur_data_new/imgs32.jld", "data")
# imgs64 = load("heur_data_new/imgs64.jld", "data")

# for i in 1:5
#     maze = imgs32[:,:,i]
#     save_monotonic_mac(maze, model, coords, string(path,string("/img32_", i, ".png")))
#     # Plots.display(show_monotonic_mac(maze, model, coords))
#     # sleep(0.1)
# end

# for i in 1:5
#     maze = imgs64[:,:,i]
#     save_monotonic_mac(maze, model, coords, string(path,string("/img64_", i, ".png")))
#     # Plots.display(show_monotonic_mac(maze, model, coords))
#     # sleep(0.1)
# end

# ugs = get_ugly_mazes()
# for i in 1:length(ugs)
#     maze = Float32.(reshape(ugs[i], size(ugs[i],1), size(ugs[i],2), 1))
#     save_monotonic_mac(maze, model, coords, string(path,string("/img_ug_", iters, "_", i, ".png")))
#     # Plots.display(show_monotonic_mac(maze, model, coords))
#     # sleep(0.1)
# end

# # Save model
# save_mac_network(path, mac)
m = cpu(model)
@save string(path,"/mac_q.bson") m