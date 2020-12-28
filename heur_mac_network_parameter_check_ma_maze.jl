using OhMyREPL, Flux, MLDataUtils, CSV, Statistics, LinearAlgebra, DataStructures, Distances, CuArrays, BSON, StatsBase, CUDAnative
using Flux: @epochs, throttle
using BSON: @load, @save 
# import Base.minimum
# import Base.maximum

include("utils_heur.jl")
include("solver_maze.jl")
include("plots_maybe.jl")
include("mazegen_prim.jl")    
include("network_stats_demo.jl")
include("MAC_network_self_stopping3.jl")

# params: mzs_in_block, h_filters, iters, add_coords, self_attention, gating, device_no 
args = parse.(Int, ARGS)
# args = [5, 16, 1, 0, 0, 0, 0]
dev_no = args[7]
CUDAnative.device!(dev_no) 

coords = false
if args[4] == 1
    coords = true
end

imgs = load("heur_data_new/ma_maze.jld")["data"] |> gpu
labels = load("heur_data_new/ma_maze_labels.jld")["data"] |> gpu
test_imgs = load("heur_data_new/ma_maze_test.jld")["data"] |> gpu
test_labels = load("heur_data_new/ma_maze_test_labels.jld")["data"] |> gpu

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

iters = args[3]
mac = create_MAC_network(size(batches[1][1],3), args[2], args[5], args[6])
ps = params(mac)
model = Chain(
    x -> run_MAC(mac, iters, x)
) |> gpu

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

# ixs = rand(1:size(labels,1),1000)
# for i in 1:10
    @epochs 10 Flux.train!(loss, ps, batches, opt)

    # if size(batches[1][1],3) == 4
    #     @show loss(cu(getobs(imgs[:,:,:,ixs])), cu(getobs(labels[ixs])))
    #     @show loss(cu(getobs(test_imgs)), cu(getobs(test_labels)))
    # elseif size(batches[1][1],3) == 6
    #     @show loss(cu(getobs(add_coord_filters(imgs[:,:,:,ixs]))), cu(getobs(labels[ixs])))
    #     @show loss(cu(getobs(add_coord_filters(test_imgs))), cu(getobs(test_labels)))
    # end
# end

CuArrays.allowscalar(true)

# if size(batches[1][1],3) == 4
#     train_loss = loss(cu(getobs(imgs[:,:,:,ixs])), cu(getobs(labels[ixs])))
#     test_loss = loss(cu(getobs(test_imgs)), cu(getobs(test_labels)))
# elseif size(batches[1][1],3) == 6
#     train_loss = loss(cu(getobs(add_coord_filters(imgs[:,:,:,ixs]))), cu(getobs(labels[ixs])))
#     test_loss = loss(cu(getobs(add_coord_filters(test_imgs))), cu(getobs(test_labels)))
# end

folder_name = string("exp_",args[1],"_",args[2],"_",args[3],"_",args[4],"_",args[5],"_",args[6],"_r")
path = string("mac_param_experiments11_ma_maze/",folder_name)
mkdir(path)

# open(string(path,"/stats.txt"), "w") do f 
#     write(f, string("Train loss = ", train_loss, "\n"))
#     write(f, string("Test loss = ", test_loss, "\n"))
# end 

# ix =  [1551, 2647, 1524, 2614,  323]
# for i in 1:length(ix)
#     # r = rand(1:size(imgs)[4])
#     maze = Float32.(onehot2img(cpu(imgs[:,:,1:4,ix[i]:ix[i]])))
#     save_monotonic_mac(maze, model, coords, string(path,string("/img_", i, ".png")))
# end

# for i in 1:length(ix)
#     r = i + 5
#     maze = Float32.(onehot2img(cpu(test_imgs[:,:,1:4,r:r])))
#     save_monotonic_mac(maze, model, coords, string(path,string("/tst_img_", i, ".png")))
# end

# Save model
save_mac_network(path, mac)