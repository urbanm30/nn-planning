using Flux, OhMyREPL, CSV, MLDataUtils, Statistics, LinearAlgebra, CuArrays, BSON, JLD, CUDAnative
using Flux: @epochs, throttle
using BSON: @save
include("utils_heur.jl")
CuArrays.allowscalar(false)

k = parse.(Int,ARGS[1])
dropout = parse.(Int,ARGS[2])
no_convs = parse.(Int,ARGS[3])
loss_type = parse.(Int,ARGS[4])
drop_val = parse.(Float32,ARGS[5])
dev_no = parse.(Int,ARGS[6])

CUDAnative.device!(dev_no)

imgs, test_imgs, labels, test_labels = load_prim_8_goal_step(8) |> gpu
train_batches = RandomBatches((imgs,labels), 10000, 10)

model_name = string("step_k", k, "_dropout", dropout, "_no_convs",no_convs, "_loss_type", loss_type, "_drop_val", drop_val)
mkdir(string("step_experiments/",model_name))

composer = Chain(
    Conv((1,1), 8 => k, swish), 
    Conv((1,1), k => 4, identity),
) |> gpu

if dropout == 0
    if no_convs == 2
        model = Chain(
            Conv((3,3), 4 => k, swish, pad = (1,1)),
            Conv((3,3), k => k, swish, pad = (1,1)),
            Conv((3,3), k => 4, identity, pad = (1,1)),
        ) |> gpu
    elseif no_convs == 4
        model = Chain(
            Conv((3,3), 4 => k, swish, pad = (1,1)),
            Conv((3,3), k => k, swish, pad = (1,1)),
            Conv((3,3), k => k, swish, pad = (1,1)),
            Conv((3,3), k => k, swish, pad = (1,1)),
            Conv((3,3), k => 4, identity, pad = (1,1)),
        ) |> gpu
    elseif no_convs == 8
        model = Chain(
            Conv((3,3), 4 => k, swish, pad = (1,1)),
            Conv((3,3), k => k, swish, pad = (1,1)),
            Conv((3,3), k => k, swish, pad = (1,1)),
            Conv((3,3), k => k, swish, pad = (1,1)),
            Conv((3,3), k => k, swish, pad = (1,1)),
            Conv((3,3), k => k, swish, pad = (1,1)),
            Conv((3,3), k => k, swish, pad = (1,1)),
            Conv((3,3), k => k, swish, pad = (1,1)),
            Conv((3,3), k => 4, identity, pad = (1,1)),
        ) |> gpu
    end
elseif dropout == 1
    if no_convs == 2
        model = Chain(
            Conv((3,3), 4 => k, swish, pad = (1,1)),
            Dropout(drop_val),
            Conv((3,3), k => k, swish, pad = (1,1)),
            Dropout(drop_val),
            Conv((3,3), k => 4, identity, pad = (1,1)),
        ) |> gpu
    elseif no_convs == 4
        model = Chain(
            Conv((3,3), 4 => k, swish, pad = (1,1)),
            Dropout(drop_val),
            Conv((3,3), k => k, swish, pad = (1,1)),
            Dropout(drop_val),
            Conv((3,3), k => k, swish, pad = (1,1)),
            Dropout(drop_val),
            Conv((3,3), k => k, swish, pad = (1,1)),
            Dropout(drop_val),
            Conv((3,3), k => 4, identity, pad = (1,1)),
        ) |> gpu
    elseif no_convs == 8
        model = Chain(
            Conv((3,3), 4 => k, swish, pad = (1,1)),
            Dropout(drop_val),
            Conv((3,3), k => k, swish, pad = (1,1)),
            Dropout(drop_val),
            Conv((3,3), k => k, swish, pad = (1,1)),
            Dropout(drop_val),
            Conv((3,3), k => k, swish, pad = (1,1)),
            Dropout(drop_val),
            Conv((3,3), k => k, swish, pad = (1,1)),
            Dropout(drop_val),
            Conv((3,3), k => k, swish, pad = (1,1)),
            Dropout(drop_val),
            Conv((3,3), k => k, swish, pad = (1,1)),
            Dropout(drop_val),
            Conv((3,3), k => k, swish, pad = (1,1)),
            Dropout(drop_val),
            Conv((3,3), k => 4, identity, pad = (1,1)),
        ) |> gpu
    end
end

resnet(model, composer, x) = composer(cat(x, model(x), dims = 3))
tensor2mat(x) = reshape(permutedims(x, (3,1,2,4)), size(x,3), :)

if loss_type == 1    
    loss(x,y) = Flux.logitcrossentropy(tensor2mat(resnet(model, composer, x)),tensor2mat(y))
elseif loss_type == 2
    loss(x,y) = Flux.mse(sf(resnet(model, composer, x), 3),y)
end

opt = ADAM()
ps = Flux.params(model)
push!(ps, Flux.params(composer)...)

tb = getobs(train_batches)
# @epochs epochs Flux.train!(loss, ps, tb, opt)
for i in 1:20
    @epochs 50 Flux.train!(loss, ps, tb, opt)
    CuArrays.allowscalar(true)

    rand_ixs = rand(1:1:size(imgs,4), 10000)
    rand_imgs = imgs[:,:,:,rand_ixs]
    rand_labels = labels[:,:,:,rand_ixs]
    o = sf(resnet(model, composer, rand_imgs), 3)
    max_wall_diff = check_walls(o, rand_labels)
    min_corr, max_wr = check_feasible_steps(rand_imgs,o,rand_labels)

    o = sf(resnet(model, composer, test_imgs), 3)
    max_wall_diff2 = check_walls(o, test_labels)
    min_corr2, max_wr2 = check_feasible_steps(cu(test_imgs),o,cu(test_labels))

    curr_epoch = i * 50
    open(string("step_experiments/",model_name,"/stats.txt"), "a") do f 
        write(f, "Epochs: ", curr_epoch, "\n")
        write(f, string("Train max_wall_diff = ", max_wall_diff, "\n"))
        write(f, string("Train min_correct_step = ", min_corr, "\n"))
        write(f, string("Train max_wrong_step = ", max_wr, "\n"))
        write(f, string("Test max_wall_diff = ", max_wall_diff2, "\n"))
        write(f, string("Test min_correct_step = ", min_corr2, "\n"))
        write(f, string("Test max_wrong_step = ", max_wr2, "\n"))
        write(f, "\n--------------------------------------------\n")
    end
    CuArrays.allowscalar(false)
end

Flux.testmode!(model, true)

# CuArrays.allowscalar(true)

# rand_ixs = rand(1:1:size(imgs,4), 10000)
# rand_imgs = imgs[:,:,:,rand_ixs]
# rand_labels = labels[:,:,:,rand_ixs]
# o = sf(resnet(model, composer, rand_imgs), 3)
# max_wall_diff = check_walls(o, rand_labels)
# min_corr, max_wr = check_feasible_steps(rand_imgs,o,rand_labels)

# o = sf(resnet(model, composer, test_imgs), 3)
# max_wall_diff2 = check_walls(o, test_labels)
# min_corr2, max_wr2 = check_feasible_steps(cu(test_imgs),o,cu(test_labels))    

# open(string("step_experiments/",model_name,"/stats.txt"), "a") do f 
#     write(f, string("Train max_wall_diff = ", max_wall_diff, "\n"))
#     write(f, string("Train min_correct_step = ", min_corr, "\n"))
#     write(f, string("Train max_wrong_step = ", max_wr, "\n"))
#     write(f, string("Test max_wall_diff = ", max_wall_diff2, "\n"))
#     write(f, string("Test min_correct_step = ", min_corr2, "\n"))
#     write(f, string("Test max_wrong_step = ", max_wr2, "\n"))
# end 

# Save the model 
m = cpu(model)
c = cpu(composer)
@save string("step_experiments/", model_name, "/model.bson") m
@save string("step_experiments/", model_name, "/composer.bson") c

