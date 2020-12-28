using OhMyREPL, Flux, CSV, MLDataUtils, Statistics, LinearAlgebra, DataStructures, Distances, CuArrays, BSON, StatsBase, CUDAnative, CSVFiles, JLD
using Flux: @epochs, throttle
using BSON: @save 
import Base.minimum
import Base.maximum

include("utils_heur.jl")
include("plots_maybe.jl")
include("network_stats_demo.jl")

# ARGS = no_atts in block[10,?], att_blocks[1,5], epochs[?], gpu_no[0,1,2,3]
args = parse.(Int, ARGS)
# args = [5 1 2 2]
dev_no = args[4]
CUDAnative.device!(dev_no) 

imgs = load("sokoban_domain/aug_trn_onehot_data.jld")["data"] |> gpu
labels = load("sokoban_domain/aug_trn_labels.jld")["data"] |> gpu
test_imgs = load("sokoban_domain/tst_onehot_data.jld")["data"] |> gpu
test_labels = load("sokoban_domain/tst_labels.jld")["data"] |> gpu 

no_batches = 10000
batch_size = 120
no_mzs_in_block = 40
batches = make_batches_sokoban(imgs, labels, no_batches, batch_size)
batch_size_identity = CuArray(zeros(no_mzs_in_block, no_mzs_in_block) + I)

CuArrays.allowscalar(false) 

function get_attentions(no_atts, no_att_blocks)
    attention_blocks = []
    ps = []
    for i in 1:no_att_blocks
        att = []
        for j in 1:no_atts
            l = Chain(Conv((3,3), 5 => 1, pad=(1,1), swish), x -> sf(x,[1,2])) |> gpu
            push!(att, l)
            push!(ps, params(l)...)
        end
        push!(attention_blocks, att)
    end
    return attention_blocks, ps
end

apply_atts(x, atts) = cat(cat([x .* a(x) for a in atts]..., dims=3), add_coord_filters(x), dims=3) # vraci no_atts * x_filters + x_filters + 2 filtru -> proste to co chceme 

no_atts = args[1]
att_blocks = args[2]
atts, ps = get_attentions(no_atts, att_blocks)
no_filters = no_atts * 5 + 7

if att_blocks == 1 
    model = Chain(
        x -> apply_atts(x, atts[1]),
        Conv((1,1), no_filters => 24, swish),
        Conv((3,3), 24 => 24, swish),
        Conv((3,3), 24 => 48, pad=(1,1), swish),
        Conv((3,3), 48 => 96, pad=(1,1), swish),
        x -> sum(x, dims=[1,2]),    
        x -> reshape(x, :, size(x)[4]),
        Dense(96, 1),
    ) |> gpu
elseif att_blocks == 5 
    model = Chain(
        x -> apply_atts(x, atts[1]),
        Conv((1,1), no_filters => 5, swish),
        x -> apply_atts(x, atts[2]),
        Conv((3,3), no_filters => 5, swish),
        x -> apply_atts(x, atts[3]),
        Conv((3,3), no_filters => 5, pad=(1,1), swish),
        x -> apply_atts(x, atts[4]), 
        Conv((3,3), no_filters => 5, pad=(1,1), swish),
        x -> apply_atts(x, atts[5]),
        x -> sum(x, dims=[1,2]),    
        x -> reshape(x, :, size(x)[4]),
        Dense(no_filters, 1),
    ) |> gpu
end

push!(ps, params(model)...) 

function loss(x,y)
    mx = model(x)
    sums = []
    no_cycles = Int64.(batch_size/no_mzs_in_block)
    for i in 1:no_cycles
        ixs = [(i - 1) * no_mzs_in_block + 1:1:i * no_mzs_in_block;]
        data_diffs = mx[ixs] .- transpose(mx[ixs])
        labels_diffs = y[ixs] .- transpose(y[ixs])
        s = sum(max.(0, .- data_diffs .* sign.(labels_diffs) .+ 1) - batch_size_identity)
        push!(sums,s)
    end
    return sum(sums)
end

opt = ADAM()
no_epochs = args[3]
@epochs no_epochs Flux.train!(loss, ps, batches, opt)

# ixs = rand(1:size(labels,1),10000)  
# train_loss = loss(cu(getobs(imgs[:,:,:,ixs])), cu(getobs(labels[ixs])))
# test_loss =  loss(getobs(test_imgs), getobs(test_labels))

CuArrays.allowscalar(true)
model_name = string("sokoban_att_noatt", no_atts, "_attblocks", att_blocks, "_epochs", no_epochs)
mkdir(string("sokoban_heur_experiments/",model_name))
# mkdir(string("sokoban_heur_experiments/", model_name, "/plots"))
println("Folders made")

# Save stats into txt 
# open(string("sokoban_heur_experiments/",model_name,"/stats.txt"), "w") do f 
    # write(f, string("Train loss = ", train_loss, "\n"))
    # write(f, string("Test loss = ", test_loss, "\n"))
# end 
# println("Saved stats")

# Save model and attentions 
m = cpu(model)
@save string("sokoban_heur_experiments/",model_name,"/model.bson") m

if att_blocks == 1 && no_atts == 10
    a1 = cpu(atts[1][1])
    a2 = cpu(atts[1][2])
    a3 = cpu(atts[1][3])
    a4 = cpu(atts[1][4])
    a5 = cpu(atts[1][5])
    a6 = cpu(atts[1][6])
    a7 = cpu(atts[1][7])
    a8 = cpu(atts[1][8])
    a9 = cpu(atts[1][9])
    a10 = cpu(atts[1][10])
    @save string("sokoban_heur_experiments/",model_name,"/att1.bson") a1
    @save string("sokoban_heur_experiments/",model_name,"/att2.bson") a2
    @save string("sokoban_heur_experiments/",model_name,"/att3.bson") a3
    @save string("sokoban_heur_experiments/",model_name,"/att4.bson") a4
    @save string("sokoban_heur_experiments/",model_name,"/att5.bson") a5
    @save string("sokoban_heur_experiments/",model_name,"/att6.bson") a6
    @save string("sokoban_heur_experiments/",model_name,"/att7.bson") a7
    @save string("sokoban_heur_experiments/",model_name,"/att8.bson") a8
    @save string("sokoban_heur_experiments/",model_name,"/att9.bson") a9
    @save string("sokoban_heur_experiments/",model_name,"/att10.bson") a10
elseif att_blocks == 1 && no_atts == 5
    a1 = cpu(atts[1][1])
    a2 = cpu(atts[1][2])
    a3 = cpu(atts[1][3])
    a4 = cpu(atts[1][4])
    a5 = cpu(atts[1][5])
    @save string("sokoban_heur_experiments/",model_name,"/att1.bson") a1
    @save string("sokoban_heur_experiments/",model_name,"/att2.bson") a2
    @save string("sokoban_heur_experiments/",model_name,"/att3.bson") a3
    @save string("sokoban_heur_experiments/",model_name,"/att4.bson") a4
    @save string("sokoban_heur_experiments/",model_name,"/att5.bson") a5
elseif att_blocks == 1 && no_atts == 20
    a1 = cpu(atts[1][1])
    a2 = cpu(atts[1][2])
    a3 = cpu(atts[1][3])
    a4 = cpu(atts[1][4])
    a5 = cpu(atts[1][5])
    a6 = cpu(atts[1][6])
    a7 = cpu(atts[1][7])
    a8 = cpu(atts[1][8])
    a9 = cpu(atts[1][9])
    a10 = cpu(atts[1][10])
    a11 = cpu(atts[1][11])
    a12 = cpu(atts[1][12])
    a13 = cpu(atts[1][13])
    a14 = cpu(atts[1][14])
    a15 = cpu(atts[1][15])
    a16 = cpu(atts[1][16])
    a17 = cpu(atts[1][17])
    a18 = cpu(atts[1][18])
    a19 = cpu(atts[1][19])
    a20 = cpu(atts[1][20])
    @save string("sokoban_heur_experiments/",model_name,"/att1.bson") a1
    @save string("sokoban_heur_experiments/",model_name,"/att2.bson") a2
    @save string("sokoban_heur_experiments/",model_name,"/att3.bson") a3
    @save string("sokoban_heur_experiments/",model_name,"/att4.bson") a4
    @save string("sokoban_heur_experiments/",model_name,"/att5.bson") a5
    @save string("sokoban_heur_experiments/",model_name,"/att6.bson") a6
    @save string("sokoban_heur_experiments/",model_name,"/att7.bson") a7
    @save string("sokoban_heur_experiments/",model_name,"/att8.bson") a8
    @save string("sokoban_heur_experiments/",model_name,"/att9.bson") a9
    @save string("sokoban_heur_experiments/",model_name,"/att10.bson") a10
    @save string("sokoban_heur_experiments/",model_name,"/att12.bson") a12
    @save string("sokoban_heur_experiments/",model_name,"/att11.bson") a11
    @save string("sokoban_heur_experiments/",model_name,"/att13.bson") a13
    @save string("sokoban_heur_experiments/",model_name,"/att14.bson") a14
    @save string("sokoban_heur_experiments/",model_name,"/att15.bson") a15
    @save string("sokoban_heur_experiments/",model_name,"/att16.bson") a16
    @save string("sokoban_heur_experiments/",model_name,"/att17.bson") a17
    @save string("sokoban_heur_experiments/",model_name,"/att18.bson") a18
    @save string("sokoban_heur_experiments/",model_name,"/att19.bson") a19
    @save string("sokoban_heur_experiments/",model_name,"/att20.bson") a20
elseif att_blocks == 5 && no_atts == 10
    a1 = cpu(atts[1][1])
    a2 = cpu(atts[1][2])
    a3 = cpu(atts[1][3])
    a4 = cpu(atts[1][4])
    a5 = cpu(atts[1][5])
    a6 = cpu(atts[1][6])
    a7 = cpu(atts[1][7])
    a8 = cpu(atts[1][8])
    a9 = cpu(atts[1][9])
    a10 = cpu(atts[1][10])
    @save string("sokoban_heur_experiments/",model_name,"/att1.bson") a1
    @save string("sokoban_heur_experiments/",model_name,"/att2.bson") a2
    @save string("sokoban_heur_experiments/",model_name,"/att3.bson") a3
    @save string("sokoban_heur_experiments/",model_name,"/att4.bson") a4
    @save string("sokoban_heur_experiments/",model_name,"/att5.bson") a5
    @save string("sokoban_heur_experiments/",model_name,"/att6.bson") a6
    @save string("sokoban_heur_experiments/",model_name,"/att7.bson") a7
    @save string("sokoban_heur_experiments/",model_name,"/att8.bson") a8
    @save string("sokoban_heur_experiments/",model_name,"/att9.bson") a9
    @save string("sokoban_heur_experiments/",model_name,"/att10.bson") a10

    a11 = cpu(atts[2][1])
    a12 = cpu(atts[2][2])
    a13 = cpu(atts[2][3])
    a14 = cpu(atts[2][4])
    a15 = cpu(atts[2][5])
    a16 = cpu(atts[2][6])
    a17 = cpu(atts[2][7])
    a18 = cpu(atts[2][8])
    a19 = cpu(atts[2][9])
    a20 = cpu(atts[2][10])
    @save string("sokoban_heur_experiments/",model_name,"/att12.bson") a12
    @save string("sokoban_heur_experiments/",model_name,"/att11.bson") a11
    @save string("sokoban_heur_experiments/",model_name,"/att13.bson") a13
    @save string("sokoban_heur_experiments/",model_name,"/att14.bson") a14
    @save string("sokoban_heur_experiments/",model_name,"/att15.bson") a15
    @save string("sokoban_heur_experiments/",model_name,"/att16.bson") a16
    @save string("sokoban_heur_experiments/",model_name,"/att17.bson") a17
    @save string("sokoban_heur_experiments/",model_name,"/att18.bson") a18
    @save string("sokoban_heur_experiments/",model_name,"/att19.bson") a19
    @save string("sokoban_heur_experiments/",model_name,"/att20.bson") a20

    a21 = cpu(atts[3][1])
    a22 = cpu(atts[3][2])
    a23 = cpu(atts[3][3])
    a24 = cpu(atts[3][4])
    a25 = cpu(atts[3][5])
    a26 = cpu(atts[3][6])
    a27 = cpu(atts[3][7])
    a28 = cpu(atts[3][8])
    a29 = cpu(atts[3][9])
    a30 = cpu(atts[3][10])
    @save string("sokoban_heur_experiments/",model_name,"/att21.bson") a21
    @save string("sokoban_heur_experiments/",model_name,"/att22.bson") a22
    @save string("sokoban_heur_experiments/",model_name,"/att23.bson") a23
    @save string("sokoban_heur_experiments/",model_name,"/att24.bson") a24
    @save string("sokoban_heur_experiments/",model_name,"/att25.bson") a25
    @save string("sokoban_heur_experiments/",model_name,"/att26.bson") a26
    @save string("sokoban_heur_experiments/",model_name,"/att27.bson") a27
    @save string("sokoban_heur_experiments/",model_name,"/att28.bson") a28
    @save string("sokoban_heur_experiments/",model_name,"/att29.bson") a29
    @save string("sokoban_heur_experiments/",model_name,"/att30.bson") a30

    a31 = cpu(atts[4][1])
    a32 = cpu(atts[4][2])
    a33 = cpu(atts[4][3])
    a34 = cpu(atts[4][4])
    a35 = cpu(atts[4][5])
    a36 = cpu(atts[4][6])
    a37 = cpu(atts[4][7])
    a38 = cpu(atts[4][8])
    a39 = cpu(atts[4][9])
    a40 = cpu(atts[4][10])
    @save string("sokoban_heur_experiments/",model_name,"/att31.bson") a31
    @save string("sokoban_heur_experiments/",model_name,"/att32.bson") a32
    @save string("sokoban_heur_experiments/",model_name,"/att33.bson") a33
    @save string("sokoban_heur_experiments/",model_name,"/att34.bson") a34
    @save string("sokoban_heur_experiments/",model_name,"/att35.bson") a35
    @save string("sokoban_heur_experiments/",model_name,"/att36.bson") a36
    @save string("sokoban_heur_experiments/",model_name,"/att37.bson") a37
    @save string("sokoban_heur_experiments/",model_name,"/att38.bson") a38
    @save string("sokoban_heur_experiments/",model_name,"/att39.bson") a39
    @save string("sokoban_heur_experiments/",model_name,"/att40.bson") a40

    a41 = cpu(atts[5][1])
    a42 = cpu(atts[5][2])
    a43 = cpu(atts[5][3])
    a44 = cpu(atts[5][4])
    a45 = cpu(atts[5][5])
    a46 = cpu(atts[5][6])
    a47 = cpu(atts[5][7])
    a48 = cpu(atts[5][8])
    a49 = cpu(atts[5][9])
    a50 = cpu(atts[5][10])
    @save string("sokoban_heur_experiments/",model_name,"/att41.bson") a41
    @save string("sokoban_heur_experiments/",model_name,"/att42.bson") a42
    @save string("sokoban_heur_experiments/",model_name,"/att43.bson") a43
    @save string("sokoban_heur_experiments/",model_name,"/att44.bson") a44
    @save string("sokoban_heur_experiments/",model_name,"/att45.bson") a45
    @save string("sokoban_heur_experiments/",model_name,"/att46.bson") a46
    @save string("sokoban_heur_experiments/",model_name,"/att47.bson") a47
    @save string("sokoban_heur_experiments/",model_name,"/att48.bson") a48
    @save string("sokoban_heur_experiments/",model_name,"/att49.bson") a49
    @save string("sokoban_heur_experiments/",model_name,"/att50.bson") a50
elseif att_blocks == 5 && no_atts == 5
    a1 = cpu(atts[1][1])
    a2 = cpu(atts[1][2])
    a3 = cpu(atts[1][3])
    a4 = cpu(atts[1][4])
    a5 = cpu(atts[1][5])
    @save string("sokoban_heur_experiments/",model_name,"/att1.bson") a1
    @save string("sokoban_heur_experiments/",model_name,"/att2.bson") a2
    @save string("sokoban_heur_experiments/",model_name,"/att3.bson") a3
    @save string("sokoban_heur_experiments/",model_name,"/att4.bson") a4
    @save string("sokoban_heur_experiments/",model_name,"/att5.bson") a5

    a6 = cpu(atts[2][1])
    a7 = cpu(atts[2][2])
    a8 = cpu(atts[2][3])
    a9 = cpu(atts[2][4])
    a10 = cpu(atts[2][5])
    @save string("sokoban_heur_experiments/",model_name,"/att6.bson") a6
    @save string("sokoban_heur_experiments/",model_name,"/att7.bson") a7
    @save string("sokoban_heur_experiments/",model_name,"/att8.bson") a8
    @save string("sokoban_heur_experiments/",model_name,"/att9.bson") a9
    @save string("sokoban_heur_experiments/",model_name,"/att10.bson") a10

    a11 = cpu(atts[3][1])
    a12 = cpu(atts[3][2])
    a13 = cpu(atts[3][3])
    a14 = cpu(atts[3][4])
    a15 = cpu(atts[3][5])
    @save string("sokoban_heur_experiments/",model_name,"/att11.bson") a11
    @save string("sokoban_heur_experiments/",model_name,"/att12.bson") a12
    @save string("sokoban_heur_experiments/",model_name,"/att13.bson") a13
    @save string("sokoban_heur_experiments/",model_name,"/att14.bson") a14
    @save string("sokoban_heur_experiments/",model_name,"/att15.bson") a15

    a16 = cpu(atts[4][1])
    a17 = cpu(atts[4][2])
    a18 = cpu(atts[4][3])
    a19 = cpu(atts[4][4])
    a20 = cpu(atts[4][5])
    @save string("sokoban_heur_experiments/",model_name,"/att16.bson") a16
    @save string("sokoban_heur_experiments/",model_name,"/att17.bson") a17
    @save string("sokoban_heur_experiments/",model_name,"/att18.bson") a18
    @save string("sokoban_heur_experiments/",model_name,"/att19.bson") a19
    @save string("sokoban_heur_experiments/",model_name,"/att20.bson") a20

    a21 = cpu(atts[5][1])
    a22 = cpu(atts[5][2])
    a23 = cpu(atts[5][3])
    a24 = cpu(atts[5][4])
    a25 = cpu(atts[5][5])
    @save string("sokoban_heur_experiments/",model_name,"/att21.bson") a21
    @save string("sokoban_heur_experiments/",model_name,"/att22.bson") a22
    @save string("sokoban_heur_experiments/",model_name,"/att23.bson") a23
    @save string("sokoban_heur_experiments/",model_name,"/att24.bson") a24
    @save string("sokoban_heur_experiments/",model_name,"/att25.bson") a25
elseif att_blocks == 5 && no_atts == 20
    a1 = cpu(atts[1][1])
    a2 = cpu(atts[1][2])
    a3 = cpu(atts[1][3])
    a4 = cpu(atts[1][4])
    a5 = cpu(atts[1][5])
    a6 = cpu(atts[1][6])
    a7 = cpu(atts[1][7])
    a8 = cpu(atts[1][8])
    a9 = cpu(atts[1][9])
    a10 = cpu(atts[1][10])
    a11 = cpu(atts[1][11])
    a12 = cpu(atts[1][12])
    a13 = cpu(atts[1][13])
    a14 = cpu(atts[1][14])
    a15 = cpu(atts[1][15])
    a16 = cpu(atts[1][16])
    a17 = cpu(atts[1][17])
    a18 = cpu(atts[1][18])
    a19 = cpu(atts[1][19])
    a20 = cpu(atts[1][20])
    @save string("sokoban_heur_experiments/",model_name,"/att1.bson") a1
    @save string("sokoban_heur_experiments/",model_name,"/att2.bson") a2
    @save string("sokoban_heur_experiments/",model_name,"/att3.bson") a3
    @save string("sokoban_heur_experiments/",model_name,"/att4.bson") a4
    @save string("sokoban_heur_experiments/",model_name,"/att5.bson") a5
    @save string("sokoban_heur_experiments/",model_name,"/att6.bson") a6
    @save string("sokoban_heur_experiments/",model_name,"/att7.bson") a7
    @save string("sokoban_heur_experiments/",model_name,"/att8.bson") a8
    @save string("sokoban_heur_experiments/",model_name,"/att9.bson") a9
    @save string("sokoban_heur_experiments/",model_name,"/att10.bson") a10
    @save string("sokoban_heur_experiments/",model_name,"/att12.bson") a12
    @save string("sokoban_heur_experiments/",model_name,"/att11.bson") a11
    @save string("sokoban_heur_experiments/",model_name,"/att13.bson") a13
    @save string("sokoban_heur_experiments/",model_name,"/att14.bson") a14
    @save string("sokoban_heur_experiments/",model_name,"/att15.bson") a15
    @save string("sokoban_heur_experiments/",model_name,"/att16.bson") a16
    @save string("sokoban_heur_experiments/",model_name,"/att17.bson") a17
    @save string("sokoban_heur_experiments/",model_name,"/att18.bson") a18
    @save string("sokoban_heur_experiments/",model_name,"/att19.bson") a19
    @save string("sokoban_heur_experiments/",model_name,"/att20.bson") a20

    a21 = cpu(atts[2][1])
    a22 = cpu(atts[2][2])
    a23 = cpu(atts[2][3])
    a24 = cpu(atts[2][4])
    a25 = cpu(atts[2][5])
    a26 = cpu(atts[2][6])
    a27 = cpu(atts[2][7])
    a28 = cpu(atts[2][8])
    a29 = cpu(atts[2][9])
    a30 = cpu(atts[2][10])
    a31 = cpu(atts[2][11])
    a32 = cpu(atts[2][12])
    a33 = cpu(atts[2][13])
    a34 = cpu(atts[2][14])
    a35 = cpu(atts[2][15])
    a36 = cpu(atts[2][16])
    a37 = cpu(atts[2][17])
    a38 = cpu(atts[2][18])
    a39 = cpu(atts[2][19])
    a40 = cpu(atts[2][20])
    @save string("sokoban_heur_experiments/",model_name,"/att21.bson") a21
    @save string("sokoban_heur_experiments/",model_name,"/att22.bson") a22
    @save string("sokoban_heur_experiments/",model_name,"/att23.bson") a23
    @save string("sokoban_heur_experiments/",model_name,"/att24.bson") a24
    @save string("sokoban_heur_experiments/",model_name,"/att25.bson") a25
    @save string("sokoban_heur_experiments/",model_name,"/att26.bson") a26
    @save string("sokoban_heur_experiments/",model_name,"/att27.bson") a27
    @save string("sokoban_heur_experiments/",model_name,"/att28.bson") a28
    @save string("sokoban_heur_experiments/",model_name,"/att29.bson") a29
    @save string("sokoban_heur_experiments/",model_name,"/att30.bson") a30
    @save string("sokoban_heur_experiments/",model_name,"/att31.bson") a31
    @save string("sokoban_heur_experiments/",model_name,"/att32.bson") a32
    @save string("sokoban_heur_experiments/",model_name,"/att33.bson") a33
    @save string("sokoban_heur_experiments/",model_name,"/att34.bson") a34
    @save string("sokoban_heur_experiments/",model_name,"/att35.bson") a35
    @save string("sokoban_heur_experiments/",model_name,"/att36.bson") a36
    @save string("sokoban_heur_experiments/",model_name,"/att37.bson") a37
    @save string("sokoban_heur_experiments/",model_name,"/att38.bson") a38
    @save string("sokoban_heur_experiments/",model_name,"/att39.bson") a39
    @save string("sokoban_heur_experiments/",model_name,"/att40.bson") a40

    a41 = cpu(atts[3][1])
    a42 = cpu(atts[3][2])
    a43 = cpu(atts[3][3])
    a44 = cpu(atts[3][4])
    a45 = cpu(atts[3][5])
    a46 = cpu(atts[3][6])
    a47 = cpu(atts[3][7])
    a48 = cpu(atts[3][8])
    a49 = cpu(atts[3][9])
    a50 = cpu(atts[3][10])
    a51 = cpu(atts[3][11])
    a52 = cpu(atts[3][12])
    a53 = cpu(atts[3][13])
    a54 = cpu(atts[3][14])
    a55 = cpu(atts[3][15])
    a56 = cpu(atts[3][16])
    a57 = cpu(atts[3][17])
    a58 = cpu(atts[3][18])
    a59 = cpu(atts[3][19])
    a60 = cpu(atts[3][20])
    @save string("sokoban_heur_experiments/",model_name,"/att41.bson") a41
    @save string("sokoban_heur_experiments/",model_name,"/att42.bson") a42
    @save string("sokoban_heur_experiments/",model_name,"/att43.bson") a43
    @save string("sokoban_heur_experiments/",model_name,"/att44.bson") a44
    @save string("sokoban_heur_experiments/",model_name,"/att45.bson") a45
    @save string("sokoban_heur_experiments/",model_name,"/att46.bson") a46
    @save string("sokoban_heur_experiments/",model_name,"/att47.bson") a47
    @save string("sokoban_heur_experiments/",model_name,"/att48.bson") a48
    @save string("sokoban_heur_experiments/",model_name,"/att49.bson") a49
    @save string("sokoban_heur_experiments/",model_name,"/att50.bson") a50
    @save string("sokoban_heur_experiments/",model_name,"/att51.bson") a51
    @save string("sokoban_heur_experiments/",model_name,"/att52.bson") a52
    @save string("sokoban_heur_experiments/",model_name,"/att53.bson") a53
    @save string("sokoban_heur_experiments/",model_name,"/att54.bson") a54
    @save string("sokoban_heur_experiments/",model_name,"/att55.bson") a55
    @save string("sokoban_heur_experiments/",model_name,"/att56.bson") a56
    @save string("sokoban_heur_experiments/",model_name,"/att57.bson") a57
    @save string("sokoban_heur_experiments/",model_name,"/att58.bson") a58
    @save string("sokoban_heur_experiments/",model_name,"/att59.bson") a59
    @save string("sokoban_heur_experiments/",model_name,"/att60.bson") a60

    a61 = cpu(atts[4][1])
    a62 = cpu(atts[4][2])
    a63 = cpu(atts[4][3])
    a64 = cpu(atts[4][4])
    a65 = cpu(atts[4][5])
    a66 = cpu(atts[4][6])
    a67 = cpu(atts[4][7])
    a68 = cpu(atts[4][8])
    a69 = cpu(atts[4][9])
    a70 = cpu(atts[4][10])
    a71 = cpu(atts[4][11])
    a72 = cpu(atts[4][12])
    a73 = cpu(atts[4][13])
    a74 = cpu(atts[4][14])
    a75 = cpu(atts[4][15])
    a76 = cpu(atts[4][16])
    a77 = cpu(atts[4][17])
    a78 = cpu(atts[4][18])
    a79 = cpu(atts[4][19])
    a80 = cpu(atts[4][20])
    @save string("sokoban_heur_experiments/",model_name,"/att61.bson") a61
    @save string("sokoban_heur_experiments/",model_name,"/att62.bson") a62
    @save string("sokoban_heur_experiments/",model_name,"/att63.bson") a63
    @save string("sokoban_heur_experiments/",model_name,"/att64.bson") a64
    @save string("sokoban_heur_experiments/",model_name,"/att65.bson") a65
    @save string("sokoban_heur_experiments/",model_name,"/att66.bson") a66
    @save string("sokoban_heur_experiments/",model_name,"/att67.bson") a67
    @save string("sokoban_heur_experiments/",model_name,"/att68.bson") a68
    @save string("sokoban_heur_experiments/",model_name,"/att69.bson") a69
    @save string("sokoban_heur_experiments/",model_name,"/att70.bson") a70
    @save string("sokoban_heur_experiments/",model_name,"/att71.bson") a71
    @save string("sokoban_heur_experiments/",model_name,"/att72.bson") a72
    @save string("sokoban_heur_experiments/",model_name,"/att73.bson") a73
    @save string("sokoban_heur_experiments/",model_name,"/att74.bson") a74
    @save string("sokoban_heur_experiments/",model_name,"/att75.bson") a75
    @save string("sokoban_heur_experiments/",model_name,"/att76.bson") a76
    @save string("sokoban_heur_experiments/",model_name,"/att77.bson") a77
    @save string("sokoban_heur_experiments/",model_name,"/att78.bson") a78
    @save string("sokoban_heur_experiments/",model_name,"/att79.bson") a79
    @save string("sokoban_heur_experiments/",model_name,"/att80.bson") a80

    a81 = cpu(atts[5][1])
    a82 = cpu(atts[5][2])
    a83 = cpu(atts[5][3])
    a84 = cpu(atts[5][4])
    a85 = cpu(atts[5][5])
    a86 = cpu(atts[5][6])
    a87 = cpu(atts[5][7])
    a88 = cpu(atts[5][8])
    a89 = cpu(atts[5][9])
    a90 = cpu(atts[5][10])
    a91 = cpu(atts[5][11])
    a92 = cpu(atts[5][12])
    a93 = cpu(atts[5][13])
    a94 = cpu(atts[5][14])
    a95 = cpu(atts[5][15])
    a96 = cpu(atts[5][16])
    a97 = cpu(atts[5][17])
    a98 = cpu(atts[5][18])
    a99 = cpu(atts[5][19])
    a100 = cpu(atts[5][20])
    @save string("sokoban_heur_experiments/",model_name,"/att81.bson") a81
    @save string("sokoban_heur_experiments/",model_name,"/att82.bson") a82
    @save string("sokoban_heur_experiments/",model_name,"/att83.bson") a83
    @save string("sokoban_heur_experiments/",model_name,"/att84.bson") a84
    @save string("sokoban_heur_experiments/",model_name,"/att85.bson") a85
    @save string("sokoban_heur_experiments/",model_name,"/att86.bson") a86
    @save string("sokoban_heur_experiments/",model_name,"/att87.bson") a87
    @save string("sokoban_heur_experiments/",model_name,"/att88.bson") a88
    @save string("sokoban_heur_experiments/",model_name,"/att89.bson") a89
    @save string("sokoban_heur_experiments/",model_name,"/att90.bson") a90
    @save string("sokoban_heur_experiments/",model_name,"/att91.bson") a91
    @save string("sokoban_heur_experiments/",model_name,"/att92.bson") a92
    @save string("sokoban_heur_experiments/",model_name,"/att93.bson") a93
    @save string("sokoban_heur_experiments/",model_name,"/att94.bson") a94
    @save string("sokoban_heur_experiments/",model_name,"/att95.bson") a95
    @save string("sokoban_heur_experiments/",model_name,"/att96.bson") a96
    @save string("sokoban_heur_experiments/",model_name,"/att97.bson") a97
    @save string("sokoban_heur_experiments/",model_name,"/att98.bson") a98
    @save string("sokoban_heur_experiments/",model_name,"/att99.bson") a99
    @save string("sokoban_heur_experiments/",model_name,"/att100.bson") a100
end