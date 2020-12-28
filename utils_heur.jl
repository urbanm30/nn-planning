# 1: wall = [1,0,0,0]
# 0: free_space = [0,1,0,0]
# 2: agent = [0,0,1,0]
# 3: goal = [0,0,0,1]
const d = Dict(1 => [1,0,0,0], 0 => [0,1,0,0], 2 => [0,0,1,0], 3 => [0,0,0,1])
const d_step = Dict(1 => [1,0,0], 0 => [0,1,0], 2 => [0,0,1], 3 => [0,1,0])

function img2onehot(x::Matrix)
    o = zeros(Float32, size(x)..., 4)
    for ii in CartesianIndices(x)
        o[ii,:,:,:] .= d[x[ii]]
    end
    o
end

function img2onehot(x::A) where {A<:AbstractArray{T,3}} where {T}
    cat([img2onehot(x[:,:,i]) for i in 1:size(x,3)]..., dims = 4)
end

function img2onehot(x::A) where {A<:AbstractArray{T,4}} where {T}
    cat([img2onehot(x[:,:,:,i]) for i in 1:size(x,4)]..., dims = 4)
end

function img2onehotnogoal(x::Matrix)
    o = zeros(Float32, size(x)..., 3)
    for ii in CartesianIndices(x)
        o[ii,:,:,:] .= d_step[x[ii]]
    end
    o
end

function img2onehotnogoal(x::A) where {A<:AbstractArray{T,3}} where {T}
    x_new = cat([img2onehotnogoal(x[:,:,i]) for i in 1:size(x,3)]..., dims = 4)
    return x_new[:,:,1:3,:]
end

function loadfile(f,n)
    csv_in = CSV.read(f; header=false, delim=';')
    imgs = reshape(transpose(Matrix(csv_in)),n,n,:)
    if size(imgs,3) <= 5000
        data = img2onehot(imgs)
    else
        data = zeros(size(imgs,1), size(imgs,2), 4, size(imgs,3))
        for i in 1:size(imgs,3)
            data[:,:,:,i] = img2onehot(imgs[:,:,i])
        end
    end
    return data
end

function loadfile_wrap(f,n)
    csv_in = CSV.read(f; header=false, delim=';')
    imgs = reshape(transpose(Matrix(csv_in)),n,n,:)
    data = zeros(size(imgs,1) + 2, size(imgs,2) + 2, 4, size(imgs,3))
    for i in 1:size(imgs,3)
        data[:,:,:,i] = wrap_with_walls(imgs[:,:,i])
    end
    return data
end

# Load maze data and transfers into one-hot represenation
# A little complicated -> working on transforming it differently
load_data_one_hot() = (loadfile("maze8_imgs.csv",8), loadfile("maze8_labels.csv",8))
load_data_100_one_hot() = (loadfile("100_imgs.csv",100), loadfile("100_labels.csv",100))
load_data_250_one_hot() = (loadfile("250_imgs.csv",250), loadfile("250_labels.csv",250))

# Turns one-hot representation back into original maze
function onehot2img(x::A) where {A<:AbstractArray{T,4}} where{T}
    xx = mapslices(Flux.onecold, x, dims = 3)
    xx[ xx .== 2] .= 0
    xx[ xx .== 3] .= 2
    xx[ xx .== 4] .= 3
    dropdims(xx, dims = 3)
end

function load_prim_8_goal_step(maze_size)
    # data = loadfile("julia_mazes_8_step.csv", maze_size)
    # test_data = loadfile("julia_mazes_8_step_test.csv", maze_size)

    labels = load("julia_8_step_labels.jld")["data"]
    # test_labels = load("julia_mazes_8_step_test_labels.jld")["data"]

    data = load("julia_8_mazes.jld")["data"]
    # test_data = load("julia_mazes_8_step_100000_test.jld")["data"]

    # labels = load("julia_mazes_8_step_100000_labels.jld")["data"]
    # test_labels = load("julia_mazes_8_step_100000_labels_test.jld")["data"]

    return data, labels
end

function load_prim_8_goal_with_labels(maze_size, wall_pad)
    if wall_pad == 1
        data = loadfile_wrap("Julia_data/julia_mazes_8.csv",maze_size)
        test_data = loadfile_wrap("Julia_data/julia_mazes_8_test.csv", maze_size)
    else
        data = loadfile("Julia_data/julia_mazes_8.csv",maze_size)
        test_data = loadfile("Julia_data/julia_mazes_8_test.csv", maze_size)
    end

    labels = convert(Matrix, CSV.read("Julia_data/julia_mazes_8_labels.csv"; header=false, delim=';'))
    test_labels = convert(Matrix, CSV.read("Julia_data/julia_mazes_8_test_labels.csv"; header=false, delim=';'))

    return data |> gpu, labels |> gpu, test_data |> gpu, test_labels |> gpu
end

function load_prim_16_goal_with_labels(maze_size, wall_pad)
    if wall_pad == 1
        data = loadfile_wrap("Julia_data/julia_mazes_16.csv",maze_size)
        # test_data = loadfile_wrap("Julia_data/julia_mazes_16_test.csv", maze_size)
    else
        data = loadfile("Julia_data/julia_mazes_16.csv",maze_size)
        # test_data = loadfile("Julia_data/julia_mazes_16_test.csv", maze_size)
    end

    labels = convert(Matrix, CSV.read("Julia_data/julia_mazes_16_labels.csv"; header=false, delim=';'))
    # test_labels = cu(convert(Matrix, CSV.read("Julia_data/julia_mazes_16_test_labels.csv"; header=false, delim=';')))

    return data, labels#, cu(test_data), test_labels
end

function load_prim_8_goal(label_type, maze_size)
    data = loadfile("Julia_data/julia_mazes_8.csv",maze_size)
    test_data = loadfile("Julia_data/julia_mazes_8_test.csv", maze_size)

    csv_in = CSV.read("Julia_data/julia_mazes_8.csv"; header=false, delim=';')
    mazes = reshape(transpose(Matrix(csv_in)),maze_size,maze_size,:)

    csv_in_test = CSV.read("Julia_data/julia_mazes_8_test.csv"; header=false, delim=';')
    test_mazes = reshape(transpose(Matrix(csv_in_test)),maze_size,maze_size,:)

    if label_type == "euclidean"
        labels = create_labels_euclid(mazes)
        test_labels = create_labels_euclid(test_mazes)
    end
    if label_type == "ground_truth"
        labels = create_labels_gt(mazes)
        test_labels = create_labels_gt(test_mazes)
    end
    if label_type == "manhattan"
        labels = create_labels_manhattan(mazes)
        test_labels = create_labels_manhattan(test_mazes)
    end

    return data, labels, test_data, test_labels
end

function load_prim_16_goal(label_type, maze_size)
    data = loadfile("julia_mazes_16.csv",maze_size)
    test_data = loadfile("julia_mazes_16_test.csv", maze_size)

    csv_in = CSV.read("julia_mazes_16.csv"; header=false, delim=';')
    mazes = reshape(transpose(Matrix(csv_in)),maze_size,maze_size,:)

    csv_in_test = CSV.read("julia_mazes_16_test.csv"; header=false, delim=';')
    test_mazes = reshape(transpose(Matrix(csv_in_test)),maze_size,maze_size,:)

    if label_type == "euclidean"
        labels = create_labels_euclid(mazes)
        test_labels = create_labels_euclid(test_mazes)
    end
    if label_type == "ground_truth"
        labels = create_labels_gt(mazes)
        test_labels = create_labels_gt(test_mazes)
    end
    if label_type == "manhattan"
        labels = create_labels_manhattan(mazes)
        test_labels = create_labels_manhattan(test_mazes)
    end

    return data, labels, test_data, test_labels
end

function load_8_goal(label_type)
    data = loadfile("8_goal_mazes.csv",8)

    csv_in = CSV.read("8_goal_mazes.csv"; header=false, delim=';')
    mazes = reshape(transpose(Matrix(csv_in)),8,8,:)

    if label_type == "euclidean"
        labels = create_labels_euclid(mazes)
    end
    if label_type == "ground_truth"
        labels = create_labels_gt(mazes)
    end
    if label_type == "manhattan"
        labels = create_labels_manhattan(mazes)
    end

    return data, labels
end

function create_labels_gt(mazes)
    labels = Array{Float32}(undef, 1, size(mazes)[3])
    for i in 1:size(mazes)[3]
        maze = mazes[:,:,i]
        goal_coords = CartesianIndices(maze)[maze .== 3]
        # path_len = a_star(maze, false, "euclidean", [])
        path, path_len, ex_nodes = bfs(maze, false, "euclidean", [])
        labels[1, i] = path_len
    end
    return Float32.(labels)
end

function create_labels_euclid(mazes)
    labels = Array{Float32}(undef, 1, size(mazes)[3])
    for i in 1:size(mazes)[3]
        maze = mazes[:,:,i]
        goal_coords = getindex.(CartesianIndices(maze)[maze .== 3], 1:2)
        agent_coords = getindex.(CartesianIndices(maze)[maze .== 2], 1:2)
        labels[1, i] = Distances.euclidean(agent_coords,goal_coords)
    end
    return Float32.(labels)
end

function create_labels_manhattan(mazes)
    labels = Array{Float32}(undef, 1, size(mazes)[3])
    for i in 1:size(mazes)[3]
        maze = mazes[:,:,i]
        goal_coords = getindex.(CartesianIndices(maze)[maze .== 3], 1:2)
        agent_coords = getindex.(CartesianIndices(maze)[maze .== 2], 1:2)
        labels[1, i] = Distances.cityblock(agent_coords,goal_coords)
    end
    return Float32.(labels)
end

function value_difference(mx, y)
    diff = abs.(mx .- y)
    max_diff = maximum(diff)
    min_diff = minimum(diff)
    mean_diff = mean(diff)
    @show(max_diff)
    @show(min_diff)
    @show(mean_diff)
end

function all_stats(mx, y)
    diff = mx .- y
    @show(length(CartesianIndices(diff)[diff .< 0]))
    @show(length(CartesianIndices(diff)[diff .> 0]))

    below = diff[CartesianIndices(diff)[diff .< 0]]
    @show(maximum(below), minimum(below), mean(below))

    above = diff[CartesianIndices(diff)[diff .> 0]]
    @show(maximum(above), minimum(above), mean(above))
end

# Returns longest path that had error bellow th
# mx = model(x), y = labels, th = threshold value
function longest_correct_path(mx, y, th)
    diffs = abs.(mx .- y)
    ixs = CartesianIndices(diffs)[diffs .< th]
    println("Th = ", th, "; ", length(ixs), "/", length(y))
    # return maximum(sel_y)
end

function optimistic_test(mx, y)
    tmp = CartesianIndices(mx)[mx .<= y]
    println("Optimistic value: ",length(tmp),"/",length(y))
end

# function sf(x, ds)
#     mx = maximum(x, dims=ds)
#     # mx = gpu_maximum(x, dims=ds)
#     e = exp.(x .- mx)
#     e ./ sum(e, dims = ds)
# end |> gpu

# function gpu_minimum(x::CuArray{T}; dims=dims) where {T}
#     mx = fill!(similar(x, T, Base.reduced_indices(x, dims)), typemax(T))
#     println(mx)
#     Base._mapreducedim!(identity, min, mx, x)
# end

# function gpu_maximum(x::CuArray{T}; dims=dims) where {T}
#     mx = fill!(similar(x, T, Base.reduced_indices(x, dims)), typemin(T))
#     Base._mapreducedim!(identity, max, mx, x)
# end

function go_through_maze(m)
    maze = reshape(m, size(m)[1],size(m)[2])
    agent_ix = CartesianIndices(maze)[maze .== 2]
    for a_ix in agent_ix
        maze[a_ix] = 0
    end
    free_ixs = CartesianIndices(maze)[maze .== 0]

    mazes = zeros(size(maze)[1],size(maze)[2],1,length(free_ixs))
    # mazes[:,:,:,1] = maze
    maze_no = 1

    for ix in free_ixs
        m = deepcopy(maze)
        # m[agent_ix] .= 0
        m[ix] .= 2
        mazes[:,:,:,maze_no] = m
        maze_no += 1
    end
    return mazes
end

function go_through_maze_multiagent(m)
    maze = reshape(m, size(m)[1],size(m)[2])
    agent_ixs = CartesianIndices(maze)[maze .== 2]
    maze[agent_ixs] .= 0
    free_ixs = CartesianIndices(maze)[maze .== 0]
    combs = collect(combinations(free_ixs, 2))
    mazes = zeros(size(m,1), size(m,2), length(combs))
    for i in 1:length(combs)
        nm = deepcopy(maze)
        comb = combs[i]
        nm[comb[1]] = 2
        nm[comb[2]] = 2
        mazes[:,:,i] = nm
    end
    return mazes
end 

function add_coord_filters(imgs)
    x_filter = Array{Float32}(undef,size(imgs)[1],size(imgs)[2])
    y_filter = Array{Float32}(undef,size(imgs)[1],size(imgs)[2])
    for i in (1:size(imgs)[1])
        x_filter[:,i] .= i
    end

    for i in (1:size(imgs)[2])  
        y_filter[i,:] .= i
    end
    add_filter = cat(x_filter, y_filter, dims=3) |> gpu
    imgs = cat(imgs, repeat(add_filter, 1, 1, 1, size(imgs)[4]), dims=3)
    return imgs
end

function get_coord_filters()
    x_filter = Array{Float32}(undef,size(imgs)[1],size(imgs)[2])
    y_filter = Array{Float32}(undef,size(imgs)[1],size(imgs)[2])
    for i in (1:size(imgs)[1])
        x_filter[:,i] .= i
    end

    for i in (1:size(imgs)[2])
        y_filter[i,:] .= i
    end
    add_filter = reshape(cat(x_filter, y_filter, dims=3), size(x_filter,1), size(x_filter,2), :, 1) |> gpu
    return add_filter
end

function make_batches(data, labels, no_batches, batch_size, mzs_from_block)
    batches = []
    # How many same mazes are in a block, mzs_from_block is how many I want in my batch
    maze_block_count = 50
    maze_block_ixs = [0:maze_block_count:(size(data,4) - maze_block_count);]
    blocks_in_batch = Int64(batch_size / mzs_from_block)

    for i in 1:no_batches
        blocks = rand(maze_block_ixs, blocks_in_batch)
        sample_ixs = Int64.(reshape(transpose(blocks .+ transpose([1:1:mzs_from_block;])), 1, :))
        batch_data = reshape(data[:,:,:,sample_ixs], size(data,1), size(data,2), size(data,3), batch_size)
        batch_labels = labels[sample_ixs]
        push!(batches, (batch_data, batch_labels))
    end

    return batches
end

function make_batches_with_coords(data, labels, no_batches, batch_size, mzs_from_block)
    batches = []
    # How many same mazes are in a block, mzs_from_block is how many I want in my batch
    maze_block_count = 50
    maze_block_ixs = [0:maze_block_count:(size(data,4) - maze_block_count);]
    blocks_in_batch = Int64(batch_size / mzs_from_block)

    for i in 1:no_batches
        blocks = rand(maze_block_ixs, blocks_in_batch)
        sample_ixs = Int64.(reshape(transpose(blocks .+ transpose([1:1:mzs_from_block;])), 1, :))
        batch_data = reshape(data[:,:,:,sample_ixs], size(data,1), size(data,2), size(data,3), batch_size)
        batch_labels = labels[sample_ixs]
        push!(batches, (add_coord_filters(batch_data), batch_labels))
    end

    return batches
end

# Data is pre-batches -> it just has to be sampled randomly 
function make_batches_sokoban(imgs, labels, no_batches, batch_size)
    batches = []
    no_instances_in_batch = Int64(batch_size / 40)
    for i in 1:no_batches
        instance_ixs = rand(1:40:size(imgs,4) - 39,no_instances_in_batch)
        ms = Nothing
        ls = []
        for ix in instance_ixs
            if ms == Nothing 
                ms = imgs[:,:,:,ix:ix + 39]
            else 
                ms = cat(ms, imgs[:,:,:,ix:ix + 39],dims=4)
            end
            push!(ls, labels[ix:ix + 39]...)
        end
        push!(batches, (ms, cu(Float32.(ls))))
    end
    return batches 
end

function get_real_heur_scaled(maze)
    mazes = go_through_maze(maze)
    map = zeros(size(mazes)[1],size(mazes)[2])
    vals = []
    for i in 1:size(mazes)[4]
        m = mazes[:,:,:,i]
        agent_coords = CartesianIndices(m)[m .== 2]
        path_len, expanded_states = a_star(m, false, "euclidean", [])
        map[agent_coords] .= path_len
        push!(vals, path_len)
    end 

    min_r = minimum(vals)
    max_r = maximum(vals)
    map = (map .- min_r) ./ (max_r - min_r)   
    vals = (vals .- min_r) ./ (max_r - min_r)

    # map[CartesianIndices(maze)[maze .== 3]] .= minimum(vals) - 0.1
    # map[CartesianIndices(maze)[maze .== 1]] .= minimum(vals) - 0.2
    map[CartesianIndices(maze)[maze .== 3]] .= -.1
    map[CartesianIndices(maze)[maze .== 1]] .= -.2

    # h2 = plot(heatmap(map, c=:lightrainbow), aspect_ratio = 1)
    # h1 = plot(heatmap(reshape(maze, size(maze)[1], size(maze)[2])), aspect_ratio = 1)

    # plot(h1, h2, layout=(1,2))
    return map 
end

function get_euclid_heur_scaled(maze)
    mazes = go_through_maze(maze)
    map = zeros(size(mazes)[1],size(mazes)[2])
    vals = []
    for i in 1:size(mazes)[4]
        m = mazes[:,:,:,i]
        agent_coords = CartesianIndices(m)[m .== 2]
        path_len = euclidean_heur(m)
        map[agent_coords] .= path_len
        push!(vals, path_len)
    end 

    # min_r = minimum(vals)
    # max_r = maximum(vals)
    # map = (map .- min_r) ./ (max_r - min_r)   
    # vals = (vals .- min_r) ./ (max_r - min_r)

    # map[CartesianIndices(maze)[maze .== 3]] .= minimum(vals) - 0.1
    # map[CartesianIndices(maze)[maze .== 1]] .= minimum(vals) - 0.2
    map[CartesianIndices(maze)[maze .== 3]] .= -.1
    map[CartesianIndices(maze)[maze .== 1]] .= -.2

    # h2 = plot(heatmap(map, c=:lightrainbow), aspect_ratio = 1)
    # h1 = plot(heatmap(reshape(maze, size(maze)[1], size(maze)[2])), aspect_ratio = 1)

    # plot(h1, h2, layout=(1,2))
    return map 
end

# Takes img -> returns wrapped onehot
function wrap_with_walls(data)
    # 8x8 -> 10x10 padded with walls
    flat = data
    # offset = cat([OffsetArray(flat[:,:,i],2:9,2:9) for i in 1:size(flat,3)]...,dims=3)
    frame = ones(size(data,1) + 2, size(data,1) + 2, size(flat,3))
    frame[2:(size(data,1) + 1),2:(size(data,1) + 1),:] = flat
    return img2onehot(frame) 
end

# Saving labels 
# filename = "julia_mazes_16_test_labels.csv" 
# for i in 0:2
#     ix1 = i * 1000 + 1
#     ix2 = ix1 + 999
#     lbls = reshape(cpu(test_labels[ix1:ix2]),1000,1)
#     CSV.write(filename, DataFrame(lbls), writeheader=false, append=true)
#     # println(ix1, ix2)
# end

function admissibility_check(maze, model)
    gt_map = get_real_heur_scaled(maze)
    # learned_map = get_euclid_heur_scaled(maze)
    learned_map = get_monotonic_att_scaled(maze, model)
    # learned_map = get_monotonic_att_sigmoid(maze, model)

    diff_map = gt_map .- learned_map
    @show(length(diff_map[diff_map .< 0]))

    h1 = plot(heatmap(gt_map, c=:lightrainbow), aspect_ratio = 1)
    h2 = plot(heatmap(learned_map, c=:lightrainbow), aspect_ratio = 1)
    plot(h1, h2, layout=(1,2))
end

# Resnet utils

# Compare "learned" walls to original walls in the data
check_walls(model,x,y) = check_walls(model(x),y)
function check_walls(mx,y)
    # m_walls = Tracker.data(mx[:,:,1,:])
    m_walls = mx[:,:,1,:]
    # label_walls = Tracker.data(y[:,:,1,:])
    label_walls = y[:,:,1,:]
    diff = abs.(label_walls .- m_walls)
    max_diffs = maximum(diff, dims=[1,2])
    max_wall_diff = maximum(max_diffs)
    max_wall_diff_ix = argmax(max_diffs)
    # @show(max_wall_diff, max_wall_diff_ix)
    return max_wall_diff, max_wall_diff_ix
end

function can_step(maze, coords)
    if (coords[1] > 0 && coords[1] <= size(maze)[1]) && (coords[2] > 0 && coords[2] <= size(maze)[1])
        if maze[coords[1],coords[2]] == 0
            return true
        else
            return false
        end
    end
    return false
end

# I need the initial data to compute this
function check_feasible_steps(x,mx,y)
    # mx = Tracker.data(mx) |> gpu
    min_correct_step = zeros(size(mx)[4]) |> gpu
    max_wrong_step = zeros(size(mx)[4]) |> gpu
    for i in (1:size(y)[4])
        maze = mx[:,:,3,i]
        agent_layer = x[:,:,3,i] 
        corr_prob_mask = zeros(size(agent_layer)) |> gpu
        agent_ix = CartesianIndices(agent_layer)[agent_layer .== 1] 
        for a_ix in agent_ix
            neighs = [(a_ix[1] + 1,a_ix[2]),(a_ix[1] - 1,a_ix[2]),
                (a_ix[1],a_ix[2] + 1),(a_ix[1],a_ix[2] - 1)]
            for n in neighs
                # if can_step(x[:,:,2,i],n) || can_step(x[:,:,4,i],n)
                if can_step(x[:,:,1,i],n)
                    corr_prob_mask[n[1],n[2]] = 1.
                    # print(n)
                end
            end
        end
        corr_step_probs = corr_prob_mask .* maze 
        zero_coords = CartesianIndices(corr_step_probs)[corr_step_probs .== 0] 
        for c in zero_coords
            corr_step_probs[c] .= 10
        end
        wr_prob_mask = (corr_prob_mask .* -1) .+ 1 
        wr_step_probs = wr_prob_mask .* maze
        min_correct_step[i] = minimum(corr_step_probs)
        max_wrong_step[i] = maximum(wr_step_probs)
    end
    # @show(minimum(min_correct_step), maximum(max_wrong_step))
    # @show(argmin(min_correct_step),argmax(max_wrong_step))
    return minimum(min_correct_step), maximum(max_wrong_step), 
        argmin(min_correct_step), argmax(max_wrong_step)
end

# When encoding any maze for expansion network -> free space on goal 
maze_cells = Dict(1 => [1,0,0], 0 => [0,1,0], 2 => [0,0,1], 3 => [0,1,0])
function no_goal_img2onehot(x)
    o = zeros(Float32, size(x)..., 3)
    for ii in CartesianIndices(x)
        o[ii,:,:] .= maze_cells[x[ii]]
    end
    o
end

function onehot2no_goal_img(x)
    xx = mapslices(Flux.onecold, x, dims = 3)
    xx[ xx .== 3] .= 2  
    xx[ xx .== 2] .= 0
    xx[ xx .== 1] .= 1
    dropdims(xx, dims = 3)
end

# When encoding Sokoban for expansion network -> free space on goal 
cells = Dict(2 => [1,0,0,0], 1 => [0,1,0,0], 3 => [0,0,1,0], 4 => [0,1,0,0], 5 => [0,0,0,1])
function no_goal_sokoban2onehot(x)
    o = zeros(Float32, size(x)..., 4)
    for ii in CartesianIndices(x)
        o[ii,:,:,:] .= cells[x[ii]]
    end
    o
end

function onehot2no_goal_sokoban(x)
    xx = mapslices(Flux.onecold, x, dims = 3)
    ixs = CartesianIndices(xx)[xx .== 4]
    if length(ixs) > 0
        xx[ xx .== 4] .= 2 # goal to free space  
    end
    xx[ xx .== 2] .= 0
    xx[ xx .== 1] .= 2
    xx[ xx .== 0] .= 1
    dropdims(xx, dims = 3)
end

# Encoding sokoban for heur network
# floor = 1; wall = 2; agent = 3; goal = 4; box = 5
s_cells = Dict(2 => [1,0,0,0,0], 1 => [0,1,0,0,0], 3 => [0,0,1,0,0], 4 => [0,0,0,1,0], 5 => [0,0,0,0,1])
function sokoban2onehot(x)
    o = zeros(Float32, size(x)..., 5)
    for ii in CartesianIndices(x)
        o[ii,:,:,:,:] .= s_cells[x[ii]]
    end
    o
end

function onehot2sokoban(x)
    xx = mapslices(Flux.onecold, x, dims = 3)
    xx[ xx .== 2] .= 0
    xx[ xx .== 1] .= 2
    xx[ xx .== 0] .= 1
    xx[ xx .== 3] .= 3
    xx[ xx .== 4] .= 4
    xx[ xx .== 5] .= 5
    dropdims(xx, dims = 3)
end

function get_neighbors(map)
    # println("Get neighbors")
    # agent_ix = CartesianIndex(3,3)
    agent_ix = CartesianIndices(map)[map .== 3][1]
    n_d = [agent_ix[1] + 1,agent_ix[2]]
    n_u = [agent_ix[1] - 1,agent_ix[2]]
    n_r = [agent_ix[1],agent_ix[2] + 1]
    n_l = [agent_ix[1],agent_ix[2] - 1]
    
    ns = []

    if n_d[1] > 0 && n_d[1] <= size(map,1) && n_d[2] > 0 && n_d[2] <= size(map,2) 
        if map[n_d[1], n_d[2]] == 1 
            new_map = deepcopy(map)
            new_map[agent_ix] = 1
            new_map[n_d[1], n_d[2]] = 3
            push!(ns, new_map)
        elseif map[n_d[1], n_d[2]] == 4 && n_d[1] + 1 > 0 && n_d[1] + 1 <= size(map,1)
            if map[n_d[1] + 1, n_d[2]] .== 1
                new_map = deepcopy(map)
                new_map[agent_ix] = 1
                new_map[n_d[1], n_d[2]] = 3
                new_map[n_d[1] + 1, n_d[2]] = 4
                push!(ns, new_map)
            end
        end
    end

    if n_u[1] > 0 && n_u[1] <= size(map,1) && n_u[2] > 0 && n_u[2] <= size(map,2) 
        if map[n_u[1], n_u[2]] == 1 
            new_map = deepcopy(map)
            new_map[agent_ix] = 1
            new_map[n_u[1], n_u[2]] = 3
            push!(ns, new_map)
        elseif map[n_u[1], n_u[2]] == 4 && n_u[1] - 1 > 0 && n_u[1] - 1 <= size(map,1)
            if map[n_u[1] - 1, n_u[2]] .== 1
                new_map = deepcopy(map)
                new_map[agent_ix] = 1
                new_map[n_u[1], n_u[2]] = 3
                new_map[n_u[1] - 1, n_u[2]] = 4
                push!(ns, new_map)
            end
        end
    end

    if n_r[1] > 0 && n_r[1] <= size(map,1) && n_r[2] > 0 && n_r[2] <= size(map,2) 
        if map[n_r[1], n_r[2]] == 1 
            new_map = deepcopy(map)
            new_map[agent_ix] = 1
            new_map[n_r[1], n_r[2]] = 3
            push!(ns, new_map)
        elseif map[n_r[1], n_r[2]] == 4 && n_r[2] + 1 > 0 && n_r[2] + 1 <= size(map,1)
            if map[n_r[1], n_r[2] + 1] .== 1 
                new_map = deepcopy(map)
                new_map[agent_ix] = 1
                new_map[n_r[1], n_r[2]] = 3
                new_map[n_r[1], n_r[2] + 1] = 4
                push!(ns, new_map)
            end
        end
    end
    
    if n_l[1] > 0 && n_l[1] <= size(map,1) && n_l[2] > 0 && n_l[2] <= size(map,2) 
        if map[n_l[1], n_l[2]] == 1 
            new_map = deepcopy(map)
            new_map[agent_ix] = 1
            new_map[n_l[1], n_l[2]] = 3
            push!(ns, new_map)
        elseif map[n_l[1], n_l[2]] == 4 && n_l[2] - 1 > 0 && n_l[2] - 1 <= size(map,1)
            if map[n_l[1], n_l[2] - 1] .== 1
                new_map = deepcopy(map)
                new_map[agent_ix] = 1
                new_map[n_l[1], n_l[2]] = 3
                new_map[n_l[1], n_l[2] - 1] = 4
                push!(ns, new_map)
            end
        end
    end
    return ns
end

# Needs sokoban_all_step_generator.jl
function check_feasible_step_sokoban(x,mx)
    # mx = Tracker.data(mx) |> gpu
    min_correct_step = zeros(size(mx)[4]) |> gpu
    min_correct_box = zeros(size(mx)[4]) |> gpu
    max_wrong_step = zeros(size(mx)[4]) |> gpu
    max_wrong_box = zeros(size(mx)[4]) |> gpu

    for i in 1:size(x,4)
        oh = deepcopy(x[:,:,:,i])
        for j in 1:size(oh,3)
            layer = oh[:,:,j]
            ixs = CartesianIndices(layer)[layer .== 1]
            if length(ixs) > 0
                if j == 1
                    layer[ixs] .= 2
                elseif j == 2 
                    layer[ixs] .= 1
                else 
                    layer[ixs] .= j
                end 
                oh[:,:,j] = layer
            end
        end
        map = sum(oh, dims=3)
        ns = get_neighbors(map)

        corr_prob_mask = zeros(size(map,1),size(map,2)) |> gpu
        corr_box_mask = zeros(size(map,1),size(map,2)) |> gpu
        for n in ns
            ag_ix = CartesianIndices(n)[n .== 3][1]
            box_ixs = CartesianIndices(n)[n .== 4]
            corr_prob_mask[ag_ix] = 1
            if length(box_ixs) > 0
                corr_box_mask[box_ixs] .= 1
            end
        end
        corr_step_probs = corr_prob_mask .* mx[:,:,3,i]
        corr_step_probs[corr_step_probs .== 0] .= 10
        min_correct_step[i] = minimum(corr_step_probs)   
        
        corr_box_probs = corr_box_mask .* mx[:,:,4,i]
        corr_box_probs[corr_box_probs .== 0] .= 10
        min_correct_box[i] = minimum(corr_box_probs)

        wr_prob_mask = (corr_prob_mask .* - 1) .+ 1 
        wr_step_probs = wr_prob_mask .* mx[:,:,3,i]
        max_wrong_step[i] = maximum(wr_step_probs)

        wr_box_mask = (corr_box_mask .* -1) .+ 1
        wr_box_probs = wr_box_mask .* mx[:,:,4,i]
        max_wrong_box[i] = maximum(wr_box_probs)
    end
    # return min_correct_step, min_correct_box, max_wrong_step, max_wrong_box
    return minimum(min_correct_step), argmin(min_correct_step), minimum(min_correct_box), argmin(min_correct_box),
        maximum(max_wrong_step), argmax(max_wrong_step), maximum(max_wrong_box), argmax(max_wrong_box)
end

# Functions for sequence distance checking -> instead of visualizing the heur values 
function sequence_dist_check_sokoban(path, model)
    goal_coords = CartesianIndices(path[1])[path[1] .== 4]
    tmp_state = State(CartesianIndices(path[1])[path[1] .== 3], CartesianIndices(path[1])[path[1] .== 5], path[1])

    model_vals = []
    for s in path 
        push!(model_vals, Tracker.data(cpu(model(reshape(cu(sokoban2onehot(s)), size(s,1), size(s,2), 5, 1))))[1])
    end
    path_order = reverse([0:1:length(path)-1;])
    model_order = sort(collect(zip(model_vals, path_order)))

    no_inversions = 0
    for i in 1:length(model_order) - 1
        for j in i + 1:length(model_order)
            if model_order[i][2] > model_order[j][2]    
                no_inversions += 1
            end
        end
    end

    return no_inversions/(length(model_order)*(length(model_order) - 1)/2), no_inversions
end

function sequence_dist_check_maze(path, model)
    model_vals = []
    for s in path 
        push!(model_vals, Tracker.data(model(reshape(cu(img2onehot(cpu(s))), size(s,1), size(s,2), 4, 1)))[1])
    end
    path_order = reverse([0:1:length(path)-1;])
    model_order = sort(collect(zip(model_vals, path_order)))

    no_inversions = 0
    for i in 1:length(model_order) - 1
        for j in i + 1:length(model_order)
            if model_order[i][2] > model_order[j][2]
                no_inversions += 1
            end
        end
    end

    return no_inversions/(length(model_order)*(length(model_order) - 1)/2), no_inversions
end