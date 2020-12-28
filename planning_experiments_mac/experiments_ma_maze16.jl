using Flux, OhMyREPL, CuArrays, CUDAnative, JLD, DelimitedFiles, BSON
cd("planning_experiments_mac")
include("../utils_heur.jl")
include("../solver_ma_maze.jl")
include("../MAC_network_self_stopping3.jl")

# load data
data = load("data16_ma_maze.jld")["data"]

heur = "neural"
max_time = 600
CUDAnative.device!(1)

gbfs_pls = Dict()
gbfs_exs = Dict()
bfs_pls = Dict()
bfs_exs = Dict()
cd("..")

folders = readdir("mac_param_experiments11_ma_maze")
exp_folders_r = []
exp_folders_q = []
for i in 1:length(folders)
    if length(split(folders[i],"_")) > 6
        push!(exp_folders_r, string("mac_param_experiments11_ma_maze/",folders[i]))
    else
        push!(exp_folders_q, string("mac_param_experiments11_ma_maze/",folders[i]))
    end
end

for exp_folder in exp_folders_r
    gbfs_pls[exp_folder] = []
    gbfs_exs[exp_folder] = []
    bfs_pls[exp_folder] = []
    bfs_exs[exp_folder] = []

    push!(gbfs_pls[exp_folder], split(exp_folder,"/")[2])
    push!(gbfs_exs[exp_folder], split(exp_folder,"/")[2])

    push!(bfs_pls[exp_folder], split(exp_folder,"/")[2])
    push!(bfs_exs[exp_folder], split(exp_folder,"/")[2])

    splitted = split(exp_folder, "/")

    file_pls_gbfs = string("planning_experiments_mac/ma_tmps/tmp16_",splitted[1], "_", splitted[2],"_gbfs_pls.jld")
    file_exs_gbfs = string("planning_experiments_mac/ma_tmps/tmp16_",splitted[1], "_", splitted[2],"_gbfs_exs.jld")

    file_pls_bfs = string("planning_experiments_mac/ma_tmps/tmp16_",splitted[1], "_", splitted[2],"_bfs_pls.jld")
    file_exs_bfs = string("planning_experiments_mac/ma_tmps/tmp16_",splitted[1], "_", splitted[2],"_bfs_exs.jld")

    mac = load_mac_network(exp_folder)
    iters = parse(Int, split(exp_folder,"_")[8])

    model = Chain(
        x -> run_MAC(mac, iters, x)
    ) |> gpu

    if parse(Int, split(exp_folder,"_")[9]) == 1
        coords = true
    else
        coords = false
    end

    start_no = 1
    if isfile(file_pls_gbfs) && isfile(file_pls_bfs)
        pls_gbfs = load(file_pls_gbfs)["data"]
        exs_gbfs = load(file_exs_gbfs)["data"]
        pls_bfs = load(file_pls_bfs)["data"]
        exs_bfs = load(file_exs_bfs)["data"]

        start_no = length(pls_gbfs)

        gbfs_pls[exp_folder] = pls_gbfs
        gbfs_exs[exp_folder] = exs_gbfs
        bfs_pls[exp_folder] = pls_bfs
        bfs_exs[exp_folder] = exs_bfs
    end

    for i in start_no:size(data,4)
        maze = onehot2img(data[:,:,:,i:i])[:,:,1]     
        path, path_len, expanded_states = gbfs_timed(maze, true, "neural", [model, coords], max_time)        
        push!(gbfs_pls[exp_folder], path_len)
        push!(gbfs_exs[exp_folder], expanded_states)
        save(file_pls_gbfs, "data", gbfs_pls[exp_folder])
        save(file_exs_gbfs, "data", gbfs_exs[exp_folder])

        path, path_len, expanded_states = bfs_timed(maze, true, "neural", [model, coords], max_time)        
        push!(bfs_pls[exp_folder], path_len)
        push!(bfs_exs[exp_folder], expanded_states)
        save(file_pls_bfs, "data", bfs_pls[exp_folder])
        save(file_exs_bfs, "data", bfs_exs[exp_folder])
        println("Done ", i)
    end

    if start_no < 51
        open("planning_experiments_mac/ma_maze16_gbfs_true_stats.csv", "a") do f
            lens = transpose(gbfs_pls[exp_folder][2:end])
            exs = transpose(gbfs_exs[exp_folder][2:end])
            writedlm(f, lens)
            writedlm(f,exs)
        end
        open("planning_experiments_mac/ma_maze16_bfs_true_stats.csv", "a") do f
            lens = transpose(bfs_pls[exp_folder][2:end])
            exs = transpose(bfs_exs[exp_folder][2:end])
            writedlm(f, lens)
            writedlm(f,exs)
        end
    end
end

gbfs_pls = Dict()
gbfs_exs = Dict()
bfs_pls = Dict()
bfs_exs = Dict()

for exp_folder in exp_folders_q
    gbfs_pls[exp_folder] = []
    gbfs_exs[exp_folder] = []
    bfs_pls[exp_folder] = []
    bfs_exs[exp_folder] = []

    push!(gbfs_pls[exp_folder], split(exp_folder,"/")[2])
    push!(gbfs_exs[exp_folder], split(exp_folder,"/")[2])

    push!(bfs_pls[exp_folder], split(exp_folder,"/")[2])
    push!(bfs_exs[exp_folder], split(exp_folder,"/")[2])

    splitted = split(exp_folder, "/")

    file_pls_gbfs = string("planning_experiments_mac/ma_tmps/tmp16_",splitted[1], "_", splitted[2],"_gbfs_pls_q.jld")
    file_exs_gbfs = string("planning_experiments_mac/ma_tmps/tmp16_",splitted[1], "_", splitted[2],"_gbfs_exs_q.jld")

    file_pls_bfs = string("planning_experiments_mac/ma_tmps/tmp16_",splitted[1], "_", splitted[2],"_bfs_pls_q.jld")
    file_exs_bfs = string("planning_experiments_mac/ma_tmps/tmp16_",splitted[1], "_", splitted[2],"_bfs_exs_q.jld")

    @load string(exp_folder, "/mac_q.bson") m 
    model = m |> gpu

    if parse(Int, split(exp_folder,"_")[8]) == 1
        coords = true
    else
        coords = false
    end

    start_no = 1
    if isfile(file_pls_gbfs) && isfile(file_pls_bfs)
        pls_gbfs = load(file_pls_gbfs)["data"]
        exs_gbfs = load(file_exs_gbfs)["data"]
        pls_bfs = load(file_pls_bfs)["data"]
        exs_bfs = load(file_exs_bfs)["data"]

        start_no = length(pls_gbfs)

        gbfs_pls[exp_folder] = pls_gbfs
        gbfs_exs[exp_folder] = exs_gbfs
        bfs_pls[exp_folder] = pls_bfs
        bfs_exs[exp_folder] = exs_bfs
    end

    for i in start_no:size(data,4)
        maze = onehot2img(data[:,:,:,i:i])[:,:,1]       
        path, path_len, expanded_states = gbfs_timed(maze, true, "neural", [model,coords], max_time)        
        push!(gbfs_pls[exp_folder], path_len)
        push!(gbfs_exs[exp_folder], expanded_states)
        save(file_pls_gbfs, "data", gbfs_pls[exp_folder])
        save(file_exs_gbfs, "data", gbfs_exs[exp_folder])

        path, path_len, expanded_states = bfs_timed(maze, true, "neural", [model,coords], max_time)        
        push!(bfs_pls[exp_folder], path_len)
        push!(bfs_exs[exp_folder], expanded_states)
        save(file_pls_bfs, "data", bfs_pls[exp_folder])
        save(file_exs_bfs, "data", bfs_exs[exp_folder])
        println("Done q ", i)
    end

    if start_no < 51
        open("planning_experiments_mac/ma_maze16_gbfs_true_stats.csv", "a") do f
            lens = transpose(gbfs_pls[exp_folder][2:end])
            exs = transpose(gbfs_exs[exp_folder][2:end])
            writedlm(f, lens)
            writedlm(f,exs)
        end
        open("planning_experiments_mac/ma_maze16_bfs_true_stats.csv", "a") do f
            lens = transpose(bfs_pls[exp_folder][2:end])
            exs = transpose(bfs_exs[exp_folder][2:end])
            writedlm(f, lens)
            writedlm(f,exs)
        end
    end
end