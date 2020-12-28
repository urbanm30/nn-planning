using Flux, OhMyREPL, CuArrays, CUDAnative, JLD, DelimitedFiles
cd("planning_experiments_mac")
include("../utils_heur.jl")
include("../solver_mg_maze.jl")
include("../MAC_network_self_stopping3.jl")

# load data
data = load("data32_mg_maze.jld")["data"]

heur = "neural"
CUDAnative.device!(3)

gbfs_pls = Dict()
gbfs_exs = Dict()
bfs_pls = Dict()
bfs_exs = Dict()
cd("..")

folders = readdir("mac_param_experiments10_mg_maze")
exp_folders_r = []
exp_folders_q = []
for i in 1:length(folders)
    if length(split(folders[i],"_")) > 6
        push!(exp_folders_r, string("mac_param_experiments10_mg_maze/",folders[i]))
    else
        push!(exp_folders_q, string("mac_param_experiments10_mg_maze/",folders[i]))
    end
end

for exp_folder in exp_folders_r[11:end]
    gbfs_pls[exp_folder] = []
    gbfs_exs[exp_folder] = []
    bfs_pls[exp_folder] = []
    bfs_exs[exp_folder] = []

    push!(gbfs_pls[exp_folder], split(exp_folder,"/")[2])
    push!(gbfs_exs[exp_folder], split(exp_folder,"/")[2])

    push!(bfs_pls[exp_folder], split(exp_folder,"/")[2])
    push!(bfs_exs[exp_folder], split(exp_folder,"/")[2])

    splitted = split(exp_folder, "/")

    file_pls_gbfs = string("planning_experiments_mac/tmps/tmp32_",splitted[1], "_", splitted[2],"_gbfs_pls.jld")
    file_exs_gbfs = string("planning_experiments_mac/tmps/tmp32_",splitted[1], "_", splitted[2],"_gbfs_exs.jld")

    file_pls_bfs = string("planning_experiments_mac/tmps/tmp32_",splitted[1], "_", splitted[2],"_bfs_pls.jld")
    file_exs_bfs = string("planning_experiments_mac/tmps/tmp32_",splitted[1], "_", splitted[2],"_bfs_exs.jld")

    mac = load_mac_network(exp_folder)
    iters = parse(Int, split(exp_folder,"_")[8])
    if parse(Int, split(exp_folder,"_")[9]) == 1
        coords = true
    else
        coords = false
    end

    model = Chain(
        x -> run_MAC(mac, iters, x)
    ) |> gpu

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
        path, path_len, expanded_states = gbfs(maze, true, "neural", [model,coords])        
        push!(gbfs_pls[exp_folder], path_len)
        push!(gbfs_exs[exp_folder], expanded_states)
        save(file_pls_gbfs, "data", gbfs_pls[exp_folder])
        save(file_exs_gbfs, "data", gbfs_exs[exp_folder])

        path, path_len, expanded_states = bfs(maze, true, "neural", [model,coords])        
        push!(bfs_pls[exp_folder], path_len)
        push!(bfs_exs[exp_folder], expanded_states)
        save(file_pls_bfs, "data", bfs_pls[exp_folder])
        save(file_exs_bfs, "data", bfs_exs[exp_folder])
        println("Done ", i)
    end

    if start_no < 51
        open("planning_experiments_mac/mg_maze32_gbfs_true_stats.csv", "a") do f
            lens = transpose(gbfs_pls[exp_folder][2:end])
            exs = transpose(gbfs_exs[exp_folder][2:end])
            writedlm(f, lens)
            writedlm(f,exs)
        end
        open("planning_experiments_mac/mg_maze32_bfs_true_stats.csv", "a") do f
            lens = transpose(bfs_pls[exp_folder][2:end])
            exs = transpose(bfs_exs[exp_folder][2:end])
            writedlm(f, lens)
            writedlm(f,exs)
        end
    end
end
println("R-part done!")

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

    file_pls_gbfs = string("planning_experiments_mac/tmps/tmp32_",splitted[1], "_", splitted[2],"_gbfs_pls_q.jld")
    file_exs_gbfs = string("planning_experiments_mac/tmps/tmp32_",splitted[1], "_", splitted[2],"_gbfs_exs_q.jld")

    file_pls_bfs = string("planning_experiments_mac/tmps/tmp32_",splitted[1], "_", splitted[2],"_bfs_pls_q.jld")
    file_exs_bfs = string("planning_experiments_mac/tmps/tmp32_",splitted[1], "_", splitted[2],"_bfs_exs_q.jld")

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
        path, path_len, expanded_states = gbfs(maze, true, "neural", [model,coords])        
        push!(gbfs_pls[exp_folder], path_len)
        push!(gbfs_exs[exp_folder], expanded_states)
        save(file_pls_gbfs, "data", gbfs_pls[exp_folder])
        save(file_exs_gbfs, "data", gbfs_exs[exp_folder])

        path, path_len, expanded_states = bfs(maze, true, "neural", [model,coords])        
        push!(bfs_pls[exp_folder], path_len)
        push!(bfs_exs[exp_folder], expanded_states)
        save(file_pls_bfs, "data", bfs_pls[exp_folder])
        save(file_exs_bfs, "data", bfs_exs[exp_folder])
        println("Done q ", i)
    end

    if start_no < 51 
        open("planning_experiments_mac/mg_maze32_gbfs_true_stats.csv", "a") do f
            lens = transpose(gbfs_pls[exp_folder][2:end])
            exs = transpose(gbfs_exs[exp_folder][2:end])
            writedlm(f, lens)
            writedlm(f,exs)
        end
        open("planning_experiments_mac/mg_maze32_bfs_true_stats.csv", "a") do f
            lens = transpose(bfs_pls[exp_folder][2:end])
            exs = transpose(bfs_exs[exp_folder][2:end])
            writedlm(f, lens)
            writedlm(f,exs)
        end
    end
end

# --------------------------------------------------------
# # MH-GBFS experiments 

# gbfs_pls = Dict()
# gbfs_exs = Dict()
# for heur in heur_combs
#     gbfs_pls[heur] = []
#     gbfs_exs[heur] = []
# end

# cd("..")
# for i in 1:size(data,3)
#     for heur in heur_combs
#         println(heur, i)

#         if "neural" in heur
#             maze = ones(size(data,1) + 2, size(data,2) + 2)
#             maze[2:9,2:9] = data[:,:,i]         
#             path, path_len, expanded_states = mh_gbfs(maze, false, heur, hn)
#             push!(gbfs_pls[heur], path_len)
#             push!(gbfs_exs[heur], expanded_states)
#         else
#             path, path_len, expanded_states = mh_gbfs(data[:,:,i], false, heur, [])
#             push!(gbfs_pls[heur], path_len)
#             push!(gbfs_exs[heur], expanded_states)
#         end
#     end
# end

# cd("planning_experiments")
# for h in heur_combs
#     lens = reshape(cat(h, gbfs_pls[h],dims=1), 1, :)
#     exs = reshape(cat(h, gbfs_exs[h],dims=1), 1, :)
#     open("mg_maze_mh_gbfs_stats.csv", "a") do f
#         writedlm(f, lens)
#         writedlm(f,exs)
#     end
# end

# gbfs_pls = Dict()
# gbfs_exs = Dict()
# for heur in heur_combs
#     gbfs_pls[heur] = []
#     gbfs_exs[heur] = []
# end

# cd("..")
# for i in 1:size(data,3)
#     for heur in heur_combs
#         println(heur, i)

#         if "neural" in heur
#             maze = ones(size(data,1) + 2, size(data,2) + 2)
#             maze[2:9,2:9] = data[:,:,i]         
#             path, path_len, expanded_states = mh_gbfs(maze, true, heur, hn)
#             push!(gbfs_pls[heur], path_len)
#             push!(gbfs_exs[heur], expanded_states)
#         else
#             path, path_len, expanded_states = mh_gbfs(data[:,:,i], true, heur, [])
#             push!(gbfs_pls[heur], path_len)
#             push!(gbfs_exs[heur], expanded_states)
#         end
#     end
# end

# cd("planning_experiments")
# for h in heur_combs
#     lens = reshape(cat(h, gbfs_pls[h],dims=1), 1, :)
#     exs = reshape(cat(h, gbfs_exs[h],dims=1), 1, :)
#     open("mg_maze_mh_gbfs_stats.csv", "a") do f
#         writedlm(f, lens)
#         writedlm(f,exs)
#     end
# end