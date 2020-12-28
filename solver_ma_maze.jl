using DataStructures, Distances, OhMyREPL, BSON, Flux, CuArrays, CUDAnative, Combinatorics
using DataStructures: PriorityQueue
using BSON: @load

include("utils_heur.jl")
include("network_stats_demo.jl")
include("heur_att_load.jl")

# dev_no = 0
# CUDAnative.device!(dev_no) 

struct State
    maze
end

function can_step(maze, coords)
    if (coords[1] > 0 && coords[1] <= size(maze)[1]) && (coords[2] > 0 && coords[2] <= size(maze)[1])
        if maze[coords[1],coords[2]] != 1  
            return true
        else
            return false
        end
    end
    return false
end

function euclidean_heur(maze, goal_coords)
    agent_coords = CartesianIndices(maze)[maze .== 2]
    dists = Dict()
    for a in agent_coords
        dists[a] = []
        for g in goal_coords
            d = Distances.euclidean([a[1],a[2]],[g[1],g[2]])
            push!(dists[a], d)
        end
    end
    perms = collect(permutations([1:1:length(dists[agent_coords[1]]);]))
    min_sum = Inf
    for p in perms
        s = 0
        p_ix = 1
        for a in agent_coords
            s += dists[a][p[p_ix]]
            p_ix += 1
        end
        if s < min_sum
            min_sum = s
        end
    end
    return min_sum
end

function att_net_heur(maze, model, atts, apply_atts, apply_atts_wrap)
    if apply_atts_wrap == Nothing
       m = reshape(img2onehot(maze), size(maze,1), size(maze,2), :, 1)
       value = model(cu(m))
    else
        m = reshape(wrap_with_walls(maze), size(maze,1) + 2, size(maze,2) + 2, : ,1)
        value = model(cu(m))
    end
    return value
end

function hff_goal_check(maze, goal_coords)
    agent_coords = sort(CartesianIndices(maze)[maze .== 2])
    g_in = 0
    for g in goal_coords
        if g in agent_coords
            g_in += 1
        end
    end
    return length(goal_coords) == g_in
end

function hff_get_succ_state(maze)
    next_state = deepcopy(maze)
    agent_ixs = CartesianIndices(maze)[maze .== 2]
    for agent_ix in agent_ixs
        neighs = [[agent_ix[1] + 1,agent_ix[2]],[agent_ix[1] - 1,agent_ix[2]],
                [agent_ix[1],agent_ix[2] + 1],[agent_ix[1],agent_ix[2] - 1]]
        for n in neighs
            if can_step(maze, n)
                next_state[n[1],n[2]] = 2
            end
        end
    end
    return next_state
end

function hff_get_goal_maze(maze)
    goal = deepcopy(maze)
    goal_coords = CartesianIndices(goal)[goal .== 3]
    agent_coords = CartesianIndices(goal)[goal .== 2]
    goal[goal_coords] .= 2
    goal[agent_coords] .= 0
    return goal
end

function hff_find_creating_action(maze, agent_ixs)
    actions = []
    for agent_ix in agent_ixs
        ns = [[agent_ix[1] + 1,agent_ix[2]],[agent_ix[1] - 1,agent_ix[2]],
                    [agent_ix[1],agent_ix[2] + 1],[agent_ix[1],agent_ix[2] - 1]]
        valid = []
        for n in ns 
            if (n[1] > 0 && n[1] <= size(maze)[1]) && (n[2] > 0 && n[2] <= size(maze)[1]) && 
                maze[n[1],n[2]] == 2 
                push!(valid, n)
            end
        end
        push!(actions, valid)
    end
    return actions
end

function hff_heur(maze, goal_coords)
    s_arr = []
    s0 = deepcopy(maze)
    push!(s_arr, s0)
    s_curr = s0

    while !hff_goal_check(s_curr, goal_coords)
        s_next = hff_get_succ_state(s_curr)
        if s_curr == s_next
            return false
        end
        push!(s_arr, s_next)
        s_curr = s_next
    end       
    agent_on_goal = []
    for ix in goal_coords
        if s_curr[ix] == 2
            push!(agent_on_goal, ix)
        end
    end
    agent_ix = deepcopy(goal_coords)

    goal_maze = hff_get_goal_maze(maze)
    s_arr = reverse(s_arr)
    path_len = 0
    for i in 1:length(s_arr) - 1
        possible_actions = hff_find_creating_action(s_arr[i], agent_ix)
        a_ix = 1
        new_agent_ix = []
        for agent in agent_ix
            action = possible_actions[a_ix][rand(1:1:length(possible_actions[a_ix]))]
            a_ix += 1
            if CartesianIndex(agent[1],agent[2]) in goal_coords
                goal_maze[agent[1],agent[2]] = 3
            else 
                goal_maze[agent[1],agent[2]] = 0
            end
            push!(new_agent_ix, action)
            goal_maze[action[1],action[2]] = 2
        end
        agent_ix = new_agent_ix
        path_len += 1
    end
    return path_len
end

function h_max_heur(maze, goal_coords, edges)
    agent_coords = sort(CartesianIndices(maze)[maze .== 2])
    walls = CartesianIndices(maze)[maze .== 1]
    C = []
    deltas = Dict()
    deltas[agent_coords] = 0
    pq = PriorityQueue()
    enqueue!(pq, agent_coords, 0)

    while !(goal_coords in C) && (length(pq) > 0)
        node = dequeue!(pq)

        if node in C
            continue
        end
        push!(C, node)

        empty_maze = deepcopy(maze)
        empty_maze[agent_coords] .= 0 
        empty_maze[node] .= 2

        ns = get_neighbors(empty_maze, goal_coords)
        for n in ns 
            m = n.maze
            agents = CartesianIndices(m)[m .== 2]
            cost = 1 
            if length(edges) > 0
                for e in edges 
                    if e.from.pos == node && e.to.pos == agents
                        cost = e.cost
                        break
                    end
                end
            end

            if haskey(deltas, agents)
                deltas[agents] = min(deltas[node] + cost, deltas[agents])
            else
                deltas[agents] = deltas[node] + cost
            end
            
            if haskey(pq, agents) 
                pq[agents] = deltas[agents]
            else
                enqueue!(pq, agents, deltas[agents])
            end
        end
    end
    if !haskey(deltas, goal_coords)
        return Inf
    else
        return deltas[goal_coords]
    end
end

mutable struct JG_node
    pos
    creating_edges
    out_edges
end

mutable struct JG_edge
    from
    to
    cost
end

function create_JG_graph(maze, goal_coords)
    coord_to_node = Dict()
    possible_coords = CartesianIndices(maze)[maze .!== 1]
    possible_combs = sort(collect(combinations(possible_coords, 2)))
    nodes = []
    edges = []
    for c in possible_combs
        node = JG_node(c, [], [])
        push!(nodes, node)
        coord_to_node[c] = node
    end
    init_node = JG_node(CartesianIndex(-1, -1), [], [])
    s_node = coord_to_node[CartesianIndices(maze)[maze .== 2]]
    a_init = JG_edge(init_node, s_node, 0)
    push!(init_node.out_edges, a_init)
    push!(s_node.creating_edges, a_init)
    push!(edges, a_init)

    goal_node = JG_node(CartesianIndex(0, 0), [], [])
    g_node = coord_to_node[goal_coords]
    a_goal = JG_edge(g_node, goal_node, 0)
    push!(g_node.out_edges, a_goal)
    push!(goal_node.creating_edges, a_goal)
    push!(edges, a_goal)

    empty_maze = deepcopy(maze)
    empty_maze[CartesianIndices(maze)[maze .== 2]] .= 0
    
    for node in nodes
        tmp_maze = deepcopy(empty_maze)
        tmp_maze[node.pos] .= 2
        ns = get_neighbors(tmp_maze, goal_coords)
        for n in ns
            agent = CartesianIndices(n.maze)[n.maze .== 2]
            next_node = coord_to_node[agent]
            action = JG_edge(node, next_node, 1)
            push!(edges, action)
            push!(node.out_edges, action)
            push!(next_node.creating_edges, action)
        end
    end
    return coord_to_node, nodes, edges, init_node, goal_node
end

function lm_cut_heur(maze, goal_coords)
    lm_cut = 0      
    coord_to_node, nodes, edges, init_node, goal_node = create_JG_graph(maze, goal_coords)

    while h_max_heur(maze, goal_coords, edges) != 0
        N_zero = []
        q = Queue{JG_node}()
        enqueue!(q, init_node)
        push!(N_zero, init_node)
        while length(q) > 0
            node = dequeue!(q)
            for e in node.out_edges
                if e.cost == 0
                    enqueue!(q, e.to)
                    push!(N_zero, e.to)
                end
            end
        end

        N_star = []
        q = Queue{JG_node}()
        enqueue!(q, goal_node)
        push!(N_star, goal_node)
        while length(q) > 0
            node = dequeue!(q)
            for e in node.creating_edges
                if e.cost == 0
                    enqueue!(q, e.from)
                    push!(N_star, e.from)
                end
            end
        end

        N_b = []
        for n in nodes
            if !(n in N_zero) && !(n in N_star)
                push!(N_b, n)
            end
        end

        landmark = []
        min_cost = Inf
        min_cost_lm = Nothing
        for e in edges
            if e.from in N_zero
                if e.to in N_star || e.to in N_b
                    push!(landmark, e)
                    if e.cost < min_cost
                        min_cost = e.cost
                        min_cost_lm = e
                    end
                end
            end
        end

        lm_cut += min_cost
        for e in landmark
            e.cost = e.cost - min_cost
        end
    end 
    return lm_cut
end

function mac_net_heur(maze, model, coords) 
    if coords
        tmp_maze = img2onehot(maze) |> gpu
        tmp_maze2 = add_coord_filters(reshape(tmp_maze, size(tmp_maze,1),size(tmp_maze,2),size(tmp_maze,3),1))
    else
        tmp_maze = img2onehot(maze) |> gpu
        tmp_maze2 = reshape(tmp_maze, size(tmp_maze,1),size(tmp_maze,2),size(tmp_maze,3),1)
    end    
    result = model(tmp_maze2)
    return cpu(result[1])
end

function compute_heur(heur_type, maze, heur_network, goal_coords)
    if heur_type == "euclidean"
        return euclidean_heur(maze, goal_coords)
    end

    if heur_type == "neural" 
        # MAC network
        if length(heur_network) == 2
            return value = mac_net_heur(maze, heur_network[1], heur_network[2])
        # Attention network
        elseif length(heur_network) == 4
            value = att_net_heur(maze, heur_network[1], heur_network[2], heur_network[3], heur_network[4])
            return value[1]
        end
    end

    if heur_type == "hff"
        return hff_heur(maze, goal_coords)
    end

    if heur_type == "lmcut"
        return lm_cut_heur(maze, goal_coords)
    end

    if heur_type == "none"
        return 0
    end
end

function is_goal(maze, goal_coords)
    agent_coords = CartesianIndices(maze)[maze .== 2]
    return sort(agent_coords) == sort(goal_coords)
end

# Get sorted array of arrays with coordinates and look for duplicate
function find_dupe(positions)
    for i in 1:length(positions) - 1
        if positions[i] == positions[i + 1]
            return true
        end
    end
    return false
end

function find_switch(maze, position)
    agent_maze = maze[:,:,3]
    agent_coords = CartesianIndices(agent_maze)[agent_maze .== 1]
    curr_positions = []
    for ac in agent_coords
        pos = [getindex(ac,1), getindex(ac,2)]
        push!(curr_positions, pos)
    end
    curr_positions = tuple(curr_positions...)

    agent_combs = collect(combinations([1:1:length(curr_positions);],2))
    # Check switching the position for every 2 agents
    for c in agent_combs
        if curr_positions[c[1]] == position[c[2]] && curr_positions[c[2]] == position[c[1]]
            return true
        end
    end
    return false
end

# Return only valid positions, input = tuples of arrays [([x1,y1],[x2,y2],[x3,y3]...), ...]
function filter_positions(maze, positions)
    valid = []
    for p in positions
        sorted_pos = sort(collect(p))  
        if !find_dupe(sorted_pos) && !find_switch(maze, p)
            push!(valid, p)
        end
    end
    return valid
end

function get_neighbors(maze, goal_coords)
    neighbor_states = []
    agent_ixs = CartesianIndices(maze)[maze .== 2]
    ns = []
    for agent_ix in agent_ixs
        valid = []
        neighs = [[agent_ix[1] + 1,agent_ix[2]],[agent_ix[1] - 1,agent_ix[2]],
                [agent_ix[1],agent_ix[2] + 1],[agent_ix[1],agent_ix[2] - 1]]
        for n in neighs
            if can_step(maze, n)
                push!(valid, n)
            end
        end
        push!(ns, valid)
    end

    arr = []
    for n in ns 
        tmp = []
        for i in n
            push!(tmp, CartesianIndex(i...))
        end
        push!(arr, sort(tmp))
    end

    all_prods = filter_positions(img2onehot(maze), reshape(collect(Base.product(arr...)),:,1)) 
    neighbor_states = []
    for p in all_prods
        new_maze = deepcopy(maze)
        new_maze[agent_ixs] .= 0
        new_maze[goal_coords] .= 3
        for position in p
            new_maze[position[1],position[2]] = 2
        end
        new_state = State(new_maze)
        push!(neighbor_states, new_state)
    end
    return neighbor_states
end

function get_neighbors_nn(maze, model, composer, goal_ix)
    agent_ix = CartesianIndices(maze)[maze .== 2]

    new_positions = []
    for agent in agent_ix
        tmp_m = deepcopy(maze)
        tmp_m[agent_ix] .= 0
        tmp_m[goal_ix] .= 3
        tmp_m[agent] .= 2
        m = reshape(no_goal_img2onehot(tmp_m), size(maze,1), size(maze,2), 3, 1)
        r = softmax(resnet(model, composer, cu(m)), dims=3)

        argmaxes = zeros(size(m,1),size(m,2))
        agent_th = 0.15
        for i in 1:size(m,1)
            for j in 1:size(m,2)
                vec = r[i,j,:,1]
                argmaxes[i,j] = argmax(vec)
                if argmax(vec) == 2 && vec[3] > agent_th
                    argmaxes[i,j] = 3
                end
            end
        end
        new_pos = CartesianIndices(argmaxes)[argmaxes .== 3]
        p = []
        for pos in new_pos
            push!(p, pos)
        end
        push!(new_positions, sort(p))
    end

    combs = filter_positions(img2onehot(maze), collect(Base.product(new_positions...)))

    ret_states = []    
    tmp_maze = deepcopy(maze)
    tmp_maze[agent_ix] .= 0
    tmp_maze[goal_ix] .= 3
    for comb in combs
        new_maze = deepcopy(tmp_maze)
        for c in comb
            new_maze[c] .= 2
        end
        new_state = State(new_maze)
        push!(ret_states, new_state)
    end

    if length(ret_states) > 16
        println("Too many successor states")
    elseif length(ret_states) < 1
        println("No successor states found")
    end

    return ret_states
end

function reconstruct_path(parents, goal_state)
    path = []
    # println("Goal state: ", goal_state)
    push!(path, goal_state.maze)
    current = parents[goal_state]

    while current != 0
        push!(path, current.maze)
        current = parents[current]
    end
    path = reverse(path)
    return path
end

function load_nn()
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

    return model, composer
end

function resnet(model,composer,x)
    composer(cat(x, model(x), dims = 3))
end

function print_pq(pq)
    println("Priority queue:")
    for item in pq
        println(item)
    end
    println("-----------------------------------------------------")
end

# nxn maze to run, find neighbors with neural network, type of heuristic ["none", "euclidean", "neural", "hff"]
function gbfs(start_maze, nn_neighbors, heur_type, heur_network)
    goal_coords = CartesianIndices(start_maze)[start_maze .== 3]

    if nn_neighbors
        # Load neural network to make steps 
        model, composer = load_nn()
    end
    f_values = Dict()
    parents = Dict()
    cl = Set()
    pq = PriorityQueue()

    start_state = State(start_maze)
    pq[start_state] = compute_heur(heur_type, start_state.maze, heur_network, goal_coords)
    f_values[start_state] = compute_heur(heur_type, start_state.maze, heur_network, goal_coords)
    parents[start_state] = 0

    goal_node = Nothing
    expanded_nodes = 0
    while length(pq) > 0
        node = dequeue!(pq)

        if node.maze in cl
            continue
        end
        push!(cl, node.maze)

        if is_goal(node.maze, goal_coords)
            # println("Goal found")
            goal_node = node
            break
        end
        expanded_nodes += 1

        ns = Nothing
        if nn_neighbors
            ns = get_neighbors_nn(node.maze, model, composer, goal_coords)
            # for n in ns
            #     println(n)
            # end
        else
            ns = get_neighbors(node.maze, goal_coords)
            # ns_nn = get_neighbors_nn(node.maze, model, composer, goal_coords)
            # if length(ns) != length(ns_nn)
                # println("Different length: ", length(ns), " ", length(ns_nn))
            # end
        end

        for neighbor in ns
            f_value = compute_heur(heur_type, neighbor.maze, heur_network, goal_coords)

            if f_value != Inf
                enqueue!(pq, neighbor, f_value)
                parents[neighbor] = node
            end
        end
    end

    if goal_node == Nothing
        return [], Inf, expanded_nodes
    end

    path = reconstruct_path(parents, goal_node)
    return (path, length(path) - 1, expanded_nodes)
end

function bfs(start_maze, nn_neighbors, heur_type, heur_network)
    goal_coords = CartesianIndices(start_maze)[start_maze .== 3]

    if nn_neighbors
        # Load neural network to make steps 
        model, composer = load_nn()
    end

    g_values = Dict()
    f_values = Dict()
    parents = Dict()
    cl = Set()
    pq = PriorityQueue()

    start_state = State(start_maze)
    pq[start_state] = compute_heur(heur_type, start_state.maze, heur_network, goal_coords)
    g_values[start_state] = 0
    f_values[start_state] = compute_heur(heur_type, start_state.maze, heur_network, goal_coords)
    parents[start_state] = 0

    goal_node = Nothing
    expanded_nodes = 0
    while length(pq) > 0
        node = dequeue!(pq)
        expanded_nodes += 1

        if is_goal(node.maze, goal_coords)
            goal_node = node
            break
        end

        if node.maze in cl
            continue
        end
        push!(cl, node.maze)

        ns = Nothing
        if nn_neighbors
            ns = get_neighbors_nn(node.maze, model, composer, goal_coords)
        else
            ns = get_neighbors(node.maze, goal_coords)
        end

        for neighbor in ns
            if neighbor.maze in cl
                continue
            end
            g_value = g_values[node] + 1
            f_value = g_value + compute_heur(heur_type, neighbor.maze, heur_network, goal_coords)

            if !(neighbor in keys(g_values)) || g_value < g_values[neighbor]
                parents[neighbor] = node
                g_values[neighbor] = g_value
                f_values[neighbor] = f_value
                enqueue!(pq, neighbor, f_value)
            elseif g_value >= g_values[neighbor]
                continue
            end
        end
    end

    if goal_node == Nothing
        return [], Inf, expanded_nodes
    end

    path = reconstruct_path(parents, goal_node)
    return (path, length(path) - 1, expanded_nodes)
end

# Multi-heuristic GBFS
function mh_gbfs(start_maze, nn_neighbors, heur_type, heur_network)
    goal_coords = CartesianIndices(start_maze)[start_maze .== 3]

    if nn_neighbors
        # Load neural network to make steps 
        model, composer = load_nn()
    end
    f_values = Dict()
    parents = Dict()
    cl = Set()
    pq = PriorityQueue()

    start_state = State(start_maze)

    heur_vec = []
    for heur in heur_type
        h = compute_heur(heur, start_state.maze, heur_network, goal_coords)
        push!(heur_vec, h)
    end

    pq[start_state] = heur_vec
    f_values[start_state] = heur_vec
    parents[start_state] = 0

    goal_node = Nothing
    expanded_nodes = 0
    while length(pq) > 0
        node = dequeue!(pq)

        if node.maze in cl
            continue
        end
        push!(cl, node.maze)

        if is_goal(node.maze, goal_coords)
            # println("Goal found")
            goal_node = node
            break
        end
        expanded_nodes += 1

        ns = Nothing
        if nn_neighbors
            ns = get_neighbors_nn(node.maze, model, composer, goal_coords)
        else
            ns = get_neighbors(node.maze, goal_coords)
        end

        for neighbor in ns
            heur_vec = []
            for heur in heur_type
                h = compute_heur(heur, neighbor.maze, heur_network, goal_coords)
                push!(heur_vec, h)
            end

            f_value = heur_vec

            if f_value[1] != Inf
                enqueue!(pq, neighbor, f_value)
                parents[neighbor] = node
            end
        end
    end

    if goal_node == Nothing
        return [], Inf, expanded_nodes
    end

    path = reconstruct_path(parents, goal_node)
    return (path, length(path) - 1, expanded_nodes)
end

# Timed versions of the state-space searches ------------------------------------------
# max_time - maximum duration of search in seconds 
function gbfs_timed(start_maze, nn_neighbors, heur_type, heur_network, max_time)
    goal_coords = CartesianIndices(start_maze)[start_maze .== 3]

    # if nn_neighbors
        # Load neural network to make steps 
        model, composer = load_nn()
    # end
    f_values = Dict()
    parents = Dict()
    cl = Set()
    pq = PriorityQueue()

    start_state = State(start_maze)
    pq[start_state] = compute_heur(heur_type, start_state.maze, heur_network, goal_coords)
    f_values[start_state] = compute_heur(heur_type, start_state.maze, heur_network, goal_coords)
    parents[start_state] = 0

    goal_node = Nothing
    expanded_nodes = 0
    t0 = time()
    while length(pq) > 0
        node = dequeue!(pq)

        if node.maze in cl
            continue
        end
        push!(cl, node.maze)

        if is_goal(node.maze, goal_coords)
            # println("Goal found")
            goal_node = node
            break
        end
        expanded_nodes += 1

        ns = Nothing
        if nn_neighbors
            ns = get_neighbors_nn(node.maze, model, composer, goal_coords)
            # for n in ns
            #     println(n)
            # end
        else
            ns = get_neighbors(node.maze, goal_coords)
            # ns_nn = get_neighbors_nn(node.maze, model, composer, goal_coords)
            # if length(ns) != length(ns_nn)
                # println("Different length: ", length(ns), " ", length(ns_nn))
            # end
        end

        for neighbor in ns
            f_value = compute_heur(heur_type, neighbor.maze, heur_network, goal_coords)

            if f_value != Inf
                enqueue!(pq, neighbor, f_value)
                parents[neighbor] = node
            end
        end

        # Time extended 
        t1 = time()
        if t1 - t0 > max_time
            return [], Inf, expanded_nodes
        end
    end

    if goal_node == Nothing
        return [], Inf, expanded_nodes
    end

    path = reconstruct_path(parents, goal_node)
    return (path, length(path) - 1, expanded_nodes)
end

function bfs_timed(start_maze, nn_neighbors, heur_type, heur_network, max_time)
    goal_coords = CartesianIndices(start_maze)[start_maze .== 3]

    if nn_neighbors
        # Load neural network to make steps 
        model, composer = load_nn()
    end

    g_values = Dict()
    f_values = Dict()
    parents = Dict()
    cl = Set()
    pq = PriorityQueue()

    start_state = State(start_maze)
    pq[start_state] = compute_heur(heur_type, start_state.maze, heur_network, goal_coords)
    g_values[start_state] = 0
    f_values[start_state] = compute_heur(heur_type, start_state.maze, heur_network, goal_coords)
    parents[start_state] = 0

    goal_node = Nothing
    expanded_nodes = 0
    t0 = time()
    while length(pq) > 0
        node = dequeue!(pq)
        expanded_nodes += 1

        if is_goal(node.maze, goal_coords)
            goal_node = node
            break
        end

        if node.maze in cl
            continue
        end
        push!(cl, node.maze)

        ns = Nothing
        if nn_neighbors
            ns = get_neighbors_nn(node.maze, model, composer, goal_coords)
        else
            ns = get_neighbors(node.maze, goal_coords)
        end

        for neighbor in ns
            if neighbor.maze in cl
                continue
            end
            g_value = g_values[node] + 1
            f_value = g_value + compute_heur(heur_type, neighbor.maze, heur_network, goal_coords)

            if !(neighbor in keys(g_values)) || g_value < g_values[neighbor]
                parents[neighbor] = node
                g_values[neighbor] = g_value
                f_values[neighbor] = f_value
                enqueue!(pq, neighbor, f_value)
            elseif g_value >= g_values[neighbor]
                continue
            end
        end

        t1 = time()
        if t1 - t0 > max_time
            return [], Inf, expanded_nodes
        end
    end

    if goal_node == Nothing
        return [], Inf, expanded_nodes
    end

    path = reconstruct_path(parents, goal_node)
    return (path, length(path) - 1, expanded_nodes)
end

function mh_gbfs_timed(start_maze, nn_neighbors, heur_type, heur_network, max_time)
    goal_coords = CartesianIndices(start_maze)[start_maze .== 3]

    if nn_neighbors
        # Load neural network to make steps 
        model, composer = load_nn()
    end
    f_values = Dict()
    parents = Dict()
    cl = Set()
    pq = PriorityQueue()

    start_state = State(start_maze)

    heur_vec = []
    for heur in heur_type
        h = compute_heur(heur, start_state.maze, heur_network, goal_coords)
        push!(heur_vec, h)
    end

    pq[start_state] = heur_vec
    f_values[start_state] = heur_vec
    parents[start_state] = 0

    goal_node = Nothing
    expanded_nodes = 0
    t0 = time()
    while length(pq) > 0
        node = dequeue!(pq)

        if node.maze in cl
            continue
        end
        push!(cl, node.maze)

        if is_goal(node.maze, goal_coords)
            # println("Goal found")
            goal_node = node
            break
        end
        expanded_nodes += 1

        ns = Nothing
        # ns2 = Nothing
        if nn_neighbors
            ns = get_neighbors_nn(node.maze, model, composer, goal_coords)
        else
            ns = get_neighbors(node.maze, goal_coords)
        end
        # ns = get_neighbors(node.maze, goal_coords)
        # ns2 = get_neighbors_nn(node.maze, model, composer, goal_coords)

        # if length(ns) != length(ns2)
        #     println("Different number of successors!")
        #     println(expanded_nodes)
        # end

        # mzs = []
        # mzs2 = []
        # for s in ns
        #     push!(mzs, s.maze)
        # end
        # for s in ns2
        #     push!(mzs2, s.maze)
        # end
        # for m in mzs
        #     if !(m in mzs2)
        #         println("Different successors!!")
        #         println(expanded_nodes)
        #     end
        # end

        for neighbor in ns
            heur_vec = []
            for heur in heur_type
                h = compute_heur(heur, neighbor.maze, heur_network, goal_coords)
                push!(heur_vec, h)
            end

            f_value = heur_vec

            if f_value[1] != Inf
                enqueue!(pq, neighbor, f_value)
                parents[neighbor] = node
            end
        end

        t1 = time()
        if t1 - t0 > max_time
            return [], Inf, expanded_nodes
        end
    end

    if goal_node == Nothing
        return [], Inf, expanded_nodes
    end

    path = reconstruct_path(parents, goal_node)
    return (path, length(path) - 1, expanded_nodes)
end

