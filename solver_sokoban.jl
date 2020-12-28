using DataStructures, Distances, Combinatorics, JLD, BSON, Flux, CuArrays
using DataStructures: Queue, PriorityQueue  
using BSON: @load

include("utils_heur.jl")
include("heur_att_load.jl")

struct State 
    agent_coords
    box_coords
    map
end

mutable struct HFF_node
    pre_agent
    eff_agent
    pre_box
    eff_box
    walls
    prev_node
    next_nodes
    creating_action
    steps
    agent0
    box0
end

mutable struct HFF_action
    pre_agent
    eff_agent
    pre_box
    eff_box
end

function goal_check(state, goal_coords)
    if sort(state.box_coords) == sort(goal_coords)
        return true
    else
        return false
    end
end

function get_neighbors(node, goal_coords)
    agent_coords = node.agent_coords
    next_ixs = [
        CartesianIndex(agent_coords[1] + 1,agent_coords[2]),
        CartesianIndex(agent_coords[1] - 1,agent_coords[2]),
        CartesianIndex(agent_coords[1],agent_coords[2] + 1),
        CartesianIndex(agent_coords[1],agent_coords[2] - 1)
    ]

    ns = []
    for ix in next_ixs
        # In bounds check 
        if ix[1] > 0 && ix[1] <= size(node.map,1) && ix[2] > 0 && ix[2] <= size(node.map,2)
            # Free space or goal check => normal agent step 
            if ((ix in goal_coords) && !(ix in node.box_coords)) || node.map[ix] == 1
                new_map = deepcopy(node.map)
                new_map[agent_coords] = 1
                new_map[goal_coords] .= 4
                new_map[node.box_coords] .= 5
                new_map[ix] = 3
                new_state = State(ix,node.box_coords,new_map)
                push!(ns, new_state)
            # Box on the next cell 
            elseif ix in node.box_coords
                next_box = Nothing
                if agent_coords[1] < ix[1] 
                    next_box = CartesianIndex(ix[1] + 1, ix[2])
                elseif agent_coords[1] > ix[1]
                    next_box = CartesianIndex(ix[1] - 1, ix[2])
                elseif agent_coords[2] < ix[2]
                    next_box = CartesianIndex(ix[1], ix[2] + 1)
                elseif agent_coords[2] > ix[2]
                    next_box = CartesianIndex(ix[1], ix[2] - 1)
                end
                # If box can be pushed
                if next_box[1] > 0 && next_box[1] <= size(node.map,1) && next_box[2] > 0 && next_box[2] <= size(node.map,2)
                    if ((next_box in goal_coords) && !(next_box in node.box_coords)) || node.map[next_box] == 1
                        new_map = deepcopy(node.map)
                        new_map[agent_coords] = 1
                        new_map[node.box_coords] .= 1
                        new_map[goal_coords] .= 4 

                        new_boxes = [next_box] 
                        for box in node.box_coords
                            if box != ix
                                push!(new_boxes, box)
                            end
                        end
                        new_map[new_boxes] .= 5
                        new_map[ix] = 3
                        
                        new_state = State(ix, new_boxes, new_map)
                        push!(ns, new_state)
                    end
                end
            end
        end
    end
    return ns
end

function reconstruct_path(parents, goal_state)  
    p = []
    push!(p, goal_state.map)
    current = parents[goal_state]

    while current != 0
        push!(p, current.map)
        current = parents[current]
    end
    p = reverse(p)
    return p
end

function is_in_visited(visited, state)
    for s in visited 
        if s.map == state.map && s.agent_coords == state.agent_coords && s.box_coords == state.box_coords
            return true
        end
    end
    return false
end

function euclidean_heur(state, goal_coords)
    box_coords = state.box_coords
    dists = Dict()
    for b in box_coords
        dists[b] = []
        for g in goal_coords
            d = Distances.euclidean([b[1],b[2]],[g[1],g[2]])
            push!(dists[b], d)
        end
    end
    perms = collect(permutations([1:1:length(dists[box_coords[1]]);]))
    if length(perms) > 2
        println("Too many perms: ")
        println(state.map)
    end
    min_sum = Inf
    for p in perms
        s = 0
        p_ix = 1
        for b in box_coords
            s += dists[b][p[p_ix]]
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
       m = reshape(sokoban2onehot(maze), size(maze,1), size(maze,2), :, 1)
       value = model(cu(m))
    else
        m = reshape(wrap_with_walls(maze), size(maze,1) + 2, size(maze,2) + 2, : ,1)
        value = model(cu(m))
    end
    return value
end

function hff_goal_check(node, goal_coords)
    box_coords = node.eff_box
    goals = 0
    for g in goal_coords
        if g in box_coords
            goals += 1
        end
    end
    return goals == length(goal_coords)
end

function hff_get_succ_states(state, map)
    next_states = []
    for a in state.eff_agent
        ns = [CartesianIndex(a[1] + 1,a[2]),CartesianIndex(a[1] - 1,a[2]),
                CartesianIndex(a[1],a[2] + 1),CartesianIndex(a[1],a[2] - 1)]
        for n in ns
            if n[1] > 0 && n[1] <= size(map,1) && n[2] > 0 && n[2] <= size(map,2) 
                if !(n in state.walls) && !(n in state.eff_box) && !(n in state.eff_agent)
                    # Regular agent step
                     # Action definition 
                     pa = a
                     ea = n
                     pb = []
                     eb = []
                     action = HFF_action(pa, ea, pb, eb)
                     
                    # State definition
                    pre = state.eff_agent
                    eff = deepcopy(pre)
                    push!(eff, n)
                    new_state = HFF_node(pre,eff,state.eff_box, state.eff_box, state.walls,state,[], action, state.steps + 1, state.agent0, state.box0)

                    push!(next_states, new_state)
                elseif n in state.eff_box
                    nb = Nothing
                    if n[1] > a[1]
                        nb = CartesianIndex(n[1] + 1, n[2])
                    elseif n[1] < a[1]
                        nb = CartesianIndex(n[1] - 1, n[2])
                    elseif n[2] > a[2]
                        nb = CartesianIndex(n[1], n[2] + 1)
                    elseif n[2] < a[2]
                        nb = CartesianIndex(n[1], n[2] - 1)
                    end
                    if nb[1] > 0 && nb[1] <= size(map,1) && nb[2] > 0 && nb[2] <= size(map,2)
                        if !(nb in state.walls) && !(nb in state.eff_box) #&& !(nb in state.eff_agent)
                            # Creating action definition 
                            pa = a
                            ea = n
                            pb = n
                            eb = nb
                            action = HFF_action(pa, ea, pb, eb)

                            a_pre = state.eff_agent
                            a_eff = deepcopy(a_pre)
                            push!(a_eff, n)
                            b_pre = state.eff_box
                            b_eff = deepcopy(b_pre)
                            push!(b_eff, nb)
                            new_state = HFF_node(a_pre, a_eff, b_pre, b_eff, state.walls, state, [], action, state.steps + 1, state.agent0, state.box0)
                            
                            push!(next_states, new_state)
                        end
                    end
                end
            end
        end
    end
    return next_states
end

function hff_heur(state, goal_coords, t0, max_time)
    walls0 = CartesianIndices(state.map)[state.map .== 2]
    agents0 = [state.agent_coords]
    boxes0 = state.box_coords

    hff_graph = HFF_node([],agents0,[],boxes0,walls0,Nothing,[], Nothing, 0, agents0, boxes0)
    curr_node = hff_graph

    queue = Queue{HFF_node}()
    enqueue!(queue, curr_node)
    goal_node = Nothing

    while length(queue) > 0
        t1 = time()
        if t1 - t0 > max_time
            println("Time limit out")
            return false
        end
        curr_node = dequeue!(queue)
        # println(curr_node.eff_agent)
        # println(curr_node.eff_box)
        if hff_goal_check(curr_node, goal_coords)
            goal_node = curr_node
            break
        end
        s_next = hff_get_succ_states(curr_node, state.map)
        if length(s_next) == 0
            continue
        end
        for n in s_next
            push!(curr_node.next_nodes, n)
            enqueue!(queue, n)
        end
    end    
    
    if goal_node == Nothing
        return Inf
    end

    path_len = 0
    node = goal_node
    while node.creating_action != Nothing
        path_len += 1
        node = node.prev_node
    end

    return path_len
end

function h_max_goal_check(C, goal_coords)
    for e in C 
        if e[2] == goal_coords
            return true
        end
    end
    return false
end

function h_max_heur(state, goal_coords, coord_to_node)
    init = (state.agent_coords, sort(state.box_coords))
    C = []
    deltas = Dict()
    deltas[init] = 0
    pq = PriorityQueue()
    enqueue!(pq, init, 0)
    empty_map = deepcopy(state.map)
    empty_map[state.agent_coords] .= 1
    empty_map[state.box_coords] .= 1
    empty_map[goal_coords] .= 4
    no_steps = 0

    # breaker = Nothing
    while !h_max_goal_check(C, goal_coords)
    # for no_steps = 1:5901
        if length(pq) == 0
            break
        end
        node = dequeue!(pq)
        # global no_steps += 1
        # global breaker = node 
        
        if node in C 
            continue
        end
        push!(C, node)  

        map = deepcopy(empty_map)
        map[node[1]] .= 3
        map[node[2]] .= 5
        temp_state = State(node[1], node[2], map)
        ns = get_neighbors(temp_state, goal_coords)
        for n in ns 
            new_key = (n.agent_coords, sort(n.box_coords))
            cost = 1

            if coord_to_node != Nothing
                curr_node = coord_to_node[(node[1], node[2])]
                for e in curr_node.out_edges
                    if e.to.agent_pos == n.agent_coords && e.to.box_pos == n.box_coords
                        cost = e.cost
                        break
                    end
                end
            end

            if haskey(deltas, new_key)
                deltas[new_key] = min(deltas[node] + cost, deltas[new_key])
            else
                deltas[new_key] = deltas[node] + cost
            end

            if haskey(pq, new_key) 
                pq[new_key] = deltas[new_key]
            else
                enqueue!(pq, new_key, deltas[new_key])
            end
        end
    end

    if !h_max_goal_check(C, goal_coords)
        return Inf
    else
        for e in C 
            if e[2] == goal_coords
                return deltas[e]
            end
        end
        return Inf 
    end
end

mutable struct JG_node
    agent_pos
    box_pos
    creating_edges
    out_edges
end

mutable struct JG_edge
    from
    to
    cost
end

function create_JG_graph(state, goal_coords)
    coord_to_node = Dict()
    possible_coords = CartesianIndices(state.map)[state.map .!== 2.]
    possible_box_combs = sort(collect(combinations(possible_coords, length(state.box_coords))))
    nodes = []
    edges = []

    for ac in possible_coords
        for bc in possible_box_combs
            if !(ac in bc)
                node = JG_node(ac, sort(bc), [], [])
                push!(nodes, node)
                coord_to_node[(ac,sort(bc))] = node
            end
        end
    end

    init_node = JG_node(CartesianIndex(-1, -1), CartesianIndex(-1, -1), [], [])
    s_node = coord_to_node[(state.agent_coords, sort(state.box_coords))]
    a_init = JG_edge(init_node, s_node, 0)
    push!(init_node.out_edges, a_init)
    push!(s_node.creating_edges, a_init)
    push!(edges, a_init)

    goal_node = JG_node(CartesianIndex(0, 0), CartesianIndex(0, 0), [], [])
    for n in nodes
        if sort(n.box_pos) == sort(goal_coords)
            g_node = n
            a_goal = JG_edge(g_node, goal_node, 0)
            push!(g_node.out_edges, a_goal)
            push!(goal_node.creating_edges, a_goal)
            push!(edges, a_goal)
        end
    end    

    empty_maze = deepcopy(state.map)
    empty_maze[CartesianIndices(state.map)[state.map .== 3]] .= 1
    empty_maze[CartesianIndices(state.map)[state.map .== 5]] .= 1
    empty_maze[goal_coords] .= 4
    
    # a = 1
    for node in nodes
        # println(a) 
        # global a +=1 
        tmp_maze = deepcopy(empty_maze)
        tmp_maze[node.agent_pos] .= 3
        tmp_maze[node.box_pos] .= 5
        tmp_state = State(node.agent_pos, node.box_pos, tmp_maze)

        ns = get_neighbors(tmp_state, goal_coords)
        for n in ns
            agent = n.agent_coords
            boxes = sort(n.box_coords)
            next_node = coord_to_node[(agent, boxes)]
            action = JG_edge(node, next_node, 1)
            push!(edges, action)
            push!(node.out_edges, action)
            push!(next_node.creating_edges, action)
        end
    end
    return coord_to_node, nodes, edges, init_node, goal_node
end

function lm_cut_heur(state, goal_coords, t0, max_time)
    lm_cut = 0      
    coord_to_node, nodes, edges, init_node, goal_node = create_JG_graph(state, goal_coords);
    # println("graph...")

    while h_max_heur(state, goal_coords, coord_to_node) != 0
        if h_max_heur(state, goal_coords, coord_to_node) == Inf
            return Inf
        end
        
        t1 = time()
        if t1 - t0 > max_time
            return false
        end

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
        tmp_maze = sokoban2onehot(maze) |> gpu
        tmp_maze2 = add_coord_filters(reshape(tmp_maze, size(tmp_maze,1),size(tmp_maze,2),size(tmp_maze,3),1))
    else
        tmp_maze = sokoban2onehot(maze) |> gpu
        tmp_maze2 = reshape(tmp_maze, size(tmp_maze,1),size(tmp_maze,2),size(tmp_maze,3),1)
    end    
    result = model(tmp_maze2)
    return cpu(result[1])
end

function compute_heur(heur_type, state, heur_network, goal_coords, t0, max_time)
    if heur_type == "euclidean"
        return euclidean_heur(state, goal_coords)
    end
    if heur_type == "hff"
        return hff_heur(state, goal_coords, t0, max_time)
    end
    if heur_type == "neural"
        if length(heur_network) == 2
            return value = mac_net_heur(state.map, heur_network[1],heur_network[2])
        else 
            value = att_net_heur(state.map, heur_network[1], heur_network[2], heur_network[3], heur_network[4])
            return value[1]
        end
    end
    if heur_type == "lmcut"
        return lm_cut_heur(state, goal_coords, t0, max_time)
    end    
    if heur_type == "none"
        return 0.
    end
end

function resnet(model,composer,x)
    composer(cat(x, model(x), dims = 3))
end

function get_neighbors_nn(map, model, composer, goal_ix)
    agent_ix = CartesianIndices(map)[map .== 3][1]
    box_ix = CartesianIndices(map)[map .== 5]

    m = reshape(no_goal_sokoban2onehot(map), size(map,1), size(map,2), 4, 1) |> gpu
    r = softmax(resnet(model, composer, m), dims=3)

    all_pos = CartesianIndices(zeros(size(m,1),size(m,2)))
    agent_th = 0.001
    box_th = 0.001
    agent_pos = []
    box_pos = []
    # wall, free space, agent, box
    for pos in all_pos
        vec = r[pos[1], pos[2], :, 1]
        if vec[3] > agent_th
            push!(agent_pos, pos)
        end
        if vec[4] > box_th
            push!(box_pos, pos)
        end
    end

    valid_ns = get_neighbors(State(agent_ix, box_ix, map), goal_ix)
    ret_states = []
    for n in valid_ns
        valid_agent = false
        for ag in agent_pos
            if ag == n.agent_coords
                valid_agent = true
                break
            end
        end
        valid_box = false
        for b in box_pos
            if b in n.box_coords
                valid_box = true
                break
            end
        end
        if valid_agent && valid_box
            push!(ret_states,n)
        end
    end

    if length(ret_states) > 4
        println("Too many successor states")
    elseif length(ret_states) < 1
        println("No successor states found")
    end

    return ret_states
end


function load_nn()
    # @load "resnet_models/sokoban_64_5_2_250_model.bson" m
    # @load "resnet_models/sokoban_64_5_2_250_composer.bson" c

    # @load "sokoban_expansion_network_relevant/m_32_50.bson" m
    # @load "sokoban_expansion_network_relevant/c_32_50.bson" c

    # @load "sokoban_expansion_network_relevant/m_128_50.bson" m
    # @load "sokoban_expansion_network_relevant/c_128_50.bson" c

    # @load "sokoban_expansion_network_relevant/m_16_100.bson" m
    # @load "sokoban_expansion_network_relevant/c_16_100.bson" c

    # @load "sokoban_expansion_network_relevant/m_10_100.bson" m
    # @load "sokoban_expansion_network_relevant/c_10_100.bson" c

    @load "sokoban_expansion_network_relevant/m3_16_100.bson" m
    @load "sokoban_expansion_network_relevant/c3_16_100.bson" c

    model = m |> gpu
    composer = c |> gpu

    # Old trained network --------------------------------------
    # @load "resnet_models/expansion_network_sokoban_model_params.bson" m_params
    # @load "resnet_models/expansion_network_sokoban_composer_params.bson" c_params

    # k = 64
    # model = Chain(
    #     Conv((5,5), 4 => k, relu, pad = (2,2)),
    #     Conv((5,5), k => k, relu, pad = (2,2)),
    #     Conv((5,5), k => 2*k, relu, pad = (2,2)),
    #     Conv((5,5), 2*k => 3*k, relu, pad = (2,2)),
    #     Conv((5,5), 3*k => 4*k, relu, pad = (2,2)),
    #     Conv((5,5), 4*k => 4*k, relu, pad = (2,2)),
    #     Conv((5,5), 4*k => 4, identity, pad = (2,2))
    # ) |> gpu    

    # composer = Chain(
    #     Conv((1,1), 8 => k, swish), 
    #     Conv((1,1), k => 4, identity),
    # ) |> gpu

    # Flux.loadparams!(model, m_params)
    # Flux.loadparams!(composer, c_params)
    # ----------------------------------------------------------
    return model, composer
end

# Solve given generated map using bfs 
# floor = 1; wall = 2; agent = 3; goal = 4; box = 5
function gbfs(map, nn_neighbors, heur_type, heur_network)
    goal_coords = CartesianIndices(map)[map .== 4]

    parents = Dict()
    cl = Set()
    pq = PriorityQueue()

    agent_coords = CartesianIndices(map)[map .== 3][1]
    box_coords = CartesianIndices(map)[map .== 5]
    start_state = State(agent_coords, box_coords, map)

    if nn_neighbors
        model, composer = load_nn()
    end

    heur_value = compute_heur(heur_type, start_state, heur_network, goal_coords)
    pq[start_state] = heur_value
    parents[start_state] = 0

    goal_node = Nothing
    expanded_nodes = 0

    while length(pq) > 0
        node = dequeue!(pq)

        if node.map in cl
            continue
        end
        push!(cl, node.map)

        if goal_check(node, goal_coords)
            goal_node = node
            # println("Goal found")
            break
        end
        expanded_nodes += 1

        ns = Nothing
        if nn_neighbors
            ns = get_neighbors_nn(node.map, model, composer, goal_coords)
        else
            ns = get_neighbors(node, goal_coords)
        end

        for n in ns
            f = compute_heur(heur_type, n, heur_network, goal_coords)
            if f != Inf
                enqueue!(pq, n, f)
                parents[n] = node
            end
        end
    end

    if goal_node == Nothing
        return [], Inf, expanded_nodes
    end

    path = reconstruct_path(parents, goal_node)
    return (path, length(path) - 1, expanded_nodes)
end

function bfs(map, nn_neighbors, heur_type, heur_network)
    goal_coords = CartesianIndices(map)[map .== 4]

    parents = Dict()
    cl = Set()
    pq = PriorityQueue()
    g_values = Dict()

    agent_coords = CartesianIndices(map)[map .== 3][1]
    box_coords = CartesianIndices(map)[map .== 5]
    start_state = State(agent_coords, box_coords, map)

    if nn_neighbors
        model, composer = load_nn()
    end

    heur_value = compute_heur(heur_type, start_state, heur_network, goal_coords)
    pq[start_state] = heur_value
    parents[start_state] = 0
    g_values[start_state] = 0

    goal_node = Nothing
    expanded_nodes = 0

    while length(pq) > 0
        node = dequeue!(pq)

        if node.map in cl
            continue
        end
        push!(cl, node.map)

        if goal_check(node, goal_coords)
            goal_node = node
            break
        end
        expanded_nodes += 1

        ns = Nothing
        if nn_neighbors
            ns = get_neighbors_nn(node.map, model, composer, goal_coords)
        else
            ns = get_neighbors(node, goal_coords)
        end

        for n in ns
            if n.map in cl
                continue
            end
            g = g_values[node] + 1
            f = compute_heur(heur_type, n, heur_network, goal_coords) + g

            if !(n in keys(g_values)) || g_value < g_values[n]
                parents[n] = node
                g_values[n] = g
                enqueue!(pq, n, f)
            elseif g_value >= g_values[n]
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
function mh_gbfs(map, nn_neighbors, heur_type, heur_network)
    goal_coords = CartesianIndices(map)[map .== 4]

    parents = Dict()
    cl = Set()
    pq = PriorityQueue()

    agent_coords = CartesianIndices(map)[map .== 3][1]
    box_coords = CartesianIndices(map)[map .== 5]
    start_state = State(agent_coords, box_coords, map)
    
    if nn_neighbors
        model, composer = load_nn()
    end

    heur_vec = []
    for heur in heur_type
        h = compute_heur(heur, start_state, heur_network, goal_coords)
        push!(heur_vec, h)
    end

    pq[start_state] = heur_vec
    parents[start_state] = 0

    goal_node = Nothing
    expanded_nodes = 0

    while length(pq) > 0
        node = dequeue!(pq)

        if node.map in cl
            continue
        end
        push!(cl, node.map)

        if goal_check(node, goal_coords)
            goal_node = node
            break
        end
        expanded_nodes += 1
        # println(node.map)

        ns = Nothing
        if nn_neighbors
            ns = get_neighbors_nn(node.map, model, composer, goal_coords)
        else
            ns = get_neighbors(node, goal_coords)
        end

        for n in ns
            heur_vec = []
            for heur in heur_type
                h = compute_heur(heur, n, heur_network, goal_coords)
                push!(heur_vec, h)
            end

            f = heur_vec
            if f[1] != Inf
                enqueue!(pq, n, f)
                parents[n] = node
            end
        end
    end

    if goal_node == Nothing
        return [], Inf, expanded_nodes
    end

    path = reconstruct_path(parents, goal_node)
    return (path, length(path) - 1, expanded_nodes)
end

function gbfs_timed(map, nn_neighbors, heur_type, heur_network, max_time)
    goal_coords = CartesianIndices(map)[map .== 4]

    parents = Dict()
    cl = Set()
    pq = PriorityQueue()

    agent_coords = CartesianIndices(map)[map .== 3][1]
    box_coords = CartesianIndices(map)[map .== 5]
    start_state = State(agent_coords, box_coords, map)

    if nn_neighbors
        model, composer = load_nn()
    end

    t0 = time()
    heur_value = compute_heur(heur_type, start_state, heur_network, goal_coords, t0, max_time)
    # println("Heur value computed")
    # if heur_value == false && heur_type != "none"
    #     return [], Inf, 0
    # end
    pq[start_state] = heur_value
    parents[start_state] = 0

    goal_node = Nothing
    expanded_nodes = 0

    while length(pq) > 0
        node = dequeue!(pq)
        # h = heatmap(node.map)
        # Plots.display(h)
        # sleep(0.1)

        if node.map in cl
            continue
        end
        push!(cl, node.map)

        if goal_check(node, goal_coords)
            goal_node = node
            break
        end
        expanded_nodes += 1

        ns = Nothing
        if nn_neighbors
            ns = get_neighbors_nn(node.map, model, composer, goal_coords)
        else
            ns = get_neighbors(node, goal_coords)
        end

        # if length(ns) == 0
        #     println("no successor states -> expanded nodes == ", expanded_nodes)
        # else
        #     # println("successor states: ", length(ns))
        # end

        for n in ns
            f = compute_heur(heur_type, n, heur_network, goal_coords, t0, max_time)
            # println(f)
            # if f == false && heur_type != "none"
            #     return [], Inf, expanded_nodes
            # end

            if f != Inf
                enqueue!(pq, n, f)
                parents[n] = node
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

function bfs_timed(map, nn_neighbors, heur_type, heur_network, max_time)
    goal_coords = CartesianIndices(map)[map .== 4]

    parents = Dict()
    cl = Set()
    pq = PriorityQueue()
    g_values = Dict()

    agent_coords = CartesianIndices(map)[map .== 3][1]
    box_coords = CartesianIndices(map)[map .== 5]
    start_state = State(agent_coords, box_coords, map)

    if nn_neighbors
        model, composer = load_nn()
    end

    t0 = time()

    heur_value = compute_heur(heur_type, start_state, heur_network, goal_coords, t0, max_time)
    # if heur_value == false && heur_type != "none"
    #     return [], Inf, 0
    # end

    pq[start_state] = heur_value
    parents[start_state] = 0
    g_values[start_state] = 0

    goal_node = Nothing
    expanded_nodes = 0

    while length(pq) > 0
        node = dequeue!(pq)
        # Plots.display(heatmap(node.map))
        # sleep(0.1)

        if node.map in cl
            continue
        end
        push!(cl, node.map)

        if goal_check(node, goal_coords)
            goal_node = node
            break
        end
        expanded_nodes += 1

        ns = Nothing
        if nn_neighbors
            ns = get_neighbors_nn(node.map, model, composer, goal_coords)
        else
            ns = get_neighbors(node, goal_coords)
        end

        for n in ns
            if n.map in cl
                continue
            end
            g = g_values[node] + 1
            h = compute_heur(heur_type, n, heur_network, goal_coords, t0, max_time) 
            f = h + g

            # if h == false && heur_type != "none"
            #     return [], Inf, expanded_nodes
            # end

            if !(n in keys(g_values)) || g < g_values[n]
                parents[n] = node
                g_values[n] = g
                enqueue!(pq, n, f)
            elseif g >= g_values[n]
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

# Multi-heuristic GBFS
function mh_gbfs_timed(map, nn_neighbors, heur_type, heur_network, max_time)
    goal_coords = CartesianIndices(map)[map .== 4]

    parents = Dict()
    cl = Set()
    pq = PriorityQueue()

    agent_coords = CartesianIndices(map)[map .== 3][1]
    box_coords = CartesianIndices(map)[map .== 5]
    start_state = State(agent_coords, box_coords, map)
    
    if nn_neighbors
        model, composer = load_nn()
    end

    t0 = time()
    heur_vec = []
    for heur in heur_type
        h = compute_heur(heur, start_state, heur_network, goal_coords, t0, max_time)
        if h == false
            return [], Inf, 0
        end
        push!(heur_vec, h)
    end

    pq[start_state] = heur_vec
    parents[start_state] = 0

    goal_node = Nothing
    expanded_nodes = 0

    while length(pq) > 0
        node = dequeue!(pq)

        if node.map in cl
            continue
        end
        push!(cl, node.map)

        if goal_check(node, goal_coords)
            goal_node = node
            break
        end
        expanded_nodes += 1
        # println(node.map)

        ns = Nothing
        if nn_neighbors
            ns = get_neighbors_nn(node.map, model, composer, goal_coords)
        else
            ns = get_neighbors(node, goal_coords)
        end

        for n in ns
            heur_vec = []
            for heur in heur_type
                h = compute_heur(heur, n, heur_network, goal_coords, t0, max_time)
                if h == false
                    return [], Inf, expanded_nodes
                end
                push!(heur_vec, h)
            end

            f = heur_vec
            if f[1] != Inf
                enqueue!(pq, n, f)
                parents[n] = node
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


# map = [2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0; 2.0 2.0 2.0 2.0 1.0 1.0 1.0 2.0; 2.0 2.0 4.0 1.0 3.0 5.0 1.0 2.0; 2.0 2.0 1.0 1.0 2.0 2.0 5.0 2.0; 2.0 2.0 2.0 2.0 1.0 1.0 1.0 2.0; 2.0 2.0 1.0 1.0 1.0 2.0 1.0 2.0; 2.0 2.0 1.0 1.0 1.0 1.0 4.0 2.0; 2.0 2.0 2.0 2.0 2.0 2.0 2.0 2.0]

# # mh_gbfs(map, false, "hff", [])
