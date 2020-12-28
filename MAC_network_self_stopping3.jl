using Flux

struct Control
    q_lin::Dense
    cq_lin::Dense
end

Flux.@treelike Control

struct Read
    m_lin::Dense
    kb_lin::Conv
    I_lin::Conv
    Ic_lin::Conv
    r_att::Dense
end

Flux.@treelike Read

struct Write
    rm_lin::Dense
    sa_lin::Conv
    c_lin::Dense
    msa_lin::Dense
    mem_gate::Dense
    self_attention::Bool
    gating::Bool
end
Flux.@treelike Write

struct Input
    q_model::Chain
    kb_model::Chain
end
Flux.@treelike Input

struct Output
    mq_dense::Dense
    mq_lin::Dense
    mq_res::Dense
    # mq_one::Dense
end

Flux.@treelike Output

mutable struct MAC_network
    h::Int64
    B::Any
    c0::Any
    m0::Any
    ci::Any
    mi::Any
    control::Control
    read::Read
    write::Write
    stopper::Chain
    input::Input
    output::Output
end

Flux.@treelike MAC_network

function (c::Control)(mn::MAC_network, q)
    qi = c.q_lin(reshape(q, mn.h, :))
    pre_cqi = cat(mn.ci, qi, dims=1)
    cqi = c.cq_lin(pre_cqi)
    # cj = sf(cqi,1)
    cj = softmax(cqi,dims=1)
    return cj
end

function (r::Read)(mn, cj, kb)
    h = mn.h
    B = mn.B

    m_l = r.m_lin(mn.mi)
    kb_l = r.kb_lin(kb)
    I1 = kb_l .* reshape(m_l, 1, 1, :, size(m_l,2))
    I2 = r.I_lin(cat(I1, kb, dims=3))
    # I2_ci = I1 .* reshape(cj, 1, 1, :, size(cj,2))
    I2_ci = I2 .* reshape(cj, 1, 1, :, size(cj,2))
    I3 = r.Ic_lin(I2_ci)
    # I3_sf = sf(I3, 4) 
    I3_sf = softmax(I3,dims=4)
    rj = reshape(sum(kb .* I3_sf, dims=[1,2]), h, :)
end

function (w::Write)(mn, rj, cj, c_prev, m_prev)
    B = mn.B
    h = mn.h
    mj = w.rm_lin(cat(rj,mn.mi,dims=1))   
    
    if w.self_attention
        c = reshape(reshape(cj, size(cj,1), size(cj,2), 1) .* c_prev, size(c_prev,3), B, h, 1)
        sa = softmax(reshape(w.sa_lin(c), size(c_prev,3), B), dims=1)
        m_sa = reshape(sum(sa .* permutedims(m_prev, (3,2,1)), dims=1), h, B)
        mj = w.msa_lin(m_sa) .+ mj
    end

    if w.gating
        c_l = w.c_lin(cj) 
        mj = sigmoid.(c_l) .* mj + (1 .- sigmoid.(c_l)) .* mn.mi
    end
    return mj
end

function (in::Input)(data)
    q = in.q_model(data)
    kb = in.kb_model(data)
    return q, kb
end

function (o::Output)(q, mp)
    mq = cat(mp, reshape(q, size(mp,1), :), dims=1)
    mq1 = o.mq_dense(mq)
    mq2 = o.mq_lin(mq1)
    res = o.mq_res(mq2)
    return res
end

function run_MAC(mn::MAC_network, no_iters::Int64, data)
    q, kb = mn.input(data)
    mn.mi = repeat(mn.m0, 1, size(data,4))
    mn.ci = repeat(mn.c0, 1, size(data,4))
    mn.B = size(data,4)

    c_prev = reshape(mn.ci, size(mn.ci,1), size(mn.ci,2), 1)
    m_prev = reshape(mn.mi, size(mn.mi,1), size(mn.mi,2), 1)

    stopped_samples = zeros(1,size(data,4)) |> gpu
    # sample_iter_ixs = zeros(1, mn.B)
    sample_res = zeros(size(mn.ci,1), size(mn.ci,2)) |> gpu
    minus_I = Float32.(-1 .* Matrix(I,mn.B,mn.B)) |> gpu
    one_vec = ones(mn.B,1) |> gpu
    zero_vec = zeros(1, mn.B) |> gpu

    for i in 1:no_iters
        c_new, m_new = MAC_iter(mn, kb, q, c_prev, m_prev)
        mn.ci = c_new
        mn.mi = m_new

        c_prev = cat(c_prev, c_new, dims=3)
        m_prev = cat(m_prev, m_new, dims=3)

        # Output of the stopper module
        stopper_out = mn.stopper(cat(c_new,m_new,dims=1))
        # stopper_bool = Float32.(stopper_out .>= 0.5) 
        # stopper_bool = round.(stopper_out)
        stopper_bool = relu.(sign.(stopper_out .- Float32(0.5)))

        # Select data for samples stopped in this iteration -> neg(stopped_samples) and stopper_bool 
        # Those that weren't stopped yet and are being stopped this iteration 
        stopper_pick = (minus_I * transpose(stopped_samples) + one_vec + transpose(stopper_bool))
        # println(stopper_pick, typeof(stopper_pick))
        # tmp = (stopper_pick .== 2) + transpose(zero_vec)
        tmp = relu.(stopper_pick .- 1)
        # Select the already selected sample data pieces from prev iterations and add the new ones 
        sample_res = stopped_samples .* sample_res + transpose(tmp) .* mn.mi
        stopped_samples = sign.(stopped_samples + transpose(tmp))

        # If all samples already stopped by the stopper module 
        if sum(stopped_samples) == mn.B
            # println("Iteration: ", i)
            break
        end
    end

    # In case there's a sample that hasn't been stopped -> say all of them were supposed to be stopped in the last iteration 
    if sum(stopped_samples) < mn.B
        stopper_bool = ones(size(stopped_samples)) |> gpu
        stopper_pick = minus_I * transpose(stopped_samples) + one_vec + transpose(stopper_bool)
        tmp = relu.(stopper_pick .- 1)
        sample_res = stopped_samples .* sample_res + transpose(tmp) .* mn.mi
    end

    r = mn.output(q, sample_res)
    # r = mn.output(q,sample_iter_res)
    return r
end

function run_MAC_iter_debug(mn::MAC_network, no_iters::Int64, data)
    q, kb = mn.input(data)
    mn.mi = repeat(mn.m0, 1, size(data,4))
    mn.ci = repeat(mn.c0, 1, size(data,4))
    mn.B = size(data,4)

    c_prev = reshape(mn.ci, size(mn.ci,1), size(mn.ci,2), 1)
    m_prev = reshape(mn.mi, size(mn.mi,1), size(mn.mi,2), 1)

    stopped_samples = zeros(1,size(data,4)) |> gpu
    # sample_iter_ixs = zeros(1, mn.B)
    sample_res = zeros(size(mn.ci,1), size(mn.ci,2)) |> gpu
    minus_I = Float32.(-1 .* Matrix(I,mn.B,mn.B)) |> gpu
    one_vec = ones(mn.B,1) |> gpu
    zero_vec = zeros(1, mn.B) |> gpu

    for i in 1:no_iters
        c_new, m_new = MAC_iter(mn, kb, q, c_prev, m_prev)
        mn.ci = c_new
        mn.mi = m_new

        c_prev = cat(c_prev, c_new, dims=3)
        m_prev = cat(m_prev, m_new, dims=3)

        # Output of the stopper module
        stopper_out = mn.stopper(cat(c_new,m_new,dims=1))
        # stopper_bool = Float32.(stopper_out .>= 0.5) 
        # stopper_bool = round.(stopper_out)
        stopper_bool = relu.(sign.(stopper_out .- Float32(0.5)))

        # Select data for samples stopped in this iteration -> neg(stopped_samples) and stopper_bool 
        # Those that weren't stopped yet and are being stopped this iteration 
        stopper_pick = (minus_I * transpose(stopped_samples) + one_vec + transpose(stopper_bool))
        # println(stopper_pick, typeof(stopper_pick))
        # tmp = (stopper_pick .== 2) + transpose(zero_vec)
        tmp = relu.(stopper_pick .- 1)
        # Select the already selected sample data pieces from prev iterations and add the new ones 
        sample_res = stopped_samples .* sample_res + transpose(tmp) .* mn.mi
        stopped_samples = sign.(stopped_samples + transpose(tmp))

        # If all samples already stopped by the stopper module 
        if sum(stopped_samples) == mn.B
            # out_iter = i
            break
        end
    end

    # In case there's a sample that hasn't been stopped -> say all of them were supposed to be stopped in the last iteration 
    if sum(stopped_samples) < mn.B
        stopper_bool = ones(size(stopped_samples)) |> gpu
        stopper_pick = minus_I * transpose(stopped_samples) + one_vec + transpose(stopper_bool)
        tmp = relu.(stopper_pick .- 1)
        sample_res = stopped_samples .* sample_res + transpose(tmp) .* mn.mi
        # out_iter = no_iters
    end

    r = mn.output(q, sample_res)
    return r, stopped_samples
end

function MAC_iter(mn::MAC_network, kb, q, c_prev, m_prev)
    cj = mn.control(mn, q)
    rj = mn.read(mn, cj, kb)
    mj = mn.write(mn, rj, cj, c_prev, m_prev)
    return cj, mj
end

function create_MAC_network(maze_filters, h_filters, w_self_att, w_gate)
    h = h_filters

    # testing new model
    q_model = Chain(
        Conv((1,1), maze_filters=>32, swish),
        Conv((3,3), 32=>32, pad=(1,1), swish),
        Conv((3,3), 32=>48, pad=(1,1), swish),
        Conv((3,3), 48 => 96, pad=(1,1), swish),
        x -> sum(x, dims=[1,2]),    
        x -> reshape(x, :, size(x)[4]),
        Dense(96, h)
    ) |> gpu

    # Out: 4x4xhxbs
    kb_model = Chain(
        Conv((3,3), maze_filters => 8, relu),
        Conv((3,3), 8 => h, relu),
    ) |> gpu

    in = Input(q_model, kb_model)

    c = Control(
        Dense(h,h) |> gpu, # q_lin
        Dense(2*h,h) |> gpu) # cq_lin

    r = Read(
        Dense(h,h) |> gpu, # m_lin
        Conv((1,1), h => h, relu) |> gpu, # kb_lin
        Conv((1,1), 2*h => h, relu) |> gpu, # I_lin
        Conv((1,1), h => h, relu) |> gpu, # Ic_lin
        Dense(h,1) |> gpu) # r_att 

    w = Write(
        Dense(2*h,h) |> gpu, # rm_lin
        Conv((1,1), h => 1, relu) |> gpu, # sa_lin 
        Dense(h,h) |> gpu, # c_lin
        Dense(h,h) |> gpu, # msa_lin
        Dense(h,1) |> gpu, # mem_gate
        w_self_att, 
        w_gate)

    s = Chain(
            Dense(2*h, h),
            Dense(h, 1, sigmoid)
        ) |> gpu

    out = Output(
        Dense(2*h,h,swish) |> gpu, # mq_dense
        Dense(h,h,swish) |> gpu, # mq_lin
        Dense(h,1) |> gpu) # mq_res
 
    c0 = rand(h,1) |> gpu # initialize control vector 
    m0 = rand(h,1) |> gpu # initialize memory vector 
    net = MAC_network(h,Nothing,c0,m0,Nothing,Nothing,c,r,w,s,in,out)
    return net
end

function save_mac_network(path, mac)
    # Input
    file = string(path, "/mac_input.bson")
    q_model = cpu(mac.input.q_model)
    kb_model = cpu(mac.input.kb_model)
    @save file q_model kb_model

    # Output
    file = string(path, "/mac_output.bson")
    mq_dense = cpu(mac.output.mq_dense)
    mq_lin = cpu(mac.output.mq_lin)
    mq_res = cpu(mac.output.mq_res)
    @save file mq_dense mq_lin mq_res

    # Control
    file = string(path, "/mac_control.bson")
    q_lin = cpu(mac.control.q_lin)
    cq_lin = cpu(mac.control.cq_lin)
    @save file q_lin cq_lin

    # Read
    file = string(path, "/mac_read.bson")
    m_lin = cpu(mac.read.m_lin)
    kb_lin = cpu(mac.read.kb_lin)
    I_lin = cpu(mac.read.I_lin)
    Ic_lin = cpu(mac.read.Ic_lin)
    r_att = cpu(mac.read.r_att)
    @save file m_lin kb_lin I_lin Ic_lin r_att

    # Write
    file = string(path, "/mac_write.bson")
    rm_lin = cpu(mac.write.rm_lin)
    sa_lin = cpu(mac.write.sa_lin)
    c_lin = cpu(mac.write.c_lin)
    msa_lin = cpu(mac.write.msa_lin)
    mem_gate = cpu(mac.write.mem_gate)
    self_attention = mac.write.self_attention
    gating = mac.write.self_attention
    @save file rm_lin sa_lin c_lin msa_lin mem_gate self_attention gating

    # Stopper
    file = string(path, "/mac_stopper.bson")
    stopper = cpu(mac.stopper)
    @save file stopper

    # Rest of the data 
    file = string(path, "/mac_data.bson")
    h = mac.h
    # no_iters = mac.no_iters
    @save file h #no_iters

    # c0 a m0 
    file = string(path, "/mac_init.bson")
    m0 = cpu(mac.m0)
    c0 = cpu(mac.c0)
    @save file m0 c0 
end

function load_mac_network(path) 
    # Data 
    file = string(path, "/mac_data.bson")
    @load file h # no_iters 

    # m0 a c0 
    file = string(path, "/mac_init.bson")
    @load file m0 c0 
    m0 = cu(m0)
    c0 = cu(c0)

    # Input
    file = string(path, "/mac_input.bson")
    @load file q_model kb_model
    q_model = q_model |> gpu
    kb_model = kb_model |> gpu
    in = Input(q_model, kb_model) 

    # Output
    file = string(path, "/mac_output.bson")
    @load file mq_dense mq_lin mq_res
    mq_dense = mq_dense |> gpu
    mq_lin = mq_lin |> gpu
    mq_res = mq_res |> gpu
    out = Output(mq_dense, mq_lin, mq_res)

    # Controls 
    file = string(path, "/mac_control.bson")
    @load file q_lin cq_lin
    q_lin = q_lin |> gpu
    cq_lin = cq_lin |> gpu 
    c = Control(q_lin, cq_lin)

    # Reads 
    file = string(path, "/mac_read.bson")
    @load file m_lin kb_lin I_lin Ic_lin r_att
    m_lin = m_lin |> gpu
    kb_lin = kb_lin |> gpu
    I_lin = I_lin |> gpu
    Ic_lin = Ic_lin |> gpu
    r_att = r_att |> gpu
    r = Read(m_lin, kb_lin, I_lin, Ic_lin, r_att)

    # Writes 
    file = string(path, "/mac_write.bson")
    @load file rm_lin sa_lin c_lin msa_lin mem_gate self_attention gating
    rm_lin = rm_lin |> gpu 
    sa_lin = sa_lin |> gpu 
    c_lin = c_lin |> gpu 
    msa_lin = msa_lin |> gpu 
    mem_gate = mem_gate |> gpu 
    w = Write(rm_lin, sa_lin, c_lin, msa_lin, mem_gate, self_attention, gating)

    # Stopper
    file = string(path, "/mac_stopper.bson")
    @load file stopper
    s = stopper |> gpu

    net = MAC_network(h,Nothing,c0,m0,Nothing,Nothing,c,r,w,s,in,out)
    return net 
end
