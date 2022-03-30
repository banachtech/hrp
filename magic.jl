using LinearAlgebra, Statistics, DelimitedFiles

function cov2corr(C)
    σinv = diagm(1.0 ./ sqrt.(diag(C)))
    return σinv * C * σinv
end

function ivp(C::Matrix{Float64})
    iv = 1.0 ./ diag(C)
    return iv ./ sum(iv)
end

function ivp(v::Vector{Float64})
    iv = 1.0 ./ v
    return iv ./ sum(iv)
end

function clustervar(C, cluster)
    V = C[cluster, cluster]
    u = ivp(V)
    return dot(u' * V, u)
end

function cov2dist(Σ)
    R = cov2corr(Σ)
    for i ∈ 1: size(R,2)
        R[i,i] = 0.0
    end
    return sqrt.(0.5 * (1.0 .- R))
end

function cluster(D, ids)
    tmp = 999999.
    idx = []
    for i ∈ 1:length(ids)
        for j ∈ i+1:length(ids)
            if D[ids[i],ids[j]] < tmp
                tmp = D[ids[i],ids[j]]
                idx = [ids[i],ids[j]]
            end
        end
    end
    return idx
end

function update_dist(D, cl)
    n = size(D,2)
    tmp = fill(0., n+1, n+1)
    tmp[1:n, 1:n] .= D
    for i ∈ 1:n
        tmp[i,n+1] = min(D[i,cl[1]], D[i, cl[2]])
        tmp[n+1,i] = tmp[i,n+1]
    end
    return tmp
end

function update_cov(Σ, cl)
    n = size(Σ,2)
    tmp = fill(0., n+1, n+1)
    tmp[1:n, 1:n] .= Σ
    u = ivp(Σ[cl, cl])
    for i ∈ 1:n
        tmp[i,n+1] = u[1]*Σ[i,cl[1]] + u[2]*Σ[i,cl[2]]
    end
    tmp[n+1,n+1] = clustervar(Σ,cl)
    return tmp
end

function allocate!(cl, α)
    u = ivp(Σ[cl, cl])
    u[2] = 1. - u[1]
    for i in 1:2
        if cl[i] <= N
            w[cl[i]] = w[cl[i]]*u[i]*α
        else
            allocate!(clusters[cl[i]-N],u[i]*α)
        end
    end
end

function hrpweights(V)
    global Σ = deepcopy(V)
    D = cov2dist(Σ)
    global N = size(Σ,2)
    ids = 1:N
    global clusters = Vector{Int64}[]
    for i ∈ 1:N-1
        cl = cluster(D, ids)
        D = update_dist(D, cl)
        Σ = update_cov(Σ, cl)
        push!(clusters, cl)
        ids = [setdiff(ids, cl); N+i]
        sort!(ids)
    end
    global w = ones(N)
    allocate!(clusters[end], 1.0)
    return w, clusters
end

if isempty(ARGS)
    println("missing returns file path ...")
    exit
end

ret, hdr = readdlm(ARGS[1], ',', header=true)
V = cov(ret)

w, clusters = hrpweights(V)

println("clusters:")
println(clusters)
println("hrp weights:")
println([reshape(hdr,length(hdr),1) w])

f = length(ARGS) < 2 ? "weights.csv" : ARGS[2]
open(f, "w") do io 
    writedlm(io,[hdr; w'],',')
end
println("weights saved in $f")