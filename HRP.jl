module HRP

using Statistics, LinearAlgebra, Clustering

export hrp, ivp

dist(x, y) = norm(collect(skipmissing(x .- y)), 2)

function covar(x, y)
    id = findall(z -> !ismissing(z), x .- y)
    return cov(x[id], y[id])
end

function ret2dist(r)
    n = size(r, 2)
    Σ = fill(0.0, n, n)
    D = fill(0.0, n, n)
    for i = 1:n
        for j = i:n
            D[i, j] = dist(r[:, i], r[:, j])
            D[j, i] = D[i, j]
            Σ[i, j] = covar(r[:, i], r[:, j])
            Σ[j, i] = Σ[i, j]
        end
    end
    return Σ, D
end

function ivp(C)
    iv = 1.0 ./ diag(C)
    return iv ./ sum(iv)
end

function clustervar(C, cluster)
    V = C[cluster, cluster]
    u = ivp(V)
    return dot(u' * V, u)
end

function bisect(items)
    tmp = Vector{Int}[]
    for item in items
        n1 = length(item)
        if n1 > 1
            n2 = floor(Int, n1 / 2)
            push!(tmp, item[1:n2])
            push!(tmp, item[n2+1:end])
        end
    end
    return tmp
end

function update!(w, items, Σ)
    for k in 1:2:length(items)
        c1, c2 = items[k], items[k+1]
        v1, v2 = clustervar(Σ, c1), clustervar(Σ, c2)
        α = v1 / (v1 + v2)
        w[c1] .= w[c1] .* α
        w[c2] .= w[c2] .* (1.0 - α)
    end
end

function hrp(x; input_type="ret")
    Σ = fill(0.0, size(x, 2), size(x, 2))
    D = similar(Σ)
    if input_type == "ret"
        Σ, D = ret2dist(x)
    elseif input_type == "cov"
        Σ, D = x, cov2dist(x)
    end
    h = hclust(D).order
    n = length(h)
    w = fill(1.0, n)
    items = [h]
    while !isempty(items)
        items = bisect(items)
        update!(w, items, Σ)
    end
    return ivp(Σ), w
end

function hrp(Σ)
    D = sqrt.(2.0*(1.0.-cov2corr(Σ)))
    id = 1:size(D,2)
    w = fill(1.0, size(D,2))
    while length(id) > 1
        id, k = split(D, id)
        v1 = clustervar(Σ, id)
        v2 = Σ[k,k]
        α = ivp([v1,v2])[1]
        w[id] .= w[id] * α
        w[k] = w[k] * (1. - α) 
    end
    return w
end

function split(D::Matrix{Float64}, id::Union{UnitRange{Int64}, Vector{Int64}})
    if length(id) == 1
        return nothing, id
    end
    tmp = []
    for i in id
        push!(tmp, minimum([D[i,k] for k in setdiff(id, i)]))
    end
    k = id[argmax(tmp)]
    id = setdiff(id, k)
    return id, k
end

function hcluster_order(D)
    n = size(D,2)
    id = 1:n
    out = []
    for i = 1:n
        id, k = split(D, id)       
        push!(out, k)
    end
    return reverse(collect(Iterators.flatten(out)))
end

function cov2corr(C)
    σinv = diagm(1.0 ./ sqrt.(diag(C)))
    return σinv * C * σinv
end

function cov2dist(C)
    P = cov2corr(C)
    S = similar(P)
    for i = 1:size(P, 1)
        for j = i:size(P, 1)
            S[i, j] = 2.0 * (1.0 - P[i, j])
            S[j, i] = S[i, j]
        end
    end
    return S
end

end