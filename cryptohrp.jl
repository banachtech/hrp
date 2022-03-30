
using DelimitedFiles, Statistics, LinearAlgebra

include("HRP.jl")

z = readdlm(ARGS[1], ',')

w = size(z, 1)
if length(ARGS) > 1
    w = min(w, parse(Int, ARGS[2]))
end

_, whrp = HRP.hrp(z[end-w+1:end, :])

open("weights.csv", "w") do io
    writedlm(io, whrp, ',')
end

println(whrp)
