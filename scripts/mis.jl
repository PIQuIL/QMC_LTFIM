using DrWatson
@quickactivate "QMC"

using QMC

using RydbergEmulator
using LinearAlgebra
using LightGraphs

function load_instance(n, ff)
    L = ceil(Int, sqrt(n/ff))
    file = joinpath(@__DIR__, "graphs", "$L-$n-$ff.atoms")
    if isfile(file)
        return include(file)
    else
        @warn "cannot find data file: $file, generate new file"
        save(n, ff)
    end
end

function distance_matrix(graph)
    V = zeros(20, 20)
    for e in edges(graph)
        V[e.dst, e.src] = RydbergEmulator.distance(atoms[e.dst], atoms[e.src])
    end
    return UpperTriangular(V')
end


atoms = load_instance(20, 0.8)
graph = unit_disk_graph(atoms, 1.5)
V = distance_matrix(graph)
