using DrWatson
@quickactivate "QMC"

using QMC

using RydbergEmulator

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

atoms = load_instance(20, 0.8)
graph = unit_disk_graph(atoms, 1.5)
