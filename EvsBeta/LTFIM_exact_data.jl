using Plots
using ProgressMeter
using DelimitedFiles

include("LTFIM_thermal.jl")

function qmc_energy_magnetization(N, β, hx, hz)
    if β%2 == 1.
        β = convert(Int, β)
    elseif β%2 == 0.
        β = convert(Int, β)
    end

    if hz%2 == 1.
        hz = convert(Int, hz)
    elseif hz%2 == 0.
        hz = convert(Int, hz)
    end

    if hx%2 == 1.
        hx = convert(Int, hx)
    elseif hx%2 == 0.
        hx = convert(Int, hx)
    end

    path = "../data/sims/mixedstate/"
    file = path * "beta=$β.BC_name=PBC_Dim=1_J=1_M=10000_hx=$hx" * "_hz=$hz" * "_nX=$N" * "_skip=10_info.txt"

    ls = readlines(file)

    energy_line = split(ls[4], " ")
    magnetization_line = split(ls[1], " ")

    energy = parse(Float64, energy_line[length(energy_line)-2])
    energy_error = parse(Float64, energy_line[length(energy_line)])
    magnetization = parse(Float64, magnetization_line[length(magnetization_line)-2])
    magnetization_error = parse(Float64, magnetization_line[length(magnetization_line)])

    return energy, energy_error, magnetization, magnetization_error

end

header = ["N" "β" "hx" "hz" "ED E/N" "ED M/N" "ε" "±" "ϴ" "±"]

global data_full = header

for N = 2:2:10
    @show N
    for β = 0.2:0.2:2
        for hx = 0:0.5:2
            for hz = 0:0.5:2

                qmc_energy, qmc_energy_error, qmc_magnetization, qmc_magnetization_error = qmc_energy_magnetization(N, β, hx ,hz)

                ltfim = LTFIM_1D(N, hx, hz, β)
                energy = ltfim.Energy
                magnetization = ltfim.Magnetization

                data = [N β hx hz energy magnetization energy-qmc_energy qmc_energy_error magnetization-qmc_magnetization qmc_magnetization_error]
                global data_full = vcat(data_full, data)

            end
        end
    end                
end

open("energy_magnetization.txt", "w") do io
    writedlm(io, data_full)
end
