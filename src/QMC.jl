module QMC

using ProgressMeter

using Measurements
using Statistics
using FFTW

using DelimitedFiles
using JLD2
using Printf

using PushVectors
import PushVectors: PushVector
using DataStructures
using SparseArrays
using LinearAlgebra

using Random

import Base: zero, one
import Base: length, size, eltype, setindex!, getindex, firstindex, lastindex, rand, show, pop!



export BinaryQMCState, BinaryGroundState, BinaryThermalState,
        Hamiltonian, AbstractIsing, AbstractTFIM, TFIM, AbstractLTFIM, LTFIM, GeneralLTFIM, AbstractRydberg, Rydberg,
        haslongitudinalfield,
        nspins, nbonds, energy, energy_density, mc_step!, mc_step_beta!,
        resize_op_list!,
        sample, simulation_cell, magnetization, staggered_magnetization,
        num_single_site_diag, num_single_site_offdiag,
        num_single_site, num_two_site_diag, autocorrelation, correlation_time, jackknife, bootstrap, mean_and_stderr,
        lattice_bond_spins, ProbabilityAlias, ProbabilityHeap, ProbabilityVector, probability_vector


@inline function pop!(v::PushVector)
    @boundscheck isempty(v) && throw(ArgumentError("vector must be non-empty"))
    x = @inbounds v.parent[v.len]
    v.len -= 1
    x
end


include("lattice.jl")
include("probabilityvectors/probabilityvector.jl")
include("operatorsamplers/operatorsampler.jl")
include("qmc_state.jl")
include("hamiltonian.jl")
include("operatorsamplers/improved_op_sampler.jl")
include("ising/Ising.jl")
include("measurements.jl")
include("error.jl")


end
