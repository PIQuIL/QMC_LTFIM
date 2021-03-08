module QMC

using ProgressMeter

using Measurements
using Statistics
import Statistics: var, mean
using Distributions
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

import Base: zero, one, convert
import Base: length, size, eltype, setindex!, getindex, firstindex, lastindex
import Base: rand, show, pop!, push!, append!, isempty, empty!, count

# using BinningAnalysis
# import BinningAnalysis: varN, std_error

export BinaryQMCState, BinaryGroundState, BinaryThermalState,
        Hamiltonian, AbstractIsing, AbstractTFIM, TFIM, AbstractLTFIM, LTFIM, GeneralLTFIM, AbstractRydberg, Rydberg,
        haslongitudinalfield,
        nspins, nbonds, energy, energy_density, mc_step!, mc_step_beta!,
        resize_op_list!,
        sample, simulation_cell, magnetization, staggered_magnetization,
        num_single_site_diag, num_single_site_offdiag,
        num_single_site, num_two_site_diag, autocorrelation, correlation_time, jackknife, bootstrap, mean_and_stderr,
        lattice_bond_spins, ProbabilityAlias, ProbabilityHeap, ProbabilityVector, probability_vector

export Bootstrap, LogBinner, varN, std_error, convergence, has_converged,
        tau, all_taus, all_vars, all_varNs, all_means, all_std_errors


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
include("error/Error.jl")


end
