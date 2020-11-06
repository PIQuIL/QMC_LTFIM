abstract type Hamiltonian{D,O<:AbstractOperatorSampler} end

localdim(::Hamiltonian{D}) where {D} = D

zero(H::Hamiltonian{2}) = zeros(Bool, nspins(H))
zero(H::Hamiltonian) = zeros(Int, nspins(H))
one(H::Hamiltonian{2}) = ones(Bool, nspins(H))
one(H::Hamiltonian) = ones(Int, nspins(H))

nspins(H::Hamiltonian) = H.Ns
nbonds(H::Hamiltonian) = H.Nb


function energy(::BinaryThermalState, H::Hamiltonian, β::Float64, ns::Vector{T}) where {T <: Real}
    E = -mean_and_stderr(ns) / β
    return H.energy_shift + E
end

energy_density(qmc_state::BinaryQMCState, H::Hamiltonian, args...) = energy(qmc_state, H, args...) / nspins(H)


function BinaryGroundState(H::Hamiltonian{2,O}, M::Int) where {K, O <: AbstractOperatorSampler{K}}
    BinaryGroundState(zero(H), zero(H), init_op_list(2*M, K))
end


function BinaryThermalState(H::Hamiltonian{2,O}, cutoff::Int) where {K, O <: AbstractOperatorSampler{K}}
    BinaryThermalState(zero(H), zero(H), init_op_list(cutoff, K))
end
