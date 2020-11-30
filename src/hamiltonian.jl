abstract type Hamiltonian{D,O<:AbstractOperatorSampler} end

localdim(::Hamiltonian{D}) where {D} = D

zero(H::Hamiltonian{2}) = zeros(Bool, nspins(H))
zero(H::Hamiltonian) = zeros(Int, nspins(H))
one(H::Hamiltonian{2}) = ones(Bool, nspins(H))
one(H::Hamiltonian) = ones(Int, nspins(H))

nspins(H::Hamiltonian) = H.Ns
nbonds(H::Hamiltonian) = H.Nb

@inline isdiagonal(H) = op -> isdiagonal(typeof(H), op)
@inline isidentity(H) = op -> isidentity(typeof(H), op)
@inline issiteoperator(H) = op -> issiteoperator(typeof(H), op)
@inline isbondoperator(H) = op -> isbondoperator(typeof(H), op)

@inline diag_update_normalization(H::Hamiltonian) = normalization(H.op_sampler)

function energy(::BinaryThermalState, H::Hamiltonian, β::Float64, ns::Vector{T}) where {T <: Real}
    E = -mean_and_stderr(ns) / β
    return H.energy_shift + E
end

energy_density(qmc_state::BinaryQMCState, H::Hamiltonian, args...; kwargs...) = energy(qmc_state, H, args...; kwargs...) / nspins(H)


function BinaryGroundState(H::Hamiltonian{2,O}, M::Int) where {K, O <: AbstractOperatorSampler{K}}
    z = zero(H)
    BinaryGroundState(z, init_op_list(2*M, Val{K}()))::BinaryGroundState{K, typeof(z)}
end


function BinaryThermalState(H::Hamiltonian{2,O}, cutoff::Int) where {K, O <: AbstractOperatorSampler{K}}
    z = zero(H)
    BinaryThermalState(z, init_op_list(cutoff, Val{K}()))::BinaryThermalState{K, typeof(z)}
end
