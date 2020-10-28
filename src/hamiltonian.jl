abstract type Hamiltonian{D,N,O<:AbstractOperatorSampler} end

localdim(::Hamiltonian{D}) where {D} = D
dim(::Hamiltonian{D,N}) where {D,N} = N

abstract type AbstractIsing{N,O} <: Hamiltonian{2,N,O} end
abstract type AbstractTFIM{N,O} <: AbstractIsing{N,O} end

struct TFIM{N,F,O} <: AbstractTFIM{N,O}
    op_sampler::O
    J::Float64
    h::Float64
    P_normalization::Float64
    Ns::Int
    Nb::Int
end

function TFIM(bond_spin, Dim::Int, Ns::Int, Nb::Int, h::Float64, J::Float64)
    ops, p = make_prob_vector(bond_spin, Ns, J, h)
    op_sampler = HierarchicalOperatorSampler(ops, p)
    F = !signbit(J)  # true if J > 0 (ferromagnetic)
    return TFIM{Dim, F, typeof(op_sampler)}(op_sampler, J, h, sum(p), Ns, Nb)
end

struct ArbitraryInteractionTFIM{N,O} <: AbstractTFIM{N,O}
    op_sampler::O
    J::Matrix{Float64}
    h::Vector{Float64}
    P_normalization::Float64
    Ns::Int
    Nb::Int
end

function ArbitraryInteractionTFIM(J::AbstractMatrix{Float64}, h::AbstractVector{Float64}; dim::Int=1)
    @assert length(h) == size(J, 1) == size(J, 2)

    ops, p = make_prob_vector(J, h)
    Ns = length(h)
    Nb = count(op -> op[1] > 0, ops)
    op_sampler = HierarchicalOperatorSampler(ops, p)

    return ArbitraryInteractionTFIM{dim, typeof(op_sampler)}(op_sampler, Matrix(J), Vector(h), sum(p), Ns, Nb)
end

###############################################################################

abstract type AbstractLTFIM{N,O} <: AbstractIsing{N,O} end

struct LTFIM{N,O} <: AbstractLTFIM{N,O}
    op_sampler::O
    J::Float64
    hx::Float64
    hz::Float64
    P_normalization::Float64
    Ns::Int
    Nb::Int
    energy_shift::Float64
end

function LTFIM(dims::NTuple{N, Int}, J::Float64, hx::Float64, hz::Float64, pbc=true) where N
    ops, p, Ns, Nb, energy_shift = make_prob_vector(dims, J, hx, hz, pbc)
    op_sampler = OperatorSampler(ops, p)
    return LTFIM{N, typeof(op_sampler)}(op_sampler, J, hx, hz, sum(p), Ns, Nb, energy_shift)
end

###############################################################################


zero(H::Hamiltonian{2}) = falses(nspins(H))
zero(H::Hamiltonian) = zeros(nspins(H))
one(H::Hamiltonian{2}) = trues(nspins(H))
one(H::Hamiltonian) = ones(nspins(H))

nspins(H::Hamiltonian) = H.Ns
nbonds(H::Hamiltonian) = H.Nb
