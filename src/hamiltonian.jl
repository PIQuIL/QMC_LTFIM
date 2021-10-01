abstract type Hamiltonian{D,O<:AbstractOperatorSampler} end

localdim(::Hamiltonian{D}) where {D} = D

zero(H::Hamiltonian{2}) = zeros(Bool, nspins(H))
zero(H::Hamiltonian) = zeros(Int, nspins(H))
one(H::Hamiltonian{2}) = ones(Bool, nspins(H))
one(H::Hamiltonian) = ones(Int, nspins(H))

nspins(H::Hamiltonian) = H.Ns
nbonds(H::Hamiltonian) = H.Nb

@inline isdiagonal(H) = op -> isdiagonal(H, op)
@inline isidentity(H) = op -> isidentity(H, op)
@inline issiteoperator(H) = op -> issiteoperator(H, op)
@inline isbondoperator(H) = op -> isbondoperator(H, op)
@inline getsite(H) = op -> getsite(H, op)
@inline getbondsites(H) = op -> getbondsites(H, op)

@inline diag_update_normalization(H::Hamiltonian) = normalization(H.op_sampler)

energy(::Type{<:BinaryQMCState{Exponential}}, H::Hamiltonian, β::Float64, n::Int) = H.energy_shift - (n / β)
function energy(::Type{<:BinaryQMCState{Exponential}}, H::Hamiltonian, β::Float64, ns::Vector{T}) where {T <: Real}
    E = -mean_and_stderr(ns) / β
    return H.energy_shift + E
end
energy(qmc_state::BinaryQMCState, args...; kwargs...) = energy(typeof(qmc_state), args...; kwargs...)


energy_density(S::Type{<:BinaryQMCState}, H::Hamiltonian, args...; kwargs...) =
    energy(S, H, args...; kwargs...) / nspins(H)

energy_density(qmc_state::BinaryQMCState, args...; kwargs...) = energy_density(typeof(qmc_state), args...; kwargs...)

function QMCState{Power}(H::Hamiltonian{2,O}, M::Int, trialstate::Union{Nothing, AbstractTrialState}=nothing) where {K, O <: AbstractOperatorSampler{K}}
    z = zero(H)
    QMCState{Power, Bool}(z, init_op_list(2*M, Val{K}()), trialstate) #::QMCState{Power, eltype(z), K, typeof(z)}
end


function QMCState{Exponential}(H::Hamiltonian{2,O}, cutoff::Int, trialstate::Union{Nothing, AbstractTrialState}=nothing) where {K, O <: AbstractOperatorSampler{K}}
    z = zero(H)
    QMCState{Exponential, Bool}(z, init_op_list(cutoff, Val{K}()), trialstate) #::QMCState{Exponential, eltype(z), K, typeof(z)}
end
