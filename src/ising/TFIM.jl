abstract type AbstractIsing{O <: AbstractOperatorSampler} <: Hamiltonian{2,O} end
abstract type AbstractTFIM{O <: AbstractOperatorSampler} <: AbstractIsing{O} end

# H = -J ∑_{⟨ij⟩} σ_i σ_j - h ∑_i σ^x_i
struct NearestNeighbourTFIM{O} <: AbstractTFIM{O}
    op_sampler::O
    J::Float64
    hx::Float64
    Ns::Int
    energy_shift::Float64
end


# H = ∑_{ij} J_ij σ_i σ_j - ∑_i h_i σ^x_i
struct TFIM{O,M <: UpperTriangular{Float64},V <: AbstractVector{Float64}} <: AbstractTFIM{O}
    op_sampler::O
    J::M
    hx::V
    Ns::Int
    energy_shift::Float64
end


###############################################################################

# TFIM ops:
#  (-2,i,i) is an off-diagonal site operator h*sigma^x_i
#  (-1,i,i) is a diagonal site operator h
#  (0,0,0) is the identity operator I - NOT USED IN THE PROJECTOR CASE
#  (1,i,j) is a diagonal bond operator J(sigma^z_i sigma^z_j)
@inline isdiagonal(::Type{<:AbstractIsing}, op::NTuple{3,Int}) = @inbounds (op[1] != -2)
@inline isidentity(::Type{<:AbstractIsing}, op::NTuple{3,Int}) = @inbounds (op[1] == 0)
@inline issiteoperator(::Type{<:AbstractIsing}, op::NTuple{3,Int}) = @inbounds (op[1] < 0)
@inline isbondoperator(::Type{<:AbstractIsing}, op::NTuple{3,Int}) = @inbounds (op[1] > 0)
@inline isdiagonal(H::AbstractIsing, op::NTuple{3,Int}) = isdiagonal(typeof(H), op)
@inline isidentity(H::AbstractIsing, op::NTuple{3,Int}) = isidentity(typeof(H), op)
@inline issiteoperator(H::AbstractIsing, op::NTuple{3,Int}) = issiteoperator(typeof(H), op)
@inline isbondoperator(H::AbstractIsing, op::NTuple{3,Int}) = isbondoperator(typeof(H), op)

@inline getbondsites(::Type{<:AbstractIsing}, op::NTuple{3, Int}) = @inbounds (op[2], op[3])
@inline getbondsites(H::AbstractIsing, op::NTuple{3, Int}) = getbondsites(typeof(H), op)

@inline makeidentity(::Type{<:AbstractIsing}) = (0, 0, 0)
@inline makediagonalsiteop(::Type{<:AbstractIsing}, i::Int) = (-1, i, i)
@inline makeoffdiagonalsiteop(::Type{<:AbstractIsing}, i::Int) = (-2, i, i)
@inline makeidentity(H::AbstractIsing) = makeidentity(typeof(H))
@inline makediagonalsiteop(H::AbstractIsing, i::Int) = makediagonalsiteop(typeof(H), i)
@inline makeoffdiagonalsiteop(H::AbstractIsing, i::Int) = makeoffdiagonalsiteop(typeof(H), i)

@inline getbondtype(::AbstractTFIM, s1::Bool, s2::Bool) = 1

###############################################################################

function make_prob_vector(J::UpperTriangular{T}, hx::AbstractVector{T}) where T
    @assert length(hx) == size(J, 1) == size(J, 2)

    ops = Vector{NTuple{3, Int}}(undef, 0)
    p = Vector{T}(undef, 0)
    energy_shift = zero(T)

    for i in eachindex(hx)
        if !iszero(hx[i])
            push!(ops, makediagonalsiteop(AbstractTFIM, i))
            push!(p, hx[i])
            energy_shift += hx[i]
        end
    end

    # only take J_ij terms from upper triangle
    for j in axes(J, 2), i in axes(J, 1)
        if i < j  # i != j: we don't want self-interactions
            if J[i, j] != 0
                push!(ops, (1, i, j))
                push!(p, 2*abs(J[i, j]))
                energy_shift += abs(J[i, j])
            end
        end
    end

    return ops, p, energy_shift
end

function make_uniform_tfim(bond_spins::Vector{NTuple{2,Int}}, Ns::Int, J::T, hx::T) where T
    hx_ = hx * ones(T, Ns)
    J_ = zeros(T, Ns, Ns)
    for (i, j) in bond_spins
        i, j = (i <= j) ? (i, j) : (j, i)
        J_[i, j] = -J
    end

    return UpperTriangular(triu!(J_, 1)), hx_
end

###############################################################################

function TFIM(J::UpperTriangular{Float64}, hx::AbstractVector{Float64})
    @assert length(hx) == size(J, 1) == size(J, 2)

    ops, p, energy_shift = make_prob_vector(J, hx)
    Ns = length(hx)
    op_sampler = OperatorSampler(ops, p)

    return TFIM{typeof(op_sampler), typeof(J), typeof(hx)}(
        op_sampler, J, hx, Ns, energy_shift
    )
end


function TFIM(bond_spin, Ns::Int, Nb::Int, hx::Float64, J::Float64)
    J_, hx_ = make_uniform_tfim(bond_spin, Ns, J, hx)
    return TFIM(J_, hx_)
end

abstract type HXField; end
struct ConstantHX; end
struct VaryingHX; end

hxfield(::AbstractIsing) = VaryingHX()
hxfield(::NearestNeighbourTFIM) = ConstantHX()

total_hx(::ConstantHX, H::AbstractIsing) = nspins(H) * H.hx
total_hx(::VaryingHX, H::AbstractIsing) = sum(H.hx)
total_hx(H::AbstractIsing) = total_hx(hxfield(H), H)

Base.@propagate_inbounds isferromagnetic(H::TFIM, (site1, site2)::NTuple{2, Int}) = signbit(H.J[site1, site2])
haslongitudinalfield(::AbstractTFIM) = false


###############################################################################


function energy(::BinaryGroundState, H::AbstractIsing, ns::Vector{<:Real}; resampler::Function=jackknife)
    hx = total_hx(H)

    if !iszero(hx)
        E = -hx * resampler(inv, ns)
    else
        E = measurement(zero(H.energy_shift))
    end

    return H.energy_shift + E
end
