###############################################################################
# Walker's Alias Method: draw samples in O(1) time, initialize vector in O(n)
#   caveat: unsure if it's possible to build an efficient update scheme
#           for these types of probability vectors

struct ProbabilityAlias{T} <: AbstractProbabilityVector{T}
    probabilities::Vector{T}
    log_probs::Vector{T}
    normalization::T
    length::Int

    cutoffs::Vector{T}
    alias::Vector{Int}

    # initialize using Vose's algorithm
    function ProbabilityAlias{T}(probs::Vector{T}) where {T <: Real}
        if length(probs) == 0
            throw(ArgumentError("probability vector must have non-zero length!"))
        end
        if any(x -> x < zero(T), probs)
            throw(ArgumentError("weights must be non-negative!"))
        end

        probs_ = copy(probs) / sum(probs)
        N = length(probs)
        avg = inv(N)
        underfull = Stack{Int}()
        overfull = Stack{Int}()

        for (i, p) in enumerate(probs_)
            if p >= avg
                push!(overfull, i)
            else
                push!(underfull, i)
            end
        end

        cutoffs = zeros(float(T), N)
        alias = zeros(Int, N)

        while !isempty(underfull) && !isempty(overfull)
            less = pop!(underfull)
            more = pop!(overfull)

            cutoffs[less] = probs_[less] * N
            alias[less] = more

            probs_[more] += probs_[less]
            probs_[more] -= avg

            if probs_[more] >= avg
                push!(overfull, more)
            else
                push!(underfull, more)
            end
        end

        while !isempty(underfull)
            cutoffs[pop!(underfull)] = 1.0
        end

        while !isempty(overfull)
            cutoffs[pop!(overfull)] = 1.0
        end

        new{float(T)}(copy(probs), log.(probs), sum(probs), N, cutoffs, alias)
    end
end

ProbabilityAlias(p::Vector{T}) where T = ProbabilityAlias{T}(p)
@inline length(pvec::ProbabilityAlias) = pvec.length
@inline normalization(pvec::ProbabilityAlias) = pvec.normalization

function show(io::IO, p::ProbabilityAlias{T}) where T
    r = repr(p.probabilities; context=IOContext(io, :limit=>true))
    print(io, "ProbabilityAlias{$T}($r)")
end

function Base.rand(rng::AbstractRNG, pvec::ProbabilityAlias{T}) where T
    u, i::Int = modf(muladd(length(pvec), rand(rng), 1.0))
    return @inbounds (u < pvec.cutoffs[i]) ? i : pvec.alias[i]
end

using RandomNumbers.Xorshifts: AbstractXoroshiro128
# xoroshiro128 seems to be noticably faster if you just sample from it twice
function Base.rand(rng::AbstractXoroshiro128, pvec::ProbabilityAlias{T}) where T
    u = rand(rng)
    i = rand(rng, 1:length(pvec))
    return @inbounds (u < pvec.cutoffs[i]) ? i : pvec.alias[i]
end


@inline function getweight(pvec::ProbabilityAlias, i::Int)
    @boundscheck checkbounds(pvec.probabilities, i)
    return @inbounds pvec.probabilities[i]
end

@inline function getweight(pvec::ProbabilityAlias, r::AbstractArray{Int})
    @boundscheck checkbounds(pvec.probabilities, r)
    return @inbounds pvec.probabilities[r]
end

@inline function getlogweight(pvec::ProbabilityAlias, i::Int)
    @boundscheck checkbounds(pvec.log_probs, i)
    return @inbounds pvec.log_probs[i]
end

@inline function getlogweight(pvec::ProbabilityAlias, r::AbstractArray{Int})
    @boundscheck checkbounds(pvec.log_probs, r)
    return @inbounds pvec.log_probs[r]
end
