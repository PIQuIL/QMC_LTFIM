abstract type AbstractRunStats; end

struct NoStats <: AbstractRunStats; end
@inline OnlineStats.fit!(R::NoStats, val) = R
@inline OnlineStats.fit!(R::NoStats, ::Symbol, val) = R
function Base.getproperty(R::NoStats, s::Symbol)
    if s === :diagonal_update
        return NoDiagonalUpdateStats()
    elseif s === :cluster_update
        return NoClusterUpdateStats()
    else
        return Base.getfield(R, s)
    end
end
summarize(R::NoStats) = NamedTuple()

OnlineStats.merge!(S_a::NoStats, S_b::NoStats) = S_a
OnlineStats.merge!(S_a::NoStats, S_b::AbstractRunStats) = deepcopy(S_b)
OnlineStats.merge!(S_a::AbstractRunStats, S_b::NoStats) = S_a

#########################################################################################

struct RunStats{T <: Real} <: AbstractRunStats
    diagonal_update::DiagonalUpdateStats{T}
    cluster_update::ClusterUpdateStats{T}

    RunStats{T}() where T = new{T}(
        DiagonalUpdateStats(T), ClusterUpdateStats(T))
end
RunStats(T::Type=Float64) = RunStats{T}()

#########################################################################################

struct RunStatsHistogram{T <: Real, R <: StepRangeLen} <: AbstractRunStats
    diagonal_update::DiagonalUpdateHistograms{T, R}
    cluster_update::ClusterUpdateHistograms{T, R}

    RunStatsHistogram{T}(
            d::DiagonalUpdateHistograms{T, R}, c::ClusterUpdateHistograms{T, R}) where {T, R} =
        new{T, R}(d, c)
end
RunStatsHistogram{T}(size::Int) where T = RunStatsHistogram{T}(
    DiagonalUpdateHistograms{T}(size), ClusterUpdateHistograms{T}(size))
RunStatsHistogram(T::Type, size::Int) = RunStatsHistogram{T}(size)
RunStatsHistogram(size::Int) = RunStatsHistogram{Float64}(size::Int)

#########################################################################################


function OnlineStats.merge!(R_a::S, R_b::S) where {S <: AbstractRunStats}
    merge!(R_a.diagonal_update, R_b.diagonal_update)
    merge!(R_a.cluster_update, R_b.cluster_update)
    return R_a
end

summarize(R::AbstractRunStats) = (
    diagonal_update = summarize(R.diagonal_update),
    cluster_update = summarize(R.cluster_update)
)

#########################################################################################
