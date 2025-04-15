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


struct RunStats{T <: Real} <: AbstractRunStats
    diagonal_update::DiagonalUpdateStats{T}
    cluster_update::ClusterUpdateStats{T}

    RunStats{T}() where T = new{T}(
        DiagonalUpdateStats(T), ClusterUpdateStats(T))
end
RunStats(T::Type=Float64) = RunStats{T}()

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

@inline OnlineStats.fit!(R::AbstractRunStats, field::Symbol, val) = 
    (fit!(getproperty(R, field), val); R)
summarize(R::AbstractRunStats) = (
    diagonal_update = summarize(R.diagonal_update),
    cluster_update = summarize(R.cluster_update)
)

#########################################################################################
