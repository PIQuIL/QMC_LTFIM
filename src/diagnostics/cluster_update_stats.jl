abstract type AbstractClusterUpdateStats; end


struct NoClusterUpdateStats <: AbstractClusterUpdateStats; end
@inline end_step!(C::NoClusterUpdateStats) = C
add_accepted_cluster_size!(C::NoClusterUpdateStats, size::Int) = C
add_rejected_cluster_size!(C::NoClusterUpdateStats, size::Int) = C
add_cluster_size!(C::NoClusterUpdateStats, size::Int) = C

Base.getproperty(::NoClusterUpdateStats, ::Symbol) = NoUpdateStat()
summarize(::NoClusterUpdateStats) = NamedTuple()


struct ClusterUpdateStats{T <: Real} <: AbstractClusterUpdateStats
    cluster_update_accept::UpdateStat{T}
    cluster_sizes::Variance{T}
    accepted_cluster_sizes::Variance{T}
    rejected_cluster_sizes::Variance{T}

    ClusterUpdateStats{T}() where T = new{T}(
        UpdateStat(T), Variance(T), Variance(T), Variance(T))
end
ClusterUpdateStats(T::Type=Float64) = ClusterUpdateStats{T}()

#########################################################################################

struct ClusterUpdateHistograms{T <: Real, R <: StepRangeLen} <: AbstractClusterUpdateStats
    cluster_update_accept::UpdateHistogram{T, R}
    cluster_sizes::VectorHistogram
    accepted_cluster_sizes::VectorHistogram
    rejected_cluster_sizes::VectorHistogram

    ClusterUpdateHistograms{T}(
        ca::UpdateHistogram{T, R}, cs::VectorHistogram, 
        acc::VectorHistogram, rej::VectorHistogram) where {T, R} = new{T, R}(ca, cs, acc, rej)
end
ClusterUpdateHistograms{T}(size::Int) where T = ClusterUpdateHistograms{T}(
    UpdateHistogram{T}(size), VectorHistogram(size), VectorHistogram(size), VectorHistogram(size)
)
ClusterUpdateHistograms(T::Type, size::Int) = ClusterUpdateHistograms{T}(size)
ClusterUpdateHistograms(size::Int) = ClusterUpdateHistograms{Float64}(size)

#########################################################################################

end_step!(C::AbstractClusterUpdateStats) = (end_step!(C.cluster_update_accept); C)
add_accepted_cluster_size!(C::AbstractClusterUpdateStats, size::Int) = (fit!(C.accepted_cluster_sizes, size); C)
add_rejected_cluster_size!(C::AbstractClusterUpdateStats, size::Int) = (fit!(C.rejected_cluster_sizes, size); C)
add_cluster_size!(C::AbstractClusterUpdateStats, size::Int) = (fit!(C.cluster_sizes, size); C)

summarize(C::AbstractClusterUpdateStats) = (
    cluster_update_accept = summarize(C.cluster_update_accept),
    cluster_sizes = summarize(C.cluster_sizes),
    accepted_cluster_sizes = summarize(C.accepted_cluster_sizes),
    rejected_cluster_sizes = summarize(C.rejected_cluster_sizes),
)

