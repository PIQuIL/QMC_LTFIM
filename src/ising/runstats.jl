using OnlineStats


abstract type AbstractRunStats; end

struct NoStats <: AbstractRunStats; end
@inline OnlineStats.fit!(R::NoStats, ::Symbol, val) = R


struct RunStats{T <: Real} <: AbstractRunStats
    diag_update_fails::Variance{T}
    cluster_update_accept::Variance{T}
    cluster_count::Variance{T}
    cluster_sizes::Variance{T}

    RunStats{T}() where T = new{T}(Variance(T), Variance(T), Variance(T), Variance(T))
end
RunStats() = RunStats{Float64}()
@inline OnlineStats.fit!(R::RunStats, field::Symbol, val) = (fit!(getproperty(R, field), val); R)


struct RunStatsHistogram{T <: Real} <: AbstractRunStats
    diag_update_fails::KHist{T}
    cluster_update_accept::KHist{T}
    cluster_count::KHist{T}
    cluster_sizes::KHist{T}

    RunStatsHistogram{T}(size::Int) where T =
        new{T}(KHist(size, T), KHist(size, T), KHist(size, T), KHist(size, T))
end
RunStatsHistogram(size::Int) = RunStatsHistogram{Float64}(size::Int)
@inline OnlineStats.fit!(R::RunStatsHistogram, field::Symbol, val) = (fit!(getproperty(R, field), val); R)
