abstract type AbstractDiagonalUpdateStats; end


struct NoDiagonalUpdateStats <: AbstractDiagonalUpdateStats; end
@inline update_replacement_attempts!(D::NoDiagonalUpdateStats, ::Int) = D
@inline end_step!(D::NoDiagonalUpdateStats) = D
Base.getproperty(::NoDiagonalUpdateStats, ::Symbol) = NoUpdateStat()
summarize(::NoDiagonalUpdateStats) = NamedTuple()

OnlineStats.merge!(S_a::NoDiagonalUpdateStats, S_b::NoDiagonalUpdateStats) = S_a
OnlineStats.merge!(S_a::NoDiagonalUpdateStats, S_b::AbstractDiagonalUpdateStats) = deepcopy(S_b)
OnlineStats.merge!(S_a::AbstractDiagonalUpdateStats, S_b::NoDiagonalUpdateStats) = S_a

#########################################################################################

struct DiagonalUpdateStats{T <: Real} <: AbstractDiagonalUpdateStats
    operator_insertion::UpdateStat{T}
    matelem_insertion::UpdateStat{T}
    removal::UpdateStat{T}
    replace::UpdateStat{T}
    replacement_attempts::Variance{T}

    DiagonalUpdateStats{T}() where T = new{T}(
        UpdateStat(T), UpdateStat(T), UpdateStat(T), UpdateStat(T), Variance(T)
    )
end
DiagonalUpdateStats(T::Type=Float64) = DiagonalUpdateStats{T}()

#########################################################################################

struct DiagonalUpdateHistograms{T <: Real, R <: StepRangeLen} <: AbstractDiagonalUpdateStats
    operator_insertion::UpdateHistogram{T, R}
    matelem_insertion::UpdateHistogram{T, R}
    removal::UpdateHistogram{T, R}
    replace::UpdateHistogram{T, R}
    replacement_attempts::VectorHistogram

    DiagonalUpdateHistograms{T}(
        oi::UpdateHistogram{T, R}, mi::UpdateHistogram{T, R}, 
        rm::UpdateHistogram{T, R}, rp::UpdateHistogram{T, R}, 
        rps::VectorHistogram) where {T, R} = new{T, R}(oi, mi, rm, rp, rps)
end
DiagonalUpdateHistograms{T}(size::Int) where T = DiagonalUpdateHistograms{T}(
    UpdateHistogram{T}(size), UpdateHistogram{T}(size), 
    UpdateHistogram{T}(size), UpdateHistogram{T}(size), 
    VectorHistogram(size)
)
DiagonalUpdateHistograms(T::Type, size::Int) = DiagonalUpdateHistograms{T}(size)
DiagonalUpdateHistograms(size::Int) = DiagonalUpdateHistograms{Float64}(size)

#########################################################################################

@inline update_replacement_attempts!(D::AbstractDiagonalUpdateStats, n_iter::Int) = (fit!(D.replacement_attempts, n_iter); D)

function OnlineStats.merge!(D_a::U, D_b::U) where {U <: AbstractDiagonalUpdateStats}
    merge!(D_a.operator_insertion, D_b.operator_insertion)
    merge!(D_a.matelem_insertion, D_b.matelem_insertion)
    merge!(D_a.removal, D_b.removal)
    merge!(D_a.replace, D_b.replace)
    merge!(D_a.replacement_attempts, D_b.replacement_attempts)
    return D_a
end

@inline function end_step!(D::AbstractDiagonalUpdateStats)
    end_step!(D.operator_insertion)
    end_step!(D.matelem_insertion)
    end_step!(D.removal)
    end_step!(D.replace)
    return D
end

summarize(D::AbstractDiagonalUpdateStats) = (
    operator_insertion = summarize(D.operator_insertion),
    matelem_insertion = summarize(D.matelem_insertion),
    removal = summarize(D.removal),
    replace = summarize(D.replace),
    replacement_attempts = summarize(D.replacement_attempts),
)
