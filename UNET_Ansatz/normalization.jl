# Datei: normalization.jl
# ψ-Normalisierung für U-Net
# Analog zum FNO_Ansatz: Gauge-Fix → Skalierung → Normalize/Denormalize
#
# Funktionen:
#   NormMeta           - Struct mit offset, scale, strategy
#   gauge_fix_psi      - Offset fixieren
#   compute_scale      - Skalenfaktor berechnen
#   normalize_psi      - Normalisieren
#   denormalize_psi    - Denormalisieren

module Normalization

using Statistics

export NormMeta,
       gauge_fix_psi,
       compute_scale,
       normalize_psi,
       denormalize_psi

# =============================================================================
# NormMeta Struct
# =============================================================================

"""
    NormMeta

Speichert alle Normalisierungsparameter eines Samples,
damit die Transformation vollständig rückgängig gemacht werden kann.
"""
struct NormMeta
    offset::Float64
    scale::Float64
    strategy::Symbol
    # Zusätzliche Diagnostik
    p_mean::Float64     # mittlere Zehnerpotenz (nur für :powmean, sonst NaN)
    maxabs::Float64     # max|ψ_fixed|
    std_val::Float64    # std(ψ_fixed)
end

# =============================================================================
# Gauge-Fix (Offset-Invarianz)
# =============================================================================

"""
    gauge_fix_psi(ψ; phase=nothing, mode=:fluid_mean, phase_crystal=1)

Fixiert die additive Konstante von ψ.

Modi:
- `:fluid_mean` → subtrahiert mean(ψ) im Fluid (phase .!= phase_crystal).
  Benötigt `phase`-Array.
- `:global_mean` → subtrahiert mean(ψ) über das gesamte Feld.
- `:corner` → subtrahiert ψ[1,1].

Gibt `(ψ_fixed, offset)` zurück.
"""
function gauge_fix_psi(ψ::AbstractMatrix;
                       phase::Union{AbstractMatrix, Nothing} = nothing,
                       mode::Symbol = :fluid_mean,
                       phase_crystal::Real = 1)
    if mode == :fluid_mean
        if phase === nothing
            error("mode=:fluid_mean benötigt ein phase-Array")
        end
        @assert size(ψ) == size(phase) "ψ und phase müssen gleiche Größe haben"
        fluid_mask = phase .!= phase_crystal
        n_fluid = count(fluid_mask)
        if n_fluid == 0
            @warn "Keine Fluid-Zellen gefunden, Fallback auf :global_mean"
            offset = Float64(mean(ψ))
        else
            offset = Float64(mean(ψ[fluid_mask]))
        end
    elseif mode == :global_mean
        offset = Float64(mean(ψ))
    elseif mode == :corner
        offset = Float64(ψ[1, 1])
    else
        error("Unknown mode=$mode (use :fluid_mean, :global_mean, or :corner)")
    end

    ψ_fixed = Float64.(ψ) .- offset
    return ψ_fixed, offset
end

# =============================================================================
# Skalenfaktor bestimmen
# =============================================================================

"""
    compute_scale(ψ_fixed; strategy=:powmean, eps=1e-30)

Bestimmt einen Skalenfaktor `s`, sodass `ψ_fixed * s` ungefähr O(1) wird.

Strategien:
- `:std`     → s = 1 / (std(ψ_fixed) + eps)
- `:maxabs`  → s = 1 / (max|ψ_fixed| + eps)
- `:powmean` → mittlere Zehnerpotenz von |ψ|, dann s = 10^(-p_mean)

Gibt `(scale, meta::NormMeta)` zurück.
"""
function compute_scale(ψ_fixed::AbstractMatrix;
                       strategy::Symbol = :powmean,
                       eps::Real = 1e-30)
    σ = std(Float64.(ψ_fixed))
    ma = maximum(abs.(ψ_fixed))

    if strategy == :std
        s = 1.0 / (σ + eps)
        p_mean = NaN
    elseif strategy == :maxabs
        s = 1.0 / (ma + eps)
        p_mean = NaN
    elseif strategy == :powmean
        # Mittlere Zehnerpotenz der Absolutwerte
        abs_vals = abs.(Float64.(ψ_fixed))
        nonzero = abs_vals[abs_vals .> eps]
        if isempty(nonzero)
            @warn "ψ_fixed ist quasi-null, Fallback scale=1.0"
            return 1.0, NormMeta(0.0, 1.0, strategy, 0.0, ma, σ)
        end
        log_vals = log10.(nonzero)
        p_mean = mean(floor.(log_vals))
        s = 10.0^(-p_mean)
    else
        error("Unknown strategy=$strategy (use :std, :maxabs, or :powmean)")
    end

    meta = NormMeta(0.0, s, strategy, isnan(p_mean) ? NaN : p_mean, ma, σ)
    return s, meta
end

# =============================================================================
# Normalisieren & Denormalisieren
# =============================================================================

"""
    normalize_psi(ψ_fixed, scale)

Normalisiert: `ψ_norm = ψ_fixed * scale`.
"""
function normalize_psi(ψ_fixed::AbstractMatrix, scale::Real)
    return Float64.(ψ_fixed) .* scale
end

"""
    denormalize_psi(ψ_norm, scale, offset)

Denormalisiert zurück in SI-Einheiten:
`ψ = ψ_norm / scale + offset`
"""
function denormalize_psi(ψ_norm::AbstractMatrix, scale::Real, offset::Real)
    return Float64.(ψ_norm) ./ scale .+ offset
end

"""
    denormalize_psi(ψ_norm, meta::NormMeta, offset)

Convenience-Variante mit NormMeta.
"""
function denormalize_psi(ψ_norm::AbstractMatrix, meta::NormMeta, offset::Real)
    return denormalize_psi(ψ_norm, meta.scale, offset)
end

end # module
