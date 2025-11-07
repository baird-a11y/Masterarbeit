# =============================================================================
# DATA PROCESSING - CLEAN VERSION (256x256 only)
# =============================================================================
# Vereinfachtes Preprocessing für feste Auflösung

using Statistics
using StatsBase

println("Data Processing (Clean) wird geladen...")

# =============================================================================
# KONSTANTEN
# =============================================================================

const TARGET_RESOLUTION = 256
const VELOCITY_NORMALIZATION_METHOD = :robust  # :robust oder :standard

# =============================================================================
# NORMALISIERUNG
# =============================================================================

"""
    robust_normalize(data; percentile_clip=99.5)

Robuste Z-Score Normalisierung mit Outlier-Clipping.

Verwendet Perzentile um extreme Ausreißer zu clippen,
dann Standard-Normalisierung (μ=0, σ=1).

# Arguments
- `data::AbstractArray`: Zu normalisierende Daten
- `percentile_clip::Float64`: Perzentil für Clipping (default: 99.5)

# Returns
- `Tuple`: (normalized_data, μ, σ)
"""
function robust_normalize(data::AbstractArray{T}; percentile_clip::Float64=99.5) where T
    data_vec = vec(data)
    
    # Perzentile für robuste Grenzen
    lower_bound = StatsBase.percentile(data_vec, 100 - percentile_clip)
    upper_bound = StatsBase.percentile(data_vec, percentile_clip)
    
    # Clip extreme Werte
    data_clipped = clamp.(data, lower_bound, upper_bound)
    
    # Z-Score Normalisierung
    μ = mean(data_clipped)
    σ = std(data_clipped) + 1f-8  # Verhindere Division durch 0
    
    data_normalized = (data_clipped .- μ) ./ σ
    
    return Float32.(data_normalized), Float32(μ), Float32(σ)
end

"""
    denormalize(data_normalized, μ, σ)

Inverse Normalisierung.
"""
function denormalize(data_normalized::AbstractArray, μ::Float32, σ::Float32)
    return (data_normalized .* σ) .+ μ
end

# =============================================================================
# PREPROCESSING PIPELINE
# =============================================================================

"""
    preprocess_phase_field(phase::AbstractArray)

Preprocessing für Phasenfeld.

# Transformationen
1. Konvertiere zu Float32
2. Binary Encoding: 0.0 (Matrix) vs. 1.0 (Kristall)
3. Reshape zu [H, W, 1, 1] für UNet

# Returns
- `Array{Float32,4}`: [256, 256, 1, 1]
"""
function preprocess_phase_field(phase::AbstractArray{T,2}) where T
    H, W = size(phase)
    
    if (H, W) != (TARGET_RESOLUTION, TARGET_RESOLUTION)
        error("Phase-Feld Größe $((H,W)) != $(TARGET_RESOLUTION)x$(TARGET_RESOLUTION)")
    end
    
    # Binary Encoding (Phase 1=Matrix→0, Phase 2=Crystal→1)
    phase_binary = Float32.(phase .== 2)
    
    # Reshape für UNet [H, W, C, B]
    phase_tensor = reshape(phase_binary, TARGET_RESOLUTION, TARGET_RESOLUTION, 1, 1)
    
    return phase_tensor
end

"""
    preprocess_velocity_field(vx, vz; method=:robust)

Preprocessing für Geschwindigkeitsfeld.

# Transformationen
1. Stack zu [H, W, 2]
2. Normalisierung (robust oder standard)
3. Reshape zu [H, W, 2, 1] für UNet

# Returns
- `Tuple`: (velocity_tensor, vx_stats, vz_stats)
  - velocity_tensor: [256, 256, 2, 1]
  - vx_stats: (μ, σ) für Denormalisierung
  - vz_stats: (μ, σ) für Denormalisierung
"""
function preprocess_velocity_field(
    vx::AbstractArray{T,2},
    vz::AbstractArray{T,2};
    method::Symbol = VELOCITY_NORMALIZATION_METHOD
) where T
    H, W = size(vx)
    
    if (H, W) != (TARGET_RESOLUTION, TARGET_RESOLUTION)
        error("Velocity-Feld Größe $((H,W)) != $(TARGET_RESOLUTION)x$(TARGET_RESOLUTION)")
    end
    
    # Normalisierung
    if method == :robust
        vx_norm, μx, σx = robust_normalize(vx)
        vz_norm, μz, σz = robust_normalize(vz)
    else  # :standard
        μx, σx = mean(vx), std(vx) + 1f-8
        μz, σz = mean(vz), std(vz) + 1f-8
        vx_norm = Float32.((vx .- μx) ./ σx)
        vz_norm = Float32.((vz .- μz) ./ σz)
    end
    
    # Stack und Reshape [H, W, 2, 1]
    velocity_stacked = cat(vx_norm, vz_norm, dims=3)
    velocity_tensor = reshape(velocity_stacked, TARGET_RESOLUTION, TARGET_RESOLUTION, 2, 1)
    
    return velocity_tensor, (μx, σx), (μz, σz)
end

"""
    preprocess_sample(phase, vx, vz)

Komplettes Preprocessing eines LaMEM Samples.

# Returns
- `Tuple`: (phase_tensor, velocity_tensor, velocity_stats)
"""
function preprocess_sample(
    phase::AbstractArray{T1,2},
    vx::AbstractArray{T2,2},
    vz::AbstractArray{T3,2}
) where {T1,T2,T3}
    
    phase_tensor = preprocess_phase_field(phase)
    velocity_tensor, vx_stats, vz_stats = preprocess_velocity_field(vx, vz)
    
    return phase_tensor, velocity_tensor, (vx_stats, vz_stats)
end

# =============================================================================
# BATCH CREATION (vereinfacht für feste Größe)
# =============================================================================

"""
    create_batch(samples)

Erstellt Batch aus mehreren Samples.
Da alle Samples 256x256 sind, ist keine adaptive Größenanpassung nötig.

# Arguments
- `samples::Vector{Tuple}`: [(phase, crystal_params, velocity, x, z), ...]

# Returns
- `Tuple`: (phase_batch, velocity_batch, crystal_params_batch, stats_batch)
"""
function create_batch(samples::Vector)
    n_samples = length(samples)
    
    if n_samples == 0
        error("Leerer Batch!")
    end
    
    # Initialisiere Arrays
    phase_batch = zeros(Float32, TARGET_RESOLUTION, TARGET_RESOLUTION, 1, n_samples)
    velocity_batch = zeros(Float32, TARGET_RESOLUTION, TARGET_RESOLUTION, 2, n_samples)
    
    crystal_params_batch = []
    stats_batch = []
    
    for (i, sample) in enumerate(samples)
        phase, crystal_params, velocity, _, _ = sample
        
        # Extrahiere vx, vz
        vx = velocity[:, :, 1]
        vz = velocity[:, :, 2]
        
        # Preprocess
        phase_tensor, velocity_tensor, stats = preprocess_sample(phase, vx, vz)
        
        # In Batch einfügen
        phase_batch[:, :, :, i] .= phase_tensor
        velocity_batch[:, :, :, i] .= velocity_tensor
        
        push!(crystal_params_batch, crystal_params)
        push!(stats_batch, stats)
    end
    
    return phase_batch, velocity_batch, crystal_params_batch, stats_batch
end

# =============================================================================
# NORMALISIERUNG VON STOKES-FELDERN
# =============================================================================

"""
    normalize_stokes_field(v_stokes, stats_batch)

Normalisiert Stokes-Geschwindigkeitsfeld mit denselben Stats wie Target.

WICHTIG: Komplett nicht-mutierend für Zygote-Kompatibilität!

# Arguments
- `v_stokes::AbstractArray{T,4}`: Stokes-Feld [H, W, 2, B]
- `stats_batch::Vector`: [(vx_stats, vz_stats), ...] pro Batch

# Returns
- `AbstractArray{T,4}`: Normalisiertes Stokes-Feld
"""
function normalize_stokes_field(
    v_stokes::AbstractArray{T,4},
    stats_batch::Vector
) where T
    H, W, _, B = size(v_stokes)
    
    # Nicht-mutierend: List comprehension für jeden Batch
    v_normalized_list = [
        begin
            vx_stats, vz_stats = stats_batch[b]
            μx, σx = vx_stats
            μz, σz = vz_stats
            
            # Erstelle normalisiertes Array für diesen Batch
            vx_norm = (v_stokes[:, :, 1, b] .- μx) ./ σx
            vz_norm = (v_stokes[:, :, 2, b] .- μz) ./ σz
            
            cat(vx_norm, vz_norm, dims=3)
        end
        for b in 1:B
    ]
    
    # Concatenate entlang Batch-Dimension
    return cat(v_normalized_list..., dims=4)
end

# =============================================================================
# POST-PROCESSING
# =============================================================================

"""
    postprocess_velocity_prediction(pred, vx_stats, vz_stats)

Denormalisiert Geschwindigkeitsvorhersage zurück zu physikalischen Einheiten.

# Arguments
- `pred::AbstractArray{T,4}`: Prediction [H, W, 2, B]
- `vx_stats::Tuple{Float32,Float32}`: (μx, σx)
- `vz_stats::Tuple{Float32,Float32}`: (μz, σz)

# Returns
- `Tuple`: (vx, vz) in physikalischen Einheiten
"""
function postprocess_velocity_prediction(
    pred::AbstractArray{T,4},
    vx_stats::Tuple{Float32,Float32},
    vz_stats::Tuple{Float32,Float32}
) where T
    
    μx, σx = vx_stats
    μz, σz = vz_stats
    
    # Extrahiere Komponenten
    vx_norm = pred[:, :, 1, :]
    vz_norm = pred[:, :, 2, :]
    
    # Denormalisiere
    vx = denormalize(vx_norm, μx, σx)
    vz = denormalize(vz_norm, μz, σz)
    
    return vx, vz
end

# =============================================================================
# DATEN-VALIDIERUNG
# =============================================================================

"""
    validate_sample(phase, velocity)

Validiert Sample-Dimensionen und Datentypen.
"""
function validate_sample(phase::AbstractArray, velocity::AbstractArray)
    # Dimensionen
    if size(phase) != (TARGET_RESOLUTION, TARGET_RESOLUTION)
        error("Phase-Feld: $(size(phase)) != $(TARGET_RESOLUTION)x$(TARGET_RESOLUTION)")
    end
    
    if size(velocity, 1) != TARGET_RESOLUTION || size(velocity, 2) != TARGET_RESOLUTION
        error("Velocity-Feld: $(size(velocity)[1:2]) != $(TARGET_RESOLUTION)x$(TARGET_RESOLUTION)")
    end
    
    if size(velocity, 3) != 2
        error("Velocity muss 2 Kanäle haben (vx, vz), hat: $(size(velocity, 3))")
    end
    
    # Wertebereich Phasenfeld
    unique_phases = unique(phase)
    if !all(p ∈ [1, 2] for p in unique_phases)
        @warn "Unerwartete Phasen-Werte: $unique_phases (erwartet: [1, 2])"
    end
    
    # NaN/Inf Check
    if any(isnan, phase) || any(isinf, phase)
        error("Phase-Feld enthält NaN/Inf")
    end
    
    if any(isnan, velocity) || any(isinf, velocity)
        error("Velocity-Feld enthält NaN/Inf")
    end
    
    return true
end

# =============================================================================
# DIVERGENZ-BERECHNUNG (für Loss)
# =============================================================================

"""
    compute_divergence(velocity::AbstractArray{T,4})

Berechnet Divergenz ∇·v für Geschwindigkeitsfeld.

# Arguments
- `velocity::AbstractArray{T,4}`: [H, W, 2, B] mit (vx, vz)

# Returns
- `AbstractArray{T,4}`: Divergenz [H-1, W-1, 1, B]
"""
function compute_divergence(velocity::AbstractArray{T,4}) where T
    vx = velocity[:, :, 1:1, :]  # [H, W, 1, B]
    vz = velocity[:, :, 2:2, :]  # [H, W, 1, B]
    
    # Finite Differenzen (zentral)
    ∂vx_∂x = vx[2:end, :, :, :] .- vx[1:end-1, :, :, :]
    ∂vz_∂z = vz[:, 2:end, :, :] .- vz[:, 1:end-1, :, :]
    
    # Match Dimensionen (nimm kleinere)
    min_h = min(size(∂vx_∂x, 1), size(∂vz_∂z, 1))
    min_w = min(size(∂vx_∂x, 2), size(∂vz_∂z, 2))
    
    ∂vx_∂x = ∂vx_∂x[1:min_h, 1:min_w, :, :]
    ∂vz_∂z = ∂vz_∂z[1:min_h, 1:min_w, :, :]
    
    # Divergenz
    div = ∂vx_∂x .+ ∂vz_∂z
    
    return div
end

# =============================================================================
# MODUL-INFO
# =============================================================================

println("Data Processing (Clean) geladen!")
println("   - Feste 256x256 Auflösung")
println("   - Robuste Normalisierung")
println("   - Automatische Validierung")
println("")
println("Wichtige Funktionen:")
println("   - preprocess_sample(phase, vx, vz)")
println("   - create_batch(samples)")
println("   - compute_divergence(velocity)")