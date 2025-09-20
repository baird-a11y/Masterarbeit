# =============================================================================
# DATA PROCESSING MODULE
# =============================================================================


using Statistics
using StatsBase  # HINZUGEFÜGT für percentile Funktion

"""
Downsampling durch Mittelwertbildung über factor×factor Blöcke
"""
function downsample_avg(data, factor)
    h, w = size(data)
    
    if h % factor != 0 || w % factor != 0
        error("Dimensionen ($h, $w) nicht durch factor=$factor teilbar!")
    end
    
    new_h, new_w = h ÷ factor, w ÷ factor
    
    reshaped = reshape(data, factor, new_h, factor, new_w)
    averaged = mean(reshaped, dims=(1, 3))
    return dropdims(averaged, dims=(1, 3))
end

"""
Upsampling durch Pixel-Wiederholung
"""
function upsample_nearest(data, factor)
    return repeat(data, inner=(factor, factor))
end

"""
Prüft ob eine Zahl eine Potenz von 2 ist und im gültigen Bereich liegt
"""
function detect_power_of_2(size_val)
    if size_val & (size_val - 1) != 0
        error("Auflösung $size_val ist keine Potenz von 2!")
    end
    
    if size_val < 64 || size_val > 512
        error("Auflösung $size_val außerhalb erlaubtem Bereich [64, 512]!")
    end
    
    return true
end

"""
Automatische Größenanpassung für Potenzen von 2
"""
function resize_power_of_2(data, target_size)
    current_size = size(data, 1)
    
    if isa(target_size, Tuple)
        target_size_val = target_size[1]
        if target_size[1] != target_size[2]
            error("Nur quadratische Zielgrößen unterstützt: $target_size")
        end
    else
        target_size_val = target_size
    end
    
    detect_power_of_2(current_size)
    detect_power_of_2(target_size_val)
    
    if current_size == target_size_val
        return copy(data)
    elseif current_size > target_size_val
        factor = current_size ÷ target_size_val
        if current_size % target_size_val != 0
            error("Downsampling-Factor $(current_size)÷$(target_size_val) ist nicht ganzzahlig!")
        end
        return downsample_avg(data, factor)
    else
        factor = target_size_val ÷ current_size
        if target_size_val % current_size != 0
            error("Upsampling-Factor $(target_size_val)÷$(current_size) ist nicht ganzzahlig!")
        end
        return upsample_nearest(data, factor)
    end
end

"""
Erkennt Auflösung und validiert sie
"""
function detect_resolution(data)
    size_val = size(data, 1)
    
    if size_val & (size_val - 1) != 0
        error("Auflösung $size_val ist keine Potenz von 2!")
    end
    
    if size_val < 64 || size_val > 512
        error("Auflösung $size_val außerhalb erlaubtem Bereich [64, 512]!")
    end
    
    return size_val
end

"""
Vollständige Vorverarbeitung eines LaMEM-Samples
"""
function preprocess_lamem_sample(x, z, phase, vx, vz, v_stokes; target_resolution=128)
    # Größenanpassung
    phase_resized = resize_power_of_2(phase, target_resolution)
    vx_resized = resize_power_of_2(vx, target_resolution)
    vz_resized = resize_power_of_2(vz, target_resolution)
    
    # Normalisierung
    vx_norm = Float32.(vx_resized ./ v_stokes)
    vz_norm = Float32.(vz_resized ./ v_stokes)
    phase_float = Float32.(phase_resized)
    
    # Tensor-Format für UNet (H, W, C, B)
    phase_tensor = reshape(phase_float, target_resolution, target_resolution, 1, 1)
    velocity_tensor = cat(vx_norm, vz_norm, dims=3)
    velocity_tensor = reshape(velocity_tensor, target_resolution, target_resolution, 2, 1)
    
    return phase_tensor, velocity_tensor
end

"""
Robuste Normalisierung mit Clipping für Outlier
"""
function robust_normalize(data; percentile_clip=99.5)
    # Berechne Perzentile für robuste Normalisierung
    data_vec = vec(data)
    lower_bound = StatsBase.percentile(data_vec, 100 - percentile_clip)
    upper_bound = StatsBase.percentile(data_vec, percentile_clip)
    
    # Clip extreme Werte
    data_clipped = clamp.(data, lower_bound, upper_bound)
    
    # Z-Score Normalisierung
    μ = mean(data_clipped)
    σ = std(data_clipped)
    
    # Verhindere Division durch 0
    if σ < 1e-8
        return zeros(Float32, size(data)), Float32(μ), Float32(σ)
    end
    
    data_normalized = (data_clipped .- μ) ./ σ
    
    return Float32.(data_normalized), Float32(μ), Float32(σ)
end

"""
Inverse Normalisierung für Vorhersagen
"""
function denormalize(data_normalized, μ, σ)
    return data_normalized .* σ .+ μ
end

"""
Speichert Normalisierungs-Parameter für spätere Verwendung
"""
mutable struct NormalizationParams
    vx_mean::Float32
    vx_std::Float32
    vz_mean::Float32
    vz_std::Float32
    v_stokes::Float32
end

"""
Verbesserte Vorverarbeitung mit Normalisierungs-Tracking
"""
function preprocess_lamem_sample_normalized(x, z, phase, vx, vz, v_stokes; 
                                           target_resolution=128,
                                           return_norm_params=false)
    # Größenanpassung
    phase_resized = resize_power_of_2(phase, target_resolution)
    vx_resized = resize_power_of_2(vx, target_resolution)
    vz_resized = resize_power_of_2(vz, target_resolution)
    
    # Robuste Normalisierung statt einfacher Stokes-Division
    vx_norm, vx_mean, vx_std = robust_normalize(vx_resized)
    vz_norm, vz_mean, vz_std = robust_normalize(vz_resized)
    
    phase_float = Float32.(phase_resized)
    
    # Tensor-Format für UNet (H, W, C, B)
    phase_tensor = reshape(phase_float, target_resolution, target_resolution, 1, 1)
    velocity_tensor = cat(vx_norm, vz_norm, dims=3)
    velocity_tensor = reshape(velocity_tensor, target_resolution, target_resolution, 2, 1)
    
    if return_norm_params
        norm_params = NormalizationParams(vx_mean, vx_std, vz_mean, vz_std, Float32(v_stokes))
        return phase_tensor, velocity_tensor, norm_params
    else
        return phase_tensor, velocity_tensor
    end
end

println("Data Processing Module geladen!")
println("Verfügbare Funktionen:")
println("  - resize_power_of_2(data, target_size)")
println("  - preprocess_lamem_sample(...)")
println("  - preprocess_lamem_sample_normalized(...)")
println("  - robust_normalize(data)")
println("  - detect_resolution(data)")