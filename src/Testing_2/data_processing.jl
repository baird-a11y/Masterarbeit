# =============================================================================
# 3D DATA PROCESSING 
# =============================================================================


using Statistics
using StatsBase
using Interpolations

# =============================================================================
# 3D RESIZE FUNCTIONS
# =============================================================================

"""
3D Nearest Neighbor Upsampling (erweitert von deiner 2D Version)
"""
function upsample_nearest_3d(data, factor)
    old_w, old_h, old_d = size(data)
    new_w, new_h, new_d = old_w * factor, old_h * factor, old_d * factor
    
    result = zeros(eltype(data), new_w, new_h, new_d)
    
    for i in 1:new_w
        for j in 1:new_h
            for k in 1:new_d
                # Mapping zu ursprünglichen Koordinaten
                orig_i = min(div(i - 1, factor) + 1, old_w)
                orig_j = min(div(j - 1, factor) + 1, old_h)
                orig_k = min(div(k - 1, factor) + 1, old_d)
                
                result[i, j, k] = data[orig_i, orig_j, orig_k]
            end
        end
    end
    
    return result
end

"""
3D Downsampling mit Averaging (erweitert von deiner 2D Version)
"""
function downsample_average_3d(data, factor)
    old_w, old_h, old_d = size(data)
    new_w = div(old_w, factor)
    new_h = div(old_h, factor)
    new_d = div(old_d, factor)
    
    result = zeros(eltype(data), new_w, new_h, new_d)
    
    for i in 1:new_w
        for j in 1:new_h
            for k in 1:new_d
                # Bereich für Averaging
                start_i = (i-1) * factor + 1
                end_i = min(i * factor, old_w)
                start_j = (j-1) * factor + 1
                end_j = min(j * factor, old_h)
                start_k = (k-1) * factor + 1
                end_k = min(k * factor, old_d)
                
                # Durchschnitt berechnen
                block = data[start_i:end_i, start_j:end_j, start_k:end_k]
                result[i, j, k] = mean(block)
            end
        end
    end
    
    return result
end

"""
3D Resize auf Potenz von 2 (erweitert von deiner 2D resize_power_of_2 Funktion)
"""
function resize_power_of_2_3d(data, target_size)
    current_w, current_h, current_d = size(data)
    
    # Prüfe ob Target Size gültig ist
    if target_size & (target_size - 1) != 0
        error("Target size $target_size ist keine Potenz von 2!")
    end
    
    # Falls bereits korrekte Größe
    if current_w == target_size && current_h == target_size && current_d == target_size
        return data
    end
    
    # Upsampling falls zu klein
    if current_w < target_size || current_h < target_size || current_d < target_size
        max_current = max(current_w, current_h, current_d)
        factor = div(target_size, max_current)
        if factor > 1
            return upsample_nearest_3d(data, factor)
        end
    end
    
    # Downsampling falls zu groß
    if current_w > target_size || current_h > target_size || current_d > target_size
        min_current = min(current_w, current_h, current_d)
        factor = div(min_current, target_size)
        if factor > 1
            return downsample_average_3d(data, factor)
        end
    end
    
    # Direct Cropping/Padding als Fallback
    return crop_or_pad_3d(data, target_size)
end

"""
3D Crop oder Pad auf exakte Größe
"""
function crop_or_pad_3d(data, target_size)
    current_w, current_h, current_d = size(data)
    result = zeros(eltype(data), target_size, target_size, target_size)
    
    # Berechne Copy-Bereiche
    copy_w = min(current_w, target_size)
    copy_h = min(current_h, target_size)
    copy_d = min(current_d, target_size)
    
    # Kopiere Daten
    result[1:copy_w, 1:copy_h, 1:copy_d] = data[1:copy_w, 1:copy_h, 1:copy_d]
    
    return result
end

# =============================================================================
# 3D INTERPOLATION FUNCTIONS
# =============================================================================

"""
3D Lineare Interpolation mit Interpolations.jl
"""
function interpolate_3d_linear(x_coords, y_coords, z_coords, data, target_resolution)
    # Erstelle Interpolationsobjekt
    itp = interpolate((x_coords, y_coords, z_coords), data, Gridded(Linear()))
    
    # Neue gleichmäßige Koordinaten
    x_new = range(minimum(x_coords), maximum(x_coords), length=target_resolution)
    y_new = range(minimum(y_coords), maximum(y_coords), length=target_resolution)
    z_new = range(minimum(z_coords), maximum(z_coords), length=target_resolution)
    
    # Interpoliere auf neues Grid
    result = zeros(Float32, target_resolution, target_resolution, target_resolution)
    
    for (i, x_val) in enumerate(x_new)
        for (j, y_val) in enumerate(y_new)
            for (k, z_val) in enumerate(z_new)
                result[i, j, k] = itp(x_val, y_val, z_val)
            end
        end
    end
    
    return result
end

# =============================================================================
# 3D HAUPTFUNKTIONEN
# =============================================================================

"""
3D Version deiner preprocess_lamem_sample Funktion
Verarbeitet LaMEM 3D-Output zu UNet-Format
"""
function preprocess_lamem_sample_3d(x, y, z, phase, vx, vy, vz, v_stokes; target_resolution=32)
    # FIX: Extrahiere Zahl aus Tupel falls nötig
    if isa(target_resolution, Tuple)
        target_size = target_resolution[1]  # Nimm erste Dimension
        println("Robuste 3D Verarbeitung: $(size(phase)) → ($(target_size))³")
    else
        target_size = target_resolution
        println("Robuste 3D Verarbeitung: $(size(phase)) → ($(target_size))³")
    end
    
    # Sichere Größenermittlung
    current_w, current_h, current_d = size(phase)
    
    # Rest der Funktion bleibt gleich, verwende target_size statt target_resolution
    if current_w != current_h || current_h != current_d
        println("  Warnung: Nicht-kubische Daten $(size(phase)), verwende kleinste Dimension")
        current_size = min(current_w, current_h, current_d)
    else
        current_size = current_w
    end
    
    println("  Aktuelle Größe: $current_size, Ziel: $target_size")
    
    # Sichere Datenextraktion - verwende target_size überall
    if current_size >= target_size
        # Cropping von Zentrum
        start_idx = div(current_size - target_size, 2) + 1
        end_idx = start_idx + target_size - 1
        
        phase_3d = phase[start_idx:end_idx, start_idx:end_idx, start_idx:end_idx]
        vx_3d = vx[start_idx:end_idx, start_idx:end_idx, start_idx:end_idx]
        vy_3d = vy[start_idx:end_idx, start_idx:end_idx, start_idx:end_idx]
        vz_3d = vz[start_idx:end_idx, start_idx:end_idx, start_idx:end_idx]
    else
        # Padding mit Nullen
        phase_3d = zeros(eltype(phase), target_size, target_size, target_size)
        vx_3d = zeros(eltype(vx), target_size, target_size, target_size)
        vy_3d = zeros(eltype(vy), target_size, target_size, target_size)
        vz_3d = zeros(eltype(vz), target_size, target_size, target_size)
        
        # Zentrale Platzierung
        start_idx = div(target_size - current_size, 2) + 1
        end_idx = start_idx + current_size - 1
        
        phase_3d[start_idx:end_idx, start_idx:end_idx, start_idx:end_idx] = phase
        vx_3d[start_idx:end_idx, start_idx:end_idx, start_idx:end_idx] = vx
        vy_3d[start_idx:end_idx, start_idx:end_idx, start_idx:end_idx] = vy
        vz_3d[start_idx:end_idx, start_idx:end_idx, start_idx:end_idx] = vz
    end
    
    # Normalisierung
    vx_norm = Float32.(vx_3d ./ v_stokes)
    vy_norm = Float32.(vy_3d ./ v_stokes)
    vz_norm = Float32.(vz_3d ./ v_stokes)
    
    # Tensoren - verwende target_size
    phase_tensor = reshape(Float32.(phase_3d), target_size, target_size, target_size, 1, 1)
    velocity_tensor = cat(vx_norm, vy_norm, vz_norm, dims=4)
    velocity_tensor = reshape(velocity_tensor, target_size, target_size, target_size, 3, 1)
    
    println("  ✓ 3D Preprocessing erfolgreich: $(size(phase_tensor)) und $(size(velocity_tensor))")
    
    return phase_tensor, velocity_tensor
end

"""
3D Version deiner preprocess_lamem_sample_normalized Funktion
"""
function preprocess_lamem_sample_normalized_3d(x, y, z, phase, vx, vy, vz, v_stokes; 
                                              target_resolution=64,
                                              return_norm_params=false)
    # 3D Größenanpassung
    phase_resized = resize_power_of_2_3d(phase, target_resolution)
    vx_resized = resize_power_of_2_3d(vx, target_resolution)
    vy_resized = resize_power_of_2_3d(vy, target_resolution)
    vz_resized = resize_power_of_2_3d(vz, target_resolution)
    
    # Robuste Normalisierung (deine Funktion erweitert)
    vx_norm, vx_mean, vx_std = robust_normalize_3d(vx_resized)
    vy_norm, vy_mean, vy_std = robust_normalize_3d(vy_resized)
    vz_norm, vz_mean, vz_std = robust_normalize_3d(vz_resized)
    
    phase_float = Float32.(phase_resized)
    
    # 3D Tensor-Format (W, H, D, C, B)
    phase_tensor = reshape(phase_float, target_resolution, target_resolution, target_resolution, 1, 1)
    velocity_tensor = cat(vx_norm, vy_norm, vz_norm, dims=4)
    velocity_tensor = reshape(velocity_tensor, target_resolution, target_resolution, target_resolution, 3, 1)
    
    if return_norm_params
        norm_params = NormalizationParams3D(
            vx_mean, vx_std, vy_mean, vy_std, vz_mean, vz_std, Float32(v_stokes)
        )
        return phase_tensor, velocity_tensor, norm_params
    else
        return phase_tensor, velocity_tensor
    end
end

"""
3D Version deiner robust_normalize Funktion
"""
function robust_normalize_3d(data; percentile_clip=99.5)
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

# =============================================================================
# 3D UTILITY STRUCTURES
# =============================================================================

"""
3D Version deiner NormalizationParams Struktur
"""
mutable struct NormalizationParams3D
    vx_mean::Float32
    vx_std::Float32
    vy_mean::Float32  # Neue Y-Komponente
    vy_std::Float32
    vz_mean::Float32
    vz_std::Float32
    v_stokes::Float32
end

"""
3D Denormalisierung
"""
function denormalize_3d(vx_norm, vy_norm, vz_norm, params::NormalizationParams3D)
    vx = vx_norm .* params.vx_std .+ params.vx_mean
    vy = vy_norm .* params.vy_std .+ params.vy_mean
    vz = vz_norm .* params.vz_std .+ params.vz_mean
    
    return vx, vy, vz
end

# =============================================================================
# 3D RESOLUTION DETECTION - ERWEITERT VON DEINEM CODE
# =============================================================================

"""
3D Version deiner detect_resolution Funktion
"""
function detect_resolution_3d(data)
    w, h, d = size(data)
    
    # Prüfe ob kubisch
    if w != h || h != d
        error("3D Daten müssen kubisch sein! Größe: $(size(data))")
    end
    
    size_val = w
    
    # Prüfe Potenz von 2
    if size_val & (size_val - 1) != 0
        error("3D Auflösung $size_val ist keine Potenz von 2!")
    end
    
    # Prüfe erlaubten Bereich (kleiner für 3D wegen Speicher)
    if size_val < 32 || size_val > 128
        error("3D Auflösung $size_val außerhalb erlaubtem Bereich [32, 128]!")
    end
    
    return size_val
end

# =============================================================================
# 3D BATCH PROCESSING
# =============================================================================

"""
3D Version für Integration in dein bestehendes Batch-System
"""
function create_3d_sample_for_batch(sample_3d, target_resolution)
    # Entpacke 3D Sample (erweitert von deinem 2D Format)
    if length(sample_3d) >= 9  # x, y, z, phase, vx, vy, vz, extras..., v_stokes
        x, y, z, phase, vx, vy, vz = sample_3d[1:7]
        v_stokes = sample_3d[end]  # Letzter Wert ist immer v_stokes
        
        # Verwende 3D Preprocessing
        phase_tensor, velocity_tensor = preprocess_lamem_sample_3d(
            x, y, z, phase, vx, vy, vz, v_stokes; 
            target_resolution=target_resolution
        )
        
        return phase_tensor, velocity_tensor
    else
        error("3D Sample hat ungültiges Format: $(length(sample_3d)) Elemente")
    end
end

# =============================================================================
# MIGRATION UTILITIES - FÜR EINFACHEN ÜBERGANG
# =============================================================================

"""
Konvertiert 2D Sample zu 3D Sample (für Tests)
"""
function convert_2d_to_3d_sample(x_2d, z_2d, phase_2d, vx_2d, vz_2d, v_stokes; 
                                 y_extent=1.0, target_resolution=64)
    # Erstelle Y-Koordinaten
    y_2d = [0.0]  # Einzelne Schicht
    
    # Erweitere zu 3D durch Extrusion
    phase_3d = repeat(phase_2d[:, :, :], 1, 1, 1)  # Triviale 3D-Erweiterung
    vx_3d = repeat(vx_2d[:, :, :], 1, 1, 1)
    vy_3d = zeros(size(vx_3d))  # Keine Y-Bewegung
    vz_3d = repeat(vz_2d[:, :, :], 1, 1, 1)
    
    return x_2d, y_2d, z_2d, phase_3d, vx_3d, vy_3d, vz_3d, v_stokes
end

"""
Test der 3D Datenverarbeitung
"""
function test_3d_data_processing()
    println("=== 3D DATA PROCESSING TEST ===")
    
    try
        # Erstelle Test-Daten
        test_resolution = 32  # Klein für Test
        
        x = range(-1, 1, length=test_resolution)
        y = range(-1, 1, length=test_resolution)
        z = range(-1, 1, length=test_resolution)
        
        # 3D Test-Arrays
        phase_test = zeros(Float32, test_resolution, test_resolution, test_resolution)
        vx_test = randn(Float32, test_resolution, test_resolution, test_resolution)
        vy_test = randn(Float32, test_resolution, test_resolution, test_resolution)
        vz_test = randn(Float32, test_resolution, test_resolution, test_resolution)
        
        # Füge Kristall hinzu (Sphere)
        center = (test_resolution÷2, test_resolution÷2, test_resolution÷2)
        radius = test_resolution÷4
        
        for i in 1:test_resolution
            for j in 1:test_resolution
                for k in 1:test_resolution
                    dist = sqrt((i - center[1])^2 + (j - center[2])^2 + (k - center[3])^2)
                    if dist <= radius
                        phase_test[i, j, k] = 1.0
                    end
                end
            end
        end
        
        v_stokes = 1.0
        
        println("✓ 3D Test-Daten erstellt: $(size(phase_test))")
        
        # Teste 3D Preprocessing
        phase_tensor, velocity_tensor = preprocess_lamem_sample_3d(
            x, y, z, phase_test, vx_test, vy_test, vz_test, v_stokes;
            target_resolution=test_resolution
        )
        
        println("✓ 3D Preprocessing erfolgreich")
        println("✓ Phase Tensor: $(size(phase_tensor))")
        println("✓ Velocity Tensor: $(size(velocity_tensor))")
        
        # Teste 3D Resize
        resized_phase = resize_power_of_2_3d(phase_test, 16)
        println("✓ 3D Resize: $(size(phase_test)) → $(size(resized_phase))")
        
        # Teste 3D Normalisierung
        normalized, μ, σ = robust_normalize_3d(vx_test)
        println("✓ 3D Normalisierung: μ=$(round(μ, digits=3)), σ=$(round(σ, digits=3))")
        
        return true
        
    catch e
        println("✗ 3D Data Processing Test fehlgeschlagen: $e")
        return false
    end
end

