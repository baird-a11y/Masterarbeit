# =============================================================================
# RESIDUAL LEARNING EVALUATION - METRIKEN-MODUL
# =============================================================================

using Statistics
using LinearAlgebra

# =============================================================================
# PHYSIKALISCHE KONSISTENZ-METRIKEN (KRITISCH FÜR NICHT-STREAM-FUNCTION)
# =============================================================================

"""
Berechnet Kontinuitätsverletzung (Divergenz) mit vollständiger Statistik
OPTIMIERT: Zentrale Differenzen, single-pass Berechnung

Returns: (mean, max, std, l2_norm)
"""
function calculate_continuity_metrics(vx::AbstractMatrix{T}, vz::AbstractMatrix{T}) where T
    H, W = size(vx)
    
    # Preallocate divergence field
    div_field = similar(vx)
    
    # Zentrale Differenzen (2. Ordnung genau)
    @inbounds for i in 2:H-1, j in 2:W-1
        dvx_dx = (vx[i, j+1] - vx[i, j-1]) / 2.0
        dvz_dz = (vz[i+1, j] - vz[i-1, j]) / 2.0
        div_field[i, j] = abs(dvx_dx + dvz_dz)
    end
    
    # Randbehandlung (einseitige Differenzen)
    for i in [1, H], j in 1:W
        dvx_dx = (i == 1) ? (vx[i, min(j+1, W)] - vx[i, j]) : (vx[i, j] - vx[i, max(j-1, 1)])
        dvz_dz = (j == 1) ? (vz[min(i+1, H), j] - vz[i, j]) : (vz[i, j] - vz[max(i-1, 1), j])
        div_field[i, j] = abs(dvx_dx + dvz_dz)
    end
    
    # Single-pass Statistiken
    div_mean = mean(div_field)
    div_max = maximum(div_field)
    div_std = std(div_field)
    div_l2 = norm(div_field)
    
    return (div_mean, div_max, div_std, div_l2)
end

"""
Berechnet Divergenz-Ähnlichkeit zwischen Prediction und Ground Truth
WICHTIG: Misst ob das Netzwerk das PATTERN der Divergenz lernt, nicht nur Magnitude
"""
function calculate_divergence_similarity(vx_pred::AbstractMatrix, vz_pred::AbstractMatrix,
                                        vx_gt::AbstractMatrix, vz_gt::AbstractMatrix)
    # Berechne beide Divergenz-Felder
    div_pred = compute_divergence_field(vx_pred, vz_pred)
    div_gt = compute_divergence_field(vx_gt, vz_gt)
    
    # Pearson-Korrelation der Divergenz-Felder
    correlation = cor(vec(div_pred), vec(div_gt))
    
    # Normalisierte Ähnlichkeit [0, 1]
    similarity = (correlation + 1.0) / 2.0
    
    return isnan(similarity) ? 0.0 : similarity
end

"""
Hilfsfunktion: Berechnet Divergenz-Feld (nicht-absolute Werte)
"""
function compute_divergence_field(vx::AbstractMatrix{T}, vz::AbstractMatrix{T}) where T
    H, W = size(vx)
    div_field = similar(vx)
    
    @inbounds for i in 2:H-1, j in 2:W-1
        dvx_dx = (vx[i, j+1] - vx[i, j-1]) / 2.0
        dvz_dz = (vz[i+1, j] - vz[i-1, j]) / 2.0
        div_field[i, j] = dvx_dx + dvz_dz
    end
    
    # Ränder
    div_field[1, :] .= 0.0
    div_field[H, :] .= 0.0
    div_field[:, 1] .= 0.0
    div_field[:, W] .= 0.0
    
    return div_field
end

"""
Berechnet Vortizitäts-Erhaltung (curl preservation)
PHYSIK: ω = ∂vx/∂z - ∂vz/∂x
"""
function calculate_vorticity_preservation(vx_gt, vz_gt, vx_pred, vz_pred)
    # Berechne Vortizität für beide Felder
    vort_gt = compute_vorticity_field(vx_gt, vz_gt)
    vort_pred = compute_vorticity_field(vx_pred, vz_pred)
    
    # Korrelation der Vortizitäts-Felder
    preservation = cor(vec(vort_gt), vec(vort_pred))
    
    return isnan(preservation) ? 0.0 : preservation
end

"""
Hilfsfunktion: Berechnet Vortizitäts-Feld
"""
function compute_vorticity_field(vx, vz)
    H, W = size(vx)
    vort = similar(vx)
    
    @inbounds for i in 2:H-1, j in 2:W-1
        dvx_dz = (vx[i+1, j] - vx[i-1, j]) / 2.0
        dvz_dx = (vz[i, j+1] - vz[i, j-1]) / 2.0
        vort[i, j] = dvx_dz - dvz_dx
    end
    
    # Ränder auf 0
    vort[1, :] .= 0.0; vort[H, :] .= 0.0
    vort[:, 1] .= 0.0; vort[:, W] .= 0.0
    
    return vort
end

# =============================================================================
# RESIDUUM-ANALYSE (SPEZIFISCH FÜR RESIDUAL LEARNING)
# =============================================================================

"""
Berechnet umfassende Residuum-Statistiken
OPTIMIERT: Single-pass Berechnung aller Metriken

Returns: (mean, max, std, sparsity, stokes_ratio)
"""
function calculate_residual_statistics(Δv_x::AbstractMatrix, Δv_z::AbstractMatrix,
                                       v_stokes_x::AbstractMatrix, v_stokes_z::AbstractMatrix;
                                       sparsity_threshold::Float64=1e-4)
    # Residuum-Magnitude (vektorweise)
    residual_mag = sqrt.(Δv_x.^2 .+ Δv_z.^2)
    
    # Statistiken
    res_mean = mean(residual_mag)
    res_max = maximum(residual_mag)
    res_std = std(residual_mag)
    
    # Sparsity: Anteil kleiner Residuen
    sparsity = count(residual_mag .< sparsity_threshold) / length(residual_mag)
    
    # Stokes-Magnitude
    stokes_mag = sqrt.(v_stokes_x.^2 .+ v_stokes_z.^2)
    mean_stokes = mean(stokes_mag)
    
    # Ratio
    stokes_ratio = res_mean / (mean_stokes + 1e-8)
    
    return (res_mean, res_max, res_std, sparsity, stokes_ratio)
end

"""
Berechnet Verbesserung über Stokes-Baseline
KERNMETRIK für Residual Learning!

Returns: (mae_stokes, mae_residual, improvement_percent)
"""
function calculate_stokes_improvement(v_total_x, v_total_z,
                                     v_stokes_x, v_stokes_z,
                                     v_gt_x, v_gt_z)
    # Fehler der reinen Stokes-Lösung
    error_stokes_x = abs.(v_stokes_x .- v_gt_x)
    error_stokes_z = abs.(v_stokes_z .- v_gt_z)
    mae_stokes = (mean(error_stokes_x) + mean(error_stokes_z)) / 2.0
    
    # Fehler mit Residuum-Korrektur
    error_total_x = abs.(v_total_x .- v_gt_x)
    error_total_z = abs.(v_total_z .- v_gt_z)
    mae_residual = (mean(error_total_x) + mean(error_total_z)) / 2.0
    
    # Relative Verbesserung
    improvement = ((mae_stokes - mae_residual) / mae_stokes) * 100.0
    
    return (mae_stokes, mae_residual, improvement)
end

# =============================================================================
# STRUKTURELLE ÄHNLICHKEIT
# =============================================================================

"""
SSIM (Structural Similarity Index) - OPTIMIERT
Type-stable, vektorisiert
"""
function calculate_ssim(img1::AbstractMatrix{T}, img2::AbstractMatrix{T};
                       window_size::Int=11, k1::Float64=0.01, k2::Float64=0.03) where T
    if size(img1) != size(img2)
        error("Bilder müssen gleiche Größe haben")
    end
    
    # Dynamischer Bereich
    L = max(maximum(abs, img1), maximum(abs, img2))
    c1 = (k1 * L)^2
    c2 = (k2 * L)^2
    
    # Lokale Mittelwerte (einfacher Box-Filter)
    μ1 = local_mean(img1, window_size)
    μ2 = local_mean(img2, window_size)
    
    # Varianzen und Kovarianz
    σ1² = local_mean(img1.^2, window_size) .- μ1.^2
    σ2² = local_mean(img2.^2, window_size) .- μ2.^2
    σ12 = local_mean(img1 .* img2, window_size) .- μ1 .* μ2
    
    # SSIM-Formel
    numerator = (2 .* μ1 .* μ2 .+ c1) .* (2 .* σ12 .+ c2)
    denominator = (μ1.^2 .+ μ2.^2 .+ c1) .* (σ1² .+ σ2² .+ c2)
    
    ssim_map = numerator ./ (denominator .+ 1e-8)
    
    return mean(ssim_map)
end

"""
Hilfsfunktion: Lokaler Mittelwert mit Box-Filter
"""
function local_mean(img::AbstractMatrix, window_size::Int)
    H, W = size(img)
    result = similar(img)
    half_win = window_size ÷ 2
    
    for i in 1:H, j in 1:W
        i_start = max(1, i - half_win)
        i_end = min(H, i + half_win)
        j_start = max(1, j - half_win)
        j_end = min(W, j + half_win)
        
        result[i, j] = mean(view(img, i_start:i_end, j_start:j_end))
    end
    
    return result
end

"""
Kreuzkorrelation (maximale Ähnlichkeit über Shifts)
"""
function calculate_cross_correlation(field1::AbstractMatrix, field2::AbstractMatrix)
    # Normalisiere
    f1 = (field1 .- mean(field1)) ./ (std(field1) + 1e-8)
    f2 = (field2 .- mean(field2)) ./ (std(field2) + 1e-8)
    
    # Direkte Korrelation (ohne Shift für Performance)
    corr = cor(vec(f1), vec(f2))
    
    return isnan(corr) ? 0.0 : abs(corr)
end

# =============================================================================
# FEHLERMETRIKEN (BASISFUNKTIONEN)
# =============================================================================

"""
Berechnet alle Standardfehler-Metriken auf einmal
OPTIMIERT: Single-pass für bessere Cache-Nutzung

Returns: (mae_x, mae_z, rmse_x, rmse_z, max_x, max_z)
"""
function calculate_error_metrics(pred_x, pred_z, gt_x, gt_z)
    # Fehler-Felder
    error_x = pred_x .- gt_x
    error_z = pred_z .- gt_z
    
    # MAE
    mae_x = mean(abs, error_x)
    mae_z = mean(abs, error_z)
    
    # RMSE
    rmse_x = sqrt(mean(error_x.^2))
    rmse_z = sqrt(mean(error_z.^2))
    
    # Max Error
    max_x = maximum(abs, error_x)
    max_z = maximum(abs, error_z)
    
    return (mae_x, mae_z, rmse_x, rmse_z, max_x, max_z)
end

"""
Pearson-Korrelation mit NaN-Handling
"""
function safe_correlation(x::AbstractArray, y::AbstractArray)
    corr = cor(vec(x), vec(y))
    return isnan(corr) ? 0.0 : corr
end

println("✓ Residual Evaluation Metriken-Modul geladen")
println("  - Physikalische Konsistenz: Divergenz, Vortizität (multi-pass)")
println("  - Residuum-Analyse: Magnitude, Sparsity, Stokes-Ratio")
println("  - Strukturelle Ähnlichkeit: SSIM (optimiert), Korrelation")
println("  - Fehlermetriken: MAE, RMSE, Max-Error (single-pass)")