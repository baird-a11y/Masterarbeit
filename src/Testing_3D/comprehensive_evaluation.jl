# =============================================================================
# COMPREHENSIVE EVALUATION MODULE
# =============================================================================

using Statistics
using LinearAlgebra
using BSON
using CSV
using DataFrames
using JSON3
using Dates
using Serialization
using Printf, JSON3, Statistics, LinearAlgebra, Random, Dates, BSON, CSV, DataFrames, Plots, Colors, Serialization, StatsBase, Distributions, HypothesisTests, Flux, CUDA, LaMEM, GeophysicalModelGenerator

"""
Datenstruktur für umfassende Evaluierungsergebnisse
"""
struct EvaluationResult
    # Grundinformationen
    crystal_count::Int
    sample_id::Int
    timestamp::DateTime
    
    # Absolute/Relative Fehlermetriken
    mae_total::Float64
    mae_vx::Float64
    mae_vz::Float64
    rmse_total::Float64
    rmse_vx::Float64
    rmse_vz::Float64
    max_error_total::Float64
    relative_error_stokes::Float64
    
    # Physikalische Konsistenz
    continuity_violation_mean::Float64
    continuity_violation_max::Float64
    divergence_similarity::Float64
    vorticity_preservation::Float64
    
    # Strukturelle Ähnlichkeit
    pearson_correlation_vx::Float64
    pearson_correlation_vz::Float64
    ssim_vx::Float64
    ssim_vz::Float64
    cross_correlation_max::Float64
    
    # Kristall-spezifische Metriken
    alignment_error_mean::Float64
    alignment_error_max::Float64
    crystal_detection_rate::Float64
    radial_profile_similarity::Float64
    
    # Multi-Kristall-Komplexität
    interaction_complexity_index::Float64
    density_robustness_score::Float64
    
    # Zusätzliche Informationen
    processing_time::Float64
    memory_usage::Float64
end

"""
Batch-Evaluierungsergebnisse für Multiple Kristallanzahlen
"""
struct BatchEvaluationResults
    crystal_range::UnitRange{Int}
    results_per_crystal::Dict{Int, Vector{EvaluationResult}}
    aggregated_statistics::Dict{Int, Dict{String, Float64}}
    scaling_metrics::Dict{String, Vector{Float64}}
    total_samples::Int
    evaluation_timestamp::DateTime
    configuration::Dict{String, Any}
end

"""
Berechnet Structural Similarity Index Measure (SSIM) für 2D-Felder
"""
function calculate_ssim(img1, img2; window_size=11, k1=0.01, k2=0.03, L=nothing)
    if size(img1) != size(img2)
        error("Bilder müssen dieselbe Größe haben")
    end
    
    # Dynamischer Bereich falls nicht angegeben
    if L === nothing
        L = maximum([maximum(img1), maximum(img2)]) - minimum([minimum(img1), minimum(img2)])
    end
    
    C1 = (k1 * L)^2
    C2 = (k2 * L)^2
    
    # Gaussian Kernel für lokale Gewichtung (vereinfacht: uniform window)
    half_window = window_size ÷ 2
    
    ssim_map = zeros(size(img1))
    valid_pixels = 0
    
    for i in (half_window+1):(size(img1,1)-half_window)
        for j in (half_window+1):(size(img1,2)-half_window)
            # Lokale Fenster
            window1 = img1[(i-half_window):(i+half_window), (j-half_window):(j+half_window)]
            window2 = img2[(i-half_window):(i+half_window), (j-half_window):(j+half_window)]
            
            # Lokale Statistiken
            μ1 = mean(window1)
            μ2 = mean(window2)
            σ1² = var(window1)
            σ2² = var(window2)
            σ12 = cov(vec(window1), vec(window2))
            
            # SSIM für dieses Fenster
            numerator = (2*μ1*μ2 + C1) * (2*σ12 + C2)
            denominator = (μ1^2 + μ2^2 + C1) * (σ1² + σ2² + C2)
            
            if denominator > 0
                ssim_map[i,j] = numerator / denominator
                valid_pixels += 1
            end
        end
    end
    
    return valid_pixels > 0 ? mean(ssim_map[ssim_map .!= 0]) : 0.0
end

"""
Berechnet Cross-Korrelation zur Detektion räumlicher Verschiebungen
"""
function calculate_cross_correlation(field1, field2; max_shift=20)
    best_correlation = 0.0
    
    for shift_x in -max_shift:max_shift
        for shift_y in -max_shift:max_shift
            # Verschiebe field2
            shifted_field2 = circshift(field2, (shift_y, shift_x))
            
            # Berechne Korrelation
            correlation = cor(vec(field1), vec(shifted_field2))
            
            if !isnan(correlation) && correlation > best_correlation
                best_correlation = correlation
            end
        end
    end
    
    return best_correlation
end

"""
Berechnet Kontinuitätsverletzung (∂vx/∂x + ∂vz/∂z)
"""
function calculate_continuity_violation(vx, vz)
    # Numerische Gradienten (zentraler Differenzenquotient)
    dvx_dx = zeros(size(vx))
    dvz_dz = zeros(size(vz))
    
    # Innere Punkte
    for i in 2:(size(vx,1)-1)
        for j in 2:(size(vx,2)-1)
            dvx_dx[i,j] = (vx[i,j+1] - vx[i,j-1]) / 2
            dvz_dz[i,j] = (vz[i+1,j] - vz[i-1,j]) / 2
        end
    end
    
    # Divergenz
    divergence = dvx_dx + dvz_dz
    
    return mean(abs.(divergence)), maximum(abs.(divergence))
end

"""
Berechnet Vortizitäts-Erhaltung
"""
function calculate_vorticity_preservation(vx_gt, vz_gt, vx_pred, vz_pred)
    # Berechne Vortizität für beide Felder
    vorticity_gt = calculate_vorticity(vx_gt, vz_gt)
    vorticity_pred = calculate_vorticity(vx_pred, vz_pred)
    
    # Korrelation der Vortizitätsfelder
    return cor(vec(vorticity_gt), vec(vorticity_pred))
end

"""
Hilfsfunktion: Berechnet Vortizität ω = ∂vz/∂x - ∂vx/∂z
"""
function calculate_vorticity(vx, vz)
    vorticity = zeros(size(vx))
    
    for i in 2:(size(vx,1)-1)
        for j in 2:(size(vx,2)-1)
            dvz_dx = (vz[i,j+1] - vz[i,j-1]) / 2
            dvx_dz = (vx[i+1,j] - vx[i-1,j]) / 2
            vorticity[i,j] = dvz_dx - dvx_dz
        end
    end
    
    return vorticity
end

"""
Berechnet radiale Geschwindigkeitsprofile um Kristalle
"""
function calculate_radial_profile_similarity(phase_field, vz_gt, vz_pred; max_radius=50)
    crystal_centers = find_crystal_centers(phase_field)
    
    if isempty(crystal_centers)
        return 0.0
    end
    
    similarities = Float64[]
    
    for center in crystal_centers
        cx, cy = center
        
        # Radiale Profile berechnen
        profile_gt = Float64[]
        profile_pred = Float64[]
        
        for r in 1:max_radius
            # Sammle alle Punkte in diesem Radius-Ring
            values_gt = Float64[]
            values_pred = Float64[]
            
            for angle in 0:π/8:2π-π/8
                x = round(Int, cx + r * cos(angle))
                y = round(Int, cy + r * sin(angle))
                
                if 1 <= x <= size(vz_gt,2) && 1 <= y <= size(vz_gt,1)
                    push!(values_gt, vz_gt[y,x])
                    push!(values_pred, vz_pred[y,x])
                end
            end
            
            if !isempty(values_gt)
                push!(profile_gt, mean(values_gt))
                push!(profile_pred, mean(values_pred))
            end
        end
        
        # Korrelation der Profile
        if length(profile_gt) > 1
            correlation = cor(profile_gt, profile_pred)
            if !isnan(correlation)
                push!(similarities, correlation)
            end
        end
    end
    
    return isempty(similarities) ? 0.0 : mean(similarities)
end

"""
Berechnet Interaktions-Komplexitäts-Index für Multi-Kristall-Systeme
"""
function calculate_interaction_complexity_index(crystal_centers, velocity_field)
    n_crystals = length(crystal_centers)
    
    if n_crystals <= 1
        return 0.0
    end
    
    # Berechne paarweise Kristall-Abstände
    distances = Float64[]
    for i in 1:n_crystals
        for j in (i+1):n_crystals
            dist = sqrt((crystal_centers[i][1] - crystal_centers[j][1])^2 + 
                       (crystal_centers[i][2] - crystal_centers[j][2])^2)
            push!(distances, dist)
        end
    end
    
    min_distance = minimum(distances)
    
    # Geschwindigkeits-Variabilität zwischen Kristallen
    velocity_variance = var(velocity_field)
    
    # Komplexitäts-Index: höhere Werte für engere Kristalle und komplexere Felder
    complexity_index = (n_crystals^2) / (min_distance + 1e-6) * log(1 + velocity_variance)
    
    return complexity_index
end

"""
Hauptfunktion: Umfassende Evaluierung eines Modells auf einem Sample
"""
function evaluate_model_comprehensive(model, sample; target_resolution=256, sample_id=1)
    start_time = time()
    memory_before = Base.gc_num().allocd
    
    try
        # 1. Sample verarbeiten
        x, z, phase, vx, vz, exx, ezz, v_stokes = sample
        
        # Preprocessing
        phase_tensor, velocity_tensor = preprocess_lamem_sample(
            x, z, phase, vx, vz, v_stokes,
            target_resolution=target_resolution
        )
        
        # UNet Vorhersage
        prediction = cpu(model(phase_tensor))
        
        # Extrahiere 2D Arrays
        phase_2d = phase_tensor[:,:,1,1]
        gt_vx = velocity_tensor[:,:,1,1]
        gt_vz = velocity_tensor[:,:,2,1]
        pred_vx = prediction[:,:,1,1]
        pred_vz = prediction[:,:,2,1]
        
        # 2. Kristall-Analyse
        crystal_centers = find_crystal_centers(phase_2d)
        n_crystals = length(crystal_centers)
        
        gt_minima = find_velocity_minima(gt_vz, n_crystals)
        pred_minima = find_velocity_minima(pred_vz, n_crystals)
        
        # 3. Absolute/Relative Fehlermetriken
        mae_vx = mean(abs.(pred_vx - gt_vx))
        mae_vz = mean(abs.(pred_vz - gt_vz))
        mae_total = (mae_vx + mae_vz) / 2
        
        rmse_vx = sqrt(mean((pred_vx - gt_vx).^2))
        rmse_vz = sqrt(mean((pred_vz - gt_vz).^2))
        rmse_total = sqrt((rmse_vx^2 + rmse_vz^2) / 2)
        
        max_error_total = maximum([maximum(abs.(pred_vx - gt_vx)), maximum(abs.(pred_vz - gt_vz))])
        relative_error_stokes = mae_total / (abs(v_stokes) + 1e-8)
        
        # 4. Physikalische Konsistenz
        continuity_mean_gt, continuity_max_gt = calculate_continuity_violation(gt_vx, gt_vz)
        continuity_mean_pred, continuity_max_pred = calculate_continuity_violation(pred_vx, pred_vz)
        continuity_violation_mean = abs(continuity_mean_pred - continuity_mean_gt)
        continuity_violation_max = abs(continuity_max_pred - continuity_max_gt)
        
        # Divergenz-Ähnlichkeit
        divergence_similarity = 1.0 - continuity_violation_mean / (continuity_mean_gt + 1e-8)
        divergence_similarity = max(0.0, min(1.0, divergence_similarity))
        
        # Vortizitäts-Erhaltung
        vorticity_preservation = calculate_vorticity_preservation(gt_vx, gt_vz, pred_vx, pred_vz)
        if isnan(vorticity_preservation)
            vorticity_preservation = 0.0
        end
        
        # 5. Strukturelle Ähnlichkeit
        pearson_vx = cor(vec(gt_vx), vec(pred_vx))
        pearson_vz = cor(vec(gt_vz), vec(pred_vz))
        if isnan(pearson_vx); pearson_vx = 0.0; end
        if isnan(pearson_vz); pearson_vz = 0.0; end
        
        ssim_vx = calculate_ssim(gt_vx, pred_vx)
        ssim_vz = calculate_ssim(gt_vz, pred_vz)
        
        cross_correlation_max = max(
            calculate_cross_correlation(gt_vx, pred_vx),
            calculate_cross_correlation(gt_vz, pred_vz)
        )
        
        # 6. Kristall-spezifische Metriken
        alignment_error_mean = calculate_alignment_error(crystal_centers, pred_minima)
        if isinf(alignment_error_mean); alignment_error_mean = 999.0; end
        
        alignment_error_max = 0.0
        if !isempty(crystal_centers) && !isempty(pred_minima)
            alignment_error_max = maximum([
                sqrt((c[1] - m[1])^2 + (c[2] - m[2])^2) 
                for c in crystal_centers[1:min(length(crystal_centers), length(pred_minima))]
                for m in pred_minima[1:min(length(crystal_centers), length(pred_minima))]
            ])
        end
        
        crystal_detection_rate = min(length(pred_minima), n_crystals) / max(n_crystals, 1)
        
        radial_profile_similarity = calculate_radial_profile_similarity(phase_2d, gt_vz, pred_vz)
        
        # 7. Multi-Kristall-Komplexität
        interaction_complexity_index = calculate_interaction_complexity_index(crystal_centers, gt_vz)
        
        # Dichte-Robustheit (vereinfacht)
        density_robustness_score = 1.0 / (1.0 + alignment_error_mean / 10.0)
        
        # 8. Performance-Metriken
        processing_time = time() - start_time
        memory_after = Base.gc_num().allocd
        memory_usage = (memory_after - memory_before) / 1024^2  # MB
        
        # Erstelle Ergebnis-Struktur
        return EvaluationResult(
            n_crystals, sample_id, now(),
            mae_total, mae_vx, mae_vz,
            rmse_total, rmse_vx, rmse_vz,
            max_error_total, relative_error_stokes,
            continuity_violation_mean, continuity_violation_max,
            divergence_similarity, vorticity_preservation,
            pearson_vx, pearson_vz, ssim_vx, ssim_vz, cross_correlation_max,
            alignment_error_mean, alignment_error_max,
            crystal_detection_rate, radial_profile_similarity,
            interaction_complexity_index, density_robustness_score,
            processing_time, memory_usage
        )
        
    catch e
        println("Fehler bei Evaluierung von Sample $sample_id: $e")
        # Rückgabe mit Fehlerwerten
        return EvaluationResult(
            0, sample_id, now(),
            999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0, 999.0,
            999.0, 999.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            999.0, 999.0, 0.0, 0.0,
            0.0, 0.0,
            time() - start_time, 0.0
        )
    end
end

"""
Automatisierte Multi-Kristall-Batch-Evaluierung
"""
function automated_multi_crystal_evaluation(model_path, crystal_range=1:15, samples_per_count=10; 
                                          target_resolution=256, output_dir="comprehensive_evaluation",
                                          verbose=true)
    
    if verbose
        println("=== AUTOMATISIERTE MULTI-KRISTALL-EVALUIERUNG ===")
        println("Modell: $model_path")
        println("Kristallbereich: $crystal_range")
        println("Samples pro Kristallanzahl: $samples_per_count")
        println("Ausgabe: $output_dir")
    end
    
    # Ausgabe-Verzeichnis erstellen
    mkpath(output_dir)
    mkpath(joinpath(output_dir, "data"))
    mkpath(joinpath(output_dir, "visualizations"))
    
    # Modell laden
    model = load_trained_model(model_path)
    
    # Ergebnisse sammeln
    results_per_crystal = Dict{Int, Vector{EvaluationResult}}()
    total_samples = 0
    
    for n_crystals in crystal_range
        if verbose
            println("\nEvaluiere $n_crystals Kristalle...")
        end
        
        crystal_results = EvaluationResult[]
        
        for sample_id in 1:samples_per_count
            if verbose && sample_id % 5 == 1
                print("  Sample $sample_id/$samples_per_count...")
            end
            
            try
                # Sample generieren
                sample = generate_crystal_sample(n_crystals, target_resolution)
                
                # Evaluieren
                result = evaluate_model_comprehensive(model, sample, 
                                                    target_resolution=target_resolution,
                                                    sample_id=sample_id)
                
                push!(crystal_results, result)
                total_samples += 1
                
            catch e
                if verbose
                    println("Fehler bei Sample $sample_id: $e")
                end
                continue
            end
            
            # Memory cleanup
            if sample_id % 5 == 0
                GC.gc()
            end
        end
        
        results_per_crystal[n_crystals] = crystal_results
        
        if verbose
            successful = length(crystal_results)
            println("  $successful/$samples_per_count erfolgreich")
        end
    end
    
    # Aggregierte Statistiken berechnen
    aggregated_stats = calculate_aggregated_statistics(results_per_crystal)
    
    # Skalierungs-Metriken berechnen
    scaling_metrics = calculate_scaling_metrics(results_per_crystal, crystal_range)
    
    # Batch-Ergebnis erstellen
    batch_results = BatchEvaluationResults(
        crystal_range,
        results_per_crystal,
        aggregated_stats,
        scaling_metrics,
        total_samples,
        now(),
        Dict(
            "model_path" => model_path,
            "target_resolution" => target_resolution,
            "samples_per_count" => samples_per_count
        )
    )
    
    # Ergebnisse speichern
    results_path = joinpath(output_dir, "data", "raw_results.bson")
    BSON.bson(results_path, Dict("batch_results" => batch_results))
    
    if verbose
        println("\nEvaluierung abgeschlossen:")
        println("  Kristallbereiche: $crystal_range")
        println("  Gesamt-Samples: $total_samples")
        println("  Ergebnisse gespeichert: $results_path")
    end
    
    return batch_results
end

"""
Hilfsfunktion: Generiert Sample mit spezifischer Kristallanzahl
"""
function generate_crystal_sample(n_crystals, resolution)
    # Verwende LaMEM_Multi_crystal aus lamem_interface.jl
    radius_crystal = fill(0.04, n_crystals)
    
    # Einfache Grid-basierte Positionierung
    centers = []
    for i in 1:n_crystals
        x_pos = -0.6 + (i-1) % 4 * 0.4
        z_pos = 0.2 + div(i-1, 4) * 0.3
        push!(centers, (x_pos, z_pos))
    end
    
    return LaMEM_Multi_crystal(
        resolution=(resolution, resolution),
        n_crystals=n_crystals,
        radius_crystal=radius_crystal,
        cen_2D=centers
    )
end

"""
Berechnet aggregierte Statistiken für jede Kristallanzahl
"""
function calculate_aggregated_statistics(results_per_crystal)
    aggregated = Dict{Int, Dict{String, Float64}}()
    
    for (n_crystals, results) in results_per_crystal
        if isempty(results)
            continue
        end
        
        stats = Dict{String, Float64}()
        
        # Alle numerischen Felder extrahieren
        mae_values = [r.mae_total for r in results]
        rmse_values = [r.rmse_total for r in results]
        alignment_values = [r.alignment_error_mean for r in results if !isinf(r.alignment_error_mean)]
        correlation_vz_values = [r.pearson_correlation_vz for r in results]
        detection_rate_values = [r.crystal_detection_rate for r in results]
        
        # Statistiken berechnen
        stats["mae_mean"] = mean(mae_values)
        stats["mae_std"] = std(mae_values)
        stats["mae_median"] = median(mae_values)
        
        stats["rmse_mean"] = mean(rmse_values)
        stats["rmse_std"] = std(rmse_values)
        
        stats["alignment_mean"] = isempty(alignment_values) ? 999.0 : mean(alignment_values)
        stats["alignment_std"] = isempty(alignment_values) ? 0.0 : std(alignment_values)
        
        stats["correlation_vz_mean"] = mean(correlation_vz_values)
        stats["correlation_vz_std"] = std(correlation_vz_values)
        
        stats["detection_rate_mean"] = mean(detection_rate_values)
        stats["detection_rate_std"] = std(detection_rate_values)
        
        stats["sample_count"] = length(results)
        
        aggregated[n_crystals] = stats
    end
    
    return aggregated
end

"""
Berechnet Skalierungs-Metriken über Kristallanzahl-Bereich
"""
function calculate_scaling_metrics(results_per_crystal, crystal_range)
    scaling = Dict{String, Vector{Float64}}()
    
    crystal_counts = Float64[]
    mae_means = Float64[]
    alignment_means = Float64[]
    correlation_means = Float64[]
    detection_rates = Float64[]
    complexity_indices = Float64[]
    
    for n_crystals in sort(collect(keys(results_per_crystal)))
        results = results_per_crystal[n_crystals]
        
        if !isempty(results)
            push!(crystal_counts, Float64(n_crystals))
            push!(mae_means, mean([r.mae_total for r in results]))
            
            alignment_vals = [r.alignment_error_mean for r in results if !isinf(r.alignment_error_mean)]
            push!(alignment_means, isempty(alignment_vals) ? 999.0 : mean(alignment_vals))
            
            push!(correlation_means, mean([r.pearson_correlation_vz for r in results]))
            push!(detection_rates, mean([r.crystal_detection_rate for r in results]))
            push!(complexity_indices, mean([r.interaction_complexity_index for r in results]))
        end
    end
    
    scaling["crystal_counts"] = crystal_counts
    scaling["mae_progression"] = mae_means
    scaling["alignment_progression"] = alignment_means
    scaling["correlation_progression"] = correlation_means
    scaling["detection_rate_progression"] = detection_rates
    scaling["complexity_progression"] = complexity_indices
    
    return scaling
end

println("Comprehensive Evaluation Module geladen!")
println("Verfügbare Funktionen:")
println("  - evaluate_model_comprehensive(model, sample)")
println("  - automated_multi_crystal_evaluation(model_path, crystal_range, samples_per_count)")
println("  - calculate_ssim(img1, img2)")
println("  - calculate_continuity_violation(vx, vz)")
println("")
println("Haupteinstiegspunkt: automated_multi_crystal_evaluation()")