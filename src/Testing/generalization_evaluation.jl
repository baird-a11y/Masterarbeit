# =============================================================================
# GENERALISIERUNGS-EVALUIERUNG - 1 BIS 10 KRISTALLE
# =============================================================================
# Speichern als: generalization_evaluation.jl

using Statistics
using Serialization
using BSON
using Dates

"""
Erweiterte Evaluierungsmetriken für Generalisierung
"""
struct GeneralizationMetrics
    crystal_count::Int
    n_samples::Int
    
    # Absolute Fehler-Metriken
    mae_vx::Float64          # Mean Absolute Error v_x
    mae_vz::Float64          # Mean Absolute Error v_z  
    mae_total::Float64       # Gesamter MAE
    
    # Relative Fehler-Metriken
    mse_vx::Float64          # Mean Squared Error v_x
    mse_vz::Float64          # Mean Squared Error v_z
    mse_total::Float64       # Gesamter MSE
    
    # Korrelations-Metriken
    r2_vx::Float64           # R² für v_x
    r2_vz::Float64           # R² für v_z
    r2_total::Float64        # Gesamtes R²
    
    # Physikalische Plausibilität
    continuity_error::Float64        # Kontinuitätsgleichungs-Verletzung
    max_velocity_error::Float64      # Maximaler Geschwindigkeitsfehler
    velocity_magnitude_error::Float64 # Fehler in Geschwindigkeitsmagnitude
end

"""
Berechnet erweiterte Evaluierungsmetriken
"""
function calculate_generalization_metrics(predictions, targets, crystal_count::Int)
    # Flatten arrays für einfachere Berechnung
    pred_vx = vec(predictions[:,:,1,:])
    pred_vz = vec(predictions[:,:,2,:])
    true_vx = vec(targets[:,:,1,:])
    true_vz = vec(targets[:,:,2,:])
    
    n_samples = size(predictions, 4)
    
    # Absolute Fehler-Metriken
    mae_vx = mean(abs.(pred_vx .- true_vx))
    mae_vz = mean(abs.(pred_vz .- true_vz))
    mae_total = (mae_vx + mae_vz) / 2
    
    # Quadratische Fehler-Metriken
    mse_vx = mean((pred_vx .- true_vx).^2)
    mse_vz = mean((pred_vz .- true_vz).^2)
    mse_total = (mse_vx + mse_vz) / 2
    
    # Korrelations-Metriken (R²)
    r2_vx = calculate_r_squared(pred_vx, true_vx)
    r2_vz = calculate_r_squared(pred_vz, true_vz)
    r2_total = (r2_vx + r2_vz) / 2
    
    # Physikalische Plausibilität
    continuity_error = calculate_continuity_error(predictions)
    max_velocity_error = maximum(abs.([pred_vx .- true_vx; pred_vz .- true_vz]))
    
    # Geschwindigkeitsmagnitude-Fehler
    pred_magnitude = sqrt.(pred_vx.^2 .+ pred_vz.^2)
    true_magnitude = sqrt.(true_vx.^2 .+ true_vz.^2)
    velocity_magnitude_error = mean(abs.(pred_magnitude .- true_magnitude))
    
    return GeneralizationMetrics(
        crystal_count, n_samples,
        mae_vx, mae_vz, mae_total,
        mse_vx, mse_vz, mse_total,
        r2_vx, r2_vz, r2_total,
        continuity_error, max_velocity_error, velocity_magnitude_error
    )
end

"""
Berechnet R² (Bestimmtheitsmaß)
"""
function calculate_r_squared(predicted, actual)
    if length(predicted) != length(actual) || length(predicted) == 0
        return 0.0
    end
    
    ss_res = sum((actual .- predicted).^2)
    ss_tot = sum((actual .- mean(actual)).^2)
    
    if ss_tot ≈ 0.0
        return 1.0  # Perfect prediction when variance is zero
    end
    
    return max(0.0, 1.0 - ss_res / ss_tot)
end

"""
Berechnet Kontinuitätsgleichungs-Verletzung ∂v_x/∂x + ∂v_z/∂z ≈ 0
"""
function calculate_continuity_error(velocity_field)
    h, w, channels, batch = size(velocity_field)
    continuity_errors = []
    
    for b in 1:batch
        vx = velocity_field[:, :, 1, b]
        vz = velocity_field[:, :, 2, b]
        
        # Numerische Ableitungen (zentrale Differenzen)
        dvx_dx = zeros(h, w)
        dvz_dz = zeros(h, w)
        
        # Innere Punkte
        for i in 2:h-1, j in 2:w-1
            dvx_dx[i, j] = (vx[i, j+1] - vx[i, j-1]) / 2
            dvz_dz[i, j] = (vz[i+1, j] - vz[i-1, j]) / 2
        end
        
        # Kontinuitätsfehler
        continuity = abs.(dvx_dx .+ dvz_dz)
        push!(continuity_errors, mean(continuity[2:end-1, 2:end-1]))
    end
    
    return mean(continuity_errors)
end

"""
Evaluiert Modell auf spezifischer Kristallanzahl
"""
function evaluate_on_crystal_count(model, crystal_count::Int; 
    n_eval_samples=20, target_resolution=256, verbose=true)
    
    if verbose
        println("Evaluiere auf $crystal_count Kristalle(n) mit $n_eval_samples Samples...")
    end
    
    # Generiere Evaluierungsdaten
    eval_dataset = generate_evaluation_dataset(crystal_count, n_eval_samples, 
                                               resolution=target_resolution, verbose=false)
    
    if length(eval_dataset) == 0
        error("Keine Evaluierungsdaten für $crystal_count Kristalle generiert!")
    end
    
    predictions = []
    targets = []
    
    for (i, sample) in enumerate(eval_dataset)
        try
            # Sample verarbeiten
            if length(sample) >= 8
                x, z, phase, vx, vz, exx, ezz, v_stokes = sample[1:8]
            else
                error("Unvollständiges Sample: $(length(sample)) Elemente")
            end
            
            # Preprocessing
            phase_tensor, velocity_tensor = preprocess_lamem_sample(
                x, z, phase, vx, vz, v_stokes,
                target_resolution=target_resolution
            )
            
            # Vorhersage
            prediction = cpu(model(phase_tensor))
            
            push!(predictions, prediction)
            push!(targets, velocity_tensor)
            
            if verbose && i % max(1, n_eval_samples ÷ 5) == 0
                println("  Sample $i/$n_eval_samples verarbeitet")
            end
            
        catch e
            if verbose
                println("  Warnung: Evaluierungssample $i fehlgeschlagen: $e")
            end
            continue
        end
    end
    
    if length(predictions) == 0
        error("Keine erfolgreichen Evaluierungen für $crystal_count Kristalle!")
    end
    
    # Tensoren zusammenfügen
    predictions_batch = cat(predictions..., dims=4)
    targets_batch = cat(targets..., dims=4)
    
    # Metriken berechnen
    metrics = calculate_generalization_metrics(predictions_batch, targets_batch, crystal_count)
    
    if verbose
        println("  Evaluierung abgeschlossen: $(length(predictions)) erfolgreiche Samples")
        println("  MAE Total: $(round(metrics.mae_total, digits=6))")
        println("  R² Total: $(round(metrics.r2_total, digits=3))")
    end
    
    return metrics
end

"""
Vollständige Generalisierungs-Evaluierung für 1-10 Kristalle
"""
function evaluate_generalization(model; 
    crystal_range=1:10, 
    n_eval_samples_per_count=20,
    target_resolution=256, 
    verbose=true,
    save_results=true,
    results_dir="generalization_results")
    
    if verbose
        println("="^80)
        println("STARTE GENERALISIERUNGS-EVALUIERUNG")
        println("="^80)
        println("Kristallbereich: $(collect(crystal_range))")
        println("Samples pro Kristallanzahl: $n_eval_samples_per_count")
        println("Zielauflösung: $(target_resolution)x$(target_resolution)")
    end
    
    if save_results
        mkpath(results_dir)
    end
    
    all_metrics = Dict{Int, GeneralizationMetrics}()
    start_time = time()
    
    for crystal_count in crystal_range
        println("\n" * "="^60)
        println("EVALUIERE $crystal_count KRISTALLE")
        println("="^60)
        
        try
            metrics = evaluate_on_crystal_count(
                model, crystal_count, 
                n_eval_samples=n_eval_samples_per_count,
                target_resolution=target_resolution,
                verbose=verbose
            )
            
            all_metrics[crystal_count] = metrics
            
            # Zwischenergebnisse anzeigen
            if verbose
                println("\nErgebnisse für $crystal_count Kristalle:")
                println("  MAE (vx): $(round(metrics.mae_vx, digits=6))")
                println("  MAE (vz): $(round(metrics.mae_vz, digits=6))")
                println("  MAE Total: $(round(metrics.mae_total, digits=6))")
                println("  R² Total: $(round(metrics.r2_total, digits=3))")
                println("  Kontinuitätsfehler: $(round(metrics.continuity_error, digits=6))")
            end
            
        catch e
            println("FEHLER bei $crystal_count Kristallen: $e")
            continue
        end
        
        # Memory cleanup
        GC.gc()
    end
    
    end_time = time()
    total_time = (end_time - start_time) / 60
    
    # Zusammenfassung
    if verbose
        println("\n" * "="^80)
        println("GENERALISIERUNGS-EVALUIERUNG ABGESCHLOSSEN")
        println("="^80)
        println("Gesamtzeit: $(round(total_time, digits=1)) Minuten")
        println("Erfolgreich evaluiert: $(length(all_metrics)) Kristallanzahlen")
    end
    
    # Speichere Ergebnisse
    if save_results && !isempty(all_metrics)
        results_path = joinpath(results_dir, "generalization_results_$(Dates.format(now(), "yyyy-mm-dd_HH-MM")).bson")
        
        results_data = Dict(
            "metrics" => all_metrics,
            "evaluation_config" => Dict(
                "crystal_range" => collect(crystal_range),
                "n_eval_samples_per_count" => n_eval_samples_per_count,
                "target_resolution" => target_resolution,
                "evaluation_time" => total_time
            )
        )
        
        BSON.@save results_path results_data
        
        if verbose
            println("Ergebnisse gespeichert: $results_path")
        end
    end
    
    return all_metrics
end

"""
Erstellt Generalisierungs-Bericht
"""
function create_generalization_report(metrics_dict::Dict{Int, GeneralizationMetrics})
    println("\n" * "="^100)
    println("GENERALISIERUNGS-BERICHT")
    println("="^100)
    
    # Header
    header = @sprintf("%-8s %-12s %-12s %-12s %-12s %-12s %-12s %-15s", 
                     "Kristalle", "MAE_vx", "MAE_vz", "MAE_total", "R²_vx", "R²_vz", "R²_total", "Kontinuität")
    println(header)
    println("-"^100)
    
    # Daten
    crystal_counts = sort(collect(keys(metrics_dict)))
    
    for crystal_count in crystal_counts
        metrics = metrics_dict[crystal_count]
        
        row = @sprintf("%-8d %-12.6f %-12.6f %-12.6f %-12.3f %-12.3f %-12.3f %-15.6f",
                      crystal_count,
                      metrics.mae_vx, metrics.mae_vz, metrics.mae_total,
                      metrics.r2_vx, metrics.r2_vz, metrics.r2_total,
                      metrics.continuity_error)
        println(row)
    end
    
    println("-"^100)
    
    # Zusammenfassung
    if length(crystal_counts) > 1
        best_mae_crystal = crystal_counts[argmin([metrics_dict[k].mae_total for k in crystal_counts])]
        worst_mae_crystal = crystal_counts[argmax([metrics_dict[k].mae_total for k in crystal_counts])]
        best_r2_crystal = crystal_counts[argmax([metrics_dict[k].r2_total for k in crystal_counts])]
        
        println("\nZUSAMMENFASSUNG:")
        println("Bester MAE: $best_mae_crystal Kristalle ($(round(metrics_dict[best_mae_crystal].mae_total, digits=6)))")
        println("Schlechtester MAE: $worst_mae_crystal Kristalle ($(round(metrics_dict[worst_mae_crystal].mae_total, digits=6)))")
        println("Bestes R²: $best_r2_crystal Kristalle ($(round(metrics_dict[best_r2_crystal].r2_total, digits=3)))")
        
        # Generalisierungs-Trend
        mae_values = [metrics_dict[k].mae_total for k in crystal_counts]
        if length(mae_values) > 2
            trend = mae_values[end] > mae_values[1] ? "verschlechtert" : "verbessert"
            println("Trend: Performance $trend sich mit steigender Kristallanzahl")
        end
    end
    
    println("="^100)
end

println("Generalisierungs-Evaluierung geladen!")
println("Verfügbare Funktionen:")
println("  - evaluate_generalization(model, crystal_range=1:10)")
println("  - evaluate_on_crystal_count(model, crystal_count)")
println("  - create_generalization_report(metrics_dict)")