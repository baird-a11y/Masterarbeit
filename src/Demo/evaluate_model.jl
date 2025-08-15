# =============================================================================
# MODEL EVALUATION MODULE - METRIKEN-BERECHNUNG
# =============================================================================

using Statistics
using BSON
using Plots

# Module laden (falls nicht bereits geladen)
include("training.jl")           # Für load_trained_model
include("lamem_interface.jl")    # Für Sample-Generation
include("data_processing.jl")    # Für preprocess_lamem_sample

"""
Metriken-Struktur für einzelne Sample-Evaluierung
"""
struct EvaluationMetrics
    # Basic Fehler-Metriken
    mae_vx::Float64
    mae_vz::Float64
    mae_total::Float64
    
    rmse_vx::Float64
    rmse_vz::Float64
    rmse_total::Float64
    
    # Relative Fehler (bezogen auf Stokes-Geschwindigkeit)
    relative_mae_vx::Float64
    relative_mae_vz::Float64
    
    # Koordinaten-Alignment
    gt_crystal_centers::Vector{Tuple{Float64, Float64}}
    unet_velocity_minima::Vector{Tuple{Float64, Float64}}
    alignment_error_pixels::Float64
    
    # Zusätzliche Info
    n_crystals::Int
    max_velocity_magnitude::Float64
    stokes_velocity::Float64
end

"""
Berechnet umfassende Metriken für ein Sample
"""
function calculate_comprehensive_metrics(model, sample; target_resolution=256)
    try
        # Sample entpacken
        x, z, phase, vx, vz, exx, ezz, v_stokes = sample
        
        # Preprocessing
        phase_tensor, velocity_tensor = preprocess_lamem_sample(
            x, z, phase, vx, vz, v_stokes,
            target_resolution=target_resolution
        )
        
        # UNet Vorhersage
        prediction = cpu(model(phase_tensor))
        
        # Extrahiere Arrays für Berechnung
        gt_vx = velocity_tensor[:,:,1,1]
        gt_vz = velocity_tensor[:,:,2,1]
        pred_vx = prediction[:,:,1,1]
        pred_vz = prediction[:,:,2,1]
        
        # 1. BASIC FEHLER-METRIKEN
        mae_vx = mean(abs.(pred_vx .- gt_vx))
        mae_vz = mean(abs.(pred_vz .- gt_vz))
        mae_total = (mae_vx + mae_vz) / 2
        
        rmse_vx = sqrt(mean((pred_vx .- gt_vx).^2))
        rmse_vz = sqrt(mean((pred_vz .- gt_vz).^2))
        rmse_total = sqrt((rmse_vx^2 + rmse_vz^2) / 2)
        
        # 2. RELATIVE FEHLER (bezogen auf Stokes-Geschwindigkeit)
        relative_mae_vx = mae_vx / abs(v_stokes) * 100  # Prozent
        relative_mae_vz = mae_vz / abs(v_stokes) * 100  # Prozent
        
        # 3. KOORDINATEN-ALIGNMENT ANALYSE
        gt_crystal_centers = find_crystal_centers(phase_tensor[:,:,1,1])
        unet_velocity_minima = find_velocity_minima(pred_vz, length(gt_crystal_centers))
        
        # Berechne durchschnittlichen Alignment-Fehler
        alignment_error = calculate_alignment_error(gt_crystal_centers, unet_velocity_minima)
        
        # 4. ZUSÄTZLICHE INFORMATIONEN
        n_crystals = length(gt_crystal_centers)
        max_velocity_magnitude = maximum(sqrt.(gt_vx.^2 .+ gt_vz.^2))
        
        # Erstelle Metriken-Struktur
        metrics = EvaluationMetrics(
            mae_vx, mae_vz, mae_total,
            rmse_vx, rmse_vz, rmse_total,
            relative_mae_vx, relative_mae_vz,
            gt_crystal_centers, unet_velocity_minima, alignment_error,
            n_crystals, max_velocity_magnitude, v_stokes
        )
        
        return metrics, (phase_tensor, velocity_tensor, prediction)
        
    catch e
        println("Fehler bei Metriken-Berechnung: $e")
        return nothing, nothing
    end
end

"""
Findet Kristall-Zentren im Phasenfeld
"""
function find_crystal_centers(phase_field)
    centers = []
    
    # Finde alle einzigartigen Kristall-IDs (> 0)
    crystal_ids = unique(phase_field[phase_field .> 0])
    
    for crystal_id in crystal_ids
        # Finde alle Pixel mit dieser Kristall-ID
        crystal_mask = phase_field .== crystal_id
        
        if sum(crystal_mask) > 0
            # Berechne Schwerpunkt
            rows, cols = size(phase_field)
            y_indices, x_indices = findall(crystal_mask) |> x -> getindex.(x, [1, 2])
            
            center_x = mean(x_indices)
            center_y = mean(y_indices)
            
            push!(centers, (center_x, center_y))
        end
    end
    
    return centers
end

"""
Findet Geschwindigkeits-Minima (Kristall-Positionen in v_z)
"""
function find_velocity_minima(vz_field, expected_count; min_distance=15)
    minima = []
    vz_copy = copy(vz_field)
    
    for i in 1:expected_count
        # Finde globales Minimum
        min_idx = argmin(vz_copy)
        min_x, min_y = min_idx[2], min_idx[1]  # (col, row) = (x, y)
        
        push!(minima, (Float64(min_x), Float64(min_y)))
        
        # Setze Bereich um gefundenes Minimum auf hohen Wert
        rows, cols = size(vz_copy)
        for dy in -min_distance:min_distance
            for dx in -min_distance:min_distance
                new_y = min_y + dy
                new_x = min_x + dx
                
                if 1 <= new_y <= rows && 1 <= new_x <= cols
                    if sqrt(dx^2 + dy^2) <= min_distance
                        vz_copy[new_y, new_x] = 1.0  # Hoher Wert
                    end
                end
            end
        end
    end
    
    return minima
end

"""
Berechnet Alignment-Fehler zwischen Kristall-Zentren und Geschwindigkeits-Minima
"""
function calculate_alignment_error(crystal_centers, velocity_minima)
    if length(crystal_centers) == 0 || length(velocity_minima) == 0
        return Inf
    end
    
    total_error = 0.0
    matched_pairs = 0
    
    # Finde beste Zuordnung (nearest neighbor)
    for center in crystal_centers
        if length(velocity_minima) > 0
            distances = [sqrt((center[1] - minimum[1])^2 + (center[2] - minimum[2])^2) 
                        for minimum in velocity_minima]
            
            min_distance = minimum(distances)
            total_error += min_distance
            matched_pairs += 1
        end
    end
    
    return matched_pairs > 0 ? total_error / matched_pairs : Inf
end

"""
Gibt Metriken-Zusammenfassung aus
"""
function print_metrics_summary(metrics::EvaluationMetrics)
    println("="^60)
    println("EVALUIERUNGS-METRIKEN ZUSAMMENFASSUNG")
    println("="^60)
    
    println("Kristall-Information:")
    println("  Anzahl Kristalle: $(metrics.n_crystals)")
    println("  Stokes-Geschwindigkeit: $(round(metrics.stokes_velocity, digits=4))")
    println("  Max. Geschwindigkeit: $(round(metrics.max_velocity_magnitude, digits=4))")
    
    println("\nAbsolute Fehler:")
    println("  MAE v_x: $(round(metrics.mae_vx, digits=6))")
    println("  MAE v_z: $(round(metrics.mae_vz, digits=6))")
    println("  MAE Total: $(round(metrics.mae_total, digits=6))")
    
    println("\nRMSE Fehler:")
    println("  RMSE v_x: $(round(metrics.rmse_vx, digits=6))")
    println("  RMSE v_z: $(round(metrics.rmse_vz, digits=6))")
    println("  RMSE Total: $(round(metrics.rmse_total, digits=6))")
    
    println("\nRelative Fehler (% der Stokes-Geschwindigkeit):")
    println("  Relativ v_x: $(round(metrics.relative_mae_vx, digits=2))%")
    println("  Relativ v_z: $(round(metrics.relative_mae_vz, digits=2))%")
    
    println("\nKoordinaten-Alignment:")
    println("  Alignment-Fehler: $(round(metrics.alignment_error_pixels, digits=2)) Pixel")
    
    if metrics.alignment_error_pixels < 10
        println("  Status: Exzellente Koordinaten-Präzision")
    elseif metrics.alignment_error_pixels < 20
        println("  Status: Gute Koordinaten-Präzision")
    elseif metrics.alignment_error_pixels < 40
        println("  Status: Akzeptable Koordinaten-Präzision")
    else
        println("  Status: Schlechte Koordinaten-Präzision")
    end
    
    println("\nGesamtbewertung:")
    if metrics.mae_total < 0.01 && metrics.alignment_error_pixels < 15
        println("EXZELLENTE Performance")
    elseif metrics.mae_total < 0.05 && metrics.alignment_error_pixels < 30
        println("GUTE Performance")
    elseif metrics.mae_total < 0.1
        println(" AKZEPTABLE Performance")
    else
        println("VERBESSERUNG ERFORDERLICH")
    end
    
    println("="^60)
end

"""
Evaluiert mehrere Samples und gibt Durchschnittswerte aus
"""
function evaluate_multiple_samples(model, samples)
    println("Evaluiere $(length(samples)) Samples...")
    
    all_metrics = []
    successful_evaluations = 0
    
    for (i, sample) in enumerate(samples)
        print("Sample $i/$(length(samples))... ")
        
        metrics, data = calculate_comprehensive_metrics(model, sample)
        
        if metrics !== nothing
            push!(all_metrics, metrics)
            successful_evaluations += 1
            println("✓")
        else
            println("Fehlgeschlagen")
        end
    end
    
    if successful_evaluations == 0
        println("Keine erfolgreichen Evaluierungen!")
        return nothing
    end
    
    # Berechne Durchschnittswerte
    avg_mae_vx = mean([m.mae_vx for m in all_metrics])
    avg_mae_vz = mean([m.mae_vz for m in all_metrics])
    avg_mae_total = mean([m.mae_total for m in all_metrics])
    
    avg_rmse_vx = mean([m.rmse_vx for m in all_metrics])
    avg_rmse_vz = mean([m.rmse_vz for m in all_metrics])
    avg_rmse_total = mean([m.rmse_total for m in all_metrics])
    
    avg_relative_mae_vx = mean([m.relative_mae_vx for m in all_metrics])
    avg_relative_mae_vz = mean([m.relative_mae_vz for m in all_metrics])
    
    avg_alignment_error = mean([m.alignment_error_pixels for m in all_metrics])
    
    # Ausgabe
    println("\n" * "="^70)
    println("DURCHSCHNITTLICHE METRIKEN ($successful_evaluations Samples)")
    println("="^70)
    
    println("Absolute Fehler (Durchschnitt):")
    println("  MAE v_x: $(round(avg_mae_vx, digits=6))")
    println("  MAE v_z: $(round(avg_mae_vz, digits=6))")
    println("  MAE Total: $(round(avg_mae_total, digits=6))")
    
    println("\nRMSE Fehler (Durchschnitt):")
    println("  RMSE v_x: $(round(avg_rmse_vx, digits=6))")
    println("  RMSE v_z: $(round(avg_rmse_vz, digits=6))")
    println("  RMSE Total: $(round(avg_rmse_total, digits=6))")
    
    println("\nRelative Fehler (Durchschnitt):")
    println("  Relativ v_x: $(round(avg_relative_mae_vx, digits=2))%")
    println("  Relativ v_z: $(round(avg_relative_mae_vz, digits=2))%")
    
    println("\nKoordinaten-Alignment (Durchschnitt):")
    println("  Alignment-Fehler: $(round(avg_alignment_error, digits=2)) Pixel")
    
    return all_metrics
end

"""
Test-Funktion für Metriken-Berechnung
"""
function test_metrics_calculation(model_path=".bson")
    println("=== TEST: METRIKEN-BERECHNUNG ===")
    
    # 1. Modell laden
    println("1. Lade trainiertes Modell...")
    try
        model = load_trained_model(model_path)
        println("✓ Modell erfolgreich geladen")
    catch e
        println("Fehler beim Laden des Modells: $e")
        return false
    end
    
    # 2. Test-Sample generieren
    println("\n2. Generiere Test-Sample...")
    try
        test_sample = LaMEM_Multi_crystal(
            resolution=(256, 256),
            n_crystals=2,  # Einfacher Test mit 2 Kristallen
            radius_crystal=[0.05, 0.05],
            cen_2D=[(0.0, 0.3), (0.0, 0.7)]
        )
        println("✓ Test-Sample generiert")
    catch e
        println("Fehler bei Sample-Generierung: $e")
        return false
    end
    
    # 3. Metriken berechnen
    println("\n3. Berechne Metriken...")
    try
        metrics, data = calculate_comprehensive_metrics(model, test_sample)
        
        if metrics !== nothing
            println("✓ Metriken erfolgreich berechnet")
            print_metrics_summary(metrics)
            return true
        else
            println(" Metriken-Berechnung fehlgeschlagen")
            return false
        end
    catch e
        println("Fehler bei Metriken-Berechnung: $e")
        return false
    end
end

println("Model Evaluation Module geladen!")
println("Verfügbare Funktionen:")
println("  - calculate_comprehensive_metrics(model, sample)")
println("  - evaluate_multiple_samples(model, samples)")
println("  - test_metrics_calculation(model_path)")
println("")
println("Zum Testen: test_metrics_calculation()")