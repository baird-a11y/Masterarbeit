# =============================================================================
# MODEL EVALUATION MODULE - METRIKEN-BERECHNUNG MIT SPEICHERFUNKTION
# =============================================================================
# Speichern als: evaluate_model.jl
using Statistics
using BSON
using Plots
using CSV
using DataFrames
using Dates
using Random

# Module laden (falls nicht bereits geladen)
include("unet_architecture.jl") # Für SimplifiedUNet Definition (WICHTIG!)
include("training.jl")           # Für load_trained_model
include("lamem_interface.jl")    # Für Sample-Generation
include("data_processing.jl")    # Für preprocess_lamem_sample

# Konstante für Evaluierungsergebnisse
const EVALUATION_DIR = "H:\\Masterarbeit\\Auswertung\\Ten_Crystals\\Metriken"

"""
Stelle sicher, dass das Evaluierungs-Verzeichnis existiert
"""
function ensure_evaluation_directory()
    if !isdir(EVALUATION_DIR)
        try
            mkpath(EVALUATION_DIR)
            println("✓ Evaluierungs-Verzeichnis erstellt: $EVALUATION_DIR")
        catch e
            println("⚠️  Warnung: Kann Verzeichnis nicht erstellen: $e")
            println("   Verwende aktuelles Verzeichnis als Fallback")
            return "."
        end
    end
    return EVALUATION_DIR
end

struct LaMEMFidelityMetrics
    mae_vx::Float64
    mae_vz::Float64
    mae_total::Float64
    rmse_vx::Float64
    rmse_vz::Float64
    rmse_total::Float64
    correlation_vx::Float64
    correlation_vz::Float64
    correlation_total::Float64
    relative_mae_vx::Float64
    relative_mae_vz::Float64
    lamem_divergence_error::Float64
    unet_divergence_error::Float64
    divergence_similarity::Float64
    crystal_centers::Vector{Tuple{Float64, Float64}}
    lamem_velocity_patterns::Vector{Tuple{Float64, Float64}}
    unet_velocity_patterns::Vector{Tuple{Float64, Float64}}
    pattern_alignment_error::Float64
    n_crystals::Int
    max_lamem_velocity::Float64
    max_unet_velocity::Float64
    velocity_scale_ratio::Float64
end

"""
Erweiterte Struktur für gespeicherte Evaluierungs-Ergebnisse
"""
struct CrystalEvaluationResults
    model_path::String
    evaluation_timestamp::String
    crystal_count::Int
    metrics::LaMEMFidelityMetrics
    sample_info::Dict{String, Any}
    quality_assessment::String
end

"""
Erstellt ein Testsample mit Kristall-Layouts
"""
function generate_test_sample(n_crystals, target_resolution=256)
    # Optimierte Kristall-Layouts wie in der Visualisierung
    if n_crystals <= 2
        if n_crystals == 1
            centers = [(0.0, 0.0)]
            radii = [0.06]
        else
            centers = [(-0.5, 0.0), (0.5, 0.0)]
            radii = [0.05, 0.05]
        end
    elseif n_crystals <= 4
        centers = [(-0.4, -0.3), (0.4, -0.3), (-0.4, 0.3), (0.4, 0.3)][1:n_crystals]
        radii = fill(0.04, n_crystals)
    elseif n_crystals <= 8
        positions = []
        for i in 1:n_crystals
            x_pos = -0.6 + (i-1) % 3 * 0.6
            z_pos = -0.4 + div(i-1, 3) * 0.4
            push!(positions, (x_pos, z_pos))
        end
        centers = positions
        radii = fill(0.035, n_crystals)
    else
        positions = []
        for i in 1:n_crystals
            x_pos = -0.8 + (i-1) % 4 * 0.533
            z_pos = -0.6 + div(i-1, 4) * 0.4
            push!(positions, (x_pos, z_pos))
        end
        centers = positions
        radii = fill(0.025, n_crystals)
    end
    
    # Dummy-Werte für Testzwecke
    x = range(-1.0, 1.0, length=target_resolution)
    z = range(-1.0, 1.0, length=target_resolution)
    phase = zeros(target_resolution, target_resolution)
    vx = zeros(target_resolution, target_resolution)
    vz = zeros(target_resolution, target_resolution)
    exx = zeros(target_resolution, target_resolution)
    ezz = zeros(target_resolution, target_resolution)
    v_stokes = 1.0
    
    # Erzeuge einfache Kristall-Strukturen
    for i in 1:n_crystals
        cx, cz = centers[i]
        r = radii[i]
        for j in 1:target_resolution
            for k in 1:target_resolution
                x_pos = x[j]
                z_pos = z[k]
                dist = sqrt((x_pos - cx)^2 + (z_pos - cz)^2)
                if dist <= r
                    phase[j,k] = 1.0
                    # Einfache Geschwindigkeitsverteilung
                    vx[j,k] = -0.5 * (x_pos - cx)
                    vz[j,k] = -0.5 * (z_pos - cz)
                end
            end
        end
    end
    
    return (x, z, phase, vx, vz, exx, ezz, v_stokes)
end

function calculate_lamem_fidelity_metrics(model, sample; target_resolution=256)
    try
        # Sample entpacken
        x, z, phase, vx, vz, _, _, v_stokes = sample
        println("  Debug - Original Dimensionen:")
        println("    Phase: $(size(phase))")
        println("    Vx: $(size(vx))")
        println("    Vz: $(size(vz))")
        println("    Target Resolution: $(target_resolution)")
        
        # Preprocessing
        phase_tensor, velocity_tensor = preprocess_lamem_sample(
            x, z, phase, vx, vz, v_stokes,
            target_resolution=target_resolution
        )
        println("  Debug - Stokes-Geschwindigkeit: $v_stokes")
        if abs(v_stokes) < 1e-10
            println("  Warnung: Stokes-Geschwindigkeit ist ~0!")
            println("  Debug - Sample Parameter:")
            println("    x range: $(minimum(x)) bis $(maximum(x))")
            println("    z range: $(minimum(z)) bis $(maximum(z))")
            println("    Phase range: $(minimum(phase)) bis $(maximum(phase))")
            println("    Vx range: $(minimum(vx)) bis $(maximum(vx))")
            println("    Vz range: $(minimum(vz)) bis $(maximum(vz))")
        end
        
        # UNet Vorhersage
        prediction = cpu(model(phase_tensor))
        println("  Debug - UNet Prediction:")
        println("    Prediction: $(size(prediction))")
        
        # Extrahiere Arrays für Berechnung
        gt_vx = velocity_tensor[:,:,1,1]
        gt_vz = velocity_tensor[:,:,2,1]
        pred_vx = prediction[:,:,1,1]
        pred_vz = prediction[:,:,2,1]
        println("  Debug - Finale Arrays:")
        println("    GT Vx: $(size(gt_vx))")
        println("    GT Vz: $(size(gt_vz))")
        println("    Pred Vx: $(size(pred_vx))")
        println("    Pred Vz: $(size(pred_vz))")
        
        if size(gt_vx) != size(pred_vx) || size(gt_vz) != size(pred_vz)
            error("Dimensionen-Mismatch: GT $(size(gt_vx)) vs Pred $(size(pred_vx))")
        end
        
        # Fehler-Metriken
        mae_vx = mean(abs.(pred_vx .- gt_vx))
        mae_vz = mean(abs.(pred_vz .- gt_vz))
        mae_total = (mae_vx + mae_vz) / 2
        rmse_vx = sqrt(mean((pred_vx .- gt_vx).^2))
        rmse_vz = sqrt(mean((pred_vz .- gt_vz).^2))
        rmse_total = sqrt((rmse_vx^2 + rmse_vz^2) / 2)
        println("  Debug - Geschwindigkeits-Details:")
        println("    Stokes-Geschwindigkeit: $v_stokes")
        println("    GT Vx range: $(minimum(gt_vx)) bis $(maximum(gt_vx))")
        println("    GT Vz range: $(minimum(gt_vz)) bis $(maximum(gt_vz))")
        println("    Pred Vx range: $(minimum(pred_vx)) bis $(maximum(pred_vx))")
        println("    Pred Vz range: $(minimum(pred_vz)) bis $(maximum(pred_vz))")
        println("    GT Vx mean: $(mean(gt_vx)), std: $(std(gt_vx))")
        println("    GT Vz mean: $(mean(gt_vz)), std: $(std(gt_vz))")
        println("    Pred Vx mean: $(mean(pred_vx)), std: $(std(pred_vx))")
        println("    Pred Vz mean: $(mean(pred_vz)), std: $(std(pred_vz))")
        println("  Debug - Skalierungs-Test:")
        println("    Max GT velocity magnitude: $(maximum(sqrt.(gt_vx.^2 .+ gt_vz.^2)))")
        println("    Max Pred velocity magnitude: $(maximum(sqrt.(pred_vx.^2 .+ pred_vz.^2)))")
        println("    Verhältnis GT/Pred max: $(maximum(sqrt.(gt_vx.^2 .+ gt_vz.^2)) / maximum(sqrt.(pred_vx.^2 .+ pred_vz.^2)))")
        
        scale_factor_x = std(gt_vx) / (std(pred_vx) + 1e-10)
        scale_factor_z = std(gt_vz) / (std(pred_vz) + 1e-10)
        println("    Geschätzte Skalierungsfaktoren: Vx=$(round(scale_factor_x, digits=3)), Vz=$(round(scale_factor_z, digits=3))")
        
        # Relative Fehler
        if abs(v_stokes) > 1e-10
            relative_mae_vx = mae_vx / abs(v_stokes) * 100
            relative_mae_vz = mae_vz / abs(v_stokes) * 100
        else
            println("  Warnung: Stokes-Geschwindigkeit ist ~0, verwende max. Geschwindigkeit für Normalisierung")
            max_vel = maximum(sqrt.(gt_vx.^2 .+ gt_vz.^2))
            relative_mae_vx = mae_vx / max_vel * 100
            relative_mae_vz = mae_vz / max_vel * 100
        end
        
        # Korrelations-Berechnung
        correlation_vx = cor(vec(gt_vx), vec(pred_vx))
        correlation_vz = cor(vec(gt_vz), vec(pred_vz))
        lamem_combined = vcat(vec(gt_vx), vec(gt_vz))
        unet_combined = vcat(vec(pred_vx), vec(pred_vz))
        correlation_total = cor(lamem_combined, unet_combined)
        
        # Divergenz-Berechnung
        lamem_div = calculate_divergence(gt_vx, gt_vz)
        unet_div = calculate_divergence(pred_vx, pred_vz)
        lamem_divergence_error = mean(abs.(lamem_div))
        unet_divergence_error = mean(abs.(unet_div))
        divergence_similarity = cor(vec(lamem_div), vec(unet_div))
        
        # Kristallzentren und Muster
        gt_crystal_centers = find_crystal_centers(phase_tensor[:,:,1,1])
        lamem_velocity_minima = find_velocity_minima(gt_vz, length(gt_crystal_centers))
        unet_velocity_minima = find_velocity_minima(pred_vz, length(gt_crystal_centers))
        alignment_error = calculate_alignment_error(gt_crystal_centers, unet_velocity_minima)
        n_crystals = length(gt_crystal_centers)
        max_lamem_velocity = maximum(sqrt.(gt_vx.^2 .+ gt_vz.^2))
        max_unet_velocity = maximum(sqrt.(pred_vx.^2 .+ pred_vz.^2))
        velocity_scale_ratio = max_unet_velocity > 0 ? max_lamem_velocity / max_unet_velocity : 0.0
        
        metrics = LaMEMFidelityMetrics(
            mae_vx, mae_vz, mae_total,
            rmse_vx, rmse_vz, rmse_total,
            correlation_vx, correlation_vz, correlation_total,
            relative_mae_vx, relative_mae_vz,
            lamem_divergence_error, unet_divergence_error, divergence_similarity,
            gt_crystal_centers, lamem_velocity_minima, unet_velocity_minima, alignment_error,
            n_crystals, max_lamem_velocity, max_unet_velocity, velocity_scale_ratio
        )
        
        return metrics, (phase_tensor, velocity_tensor, prediction)
    catch e
        println("Fehler bei Metriken-Berechnung: $e")
        println("Stacktrace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        return nothing, nothing
    end
end

"""
Evaluiert ein Modell für eine spezifische Kristallanzahl und speichert die Ergebnisse
"""
function evaluate_and_save_crystal_metrics(model_path, n_crystals; 
                                          target_resolution=256,
                                          save_individual=true,
                                          verbose=true)
    
    if verbose
        println("=== EVALUIERUNG: $n_crystals KRISTALLE ===")
    end
    
    try
        # Modell laden
        model = load_trained_model(model_path)
        
        # Sample für spezifische Kristallanzahl generieren
        sample = generate_test_sample(n_crystals, target_resolution)
        
        # Metriken berechnen
        metrics, tensors = calculate_lamem_fidelity_metrics(model, sample, target_resolution=target_resolution)
        
        if metrics === nothing
            println("❌ Metriken-Berechnung fehlgeschlagen für $n_crystals Kristalle")
            return nothing
        end
        
        # Sample-Information sammeln
        x, z, phase, vx, vz, exx, ezz, v_stokes = sample
        sample_info = Dict(
            "v_stokes" => v_stokes,
            "phase_range" => (minimum(phase), maximum(phase)),
            "vx_range" => (minimum(vx), maximum(vx)),
            "vz_range" => (minimum(vz), maximum(vz)),
            "actual_crystals_found" => metrics.n_crystals,
            "target_resolution" => target_resolution
        )
        
        # Qualitätsbewertung
        quality = assess_quality(metrics)
        
        # Ergebnis-Struktur erstellen
        result = CrystalEvaluationResults(
            model_path,
            string(Dates.now()),
            n_crystals,
            metrics,
            sample_info,
            quality
        )
        
        # Speichern
        if save_individual
            save_crystal_evaluation(result, verbose=verbose)
        end
        
        if verbose
            print_evaluation_summary(result)
        end
        
        return result
        
    catch e
        println("❌ Evaluierung fehlgeschlagen für $n_crystals Kristalle: $e")
        return nothing
    end
end

"""
Bewertet die Qualität der Metriken
"""
function assess_quality(metrics::LaMEMFidelityMetrics)
    # Bewertungsskala basierend auf den definierten Kriterien
    if metrics.mae_total < 0.01 && metrics.correlation_total > 0.95
        return "Exzellent"
    elseif metrics.mae_total < 0.05 && metrics.correlation_total > 0.85
        return "Gut"
    elseif metrics.mae_total < 0.1 && metrics.correlation_total > 0.70
        return "Akzeptabel"
    elseif metrics.correlation_total > 0.50
        return "Schwach"
    else
        return "Unzureichend"
    end
end

"""
Speichert die Evaluierungsergebnisse für eine Kristallanzahl
"""
function save_crystal_evaluation(result::CrystalEvaluationResults; verbose=true)
    eval_dir = ensure_evaluation_directory()
    
    # Dateiname mit Timestamp und Kristallanzahl
    timestamp = replace(result.evaluation_timestamp, ":" => "-", "." => "-")
    filename = "metrics_$(result.crystal_count)_crystals_$(timestamp).bson"
    filepath = joinpath(eval_dir, filename)
    
    # Als BSON speichern (vollständige Daten)
    BSON.@save filepath result
    
    if verbose
        println("✓ Metriken gespeichert: $filepath")
    end
    
    return filepath
end

"""
Evaluiert Modell für alle Kristallanzahlen von 1 bis max_crystals
"""
function evaluate_crystal_range(model_path, max_crystals; 
                               target_resolution=256,
                               save_summary=true,
                               verbose=true)
    
    println("=== EVALUIERUNG: 1 BIS $max_crystals KRISTALLE ===")
    println("Modell: $model_path")
    println("Zielauflösung: $(target_resolution)x$(target_resolution)")
    
    results = []
    
    for n_crystals in 1:max_crystals
        println("\n--- Evaluiere $n_crystals Kristalle ---")
        
        result = evaluate_and_save_crystal_metrics(
            model_path, n_crystals,
            target_resolution=target_resolution,
            save_individual=true,
            verbose=verbose
        )
        
        if result !== nothing
            push!(results, result)
            println("✓ $n_crystals Kristalle abgeschlossen")
        else
            println("❌ $n_crystals Kristalle fehlgeschlagen")
        end
    end
    
    # Zusammenfassung speichern
    if save_summary && length(results) > 0
        save_evaluation_summary(results, model_path, verbose=verbose)
    end
    
    println("\n✓ Evaluierung 1-$max_crystals Kristalle abgeschlossen!")
    println("Ergebnisse gespeichert in: $(ensure_evaluation_directory())")
    
    return results
end

"""
Speichert eine CSV-Zusammenfassung aller Evaluierungen
"""
function save_evaluation_summary(results, model_path; verbose=true)
    eval_dir = ensure_evaluation_directory()
    
    # DataFrame erstellen
    df_data = []
    
    for result in results
        m = result.metrics
        push!(df_data, [
            result.crystal_count,
            m.n_crystals,  # Tatsächlich gefundene Kristalle
            round(m.mae_total, digits=6),
            round(m.rmse_total, digits=6),
            round(m.correlation_total, digits=4),
            round(m.correlation_vx, digits=4),
            round(m.correlation_vz, digits=4),
            round(m.relative_mae_vx, digits=2),
            round(m.relative_mae_vz, digits=2),
            round(m.pattern_alignment_error, digits=2),
            round(m.velocity_scale_ratio, digits=3),
            round(m.divergence_similarity, digits=4),
            result.quality_assessment,
            result.evaluation_timestamp
        ])
    end
    
    df = DataFrame(df_data, [
        "Target_Crystals", "Found_Crystals", "MAE_Total", "RMSE_Total",
        "Correlation_Total", "Correlation_Vx", "Correlation_Vz",
        "Relative_MAE_Vx_Percent", "Relative_MAE_Vz_Percent",
        "Alignment_Error_Pixels", "Velocity_Scale_Ratio", "Divergence_Similarity",
        "Quality_Assessment", "Timestamp"
    ])
    
    # CSV speichern
    model_name = splitpath(model_path)[end]
    csv_filename = "evaluation_summary_$(replace(model_name, ".bson" => "")).csv"
    csv_filepath = joinpath(eval_dir, csv_filename)
    
    CSV.write(csv_filepath, df)
    
    if verbose
        println("✓ Evaluierungs-Zusammenfassung gespeichert: $csv_filepath")
        println("\nZusammenfassung der Ergebnisse:")
        println(df)
    end
    
    return csv_filepath
end

"""
Druckt eine übersichtliche Zusammenfassung eines Evaluierungsergebnisses
"""
function print_evaluation_summary(result::CrystalEvaluationResults)
    m = result.metrics
    
    println("📊 EVALUIERUNGS-ZUSAMMENFASSUNG")
    println("─"^50)
    println("Kristallanzahl: $(result.crystal_count) (gefunden: $(m.n_crystals))")
    println("Qualität: $(result.quality_assessment)")
    println()
    println("🎯 HAUPT-QUALITÄTSMETRIKEN:")
    println("  MAE Total: $(round(m.mae_total, digits=6))")
    println("  RMSE Total: $(round(m.rmse_total, digits=6))")
    println("  Korrelation Total: $(round(m.correlation_total, digits=4))")
    println()
    println("📊 STRUKTURELLE ÄHNLICHKEIT:")
    println("  Korrelation Vx: $(round(m.correlation_vx, digits=4))")
    println("  Korrelation Vz: $(round(m.correlation_vz, digits=4))")
    println("  Alignment-Fehler: $(round(m.pattern_alignment_error, digits=2)) Pixel")
    println()
    println("⚡ PHYSIK-KONSISTENZ:")
    println("  Geschwindigkeits-Skalierung: $(round(m.velocity_scale_ratio, digits=3))")
    println("  Divergenz-Ähnlichkeit: $(round(m.divergence_similarity, digits=4))")
    println("  Relative MAE Vx: $(round(m.relative_mae_vx, digits=2))%")
    println("  Relative MAE Vz: $(round(m.relative_mae_vz, digits=2))%")
end

# =============================================================================
# URSPRÜNGLICHE KRISTALL-ERKENNUNGS-FUNKTIONEN
# =============================================================================

"""
Findet Kristall-Zentren im Phasenfeld durch Clustering
"""
function find_crystal_centers(phase_field; min_crystal_size=50)
    # Finde alle Pixel mit Phase > 0.5 (Kristall-Bereiche)
    crystal_mask = phase_field .> 0.5
    
    if sum(crystal_mask) == 0
        return []  # Keine Kristalle gefunden
    end
    
    # Connected Components Analysis (vereinfacht)
    labeled_regions, num_regions = find_connected_components(crystal_mask)
    
    crystal_centers = []
    
    for region_id in 1:num_regions
        # Finde alle Pixel dieser Region
        region_mask = labeled_regions .== region_id
        region_size = sum(region_mask)
        
        # Filtere zu kleine Regionen (Rauschen)
        if region_size < min_crystal_size
            continue
        end
        
        # Berechne Schwerpunkt der Region
        indices = findall(region_mask)
        
        if length(indices) > 0
            center_y = mean([idx[1] for idx in indices])
            center_x = mean([idx[2] for idx in indices])
            
            push!(crystal_centers, (center_x, center_y))
        end
    end
    
    return crystal_centers
end

"""
Vereinfachte Connected Components Analysis
"""
function find_connected_components(binary_mask)
    H, W = size(binary_mask)
    labeled = zeros(Int, H, W)
    current_label = 0
    
    # 4-connected neighbors
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for i in 1:H
        for j in 1:W
            if binary_mask[i, j] && labeled[i, j] == 0
                current_label += 1
                
                # Flood fill für diese Komponente
                stack = [(i, j)]
                labeled[i, j] = current_label
                
                while !isempty(stack)
                    y, x = pop!(stack)
                    
                    for (dy, dx) in neighbors
                        ny, nx = y + dy, x + dx
                        
                        if 1 <= ny <= H && 1 <= nx <= W
                            if binary_mask[ny, nx] && labeled[ny, nx] == 0
                                labeled[ny, nx] = current_label
                                push!(stack, (ny, nx))
                            end
                        end
                    end
                end
            end
        end
    end
    
    return labeled, current_label
end

"""
Findet Geschwindigkeits-Minima (z.B. stärkste negative v_z Werte)
"""
function find_velocity_minima(velocity_field, expected_count; search_radius=20)
    minima_positions = []
    velocity_copy = copy(velocity_field)
    
    for i in 1:expected_count
        # Finde globales Minimum
        min_idx = argmin(velocity_copy)
        min_position = (min_idx[2], min_idx[1])  # (x, y) Format
        
        push!(minima_positions, min_position)
        
        # Lösche Umgebung um dieses Minimum für nächste Suche
        y_center, x_center = min_idx[1], min_idx[2]
        H, W = size(velocity_copy)
        
        y_range = max(1, y_center - search_radius):min(H, y_center + search_radius)
        x_range = max(1, x_center - search_radius):min(W, x_center + search_radius)
        
        velocity_copy[y_range, x_range] .= Inf  # Ausschließen für weitere Suche
    end
    
    return minima_positions
end

"""
Berechnet durchschnittlichen Alignment-Fehler zwischen zwei Punktlisten
"""
function calculate_alignment_error(points1, points2)
    if length(points1) == 0 || length(points2) == 0
        return Inf
    end
    
    # Wenn unterschiedliche Anzahl Punkte: Verwende Minimum
    n_points = min(length(points1), length(points2))
    
    if n_points == 0
        return Inf
    end
    
    total_distance = 0.0
    
    # Finde beste Zuordnung (einfachster Ansatz: nearest neighbor)
    used_indices = Set{Int}()
    
    for i in 1:n_points
        point1 = points1[i]
        
        best_distance = Inf
        best_j = 0
        
        for j in 1:length(points2)
            if j in used_indices
                continue
            end
            
            point2 = points2[j]
            distance = sqrt((point1[1] - point2[1])^2 + (point1[2] - point2[2])^2)
            
            if distance < best_distance
                best_distance = distance
                best_j = j
            end
        end
        
        if best_j > 0
            total_distance += best_distance
            push!(used_indices, best_j)
        end
    end
    
    return total_distance / n_points
end

"""
Berechnet Divergenz eines Geschwindigkeitsfeldes (vereinfacht)
"""
function calculate_divergence(vx, vz)
    H, W = size(vx)
    divergence = zeros(H, W)
    
    for i in 2:H-1
        for j in 2:W-1
            # Finite Differenzen für Divergenz
            dvx_dx = (vx[i, j+1] - vx[i, j-1]) / 2
            dvz_dz = (vz[i+1, j] - vz[i-1, j]) / 2
            divergence[i, j] = dvx_dx + dvz_dz
        end
    end
    
    return divergence
end

"""
Test-Funktion für Kristall-Erkennung
"""
function test_crystal_detection()
    println("=== TEST: KRISTALL-ERKENNUNG ===")
    
    # Erstelle Test-Phasenfeld mit 2 Kristallen
    test_phase = zeros(256, 256)
    
    # Kristall 1: Zentriert bei (64, 128)
    for i in 55:75, j in 119:137
        if (i-65)^2 + (j-128)^2 <= 100  # Radius ~10
            test_phase[i, j] = 1.0
        end
    end
    
    # Kristall 2: Zentriert bei (192, 128)  
    for i in 183:201, j in 119:137
        if (i-192)^2 + (j-128)^2 <= 100  # Radius ~10
            test_phase[i, j] = 1.0
        end
    end
    
    # Teste Kristall-Erkennung
    centers = find_crystal_centers(test_phase)
    
    println("Gefundene Kristall-Zentren: $centers")
    println("Erwartet: ungefähr [(64, 128), (192, 128)]")
    
    # Teste Geschwindigkeits-Minima
    test_velocity = randn(256, 256)
    test_velocity[65, 128] = -5.0  # Starkes Minimum bei Kristall 1
    test_velocity[192, 128] = -4.0  # Starkes Minimum bei Kristall 2
    
    minima = find_velocity_minima(test_velocity, 2)
    println("Gefundene Geschwindigkeits-Minima: $minima")
    
    # Teste Alignment-Berechnung
    alignment_error = calculate_alignment_error(centers, minima)
    println("Alignment-Fehler: $(round(alignment_error, digits=2)) Pixel")
    
    return length(centers) == 2 && alignment_error < 20
end

"""
Test-Funktion für die Evaluierung
"""
function test_lamem_fidelity_evaluation(model_path="test_model.bson"; max_crystals=2)
    println("=== TEST: LAMEM FIDELITY EVALUATION ===")
    
    try
        # Teste Kristall-Erkennung
        detection_ok = test_crystal_detection()
        println("Kristall-Erkennung: $(detection_ok ? "OK" : "FEHLER")")
        
        # Teste Evaluierung für kleine Kristallanzahl
        println("\nTestevaluierung...")
        results = evaluate_crystal_range(model_path, max_crystals; 
                                       target_resolution=128,
                                       save_summary=false,
                                       verbose=true)
        
        if length(results) > 0
            println("✓ Testevaluierung erfolgreich abgeschlossen")
            return true
        else
            println("❌ Testevaluierung fehlgeschlagen")
            return false
        end
    catch e
        println("❌ Testfehler: $e")
        return false
    end
end

# ... (alle vorhandenen Funktionen bleiben unverändert) ...

"""
Test-Funktion für die Evaluierung
"""
function test_lamem_fidelity_evaluation(model_path="test_model.bson"; max_crystals=2)
    println("=== TEST: LAMEM FIDELITY EVALUATION ===")
    
    try
        # Teste Kristall-Erkennung
        detection_ok = test_crystal_detection()
        println("Kristall-Erkennung: $(detection_ok ? "OK" : "FEHLER")")
        
        # Teste Evaluierung für kleine Kristallanzahl
        println("\nTestevaluierung...")
        results = evaluate_crystal_range(model_path, max_crystals; 
                                       target_resolution=128,
                                       save_summary=false,
                                       verbose=true)
        
        if length(results) > 0
            println("✓ Testevaluierung erfolgreich abgeschlossen")
            return true
        else
            println("❌ Testevaluierung fehlgeschlagen")
            return false
        end
    catch e
        println("❌ Testfehler: $e")
        return false
    end
end

# =============================================================================
# ERWEITERTE EVALUATION UND PLOTTING-FUNKTIONEN
# =============================================================================

"""
Erstellt automatisch alle Evaluierungen und Plots für verschiedene Kristallanzahlen
"""
function auto_evaluate_and_plot(model_path, max_crystals=10; target_resolution=256, verbose=true)
    println("=== AUTOMATISCHE EVALUATION UND PLOT-ERSTELLUNG ===")
    println("Modell: $model_path")
    println("Max. Kristalle: $max_crystals")
    println("Auflösung: $(target_resolution)x$(target_resolution)")
    
    # 1. Evaluierung durchführen
    if verbose
        println("\n1. Führe Evaluierung durch...")
    end
    results = evaluate_crystal_range(model_path, max_crystals; 
                                   target_resolution=target_resolution,
                                   save_summary=true,
                                   verbose=verbose)
    
    if length(results) == 0
        println("❌ Keine Ergebnisse erhalten")
        return nothing
    end
    
    # 2. Plots erstellen
    if verbose
        println("\n2. Erstelle Plots...")
    end
    plots = create_comprehensive_comparison_plots(results)
    
    # 3. Zusammenfassung anzeigen
    if verbose
        println("\n3. Erstelle Zusammenfassung...")
    end
    print_evaluation_summary_table(results)
    
    println("\n✓ Automatische Evaluierung und Plot-Erstellung abgeschlossen!")
    println("Ergebnisse gespeichert in: $(ensure_evaluation_directory())")
    
    return results, plots
end

println("Model Evaluation Module geladen!")
println("Verfügbare Funktionen:")
println("  - calculate_lamem_fidelity_metrics(model, sample)")
println("  - evaluate_multiple_samples(model, samples)")
println("  - test_lamem_fidelity_evaluation(model_path)")
println("")
println("Zum Testen: test_lamem_fidelity_evaluation()")