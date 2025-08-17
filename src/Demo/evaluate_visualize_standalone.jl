# =============================================================================
# KOMBINIERTE EVALUIERUNG UND VISUALISIERUNG (OHNE LAMEM-ABHÄNGIGKEIT)
# =============================================================================
# Speichern als: evaluate_visualize_standalone.jl

using Plots
using Statistics
using Colors
using BSON
using CSV
using DataFrames
using JSON3
using Serialization
using Dates
using Flux

# KONFIGURATION - SPEICHER-VERZEICHNIS
const OUTPUT_BASE_DIR = "H:\\Masterarbeit\\Auswertung\\Ten_Crystals"

"""
Stelle sicher, dass das Ausgabeverzeichnis existiert
"""
function ensure_output_directory(subdir="")
    if isempty(subdir)
        target_dir = OUTPUT_BASE_DIR
    else
        target_dir = joinpath(OUTPUT_BASE_DIR, subdir)
    end
    
    if !isdir(target_dir)
        try
            mkpath(target_dir)
            println("✓ Ausgabeverzeichnis erstellt: $target_dir")
        catch e
            println("⚠ Warnung: Kann Verzeichnis nicht erstellen: $e")
            println("   Verwende aktuelles Verzeichnis als Fallback")
            return "."
        end
    end
    return target_dir
end

# Lade nur die Module ohne LaMEM-Abhängigkeit
println("Lade Module (ohne LaMEM)...")

# Prüfe welche Module verfügbar sind
modules_to_load = [
    ("data_processing.jl", "Data Processing"),
    ("unet_architecture.jl", "UNet Architecture"), 
    ("training.jl", "Training")
]

for (module_file, module_name) in modules_to_load
    if isfile(module_file)
        try
            include(module_file)
            println("✓ $module_name geladen")
        catch e
            println("⚠ Fehler beim Laden von $module_name: $e")
        end
    else
        println("⚠ $module_file nicht gefunden")
    end
end

# =============================================================================
# KRISTALL-ERKENNUNGS-FUNKTIONEN (EIGENSTÄNDIG)
# =============================================================================

"""
Findet Kristall-Zentren im Phasenfeld durch Clustering
"""
function find_crystal_centers(phase_field; min_crystal_size=50)
    crystal_mask = phase_field .> 0.5
    
    if sum(crystal_mask) == 0
        return Tuple{Float64, Float64}[]
    end
    
    labeled_regions, num_regions = find_connected_components(crystal_mask)
    crystal_centers = Tuple{Float64, Float64}[]
    
    for region_id in 1:num_regions
        region_mask = labeled_regions .== region_id
        region_size = sum(region_mask)
        
        if region_size < min_crystal_size
            continue
        end
        
        indices = findall(region_mask)
        if length(indices) > 0
            center_y = mean([idx[1] for idx in indices])
            center_x = mean([idx[2] for idx in indices])
            push!(crystal_centers, (Float64(center_x), Float64(center_y)))
        end
    end
    
    return crystal_centers
end

"""
Connected Components Analysis
"""
function find_connected_components(binary_mask)
    H, W = size(binary_mask)
    labeled = zeros(Int, H, W)
    current_label = 0
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for i in 1:H, j in 1:W
        if binary_mask[i, j] && labeled[i, j] == 0
            current_label += 1
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
    
    return labeled, current_label
end

"""
Findet Geschwindigkeits-Minima
"""
function find_velocity_minima(velocity_field, expected_count; search_radius=20)
    minima_positions = Tuple{Float64, Float64}[]
    velocity_copy = copy(velocity_field)
    
    for i in 1:expected_count
        min_idx = argmin(velocity_copy)
        min_position = (Float64(min_idx[2]), Float64(min_idx[1]))
        push!(minima_positions, min_position)
        
        y_center, x_center = min_idx[1], min_idx[2]
        H, W = size(velocity_copy)
        y_range = max(1, y_center - search_radius):min(H, y_center + search_radius)
        x_range = max(1, x_center - search_radius):min(W, x_center + search_radius)
        velocity_copy[y_range, x_range] .= Inf
    end
    
    return minima_positions
end

"""
Berechnet Alignment-Fehler
"""
function calculate_alignment_error(points1, points2)
    if length(points1) == 0 || length(points2) == 0
        return -1.0
    end
    
    pos1 = [(Float64(p[1]), Float64(p[2])) for p in points1]
    pos2 = [(Float64(p[1]), Float64(p[2])) for p in points2]
    n_points = min(length(pos1), length(pos2))
    
    if n_points == 0
        return -1.0
    end
    
    total_distance = 0.0
    used_indices = Set{Int}()
    
    for i in 1:n_points
        point1 = pos1[i]
        best_distance = 999999.0
        best_j = 0
        
        for j in 1:length(pos2)
            if j in used_indices
                continue
            end
            
            point2 = pos2[j]
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

# =============================================================================
# ANALYSE-FUNKTIONEN
# =============================================================================

"""
Analysiert bereits vorhandene LaMEM-Daten
"""
function analyze_lamem_sample(sample, model; 
                             target_resolution=256,
                             output_dir="analysis_results",
                             save_results=true,
                             title_prefix="")
    
    start_time = time()
    println("=== ANALYSIERE LAMEM-SAMPLE ===")
    
    try
        # 1. Output-Verzeichnis vorbereiten
        if save_results
            # Verwende den konfigurierten Base-Pfad
            full_output_dir = joinpath(OUTPUT_BASE_DIR, output_dir)
            ensure_output_directory(output_dir)
            println("Speichere Ergebnisse in: $full_output_dir")
        else
            full_output_dir = output_dir
        end
        
        # 2. Sample verarbeiten
        println("1. Verarbeite Sample...")
        x, z, phase, vx, vz, exx, ezz, v_stokes = sample
        
        # Verwende preprocess_lamem_sample falls verfügbar
        if isdefined(Main, :preprocess_lamem_sample)
            phase_tensor, velocity_tensor = preprocess_lamem_sample(
                x, z, phase, vx, vz, v_stokes,
                target_resolution=target_resolution
            )
        else
            # Fallback: Manuelle Verarbeitung
            phase_tensor, velocity_tensor = manual_preprocess(
                phase, vx, vz, v_stokes, target_resolution
            )
        end
        
        # 3. UNet-Vorhersage
        println("2. UNet-Vorhersage...")
        prediction = cpu(model(phase_tensor))
        
        # 4. Extrahiere 2D-Arrays
        phase_2d = phase_tensor[:,:,1,1]
        gt_vx = velocity_tensor[:,:,1,1]
        gt_vz = velocity_tensor[:,:,2,1]
        pred_vx = prediction[:,:,1,1]
        pred_vz = prediction[:,:,2,1]
        
        # 5. Kristall-Analyse
        println("3. Kristall-Erkennung...")
        crystal_centers = find_crystal_centers(phase_2d)
        gt_minima_vz = find_velocity_minima(gt_vz, length(crystal_centers))
        pred_minima_vz = find_velocity_minima(pred_vz, length(crystal_centers))
        
        # 6. Metriken berechnen
        println("4. Berechne Metriken...")
        metrics = calculate_detailed_metrics(
            crystal_centers, gt_minima_vz, pred_minima_vz,
            gt_vx, gt_vz, pred_vx, pred_vz
        )
        
        # 7. Visualisierung
        if save_results
            println("5. Erstelle Visualisierungen...")
            
            vis_paths = create_comprehensive_visualizations(
                phase_2d, gt_vx, gt_vz, pred_vx, pred_vz,
                crystal_centers, gt_minima_vz, pred_minima_vz,
                target_resolution, full_output_dir, title_prefix
            )
            
            # Speichere Ergebnisse
            save_analysis_results(metrics, vis_paths, full_output_dir)
        end
        
        # 8. Ausgabe
        processing_time = (time() - start_time) * 1000
        print_detailed_summary(metrics, processing_time)
        
        return metrics
        
    catch e
        println("Fehler bei Analyse: $e")
        rethrow(e)
    end
end

"""
Manuelle Preprocessing-Funktion falls preprocess_lamem_sample nicht verfügbar
"""
function manual_preprocess(phase, vx, vz, v_stokes, target_resolution)
    # Größenanpassung auf target_resolution
    current_size = size(phase, 1)
    
    if current_size != target_resolution
        # Einfache Interpolation für Größenanpassung
        phase_resized = resize_array(phase, target_resolution)
        vx_resized = resize_array(vx, target_resolution)
        vz_resized = resize_array(vz, target_resolution)
    else
        phase_resized = phase
        vx_resized = vx
        vz_resized = vz
    end
    
    # Normalisierung
    vx_norm = Float32.(vx_resized ./ v_stokes)
    vz_norm = Float32.(vz_resized ./ v_stokes)
    phase_float = Float32.(phase_resized)
    
    # Tensor-Format
    phase_tensor = reshape(phase_float, target_resolution, target_resolution, 1, 1)
    velocity_tensor = cat(vx_norm, vz_norm, dims=3)
    velocity_tensor = reshape(velocity_tensor, target_resolution, target_resolution, 2, 1)
    
    return phase_tensor, velocity_tensor
end

"""
Einfache Array-Größenanpassung
"""
function resize_array(arr, target_size)
    current_size = size(arr, 1)
    if current_size == target_size
        return arr
    end
    
    # Nearest neighbor resampling
    scale_factor = target_size / current_size
    new_arr = zeros(Float64, target_size, target_size)
    
    for i in 1:target_size, j in 1:target_size
        orig_i = max(1, min(current_size, round(Int, i / scale_factor)))
        orig_j = max(1, min(current_size, round(Int, j / scale_factor)))
        new_arr[i, j] = arr[orig_i, orig_j]
    end
    
    return new_arr
end

"""
Berechnet detaillierte Metriken
"""
function calculate_detailed_metrics(crystal_centers, gt_minima, pred_minima, gt_vx, gt_vz, pred_vx, pred_vz)
    return Dict(
        "n_crystals" => length(crystal_centers),
        "lamem_alignment_error" => calculate_alignment_error(crystal_centers, gt_minima),
        "unet_alignment_error" => calculate_alignment_error(crystal_centers, pred_minima),
        "gt_vs_pred_alignment" => calculate_alignment_error(gt_minima, pred_minima),
        "mae_vx" => mean(abs.(pred_vx .- gt_vx)),
        "mae_vz" => mean(abs.(pred_vz .- gt_vz)),
        "mae_total" => (mean(abs.(pred_vx .- gt_vx)) + mean(abs.(pred_vz .- gt_vz))) / 2,
        "rmse_vx" => sqrt(mean((pred_vx .- gt_vx).^2)),
        "rmse_vz" => sqrt(mean((pred_vz .- gt_vz).^2)),
        "correlation_vx" => cor(vec(gt_vx), vec(pred_vx)),
        "correlation_vz" => cor(vec(gt_vz), vec(pred_vz)),
        "crystal_centers" => crystal_centers,
        "gt_minima" => gt_minima,
        "pred_minima" => pred_minima
    )
end

"""
Erstellt umfassende Visualisierungen
"""
function create_comprehensive_visualizations(phase_2d, gt_vx, gt_vz, pred_vx, pred_vz,
                                           crystal_centers, gt_minima, pred_minima,
                                           resolution, output_dir, title_prefix)
    paths = Dict{String, String}()
    timestamp_str = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    
    try
        # 1. 3-Panel Hauptplot
        p1 = create_phase_plot(phase_2d, crystal_centers, resolution)
        p2 = create_velocity_plot(gt_vz, gt_minima, resolution, "LaMEM: v_z")
        p3 = create_velocity_plot(pred_vz, pred_minima, resolution, "UNet: v_z")
        
        main_plot = plot(p1, p2, p3, layout=(1, 3), size=(1200, 400),
                        plot_title="$title_prefix - Geschwindigkeitsfeld-Analyse")
        
        main_path = joinpath(output_dir, "main_analysis_$(timestamp_str).png")
        savefig(main_plot, main_path)
        paths["main_plot"] = main_path
        
        # 2. Differenz-Plots
        diff_vx = pred_vx .- gt_vx
        diff_vz = pred_vz .- gt_vz
        diff_max = max(maximum(abs.(diff_vx)), maximum(abs.(diff_vz)))
        
        p_diff_vx = heatmap(diff_vx, c=:RdBu, aspect_ratio=:equal, 
                           title="Differenz: UNet - LaMEM (v_x)",
                           clims=(-diff_max, diff_max))
        p_diff_vz = heatmap(diff_vz, c=:RdBu, aspect_ratio=:equal,
                           title="Differenz: UNet - LaMEM (v_z)",
                           clims=(-diff_max, diff_max))
        
        diff_plot = plot(p_diff_vx, p_diff_vz, layout=(1, 2), size=(800, 400))
        diff_path = joinpath(output_dir, "velocity_differences_$(timestamp_str).png")
        savefig(diff_plot, diff_path)
        paths["differences"] = diff_path
        
        # 3. Korrelations-Plots
        scatter_vx = scatter(vec(gt_vx), vec(pred_vx), alpha=0.5, 
                           title="Korrelation v_x", xlabel="LaMEM", ylabel="UNet")
        plot!(scatter_vx, [minimum(gt_vx), maximum(gt_vx)], [minimum(gt_vx), maximum(gt_vx)], 
              color=:red, linewidth=2, label="Perfekte Korrelation")
        
        scatter_vz = scatter(vec(gt_vz), vec(pred_vz), alpha=0.5,
                           title="Korrelation v_z", xlabel="LaMEM", ylabel="UNet")
        plot!(scatter_vz, [minimum(gt_vz), maximum(gt_vz)], [minimum(gt_vz), maximum(gt_vz)],
              color=:red, linewidth=2, label="Perfekte Korrelation")
        
        corr_plot = plot(scatter_vx, scatter_vz, layout=(1, 2), size=(800, 400))
        corr_path = joinpath(output_dir, "correlations_$(timestamp_str).png")
        savefig(corr_plot, corr_path)
        paths["correlations"] = corr_path
        
        println("  Visualisierungen erstellt: $(length(paths))")
        return paths
        
    catch e
        println("Fehler bei Visualisierungen: $e")
        return paths
    end
end

"""
Hilfsfunktionen für Plots
"""
function create_phase_plot(phase_2d, crystal_centers, resolution)
    p = heatmap(1:resolution, 1:resolution, phase_2d,
                c=:grays, aspect_ratio=:equal, title="Phasenfeld",
                xlabel="x", ylabel="z")
    
    for (i, center) in enumerate(crystal_centers)
        scatter!(p, [center[1]], [center[2]], 
                markersize=8, markercolor=:white, 
                markerstrokecolor=:black, markerstrokewidth=2,
                label=i==1 ? "Kristall-Zentren" : "")
    end
    
    return p
end

function create_velocity_plot(vz_field, velocity_minima, resolution, plot_title)
    vz_max = maximum(abs.(vz_field))
    
    p = heatmap(1:resolution, 1:resolution, vz_field,
                c=:RdBu, aspect_ratio=:equal, title=plot_title,
                xlabel="x", ylabel="z", clims=(-vz_max, vz_max))
    
    for (i, minimum) in enumerate(velocity_minima)
        scatter!(p, [minimum[1]], [minimum[2]], 
                markersize=10, markershape=:star5, markercolor=:yellow,
                markerstrokecolor=:black, markerstrokewidth=1,
                label=i==1 ? "v_z Minima" : "")
    end
    
    return p
end

"""
Speichert Analyse-Ergebnisse
"""
function save_analysis_results(metrics, vis_paths, output_dir)
    timestamp_str = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    
    # JSON Export
    results_dict = Dict(
        "timestamp" => string(now()),
        "metrics" => metrics,
        "visualization_paths" => vis_paths
    )
    
    json_path = joinpath(output_dir, "analysis_results_$(timestamp_str).json")
    open(json_path, "w") do io
        JSON3.pretty(io, results_dict)
    end
    
    # CSV Export
    csv_data = DataFrame(
        timestamp = [string(now())],
        n_crystals = [metrics["n_crystals"]],
        lamem_alignment = [metrics["lamem_alignment_error"]],
        unet_alignment = [metrics["unet_alignment_error"]],
        mae_total = [metrics["mae_total"]],
        correlation_vx = [metrics["correlation_vx"]],
        correlation_vz = [metrics["correlation_vz"]]
    )
    
    csv_path = joinpath(output_dir, "analysis_summary_$(timestamp_str).csv")
    CSV.write(csv_path, csv_data)
    
    println("  Ergebnisse gespeichert: JSON + CSV")
end

"""
Gibt detaillierte Zusammenfassung aus
"""
function print_detailed_summary(metrics, processing_time)
    println("\n" * "="^60)
    println("DETAILLIERTE ANALYSE-ZUSAMMENFASSUNG")
    println("="^60)
    
    println("Verarbeitungszeit: $(round(processing_time, digits=2)) ms")
    println("Kristalle gefunden: $(metrics["n_crystals"])")
    
    println("\nAlignment-Fehler:")
    println("  LaMEM (GT): $(round(metrics["lamem_alignment_error"], digits=2)) Pixel")
    println("  UNet: $(round(metrics["unet_alignment_error"], digits=2)) Pixel")
    println("  GT vs UNet: $(round(metrics["gt_vs_pred_alignment"], digits=2)) Pixel")
    
    println("\nGeschwindigkeits-Metriken:")
    println("  MAE Total: $(round(metrics["mae_total"], digits=6))")
    println("  MAE v_x: $(round(metrics["mae_vx"], digits=6))")
    println("  MAE v_z: $(round(metrics["mae_vz"], digits=6))")
    println("  RMSE v_x: $(round(metrics["rmse_vx"], digits=6))")
    println("  RMSE v_z: $(round(metrics["rmse_vz"], digits=6))")
    
    println("\nKorrelationen:")
    println("  Korrelation v_x: $(round(metrics["correlation_vx"], digits=3))")
    println("  Korrelation v_z: $(round(metrics["correlation_vz"], digits=3))")
    
    # Bewertung
    mae_quality = if metrics["mae_total"] < 0.01
        "Exzellent"
    elseif metrics["mae_total"] < 0.05
        "Gut"
    elseif metrics["mae_total"] < 0.1
        "Akzeptabel"
    else
        "Verbesserungsbedürftig"
    end
    
    corr_quality = if min(metrics["correlation_vx"], metrics["correlation_vz"]) > 0.9
        "Exzellent"
    elseif min(metrics["correlation_vx"], metrics["correlation_vz"]) > 0.8
        "Gut"
    elseif min(metrics["correlation_vx"], metrics["correlation_vz"]) > 0.7
        "Akzeptabel"
    else
        "Verbesserungsbedürftig"
    end
    
    println("\nGesamtbewertung:")
    println("  MAE-Qualität: $mae_quality")
    println("  Korrelations-Qualität: $corr_quality")
    
    println("="^60)
end

# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

"""
Sichere Modell-Lade-Funktion
"""
function load_model_safe(model_path::String)
    if !isfile(model_path)
        alternative_paths = [
            joinpath("ten_crystal_checkpoints", "best_model.bson"),
            joinpath("checkpoints", "best_model.bson"),
            "model.bson",
            "best_model.bson"
        ]
        
        for alt_path in alternative_paths
            if isfile(alt_path)
                model_path = alt_path
                break
            end
        end
        
        if !isfile(model_path)
            error("Modelldatei nicht gefunden: $model_path")
        end
    end
    
    try
        if isdefined(Main, :load_trained_model)
            return load_trained_model(model_path)
        end
    catch e
        println("load_trained_model nicht verfügbar: $e")
    end
    
    # Fallback: BSON
    model_dict = BSON.load(model_path)
    for key in [:model, :best_model, :final_model, :trained_model]
        if haskey(model_dict, key)
            return model_dict[key]
        end
    end
    
    error("Kein gültiges Modell gefunden")
end

"""
Test-Funktion ohne LaMEM-Abhängigkeit
"""
function test_standalone_analysis()
    println("=== TEST: EIGENSTÄNDIGE ANALYSE ===")
    
    try
        # Erstelle Test-Daten
        println("1. Erstelle Test-Daten...")
        sample = create_test_sample()
        
        # Lade Modell
        println("2. Lade Modell...")
        model = load_model_safe("H:/Masterarbeit/Modelle/ten_crystal_modells/best_model.bson")
        
        # Analysiere - Speichert automatisch in H:\Masterarbeit\Auswertung\Ten_Crystals\
        println("3. Starte Analyse...")
        metrics = analyze_lamem_sample(
            sample, model,
            output_dir="standalone_test",  # Wird als Unterordner erstellt
            title_prefix="Standalone Test"
        )
        
        println("✓ Test erfolgreich abgeschlossen!")
        println("📁 Ergebnisse gespeichert in: $(joinpath(OUTPUT_BASE_DIR, "standalone_test"))")
        return true
        
    catch e
        println("✗ Test fehlgeschlagen: $e")
        return false
    end
end

"""
Organisierte Analyse für Masterarbeit-Experimente
"""
function analyze_experiment(model, sample, experiment_name; 
                           description="", 
                           experiment_group="general")
    
    println("=== MASTERARBEIT-EXPERIMENT: $experiment_name ===")
    
    # Erstelle strukturierte Verzeichnisse
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    subdir = joinpath(experiment_group, "$(experiment_name)_$(timestamp)")
    
    # Analysiere
    metrics = analyze_lamem_sample(
        sample, model,
        output_dir=subdir,
        title_prefix=experiment_name
    )
    
    # Zusätzliche Experiment-Dokumentation
    full_path = joinpath(OUTPUT_BASE_DIR, subdir)
    experiment_info = Dict(
        "experiment_name" => experiment_name,
        "description" => description,
        "timestamp" => string(now()),
        "experiment_group" => experiment_group,
        "metrics_summary" => Dict(
            "n_crystals" => metrics["n_crystals"],
            "mae_total" => metrics["mae_total"],
            "correlation_vz" => metrics["correlation_vz"],
            "unet_alignment_error" => metrics["unet_alignment_error"]
        )
    )
    
    # Speichere Experiment-Info
    info_path = joinpath(full_path, "experiment_info.json")
    open(info_path, "w") do io
        JSON3.pretty(io, experiment_info)
    end
    
    # Update Master-Log
    update_master_experiment_log(experiment_info)
    
    println("📁 Experiment gespeichert: $full_path")
    return metrics
end

"""
Aktualisiert Master-Experiment-Log
"""
function update_master_experiment_log(experiment_info)
    master_log_path = joinpath(OUTPUT_BASE_DIR, "master_experiment_log.csv")
    
    # Neue Zeile für Master-Log
    new_row = DataFrame(
        timestamp = [experiment_info["timestamp"]],
        experiment_name = [experiment_info["experiment_name"]],
        experiment_group = [experiment_info["experiment_group"]],
        description = [experiment_info["description"]],
        n_crystals = [experiment_info["metrics_summary"]["n_crystals"]],
        mae_total = [experiment_info["metrics_summary"]["mae_total"]],
        correlation_vz = [experiment_info["metrics_summary"]["correlation_vz"]],
        unet_alignment_error = [experiment_info["metrics_summary"]["unet_alignment_error"]]
    )
    
    if isfile(master_log_path)
        # Lade existierende Daten und hänge an
        existing_df = CSV.read(master_log_path, DataFrame)
        master_df = vcat(existing_df, new_row)
    else
        master_df = new_row
    end
    
    CSV.write(master_log_path, master_df)
    println("📋 Master-Log aktualisiert: $master_log_path")
end

"""
Erstellt Test-Sample
"""
function create_test_sample()
    # Grid-Koordinaten
    x_vec = collect(range(-1, 1, length=256))
    z_vec = collect(range(-1, 1, length=256))
    
    # Phasenfeld mit 2 Kristallen
    phase = zeros(Float64, 256, 256)
    
    # Kristall 1
    for i in 100:120, j in 100:120
        if (i-110)^2 + (j-110)^2 <= 100
            phase[i, j] = 1.0
        end
    end
    
    # Kristall 2
    for i in 140:160, j in 140:160
        if (i-150)^2 + (j-150)^2 <= 100
            phase[i, j] = 1.0
        end
    end
    
    # Geschwindigkeitsfelder mit realistischen Dipol-Strömungen
    vx = zeros(Float64, 256, 256)
    vz = zeros(Float64, 256, 256)
    
    for i in 1:256, j in 1:256
        # Kristall 1 Einfluss
        r1 = sqrt((i-110)^2 + (j-110)^2)
        if r1 < 50 && r1 > 0
            vx[i, j] += (i-110) / r1 * 0.05
            vz[i, j] -= 0.2
        end
        
        # Kristall 2 Einfluss
        r2 = sqrt((i-150)^2 + (j-150)^2)
        if r2 < 50 && r2 > 0
            vx[i, j] += (i-150) / r2 * 0.05
            vz[i, j] -= 0.15
        end
    end
    
    # Rauschen
    vx .+= randn(256, 256) * 0.01
    vz .+= randn(256, 256) * 0.01
    
    # Dummy-Werte
    exx = zeros(Float64, 256, 256)
    ezz = zeros(Float64, 256, 256)
    v_stokes = 0.5
    
    return (x_vec, z_vec, phase, vx, vz, exx, ezz, v_stokes)
end

println("✓ Eigenständige Evaluierung und Visualisierung geladen!")
println("📁 Speicher-Verzeichnis: $OUTPUT_BASE_DIR")
println("Hauptfunktionen:")
println("  - analyze_lamem_sample(sample, model) - Analysiere existierende LaMEM-Daten")
println("  - test_standalone_analysis() - Test ohne LaMEM-Generierung")
println("  - analyze_experiment(model, sample, name) - Organisierte Masterarbeit-Experimente")
println("")
println("Zum Testen: test_standalone_analysis()")
println("Alle Ergebnisse werden automatisch in H:\\Masterarbeit\\Auswertung\\Ten_Crystals\\ gespeichert")