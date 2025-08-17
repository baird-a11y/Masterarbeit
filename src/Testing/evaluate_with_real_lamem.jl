# =============================================================================
# EVALUIERUNG MIT ECHTEN LAMEM-DATEN
# =============================================================================
# Speichern als: evaluate_with_real_lamem.jl

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

# LaMEM laden (ignoriere Precompilation-Warnings)
println("Lade LaMEM (ignoriere Warnings)...")
using LaMEM, GeophysicalModelGenerator

# KONFIGURATION
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

# Lade die erforderlichen Module
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
# ECHTE LAMEM-SIMULATIONEN
# =============================================================================

"""
Erstellt echte LaMEM-Simulation (robuste Version)
"""
function create_real_lamem_sample(;
    resolution=(256, 256),
    n_crystals=2,
    radius_crystal=[0.05, 0.05],
    cen_2D=[(-0.3, 0.0), (0.3, 0.0)],
    η_magma=1e20,
    Δρ=200)
    
    println("Erstelle echte LaMEM-Simulation...")
    println("  Kristalle: $n_crystals")
    println("  Positionen: $cen_2D")
    println("  Radien: $radius_crystal")
    
    try
        # Berechne abgeleitete Parameter
        η_crystal = 1e4 * η_magma           
        ρ_magma = 2700
        ρ_crystal = ρ_magma + Δρ            
        
        # Create LaMEM model
        target_h, target_w = resolution
        nel_h, nel_w = target_h - 1, target_w - 1
        
        model = Model(
            Grid(nel=(nel_h, nel_w), x=[-1,1], z=[-1,1]), 
            Time(nstep_max=1), 
            Output(out_strain_rate=1)
        )
        
        # Define phases
        matrix = Phase(ID=0, Name="matrix", eta=η_magma, rho=ρ_magma)
        crystal = Phase(ID=1, Name="crystal", eta=η_crystal, rho=ρ_crystal)
        add_phase!(model, crystal, matrix)

        # Add crystals
        for i = 1:n_crystals
            current_radius = length(radius_crystal) >= i ? radius_crystal[i] : radius_crystal[1]
            
            if length(cen_2D) >= i
                current_center = cen_2D[i]
            else
                # Fallback: Zufällige Position
                x_pos = rand(-0.6:0.1:0.6)
                z_pos = rand(0.2:0.1:0.8)
                current_center = (x_pos, z_pos)
            end
            
            println("    Kristall $i: Zentrum $(current_center), Radius $current_radius")
            
            add_sphere!(model, 
                cen=(current_center[1], 0.0, current_center[2]), 
                radius=current_radius, 
                phase=ConstantPhase(1)
            )
        end

        # Run LaMEM
        println("  Starte LaMEM-Simulation...")
        run_lamem(model, 1, cores=1, mpiexec="")  # Single-core für Stabilität
        
        # Read results
        data, _ = read_LaMEM_timestep(model, 1)

        # Extract data
        x_vec_1D = data.x.val[:,1,1]
        z_vec_1D = data.z.val[1,1,:]
        phase = data.fields.phase[:,1,:]
        Vx = data.fields.velocity[1][:,1,:]          
        Vz = data.fields.velocity[3][:,1,:]          
        Exx = data.fields.strain_rate[1][:,1,:]      
        Ezz = data.fields.strain_rate[9][:,1,:]      

        # Stokes velocity calculation
        ref_radius = radius_crystal[1]
        V_stokes = 2/9 * Δρ * 9.81 * (ref_radius * 1000)^2 / η_magma  
        V_stokes_cm_year = V_stokes * 100 * (3600*24*365.25)      
        
        println("  ✓ LaMEM-Simulation erfolgreich!")
        println("    Phasen-Bereich: $(minimum(phase)) bis $(maximum(phase))")
        println("    Vx-Bereich: $(minimum(Vx)) bis $(maximum(Vx))")
        println("    Vz-Bereich: $(minimum(Vz)) bis $(maximum(Vz))")
        println("    Stokes-Geschwindigkeit: $(round(V_stokes_cm_year, digits=3)) cm/Jahr")
        
        return x_vec_1D, z_vec_1D, phase, Vx, Vz, Exx, Ezz, V_stokes_cm_year
        
    catch e
        println("✗ LaMEM-Simulation fehlgeschlagen: $e")
        rethrow(e)
    end
end

"""
Vordefinierte LaMEM-Konfigurationen für systematische Tests
"""
function get_lamem_test_configs()
    return Dict(
        "horizontal_layout" => Dict(
            :n_crystals => 2,
            :radius_crystal => [0.05, 0.05],
            :cen_2D => [(-0.4, 0.0), (0.4, 0.0)],
            :description => "2 Kristalle horizontal angeordnet"
        ),
        
        "vertical_layout" => Dict(
            :n_crystals => 2,
            :radius_crystal => [0.05, 0.05],
            :cen_2D => [(0.0, -0.3), (0.0, 0.3)],
            :description => "2 Kristalle vertikal angeordnet"
        ),
        
        "diagonal_layout" => Dict(
            :n_crystals => 2,
            :radius_crystal => [0.05, 0.05],
            :cen_2D => [(-0.3, -0.3), (0.3, 0.3)],
            :description => "2 Kristalle diagonal angeordnet"
        ),
        
        "three_crystals" => Dict(
            :n_crystals => 3,
            :radius_crystal => [0.05, 0.04, 0.06],
            :cen_2D => [(-0.3, 0.0), (0.0, 0.4), (0.3, -0.2)],
            :description => "3 Kristalle in Dreieck-Formation"
        ),
        
        "five_crystals" => Dict(
            :n_crystals => 5,
            :radius_crystal => [0.04, 0.04, 0.04, 0.04, 0.04],
            :cen_2D => [(-0.4, -0.2), (-0.4, 0.2), (0.0, 0.0), (0.4, -0.2), (0.4, 0.2)],
            :description => "5 Kristalle in X-Formation"
        )
    )
end

# =============================================================================
# KRISTALL-ERKENNUNGS-FUNKTIONEN (KOPIERT AUS STANDALONE VERSION)
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

# =============================================================================
# ANALYSE-FUNKTIONEN
# =============================================================================

"""
Analysiert echte LaMEM-Daten mit UNet
"""
function analyze_real_lamem_sample(sample, model; 
                                  target_resolution=256,
                                  output_dir="real_lamem_analysis",
                                  save_results=true,
                                  title_prefix="")
    
    start_time = time()
    println("=== ANALYSIERE ECHTE LAMEM-DATEN ===")
    
    try
        # Output-Verzeichnis vorbereiten
        if save_results
            full_output_dir = joinpath(OUTPUT_BASE_DIR, output_dir)
            ensure_output_directory(output_dir)
            println("Speichere Ergebnisse in: $full_output_dir")
        else
            full_output_dir = output_dir
        end
        
        # Sample verarbeiten
        println("1. Verarbeite LaMEM-Sample...")
        x, z, phase, vx, vz, exx, ezz, v_stokes = sample
        
        println("   Original Dimensionen:")
        println("     Phase: $(size(phase))")
        println("     Vx: $(size(vx))")
        println("     Vz: $(size(vz))")
        println("     Stokes-Geschwindigkeit: $v_stokes cm/Jahr")
        
        # Verwende preprocess_lamem_sample falls verfügbar
        if isdefined(Main, :preprocess_lamem_sample)
            phase_tensor, velocity_tensor = preprocess_lamem_sample(
                x, z, phase, vx, vz, v_stokes,
                target_resolution=target_resolution
            )
            println("   ✓ Preprocessing mit preprocess_lamem_sample")
        else
            error("preprocess_lamem_sample nicht verfügbar! Lade data_processing.jl")
        end
        
        # UNet-Vorhersage
        println("2. UNet-Vorhersage...")
        prediction = cpu(model(phase_tensor))
        
        # Extrahiere 2D-Arrays
        phase_2d = phase_tensor[:,:,1,1]
        gt_vx = velocity_tensor[:,:,1,1]
        gt_vz = velocity_tensor[:,:,2,1]
        pred_vx = prediction[:,:,1,1]
        pred_vz = prediction[:,:,2,1]
        
        println("   Verarbeitete Dimensionen:")
        println("     Phase 2D: $(size(phase_2d))")
        println("     GT vz Bereich: $(minimum(gt_vz)) bis $(maximum(gt_vz))")
        println("     UNet vz Bereich: $(minimum(pred_vz)) bis $(maximum(pred_vz))")
        
        # Kristall-Analyse
        println("3. Kristall-Erkennung...")
        crystal_centers = find_crystal_centers(phase_2d)
        gt_minima_vz = find_velocity_minima(gt_vz, length(crystal_centers))
        pred_minima_vz = find_velocity_minima(pred_vz, length(crystal_centers))
        
        println("   Kristalle gefunden: $(length(crystal_centers))")
        println("   Kristall-Zentren: $crystal_centers")
        println("   GT Minima: $gt_minima_vz")
        println("   UNet Minima: $pred_minima_vz")
        
        # Metriken berechnen
        println("4. Berechne Metriken...")
        metrics = calculate_detailed_metrics(
            crystal_centers, gt_minima_vz, pred_minima_vz,
            gt_vx, gt_vz, pred_vx, pred_vz
        )
        
        # Visualisierung
        if save_results
            println("5. Erstelle Visualisierungen...")
            
            vis_paths = create_real_lamem_visualizations(
                phase_2d, gt_vx, gt_vz, pred_vx, pred_vz,
                crystal_centers, gt_minima_vz, pred_minima_vz,
                target_resolution, full_output_dir, title_prefix
            )
            
            # Speichere Ergebnisse
            save_real_lamem_results(metrics, vis_paths, full_output_dir, sample)
        end
        
        # Ausgabe
        processing_time = (time() - start_time) * 1000
        print_real_lamem_summary(metrics, processing_time)
        
        return metrics
        
    catch e
        println("Fehler bei LaMEM-Analyse: $e")
        rethrow(e)
    end
end

"""
Erstellt Visualisierungen für echte LaMEM-Daten
"""
function create_real_lamem_visualizations(phase_2d, gt_vx, gt_vz, pred_vx, pred_vz,
                                        crystal_centers, gt_minima, pred_minima,
                                        resolution, output_dir, title_prefix)
    paths = Dict{String, String}()
    timestamp_str = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    
    try
        # 1. 3-Panel Hauptplot (wie Image 1)
        p1 = create_phase_plot(phase_2d, crystal_centers, resolution)
        p2 = create_velocity_plot(gt_vz, gt_minima, resolution, "LaMEM: v_z")
        p3 = create_velocity_plot(pred_vz, pred_minima, resolution, "UNet: v_z")
        
        main_plot = plot(p1, p2, p3, layout=(1, 3), size=(1200, 400),
                        plot_title="$title_prefix - Geschwindigkeitsfeld-Vorhersage")
        
        main_path = joinpath(output_dir, "real_lamem_analysis_$(timestamp_str).png")
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
        diff_path = joinpath(output_dir, "lamem_velocity_differences_$(timestamp_str).png")
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
        corr_path = joinpath(output_dir, "lamem_correlations_$(timestamp_str).png")
        savefig(corr_plot, corr_path)
        paths["correlations"] = corr_path
        
        println("  ✓ Visualisierungen erstellt: $(length(paths))")
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
Speichert echte LaMEM-Ergebnisse
"""
function save_real_lamem_results(metrics, vis_paths, output_dir, original_sample)
    timestamp_str = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    
    # Erweiterte Daten mit LaMEM-Infos
    results_dict = Dict(
        "timestamp" => string(now()),
        "data_source" => "real_lamem_simulation",
        "lamem_info" => Dict(
            "stokes_velocity" => original_sample[8],
            "original_dimensions" => size(original_sample[3]),
            "velocity_ranges" => Dict(
                "vx_min" => minimum(original_sample[4]),
                "vx_max" => maximum(original_sample[4]),
                "vz_min" => minimum(original_sample[5]),
                "vz_max" => maximum(original_sample[5])
            )
        ),
        "metrics" => metrics,
        "visualization_paths" => vis_paths
    )
    
    # JSON Export
    json_path = joinpath(output_dir, "real_lamem_results_$(timestamp_str).json")
    open(json_path, "w") do io
        JSON3.pretty(io, results_dict)
    end
    
    # CSV Export  
    csv_data = DataFrame(
        timestamp = [string(now())],
        data_source = ["real_lamem"],
        n_crystals = [metrics["n_crystals"]],
        stokes_velocity = [original_sample[8]],
        lamem_alignment = [metrics["lamem_alignment_error"]],
        unet_alignment = [metrics["unet_alignment_error"]],
        mae_total = [metrics["mae_total"]],
        correlation_vx = [metrics["correlation_vx"]],
        correlation_vz = [metrics["correlation_vz"]]
    )
    
    csv_path = joinpath(output_dir, "real_lamem_summary_$(timestamp_str).csv")
    CSV.write(csv_path, csv_data)
    
    println("  ✓ Echte LaMEM-Ergebnisse gespeichert: JSON + CSV")
end

"""
Gibt Zusammenfassung für echte LaMEM-Daten aus
"""
function print_real_lamem_summary(metrics, processing_time)
    println("\n" * "="^60)
    println("ECHTE LAMEM-DATEN ANALYSE-ZUSAMMENFASSUNG")
    println("="^60)
    
    println("Datenquelle: Echte LaMEM-Simulation")
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
    
    println("\nGesamtbewertung (ECHTE LaMEM-Daten):")
    println("  MAE-Qualität: $mae_quality")
    println("  Korrelations-Qualität: $corr_quality")
    
    println("="^60)
end

# =============================================================================
# SICHERE MODELL-LADE-FUNKTION
# =============================================================================

# =============================================================================
# SICHERE MODELL-LADE-FUNKTION UND TEST-FUNKTIONEN
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

# =============================================================================
# HAUPTTEST-FUNKTIONEN
# =============================================================================

"""
Test mit echter LaMEM-Simulation (horizontal layout wie Image 1)
"""
function test_real_lamem_analysis()
    println("=== TEST: ECHTE LAMEM-SIMULATION ===")
    
    try
        # 1. Lade Modell
        println("1. Lade UNet-Modell...")
        model = load_model_safe("H:/Masterarbeit/Modelle/ten_crystal_modells/best_model.bson")
        println("   ✓ Modell geladen")
        
        # 2. Erstelle echte LaMEM-Simulation (horizontal wie Image 1)
        println("2. Erstelle echte LaMEM-Simulation...")
        real_sample = create_real_lamem_sample(
            resolution=(256, 256),
            n_crystals=2,
            radius_crystal=[0.05, 0.05],
            cen_2D=[(-0.4, 0.0), (0.4, 0.0)],  # Horizontal angeordnet
            η_magma=1e20,
            Δρ=200
        )
        println("   ✓ LaMEM-Simulation erfolgreich")
        
        # 3. Analysiere mit UNet
        println("3. Starte UNet-Analyse...")
        metrics = analyze_real_lamem_sample(
            real_sample, model,
            output_dir="real_lamem_test",
            title_prefix="Echte LaMEM-Simulation"
        )
        
        println("✓ Test mit echten LaMEM-Daten erfolgreich!")
        println("📁 Ergebnisse gespeichert in: $(joinpath(OUTPUT_BASE_DIR, "real_lamem_test"))")
        return true
        
    catch e
        println("✗ Test fehlgeschlagen: $e")
        println("\nMögliche Lösungen:")
        println("1. Stelle sicher, dass LaMEM korrekt installiert ist")
        println("2. Überprüfe ob preprocess_lamem_sample verfügbar ist")
        println("3. Teste mit gespeicherten LaMEM-Daten")
        return false
    end
end

"""
Systematische Tests mit verschiedenen Kristall-Konfigurationen
"""
function test_systematic_lamem_configurations()
    println("=== SYSTEMATISCHE LAMEM-KONFIGURATIONEN ===")
    
    try
        # Lade Modell
        model = load_model_safe("H:/Masterarbeit/Modelle/ten_crystal_modells/best_model.bson")
        configs = get_lamem_test_configs()
        
        results_summary = []
        
        for (config_name, config) in configs
            println("\n--- Teste Konfiguration: $config_name ---")
            println("   $(config[:description])")
            
            try
                # Erstelle LaMEM-Sample
                sample = create_real_lamem_sample(
                    resolution=(256, 256),
                    n_crystals=config[:n_crystals],
                    radius_crystal=config[:radius_crystal],
                    cen_2D=config[:cen_2D]
                )
                
                # Analysiere
                metrics = analyze_real_lamem_sample(
                    sample, model,
                    output_dir="systematic_$(config_name)",
                    title_prefix="$(config_name) - $(config[:description])"
                )
                
                # Sammle Ergebnisse
                push!(results_summary, (
                    config_name = config_name,
                    n_crystals = config[:n_crystals],
                    mae_total = metrics["mae_total"],
                    correlation_vz = metrics["correlation_vz"],
                    unet_alignment = metrics["unet_alignment_error"],
                    description = config[:description]
                ))
                
                println("   ✓ $config_name erfolgreich: MAE=$(round(metrics["mae_total"], digits=4))")
                
            catch e
                println("   ✗ $config_name fehlgeschlagen: $e")
                continue
            end
        end
        
        # Erstelle Vergleichstabelle
        if !isempty(results_summary)
            summary_df = DataFrame(results_summary)
            summary_path = joinpath(OUTPUT_BASE_DIR, "systematic_comparison.csv")
            CSV.write(summary_path, summary_df)
            
            println("\n" * "="^60)
            println("SYSTEMATISCHER VERGLEICH ABGESCHLOSSEN")
            println("="^60)
            println("Vergleichstabelle: $summary_path")
            
            # Zeige beste Konfiguration
            best_idx = argmin(summary_df.mae_total)
            best_config = summary_df[best_idx, :]
            println("\nBeste Konfiguration:")
            println("  Name: $(best_config.config_name)")
            println("  Beschreibung: $(best_config.description)")
            println("  MAE Total: $(round(best_config.mae_total, digits=4))")
            println("  Korrelation v_z: $(round(best_config.correlation_vz, digits=3))")
            
            return true
        else
            println("Keine erfolgreichen Konfigurationen")
            return false
        end
        
    catch e
        println("Fehler bei systematischen Tests: $e")
        return false
    end
end

"""
Erstelle und speichere mehrere LaMEM-Samples für spätere Verwendung
"""
function generate_lamem_sample_batch(n_samples=10)
    println("=== ERSTELLE LAMEM-SAMPLE-BATCH ===")
    println("Generiere $n_samples LaMEM-Samples...")
    
    samples = []
    configs = get_lamem_test_configs()
    config_names = collect(keys(configs))
    
    for i in 1:n_samples
        println("\nGeneriere Sample $i/$n_samples...")
        
        # Wähle zufällige Konfiguration
        config_name = rand(config_names)
        config = configs[config_name]
        
        # Leichte Variation der Parameter
        varied_config = Dict(
            :n_crystals => config[:n_crystals],
            :radius_crystal => [r + rand(-0.01:0.001:0.01) for r in config[:radius_crystal]],
            :cen_2D => [(x + rand(-0.1:0.02:0.1), z + rand(-0.1:0.02:0.1)) for (x, z) in config[:cen_2D]]
        )
        
        try
            sample = create_real_lamem_sample(
                resolution=(256, 256),
                n_crystals=varied_config[:n_crystals],
                radius_crystal=varied_config[:radius_crystal],
                cen_2D=varied_config[:cen_2D],
                η_magma=10^(rand() * 2 + 19),  # 1e19 bis 1e21
                Δρ=rand(150:50:300)           # 150-300
            )
            
            push!(samples, (sample=sample, config_name=config_name, config=varied_config))
            println("   ✓ Sample $i erfolgreich (basierend auf $config_name)")
            
        catch e
            println("   ✗ Sample $i fehlgeschlagen: $e")
            continue
        end
        
        # Memory cleanup
        if i % 3 == 0
            GC.gc()
        end
    end
    
    # Speichere Batch
    if !isempty(samples)
        batch_path = joinpath(OUTPUT_BASE_DIR, "lamem_sample_batch_$(length(samples))_samples.jls")
        serialize(batch_path, samples)
        
        println("\n✓ Sample-Batch gespeichert: $batch_path")
        println("  Samples erstellt: $(length(samples))/$n_samples")
        println("  Verwendung: samples = deserialize(\"$batch_path\")")
        
        return batch_path
    else
        println("\n✗ Keine Samples erfolgreich erstellt")
        return nothing
    end
end

"""
Analysiere gespeicherten Sample-Batch
"""
function analyze_sample_batch(batch_path=nothing)
    if batch_path === nothing
        # Finde neuesten Batch
        batch_files = filter(f -> contains(f, "lamem_sample_batch") && endswith(f, ".jls"), 
                           readdir(OUTPUT_BASE_DIR))
        
        if isempty(batch_files)
            println("Keine Sample-Batches gefunden. Erstelle zuerst mit generate_lamem_sample_batch()")
            return false
        end
        
        batch_path = joinpath(OUTPUT_BASE_DIR, sort(batch_files)[end])
    end
    
    println("=== ANALYSIERE SAMPLE-BATCH ===")
    println("Lade Batch: $batch_path")
    
    try
        samples_data = deserialize(batch_path)
        model = load_model_safe("H:/Masterarbeit/Modelle/ten_crystal_modells/best_model.bson")
        
        println("Samples im Batch: $(length(samples_data))")
        
        batch_results = []
        
        for (i, sample_data) in enumerate(samples_data)
            println("\nAnalysiere Sample $i/$(length(samples_data)) ($(sample_data.config_name))")
            
            try
                metrics = analyze_real_lamem_sample(
                    sample_data.sample, model,
                    output_dir="batch_analysis_sample_$(i)",
                    title_prefix="Batch Sample $i - $(sample_data.config_name)"
                )
                
                push!(batch_results, merge(
                    Dict("sample_id" => i, "config_name" => sample_data.config_name),
                    Dict(k => v for (k, v) in metrics if isa(v, Real))
                ))
                
                println("   ✓ Sample $i: MAE=$(round(metrics["mae_total"], digits=4))")
                
            catch e
                println("   ✗ Sample $i fehlgeschlagen: $e")
                continue
            end
        end
        
        # Erstelle Batch-Zusammenfassung
        if !isempty(batch_results)
            batch_df = DataFrame(batch_results)
            summary_path = joinpath(OUTPUT_BASE_DIR, "batch_analysis_summary.csv")
            CSV.write(summary_path, batch_df)
            
            println("\n" * "="^60)
            println("BATCH-ANALYSE ZUSAMMENFASSUNG")
            println("="^60)
            println("Analysierte Samples: $(length(batch_results))")
            println("Durchschnittliche MAE: $(round(mean(batch_df.mae_total), digits=4))")
            println("Durchschnittliche Korrelation v_z: $(round(mean(batch_df.correlation_vz), digits=3))")
            println("Zusammenfassung: $summary_path")
            
            return true
        else
            println("Keine erfolgreichen Analysen")
            return false
        end
        
    catch e
        println("Fehler beim Laden des Batches: $e")
        return false
    end
end

# =============================================================================
# EINFACHE START-FUNKTION
# =============================================================================

"""
Einfacher Start-Test für echte LaMEM-Daten
"""
function start_real_lamem_test()
    println("🚀 STARTE ECHTE LAMEM-ANALYSE")
    println("="^50)
    
    # Prüfe Abhängigkeiten
    println("1. Prüfe Abhängigkeiten...")
    dependencies_ok = true
    
    if !isdefined(Main, :preprocess_lamem_sample)
        println("   ⚠ preprocess_lamem_sample nicht verfügbar")
        dependencies_ok = false
    end
    
    if !isfile("H:/Masterarbeit/Modelle/ten_crystal_modells/best_model.bson")
        println("   ⚠ UNet-Modell nicht gefunden")
        dependencies_ok = false
    end
    
    if dependencies_ok
        println("   ✓ Alle Abhängigkeiten verfügbar")
        
        # Starte Test
        println("\n2. Starte Test...")
        success = test_real_lamem_analysis()
        
        if success
            println("\n🎉 ERFOLG!")
            println("Du hast jetzt echte LaMEM-Daten analysiert!")
            println("Schaue dir die Plots in H:\\Masterarbeit\\Auswertung\\Ten_Crystals\\ an")
        else
            println("\n💡 ALTERNATIVE:")
            println("Versuche: generate_lamem_sample_batch(5)")
        end
        
        return success
    else
        println("\n❌ ABHÄNGIGKEITEN FEHLEN")
        println("Bitte lade zuerst:")
        println("   include(\"data_processing.jl\")")
        println("   include(\"unet_architecture.jl\")")
        println("   include(\"training.jl\")")
        return false
    end
end

println("✓ Echte LaMEM-Evaluierung geladen!")
println("📁 Speicher-Verzeichnis: $OUTPUT_BASE_DIR")
println("\nHauptfunktionen:")
println("  🎯 start_real_lamem_test() - Einfacher Start")
println("  🔬 test_real_lamem_analysis() - Einzelner Test")
println("  📊 test_systematic_lamem_configurations() - Systematische Tests")
println("  📦 generate_lamem_sample_batch(n) - Erstelle Sample-Batch")
println("  📈 analyze_sample_batch() - Analysiere Sample-Batch")
println("")
println("🚀 Zum Starten: start_real_lamem_test()")
println("Alle Ergebnisse werden in H:\\Masterarbeit\\Auswertung\\Ten_Crystals\\ gespeichert")