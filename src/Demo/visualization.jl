# =============================================================================
# VISUALIZATION MODULE - 3-PANEL PLOTS (KORRIGIERT)
# =============================================================================
# Speichern als: visualization.jl

using Plots
using Statistics
using Colors
using BSON

# Module laden (alle erforderlichen Abhängigkeiten)
include("lamem_interface.jl")     # Für LaMEM_Multi_crystal
include("data_processing.jl")     # Für preprocess_lamem_sample
include("unet_architecture.jl")   # Für SimplifiedUNet Definition
include("training.jl")            # Für load_trained_model
include("evaluate_model.jl")      # Für Metriken-Berechnung

# Konstante für Ausgabeverzeichnis
const OUTPUT_DIR = "H:\\Masterarbeit\\Auswertung\\Ten_Crystals"

"""
Stelle sicher, dass das Ausgabeverzeichnis existiert
"""
function ensure_output_directory()
    if !isdir(OUTPUT_DIR)
        try
            mkpath(OUTPUT_DIR)
            println("Ausgabeverzeichnis erstellt: $OUTPUT_DIR")
        catch e
            println("Warnung: Kann Verzeichnis nicht erstellen: $e")
            println("Verwende aktuelles Verzeichnis als Fallback")
            return "."
        end
    end
    return OUTPUT_DIR
end

"""
Lädt ein gespeichertes Modell (falls load_trained_model nicht verfügbar)
"""
function load_model_safe(model_path::String)
    println("Lade Modell: $model_path")
    
    if !isfile(model_path)
        error("Modelldatei nicht gefunden: $model_path")
    end
    
    try
        # Versuche die training.jl Funktion zu verwenden
        return load_trained_model(model_path)
    catch e
        println("load_trained_model nicht verfügbar, verwende BSON.load...")
        
        # Fallback: Direktes BSON laden
        model_dict = BSON.load(model_path)
        
        # Versuche verschiedene Schlüssel
        for key in [:model, :best_model, :final_model, :trained_model]
            if haskey(model_dict, key)
                model = model_dict[key]
                println("Modell unter Schlüssel '$key' gefunden")
                return model
            end
        end
        
        # Fallback: Nehme ersten Wert der ein Modell sein könnte
        for (key, value) in model_dict
            if isa(value, Flux.Chain) || hasproperty(value, :layers) || string(typeof(value)) |> x -> contains(x, "UNet")
                println("Modell unter Schlüssel '$key' gefunden")
                return value
            end
        end
        
        error("Kein Modell in der BSON-Datei gefunden")
    end
end

"""
Erstellt 3-Panel Visualisierung
"""
function create_three_panel_plot(model, sample; 
                                 target_resolution=256, 
                                 save_path=nothing,
                                 title_prefix="")
    
    println("Erstelle 3-Panel Visualisierung...")
    
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
        gt_vz = velocity_tensor[:,:,2,1]  # Fokus auf v_z
        pred_vz = prediction[:,:,2,1]
        
        # 2. Kristall-Zentren und Geschwindigkeits-Minima finden
        crystal_centers = find_crystal_centers(phase_2d)
        gt_minima = find_velocity_minima(gt_vz, length(crystal_centers))
        pred_minima = find_velocity_minima(pred_vz, length(crystal_centers))
        
        # 3. Berechne Alignment-Fehler für Titel
        alignment_error = calculate_alignment_error(crystal_centers, pred_minima)
        
        println("  Kristalle gefunden: $(length(crystal_centers))")
        println("  Alignment-Fehler: $(round(alignment_error, digits=2)) Pixel")
        
        # 4. Erstelle die 3 Plots
        
        # PLOT 1: Phasenfeld
        p1 = create_phase_plot(phase_2d, crystal_centers, target_resolution)
        
        # PLOT 2: LaMEM v_z (Ground Truth)
        p2 = create_velocity_plot(gt_vz, gt_minima, target_resolution, "LaMEM: v_z")
        
        # PLOT 3: UNet v_z (Vorhersage)  
        p3 = create_velocity_plot(pred_vz, pred_minima, target_resolution, "UNet: v_z")
        
        # 5. Kombiniere zu Layout mit Koordinaten-Info
        coord_info = "Koordinaten-Analyse ($(target_resolution), $(target_resolution)):\n"
        coord_info *= "GT Alignment: $(round(calculate_alignment_error(crystal_centers, gt_minima), digits=1)) px\n"
        coord_info *= "UNet Alignment: $(round(alignment_error, digits=1)) px\n"
        coord_info *= "GT vs UNet: $(round(calculate_alignment_error(gt_minima, pred_minima), digits=1)) px\n"
        coord_info *= "\nStatus: "
        coord_info *= alignment_error < 15 ? "OK" : "Schlecht"
        
        # Layout erstellen
        layout_title = isempty(title_prefix) ? "Geschwindigkeitsfeld-Vorhersage" : "$title_prefix - Geschwindigkeitsfeld-Vorhersage"
        
        final_plot = plot(p1, p2, p3, 
                         layout=(1, 3), 
                         size=(1200, 400),
                         plot_title=layout_title)
        
        # Koordinaten-Info ausgeben
        coord_text = "Koordinaten-Analyse ($(target_resolution), $(target_resolution)):\n"
        coord_text *= "GT Alignment: $(round(calculate_alignment_error(crystal_centers, gt_minima), digits=1)) px\n"
        coord_text *= "UNet Alignment: $(round(alignment_error, digits=1)) px\n"
        coord_text *= "GT vs UNet: $(round(calculate_alignment_error(gt_minima, pred_minima), digits=1)) px\n"
        coord_text *= "Status: " * (alignment_error < 15 ? "OK" : "Schlecht")
        
        println("  $coord_text")
        
        # 6. Speichern falls gewünscht mit korrektem Pfad
        if save_path !== nothing
            output_dir = ensure_output_directory()
            full_save_path = joinpath(output_dir, save_path)
            
            savefig(final_plot, full_save_path)
            println("Plot gespeichert: $full_save_path")
        end
        
        return final_plot, crystal_centers, gt_minima, pred_minima
        
    catch e
        println("Fehler bei Visualisierung: $e")
        return nothing, [], [], []
    end
end

"""
Erstellt Phasenfeld-Plot (Panel 1)
"""
function create_phase_plot(phase_2d, crystal_centers, resolution)
    # Schwarz-Weiß Phasenfeld
    p = heatmap(1:resolution, 1:resolution, phase_2d,
                c=:grays, 
                aspect_ratio=:equal,
                title="Phasenfeld",
                xlabel="x", ylabel="z",
                xlims=(1, resolution), ylims=(1, resolution))
    
    # Kristall-Zentren markieren
    for (i, center) in enumerate(crystal_centers)
        x_coord, y_coord = center
        
        # Weiße Punkte für bessere Sichtbarkeit
        scatter!(p, [x_coord], [y_coord], 
                markersize=8, 
                markercolor=:white, 
                markerstrokecolor=:black,
                markerstrokewidth=2,
                label=i==1 ? "Kristall-Zentren" : "")
        
        # Rote Punkte als Alternative
        if i <= 2  # Nur erste 2 rot markieren
            scatter!(p, [x_coord], [y_coord], 
                    markersize=6, 
                    markercolor=:red, 
                    markerstrokewidth=0,
                    label="")
        end
    end
    
    return p
end

"""
Erstellt Geschwindigkeitsfeld-Plot (Panel 2 & 3)
"""
function create_velocity_plot(vz_field, velocity_minima, resolution, plot_title)
    # Bestimme Farbbereich symmetrisch um 0
    vz_max = maximum(abs.(vz_field))
    
    # Velocity Heatmap mit verfügbarer Farbpalette
    p = heatmap(1:resolution, 1:resolution, vz_field,
                c=:RdBu,  # Standard Rot-Blau Farbschema
                aspect_ratio=:equal,
                title=plot_title,
                xlabel="x", ylabel="z",
                xlims=(1, resolution), ylims=(1, resolution),
                clims=(-vz_max, vz_max))
    
    # Geschwindigkeits-Minima markieren (gelbe Sterne)
    for (i, minimum) in enumerate(velocity_minima)
        x_coord, y_coord = minimum
        
        scatter!(p, [x_coord], [y_coord], 
                markersize=10, 
                markershape=:star5,
                markercolor=:yellow, 
                markerstrokecolor=:black,
                markerstrokewidth=1,
                label=i==1 ? "v_z Minima" : "")
    end
    
    return p
end

"""
Test-Funktion für Visualisierung (korrigiert)
"""
function test_visualization(model_path="H:/Masterarbeit/Modelle/ten_crystal_modells/best_model.bson")
    println("=== TEST: 3-PANEL VISUALISIERUNG ===")
    
    try
        # 1. Modell laden (mit Fallback)
        println("1. Lade Modell...")
        model = load_model_safe(model_path)
        println("Modell geladen")
        
        # 2. Test-Sample generieren
        println("2. Generiere Test-Sample...")
        test_sample = LaMEM_Multi_crystal(
            resolution=(256, 256),
            n_crystals=2,
            radius_crystal=[0.05, 0.05],
            cen_2D=[(0.0, 0.3), (0.0, 0.7)]
        )
        println("Test-Sample generiert")
        
        # 3. Visualisierung erstellen
        println("3. Erstelle 3-Panel Plot...")
        plot_result, crystal_centers, gt_minima, pred_minima = create_three_panel_plot(
            model, test_sample,
            save_path="test_visualization.png",
            title_prefix="Test"
        )
        
        if plot_result !== nothing
            println("Visualisierung erfolgreich erstellt")
            
            # Zeige Details
            println("\nDetails:")
            println("  Kristall-Zentren: $crystal_centers")
            println("  GT Minima: $gt_minima") 
            println("  UNet Minima: $pred_minima")
            
            # Zeige Plot
            display(plot_result)
            
            return true
        else
            println("Visualisierung fehlgeschlagen")
            return false
        end
        
    catch e
        println("Fehler bei Visualisierung-Test: $e")
        println("Stacktrace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        return false
    end
end

"""
Interaktive Visualisierung für verschiedene Kristallanzahlen
"""
function interactive_visualization(model_path="H:/Masterarbeit/Modelle/ten_crystal_modells/best_model.bson")
    println("=== INTERAKTIVE KRISTALL-VISUALISIERUNG ===")
    
    # Modell laden
    local model
    try
        model = load_model_safe(model_path)
        println("Modell geladen: $model_path")
    catch e
        println("Fehler beim Laden des Modells: $e")
        return
    end
    
    println("\nBilder werden gespeichert in: $OUTPUT_DIR")
    println("\nInteraktive Visualisierung gestartet...")
    println("Gib eine Kristallanzahl ein (1-15) oder 0 zum Beenden.")
    
    while true
        print("\nKristallanzahl: ")
        input = readline()
        
        if input == "0"
            println("Beende interaktive Visualisierung.")
            break
        end
        
        try
            n_crystals = parse(Int, input)
            
            if n_crystals < 1 || n_crystals > 15
                println("Bitte Zahl zwischen 1 und 15 eingeben!")
                continue
            end
            
            println("\nGeneriere $n_crystals-Kristall System...")
            
            # Sample generieren mit besserer Kristall-Verteilung
            local sample
            try
                if n_crystals <= 2
                    # Für 1-2 Kristalle: Links-Rechts Layout
                    if n_crystals == 1
                        centers = [(0.0, 0.0)]
                        radii = [0.06]
                    else
                        centers = [(-0.5, 0.0), (0.5, 0.0)]
                        radii = [0.05, 0.05]
                    end
                elseif n_crystals <= 4
                    # Für 3-4 Kristalle: Verteiltes Layout
                    centers = [(-0.4, -0.3), (0.4, -0.3), (-0.4, 0.3), (0.4, 0.3)][1:n_crystals]
                    radii = fill(0.04, n_crystals)
                elseif n_crystals <= 8
                    # Für 5-8 Kristalle: Grid-Layout
                    positions = []
                    for i in 1:n_crystals
                        x_pos = -0.6 + (i-1) % 3 * 0.6
                        z_pos = -0.4 + div(i-1, 3) * 0.4
                        push!(positions, (x_pos, z_pos))
                    end
                    centers = positions
                    radii = fill(0.035, n_crystals)
                else
                    # Für 9-15 Kristalle: Dichtes Grid mit kleineren Radien
                    positions = []
                    for i in 1:n_crystals
                        x_pos = -0.8 + (i-1) % 4 * 0.533
                        z_pos = -0.6 + div(i-1, 4) * 0.4
                        push!(positions, (x_pos, z_pos))
                    end
                    centers = positions
                    radii = fill(0.025, n_crystals)
                end
                
                sample = LaMEM_Multi_crystal(
                    resolution=(256, 256),
                    n_crystals=n_crystals,
                    radius_crystal=radii,
                    cen_2D=centers
                )
                
            catch e
                println("Fehler bei Sample-Generierung: $e")
                continue
            end
            
            # Visualisierung erstellen
            save_name = "visualization_$(n_crystals)_crystals.png"
            
            local plot_result
            try
                plot_result, _, _, _ = create_three_panel_plot(
                    model, sample,
                    save_path=save_name,
                    title_prefix="$n_crystals Kristalle"
                )
                
                if plot_result !== nothing
                    println("Visualisierung erstellt und gespeichert: $(joinpath(OUTPUT_DIR, save_name))")
                    display(plot_result)
                else
                    println("Visualisierung fehlgeschlagen")
                end
                
            catch e
                println("Fehler bei Visualisierung: $e")
                continue
            end
            
        catch e
            if isa(e, ArgumentError) && contains(string(e), "invalid base 10 digit")
                println("Bitte nur Zahlen eingeben (z.B. 5)")
            else
                println("Fehler: $e")
            end
            continue
        end
    end
end

"""
Batch-Visualisierung für mehrere Kristallanzahlen

"""
function create_crystal_comparison_plots(model_path="H:/Masterarbeit/Modelle/ten_crystal_modells/best_model.bson", 
                                        crystal_counts=[1, 3, 5, 8, 10])
    println("=== BATCH-VISUALISIERUNG ===")
    println("Erstelle Plots für Kristallanzahlen: $crystal_counts")
    println("Speichern in: $OUTPUT_DIR")
    
    model = load_model_safe(model_path)
    
    for n_crystals in crystal_counts
        println("\nErstelle Plot für $n_crystals Kristalle...")
        
        try
            # Sample generieren
            centers = [(rand(-0.5:0.1:0.5), rand(0.2:0.1:0.8)) for _ in 1:n_crystals]
            radii = fill(0.05, n_crystals)
            
            sample = LaMEM_Multi_crystal(
                resolution=(256, 256),
                n_crystals=n_crystals,
                radius_crystal=radii,
                cen_2D=centers
            )
            
            # Visualisierung
            save_name = "comparison_$(n_crystals)_crystals.png"
            
            plot_result, _, _, _ = create_three_panel_plot(
                model, sample,
                save_path=save_name,
                title_prefix="$n_crystals Kristalle - Vergleich"
            )
            
            if plot_result !== nothing
                println("$(joinpath(OUTPUT_DIR, save_name)) erstellt")
            else
                println("Fehler bei $n_crystals Kristallen")
            end
            
        catch e
            println("Fehler bei $n_crystals Kristallen: $e")
            continue
        end
    end
    
    println("\nBatch-Visualisierung abgeschlossen")
    println("Alle Bilder gespeichert in: $OUTPUT_DIR")
end

println("Visualization Module geladen!")
println("Ausgabeverzeichnis: $OUTPUT_DIR")
println("Verfügbare Funktionen:")
println("  - test_visualization() - Einfacher Test")
println("  - interactive_visualization() - Interaktive Eingabe") 
println("  - create_crystal_comparison_plots() - Batch-Erstellung")
println("")
println("Zum Testen: test_visualization()")