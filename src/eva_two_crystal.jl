using Flux, BSON, Statistics
using GLMakie
using LaMEM, GeophysicalModelGenerator

println("=== ERWEITERTE V_Z KOORDINATEN-DEBUGGING (1-2 KRISTALLE) ===")

# ==================== EINFACHE MODELL-DEFINITIONEN ====================

struct VelocityUNet
    encoder1; encoder2; encoder3; encoder4; bottleneck
    decoder4; decoder4_1; decoder3; decoder3_1
    decoder2; decoder2_1; decoder1; decoder1_1
end

Flux.@functor VelocityUNet

function crop_and_concat(x, skip, dims=3)
    x_size = size(x)
    skip_size = size(skip)
    
    height_diff = skip_size[1] - x_size[1]
    width_diff = skip_size[2] - x_size[2]
    
    if height_diff < 0 || width_diff < 0
        target_h = max(x_size[1], skip_size[1])
        target_w = max(x_size[2], skip_size[2])
        
        padded_skip = zeros(eltype(skip), target_h, target_w, skip_size[3], skip_size[4])
        
        h_offset = max(0, -height_diff) ÷ 2 + 1
        w_offset = max(0, -width_diff) ÷ 2 + 1
        
        padded_skip[h_offset:h_offset+skip_size[1]-1,
                    w_offset:w_offset+skip_size[2]-1, :, :] .= skip
        
        return cat(x, padded_skip, dims=dims)
    else
        h_offset = height_diff ÷ 2 + 1
        w_offset = width_diff ÷ 2 + 1
        
        h_end = h_offset + x_size[1] - 1
        w_end = w_offset + x_size[2] - 1
        
        h_end = min(h_end, skip_size[1])
        w_end = min(w_end, skip_size[2])
        
        cropped_skip = skip[h_offset:h_end, w_offset:w_end, :, :]
        
        return cat(x, cropped_skip, dims=dims)
    end
end

function (model::VelocityUNet)(x)
    e1 = model.encoder1(x)
    e2 = model.encoder2(e1)
    e3 = model.encoder3(e2)
    e4 = model.encoder4(e3)
    b = model.bottleneck(e4)
    
    d4 = model.decoder4(b)
    d4 = model.decoder4_1(crop_and_concat(d4, e4))
    
    d3 = model.decoder3(d4)
    d3 = model.decoder3_1(crop_and_concat(d3, e3))
    
    d2 = model.decoder2(d3)
    d2 = model.decoder2_1(crop_and_concat(d2, e2))
    
    d1 = model.decoder1(d2)
    d1 = model.decoder1_1(crop_and_concat(d1, e1))
    
    return d1
end

# ==================== ERWEITERTE LAMEM FUNCTION ====================

function LaMEM_Multi_crystal(; nx=256, nz=256, η=1e20, Δρ=200, cen_2D=[(0.0, 0.0)], R=[0.1])
    """
    LaMEM Funktion für 1-2 Kristalle
    """
    η_crystal = 1e4*η
    ρ_magma = 2700

    model = Model(Grid(nel=(nx,nz), x=[-1,1], z=[-1,1]), Time(nstep_max=1), Output(out_strain_rate=1))
    matrix = Phase(ID=0, Name="matrix", eta=η, rho=ρ_magma)
    
    # Kristall-Phasen erstellen (bis zu 2)
    crystal_phases = []
    for i in 1:length(cen_2D)
        # Falls Δρ ein Array ist, verwende verschiedene Werte
        if isa(Δρ, Array)
            rho_crystal = ρ_magma + Δρ[min(i, length(Δρ))]
        else
            rho_crystal = ρ_magma + Δρ
        end
        
        crystal = Phase(ID=i, Name="crystal_$i", eta=η_crystal, rho=rho_crystal)
        push!(crystal_phases, crystal)
    end
    
    add_phase!(model, matrix)
    for crystal in crystal_phases
        add_phase!(model, crystal)
    end

    # Kristalle als Sphären hinzufügen
    for i = 1:length(cen_2D)
        add_sphere!(model, 
                   cen=(cen_2D[i][1], 0.0, cen_2D[i][2]), 
                   radius=R[min(i, length(R))], 
                   phase=ConstantPhase(i))
    end
   
    run_lamem(model, 1)
    data, _ = read_LaMEM_timestep(model, 1)

    x_vec_1D = data.x.val[:,1,1]
    z_vec_1D = data.z.val[1,1,:]
    phase = data.fields.phase[:,1,:]
    Vx = data.fields.velocity[1][:,1,:]
    Vz = data.fields.velocity[3][:,1,:]
    
    # Stokes-Geschwindigkeit basierend auf erstem Kristall
    first_delta_rho = isa(Δρ, Array) ? Δρ[1] : Δρ
    first_radius = isa(R, Array) ? R[1] : R
    V_stokes = 2/9*first_delta_rho*9.81*(first_radius*1000)^2/(η)
    V_stokes_cm_year = V_stokes * 100 * (3600*24*365.25)

    return x_vec_1D, z_vec_1D, phase, Vx, Vz, V_stokes_cm_year
end

# ==================== ERWEITERTE HILFSFUNKTIONEN ====================

function load_velocity_model(model_path)
    println("Lade Modell: $model_path")
    if !isfile(model_path)
        error("Modelldatei nicht gefunden: $model_path")
    end
    
    model_dict = BSON.load(model_path)
    model_cpu = nothing
    
    for key in [:final_model_cpu, :model_cpu, :model, :trained_model, :final_velocity_model]
        if haskey(model_dict, key)
            model_cpu = model_dict[key]
            println("Modell unter Schlüssel '$key' gefunden")
            break
        end
    end
    
    if model_cpu === nothing
        model_cpu = first(values(model_dict))
        println("Verwende ersten Wert aus BSON-Datei")
    end
    
    return model_cpu
end

function find_crystal_centers_multi(phase_field)
    """
    Findet die Zentren aller Kristalle im Phasenfeld (bis zu 2)
    Returns: Array von (x_center, z_center) für jeden Kristall
    """
    crystal_centers = []
    
    # Finde alle Kristall-IDs (größer als 0.5)
    unique_phases = unique(phase_field[phase_field .> 0.5])
    sort!(unique_phases)  # Sortiere für konsistente Reihenfolge
    
    for phase_id in unique_phases
        # Finde alle Punkte für diese Phase
        crystal_indices = findall(x -> abs(x - phase_id) < 0.1, phase_field)
        
        if !isempty(crystal_indices)
            x_coords = [idx[1] for idx in crystal_indices]
            z_coords = [idx[2] for idx in crystal_indices]
            
            center = (mean(x_coords), mean(z_coords))
            push!(crystal_centers, center)
        end
    end
    
    return crystal_centers
end

function find_velocity_minima_multi(vz_field, num_minima=2)
    """
    Findet die N stärksten lokalen Minima in v_z
    """
    # Sortiere alle Punkte nach v_z Wert
    linear_indices = sortperm(vec(vz_field))
    
    minima = []
    min_distance = 20  # Mindestabstand zwischen Minima in Pixeln
    
    for idx in linear_indices
        cart_idx = CartesianIndices(vz_field)[idx]
        pos = (cart_idx[1], cart_idx[2])
        value = vz_field[idx]
        
        # Prüfe ob weit genug von bestehenden Minima entfernt
        too_close = false
        for existing_min in minima
            distance = sqrt((pos[1] - existing_min[1])^2 + (pos[2] - existing_min[2])^2)
            if distance < min_distance
                too_close = true
                break
            end
        end
        
        if !too_close
            push!(minima, (pos[1], pos[2], value))
            
            if length(minima) >= num_minima
                break
            end
        end
    end
    
    return minima
end

# ==================== HAUPTFUNKTION: ERWEITERTE V_Z KOORDINATEN-DEBUG ====================

function debug_vz_coordinates_adaptive(model_path; target_size=(256, 256), num_crystals=1)
    """
    Erweiterte Debug-Funktion für 1-2 Kristalle
    """
    println("\n=== ADAPTIVE V_Z KOORDINATEN-DEBUGGING (1-$(num_crystals) KRISTALLE) ===")
    
    # 1. Modell laden
    model = load_velocity_model(model_path)
    
    # 2. LaMEM Ground Truth generieren (abhängig von num_crystals)
    println("\n1. Generiere LaMEM Ground Truth für $num_crystals Kristall(e)...")
    
    if num_crystals == 1
        # Ein Kristall
        x, z, phase_gt, vx_gt, vz_gt, v_stokes = LaMEM_Multi_crystal(
            nx=target_size[1], nz=target_size[2], 
            η=1e20, Δρ=200, 
            cen_2D=[(0.0, 0.5)], 
            R=[0.05]
        )
        crystal_params = "1 Kristall: Position (0.0, 0.5), Radius 0.05"
    else
        # Zwei Kristalle
        x, z, phase_gt, vx_gt, vz_gt, v_stokes = LaMEM_Multi_crystal(
            nx=target_size[1], nz=target_size[2], 
            η=1e20, Δρ=[200, 250], 
            cen_2D=[(-0.3, 0.4), (0.3, 0.6)], 
            R=[0.04, 0.05]
        )
        crystal_params = "2 Kristalle: Positionen (-0.3, 0.4), (0.3, 0.6), Radien 0.04, 0.05"
    end
    
    println("   Dimensionen - Phase: $(size(phase_gt)), Vz: $(size(vz_gt))")
    println("   Parameter: $crystal_params")
    
    # 3. Kristall-Positionen in Ground Truth finden
    println("\n2. Analysiere Ground Truth...")
    gt_crystal_centers = find_crystal_centers_multi(phase_gt)
    gt_vz_minima = find_velocity_minima_multi(vz_gt, length(gt_crystal_centers))
    
    println("   Gefundene Kristall-Zentren (GT): $(length(gt_crystal_centers))")
    for (i, center) in enumerate(gt_crystal_centers)
        println("     Kristall $i: $(round.(center, digits=2))")
    end
    
    println("   Gefundene v_z Minima (GT): $(length(gt_vz_minima))")
    for (i, minimum) in enumerate(gt_vz_minima)
        println("     Minimum $i: Position $(round.(minimum[1:2], digits=1)), Wert: $(round(minimum[3], digits=6))")
    end
    
    # Berechne Alignment-Fehler für jeden Kristall
    gt_alignment_errors = []
    for (i, crystal_center) in enumerate(gt_crystal_centers)
        if i <= length(gt_vz_minima)
            vz_min = gt_vz_minima[i]
            error = sqrt((crystal_center[1] - vz_min[1])^2 + (crystal_center[2] - vz_min[2])^2)
            push!(gt_alignment_errors, error)
            println("   GT Alignment-Fehler Kristall $i: $(round(error, digits=1)) Pixel")
        end
    end
    
    # 4. UNet Input vorbereiten
    println("\n3. Bereite UNet Input vor...")
    actual_size = size(phase_gt)
    if actual_size != target_size
        println("   Warnung: LaMEM liefert $(actual_size) statt $(target_size) - passe an")
        target_size = actual_size
    end
    
    phase_input = reshape(Float32.(phase_gt), actual_size[1], actual_size[2], 1, 1)
    
    # 5. UNet Vorhersage
    println("\n4. UNet Vorhersage...")
    local vz_pred
    try
        prediction = model(phase_input)
        vz_pred = prediction[:, :, 2, 1]  # Nur v_z Komponente
        
        println("   UNet Ausgabe-Dimensionen: $(size(prediction))")
        println("   v_z Vorhersage-Dimensionen: $(size(vz_pred))")
    catch e
        println("   UNet Fehler: $e")
        println("   Input Dimensionen waren: $(size(phase_input))")
        return nothing
    end
    
    # 6. UNet Koordinaten analysieren
    println("\n5. Analysiere UNet Vorhersage...")
    pred_crystal_centers = find_crystal_centers_multi(phase_gt)  # Gleiche Phase wie GT
    pred_vz_minima = find_velocity_minima_multi(vz_pred, length(pred_crystal_centers))
    
    println("   UNet v_z Minima: $(length(pred_vz_minima))")
    for (i, minimum) in enumerate(pred_vz_minima)
        println("     Minimum $i: Position $(round.(minimum[1:2], digits=1)), Wert: $(round(minimum[3], digits=6))")
    end
    
    # Alignment-Fehler für UNet
    pred_alignment_errors = []
    for (i, crystal_center) in enumerate(pred_crystal_centers)
        if i <= length(pred_vz_minima)
            vz_min = pred_vz_minima[i]
            error = sqrt((crystal_center[1] - vz_min[1])^2 + (crystal_center[2] - vz_min[2])^2)
            push!(pred_alignment_errors, error)
            println("   UNet Alignment-Fehler Kristall $i: $(round(error, digits=1)) Pixel")
        end
    end
    
    # 7. Koordinaten-Vergleich
    println("\n6. Koordinaten-Vergleich...")
    gt_to_unet_errors = []
    for (i, gt_min) in enumerate(gt_vz_minima)
        if i <= length(pred_vz_minima)
            pred_min = pred_vz_minima[i]
            error = sqrt((gt_min[1] - pred_min[1])^2 + (gt_min[2] - pred_min[2])^2)
            push!(gt_to_unet_errors, error)
            println("   GT vs UNet Kristall $i: $(round(error, digits=1)) Pixel Unterschied")
        end
    end
    
    # 8. Visualisierung mit adaptiver Markierung
    println("\n7. Erstelle adaptive Visualisierung...")
    
    fig = Figure(resolution=(1200, 400))
    
    # Farben für bis zu 2 Kristalle
    crystal_colors = [:red, :blue]
    minima_colors = [:yellow, :orange]
    
    # Plot 1: Phasenfeld
    ax1 = Axis(fig[1, 1], title="Phasenfeld ($num_crystals Kristall$(num_crystals > 1 ? "e" : ""))", 
               xlabel="x", ylabel="z", aspect=DataAspect())
    heatmap!(ax1, phase_gt, colormap=:grays)
    
    # Markiere Kristall-Zentren
    for (i, center) in enumerate(gt_crystal_centers)
        color = crystal_colors[min(i, length(crystal_colors))]
        scatter!(ax1, [center[2]], [center[1]], color=color, markersize=12, 
                marker=:circle, strokewidth=2, strokecolor=:white)
    end
    
    # Plot 2: LaMEM v_z
    ax2 = Axis(fig[1, 2], title="LaMEM: v_z", xlabel="x", ylabel="z", aspect=DataAspect())
    heatmap!(ax2, vz_gt ./ v_stokes, colormap=:RdBu, colorrange=(-3, 1))
    
    # Konturen für alle Kristalle
    if num_crystals == 1
        contour!(ax2, phase_gt, levels=[0.5], color=:black, linewidth=2)
    else
        contour!(ax2, phase_gt, levels=[0.5, 1.5], color=:black, linewidth=2)
    end
    
    # # Markiere GT Minima
    # for (i, minimum) in enumerate(gt_vz_minima)
    #     color = minima_colors[min(i, length(minima_colors))]
    #     scatter!(ax2, [minimum[2]], [minimum[1]], color=color, markersize=12,
    #             marker=:star5, strokewidth=2, strokecolor=:black)
    # end
    
    # Plot 3: UNet v_z
    ax3 = Axis(fig[1, 3], title="UNet: v_z", xlabel="x", ylabel="z", aspect=DataAspect())
    heatmap!(ax3, vz_pred, colormap=:RdBu, colorrange=(-3, 1))
    
    # Konturen
    if num_crystals == 1
        contour!(ax3, phase_gt, levels=[0.5], color=:black, linewidth=2)
    else
        contour!(ax3, phase_gt, levels=[0.5, 1.5], color=:black, linewidth=2)
    end
    
    # # Markiere UNet Minima
    # for (i, minimum) in enumerate(pred_vz_minima)
    #     color = minima_colors[min(i, length(minima_colors))]
    #     scatter!(ax3, [minimum[2]], [minimum[1]], color=color, markersize=12,
    #             marker=:star5, strokewidth=2, strokecolor=:black)
    # end
    
    # Adaptive Statistiken
    avg_gt_alignment = length(gt_alignment_errors) > 0 ? mean(gt_alignment_errors) : NaN
    avg_pred_alignment = length(pred_alignment_errors) > 0 ? mean(pred_alignment_errors) : NaN
    avg_gt_to_unet = length(gt_to_unet_errors) > 0 ? mean(gt_to_unet_errors) : NaN
    
    stats_text = """
    $num_crystals-Kristall Koordinaten-Analyse ($(actual_size)):
    Durchschn. GT Alignment: $(round(avg_gt_alignment, digits=1)) px
    Durchschn. UNet Alignment: $(round(avg_pred_alignment, digits=1)) px
    Durchschn. GT vs UNet: $(round(avg_gt_to_unet, digits=1)) px
    
    Status: $(avg_gt_to_unet < 30 ? "OK" : "Problem") | Kristalle erkannt: $(length(gt_crystal_centers))/$num_crystals
    """
    
    Label(fig[2, 1:3], stats_text, fontsize=12)
    
    filename = "$(num_crystals)_crystal_debug.png"
    save(filename, fig)
    println("   Gespeichert: $filename")
    
    display(fig)
    
    # 9. Adaptive Zusammenfassung
    println("\n" * "="^50)
    println("ZUSAMMENFASSUNG ($num_crystals KRISTALL$(num_crystals > 1 ? "E" : "")):")
    
    if length(gt_crystal_centers) == num_crystals
        println("Kristall-Erkennung: Erfolgreich")
    else
        println("Kristall-Erkennung: Problematisch ($(length(gt_crystal_centers))/$num_crystals gefunden)")
    end
    
    if avg_gt_to_unet < 30 && avg_pred_alignment < 20
        println("UNet-Leistung: Gut")
    elseif avg_gt_alignment > 10
        println("Ground Truth Problem: LaMEM-Daten inkonsistent")
    elseif avg_gt_to_unet > 30
        println("Koordinaten-Problem: UNet verwendet andere Koordinaten")
    else
        println("Lern-Problem: UNet lernt falsche Beziehungen")
    end
    
    println("="^50)
    
    return (
        gt_crystal_centers = gt_crystal_centers,
        gt_vz_minima = gt_vz_minima,
        pred_vz_minima = pred_vz_minima,
        gt_alignment_errors = gt_alignment_errors,
        pred_alignment_errors = pred_alignment_errors,
        gt_to_unet_errors = gt_to_unet_errors,
        phase_gt = phase_gt,
        vz_gt = vz_gt,
        vz_pred = vz_pred,
        actual_size = actual_size,
        num_crystals_found = length(gt_crystal_centers)
    )
end

# ==================== CONVENIENCE FUNKTIONEN ====================

function debug_single_crystal(model_path; target_size=(256, 256))
    """
    Debug-Funktion für ein Kristall
    """
    return debug_vz_coordinates_adaptive(model_path, target_size=target_size, num_crystals=1)
end

function debug_two_crystals(model_path; target_size=(256, 256))
    """
    Debug-Funktion für zwei Kristalle
    """
    return debug_vz_coordinates_adaptive(model_path, target_size=target_size, num_crystals=2)
end

# ==================== AUSFÜHRUNG ====================

println("\n" * "="^60)
println("ERWEITERTE KRISTALL-DEBUGGING FUNKTIONEN")
println("="^60)
println("Verwendung:")
println("  # Ein Kristall:")
println("  result1 = debug_single_crystal(\"dein_model.bson\")")
println("  ")
println("  # Zwei Kristalle:")
println("  result2 = debug_two_crystals(\"dein_model.bson\")")
println("  ")
println("  # Adaptiv (automatisch):")
println("  result = debug_vz_coordinates_adaptive(\"dein_model.bson\", num_crystals=2)")
println("="^60)

# # Beispiel-Ausführung (Pfad anpassen!):
# println("\nBeispiel: Ein Kristall")
# result_1 = debug_single_crystal("final_model_two_crystals_demo.bson", target_size=(256, 256))

println("\nBeispiel: Zwei Kristalle")
result_2 = debug_two_crystals("final_model_two_crystals_demo.bson", target_size=(256, 256))