using Flux, BSON, Statistics
using GLMakie
using LaMEM, GeophysicalModelGenerator

println("=== VEREINFACHTES V_Z KOORDINATEN-DEBUGGING ===")

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

# ==================== LAMEM FUNCTION ====================

function LaMEM_Single_crystal(; nx=256, nz=256, η=1e20, Δρ=200, cen_2D=[(0.0, 0.0)], R=[0.1])
    η_crystal = 1e4*η
    ρ_magma = 2700

    model = Model(Grid(nel=(nx,nz), x=[-1,1], z=[-1,1]), Time(nstep_max=1), Output(out_strain_rate=1))
    matrix = Phase(ID=0, Name="matrix", eta=η, rho=ρ_magma)
    crystal = Phase(ID=1, Name="crystal", eta=η_crystal, rho=ρ_magma+Δρ)
    add_phase!(model, crystal, matrix)

    for i = 1:length(cen_2D)
        add_sphere!(model, cen=(cen_2D[i][1], 0.0, cen_2D[i][2]), radius=R[i], phase=ConstantPhase(1))
    end
   
    run_lamem(model, 1)
    data, _ = read_LaMEM_timestep(model, 1)

    x_vec_1D = data.x.val[:,1,1]
    z_vec_1D = data.z.val[1,1,:]
    phase = data.fields.phase[:,1,:]
    Vx = data.fields.velocity[1][:,1,:]
    Vz = data.fields.velocity[3][:,1,:]
    
    V_stokes = 2/9*Δρ*9.81*(R[1]*1000)^2/(η)
    V_stokes_cm_year = V_stokes * 100 * (3600*24*365.25)

    return x_vec_1D, z_vec_1D, phase, Vx, Vz, V_stokes_cm_year
end

# ==================== HILFSFUNKTIONEN ====================

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

function find_crystal_center(phase_field)
    """Findet das Zentrum des Kristalls im Phasenfeld"""
    crystal_indices = findall(x -> x > 0.5, phase_field)
    if isempty(crystal_indices)
        return nothing
    end
    
    x_coords = [idx[1] for idx in crystal_indices]
    z_coords = [idx[2] for idx in crystal_indices]
    
    return (mean(x_coords), mean(z_coords))
end

function find_velocity_minimum(vz_field)
    """Findet die Position der stärksten Abwärtsbewegung"""
    min_idx = argmin(vz_field)
    return (min_idx[1], min_idx[2], vz_field[min_idx])
end

# ==================== HAUPTFUNKTION: V_Z KOORDINATEN-DEBUG ====================

function debug_vz_coordinates(model_path; target_size=(256, 256))
    println("\n=== V_Z KOORDINATEN-DEBUGGING ===")
    
    # 1. Modell laden
    model = load_velocity_model(model_path)
    
    # 2. LaMEM Ground Truth generieren
    println("\n1. Generiere LaMEM Ground Truth...")
    x, z, phase_gt, vx_gt, vz_gt, v_stokes = LaMEM_Single_crystal(
        nx=target_size[1], nz=target_size[2], 
        η=1e20, Δρ=200, 
        cen_2D=[(0.0, 0.5)], 
        R=[0.05]
    )
    
    println("   Dimensionen - Phase: $(size(phase_gt)), Vz: $(size(vz_gt))")
    
    # 3. Kristall-Position in Ground Truth finden
    println("\n2. Analysiere Ground Truth...")
    gt_crystal_center = find_crystal_center(phase_gt)
    gt_vz_min_pos = find_velocity_minimum(vz_gt)
    
    println("   Kristall-Zentrum (GT): $gt_crystal_center")
    println("   Min v_z Position (GT): $(gt_vz_min_pos[1:2]), Wert: $(round(gt_vz_min_pos[3], digits=6))")
    
    gt_alignment_error = sqrt((gt_crystal_center[1] - gt_vz_min_pos[1])^2 + (gt_crystal_center[2] - gt_vz_min_pos[2])^2)
    println("   GT Alignment-Fehler: $(round(gt_alignment_error, digits=1)) Pixel")
    
    # 4. UNet Input vorbereiten
    println("\n3. Bereite UNet Input vor...")
    println("   Tatsächliche Phase-Dimensionen: $(size(phase_gt))")
    println("   Tatsächliche Vz-Dimensionen: $(size(vz_gt))")
    
    # Größe anpassen falls LaMEM andere Dimensionen liefert
    actual_size = size(phase_gt)
    if actual_size != target_size
        println("   ⚠ LaMEM liefert $(actual_size) statt $(target_size) - passe an")
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
        println("    UNet Fehler: $e")
        println("   Input Dimensionen waren: $(size(phase_input))")
        return nothing
    end
    
    # 6. UNet Koordinaten analysieren
    println("\n5. Analysiere UNet Vorhersage...")
    pred_crystal_center = find_crystal_center(phase_gt)  # Gleiche Phase wie GT
    pred_vz_min_pos = find_velocity_minimum(vz_pred)
    
    println("   Kristall-Zentrum (Input): $pred_crystal_center")
    println("   Min v_z Position (UNet): $(pred_vz_min_pos[1:2]), Wert: $(round(pred_vz_min_pos[3], digits=6))")
    
    pred_alignment_error = sqrt((pred_crystal_center[1] - pred_vz_min_pos[1])^2 + (pred_crystal_center[2] - pred_vz_min_pos[2])^2)
    println("   UNet Alignment-Fehler: $(round(pred_alignment_error, digits=1)) Pixel")
    
    # 7. Koordinaten-Vergleich
    println("\n6. Koordinaten-Vergleich...")
    gt_to_unet_error = sqrt((gt_vz_min_pos[1] - pred_vz_min_pos[1])^2 + (gt_vz_min_pos[2] - pred_vz_min_pos[2])^2)
    println("   GT vs UNet v_z Position: $(round(gt_to_unet_error, digits=1)) Pixel Unterschied")
    
    # 8. Koordinaten-Diagnose
    println("\n7. DIAGNOSE:")
    if gt_alignment_error < 10
        println("    Ground Truth: Kristall und v_z gut ausgerichtet")
    else
        println("    Ground Truth: Kristall und v_z schlecht ausgerichtet!")
    end
    
    if pred_alignment_error < 20
        println("    UNet: Lernt korrekte Kristall-Geschwindigkeit Beziehung")
    else
        println("    UNet: Lernt FALSCHE Kristall-Geschwindigkeit Beziehung!")
    end
    
    if gt_to_unet_error < 30
        println("    Koordinaten: GT und UNet verwenden ähnliche Koordinaten")
    else
        println("    Koordinaten: GT und UNet verwenden VERSCHIEDENE Koordinaten!")
    end
    
    # 9. Einfache Visualisierung - NUR V_Z!
    println("\n8. Erstelle v_z Visualisierung...")
    
    fig = Figure(resolution=(1200, 400))
    
    # Plot 1: Phasenfeld
    ax1 = Axis(fig[1, 1], title="Phasenfeld", xlabel="x", ylabel="z", aspect=DataAspect())
    heatmap!(ax1, phase_gt, colormap=:grays)
    #scatter!(ax1, [gt_crystal_center[2]], [gt_crystal_center[1]], color=:red, markersize=15, label="Kristall-Zentrum")
    
    # Plot 2: LaMEM v_z
    ax2 = Axis(fig[1, 2], title="LaMEM: v_z", xlabel="x", ylabel="z", aspect=DataAspect())
    heatmap!(ax2, vz_gt ./ v_stokes, colormap=:RdBu, colorrange=(-3, 1))
    #scatter!(ax2, [gt_vz_min_pos[2]], [gt_vz_min_pos[1]], color=:yellow, markersize=15, label="Min v_z")
    contour!(ax2, phase_gt, levels=[0.5], color=:black, linewidth=2)
    
    # Plot 3: UNet v_z
    ax3 = Axis(fig[1, 3], title="UNet: v_z", xlabel="x", ylabel="z", aspect=DataAspect())
    heatmap!(ax3, vz_pred, colormap=:RdBu, colorrange=(-3, 1))
    #scatter!(ax3, [pred_vz_min_pos[2]], [pred_vz_min_pos[1]], color=:yellow, markersize=15, label="Min v_z")
    contour!(ax3, phase_gt, levels=[0.5], color=:black, linewidth=2)
    
    # Statistiken hinzufügen
    stats_text = """
    Koordinaten-Analyse ($(actual_size)):
    GT Alignment: $(round(gt_alignment_error, digits=1)) px
    UNet Alignment: $(round(pred_alignment_error, digits=1)) px
    GT vs UNet: $(round(gt_to_unet_error, digits=1)) px
    
    Status: $(gt_to_unet_error < 30 ? " OK" : " Problem")
    """
    
    Label(fig[2, 1:3], stats_text, fontsize=12)
    
    save("vz_coordinate_debug.png", fig)
    println("   Gespeichert: vz_coordinate_debug.png")
    
    display(fig)
    
    # 10. Zusammenfassung
    println("\n" * "="^50)
    println("ZUSAMMENFASSUNG:")
    if gt_to_unet_error < 30 && pred_alignment_error < 20
        println("🎉 KEIN KOORDINATEN-PROBLEM! UNet lernt korrekt.")
    elseif gt_alignment_error > 10
        println(" PROBLEM IN GROUND TRUTH! LaMEM-Daten inkonsistent.")
        println("   → Prüfe LaMEM Datengenerierung")
    elseif gt_to_unet_error > 30
        println(" KOORDINATEN-PROBLEM! UNet verwendet andere Koordinaten als GT.")
        println("   → Prüfe Datenvorverarbeitung im Training")
    elseif pred_alignment_error > 20
        println(" LERN-PROBLEM! UNet lernt falsche Kristall-Geschwindigkeit Beziehung.")
        println("   → Mehr/bessere Trainingsdaten nötig")
    end
    println("="^50)
    
    return (
        gt_crystal_center = gt_crystal_center,
        gt_vz_min_pos = gt_vz_min_pos,
        pred_vz_min_pos = pred_vz_min_pos,
        gt_alignment_error = gt_alignment_error,
        pred_alignment_error = pred_alignment_error,
        gt_to_unet_error = gt_to_unet_error,
        phase_gt = phase_gt,
        vz_gt = vz_gt,
        vz_pred = vz_pred,
        actual_size = actual_size
    )
end

# ==================== AUSFÜHRUNG ====================

println("Verwende: debug_vz_coordinates(\"dein_model.bson\")")
println("Beispiel:")

# Ausführung (Pfad anpassen!):
result = debug_vz_coordinates("final_model_30_200_2.bson", target_size=(256, 256))