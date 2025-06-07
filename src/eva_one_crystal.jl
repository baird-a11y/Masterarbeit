using Flux, BSON, Statistics
using GLMakie  # Für Visualisierung
using LaMEM, GeophysicalModelGenerator  # Für Ground Truth Vergleich

println("=== VELOCITY UNET - KOMPLETTE EVALUIERUNG ===")

# ==================== MODELL-DEFINITIONEN ====================

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
        
        if x_size[1] > target_h || x_size[2] > target_w
            x_crop_h = min(x_size[1], target_h)
            x_crop_w = min(x_size[2], target_w)
            x_cropped = x[1:x_crop_h, 1:x_crop_w, :, :]
        else
            x_cropped = x
        end
        
        return cat(x_cropped, padded_skip, dims=dims)
    else
        h_offset = height_diff ÷ 2 + 1
        w_offset = width_diff ÷ 2 + 1
        
        h_end = h_offset + x_size[1] - 1
        w_end = w_offset + x_size[2] - 1
        
        h_end = min(h_end, skip_size[1])
        w_end = min(w_end, skip_size[2])
        h_offset = max(1, h_end - x_size[1] + 1)
        w_offset = max(1, w_end - x_size[2] + 1)
        
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

# ==================== LAMEM GROUND TRUTH ====================

function LaMEM_Single_crystal(; nx=128, nz=128, η=1e20, Δρ=200, cen_2D=[(0.0, 0.0)], R=[0.1])
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
    Exx = data.fields.strain_rate[1][:,1,:]
    Ezz = data.fields.strain_rate[9][:,1,:]
    rho = data.fields.density[:,1,:]
    log10eta = data.fields.visc_creep[:,1,:]

    V_stokes = 2/9*Δρ*9.81*(R[1]*1000)^2/(η)
    V_stokes_cm_year = V_stokes * 100 * (3600*24*365.25)

    return x_vec_1D, z_vec_1D, phase, Vx, Vz, Exx, Ezz, V_stokes_cm_year
end

# ==================== HILFSFUNKTIONEN ====================

function standardize_size(data, target_h=128, target_w=128)
    h, w = size(data)
    final = zeros(eltype(data), target_h, target_w)
    h_range = 1:min(h, target_h)
    w_range = 1:min(w, target_w)
    final[h_range, w_range] .= view(data, h_range, w_range)
    return final
end

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

# ==================== 1. BASIS-FUNKTIONALITÄTSTEST ====================

function test_basic_functionality(model, target_size=(128, 128))
    println("\n=== 1. BASIS-FUNKTIONALITÄTSTEST ===")
    
    h, w = target_size
    test_input = randn(Float32, h, w, 1, 1)
    
    try
        output = model(test_input)
        println("✓ Modell funktioniert!")
        println("  Input: $(size(test_input))")
        println("  Output: $(size(output))")
        
        if size(output, 3) == 2
            vx = output[:, :, 1, 1]
            vz = output[:, :, 2, 1]
            
            println("  v_x Bereich: [$(round(minimum(vx), digits=3)), $(round(maximum(vx), digits=3))]")
            println("  v_z Bereich: [$(round(minimum(vz), digits=3)), $(round(maximum(vz), digits=3))]")
            
            if std(vx) > 0.001 && std(vz) > 0.001
                println("✓ Sinnvolle Variationen in Geschwindigkeitsfeldern")
                return true
            else
                println("⚠ Konstante Ausgabewerte - mögliches Problem")
                return false
            end
        else
            println("✗ Falsche Anzahl Output-Kanäle: $(size(output, 3))")
            return false
        end
    catch e
        println("✗ Modell-Fehler: $e")
        return false
    end
end

# ==================== 2. KOORDINATENSYSTEM-CHECK ====================

function check_coordinate_system(model, target_size=(128, 128))
    println("\n=== 2. KOORDINATENSYSTEM-CHECK ===")
    
    h, w = target_size
    positions = [
        ("oben", h÷4, w÷2),
        ("mitte", h÷2, w÷2),
        ("unten", 3*h÷4, w÷2)
    ]
    
    results = []
    
    for (name, center_x, center_y) in positions
        # Erstelle Phasenfeld mit Kristall
        phase_field = zeros(Float32, h, w, 1, 1)
        radius = 15
        
        for i in 1:h, j in 1:w
            if sqrt((i-center_x)^2 + (j-center_y)^2) < radius
                phase_field[i, j, 1, 1] = 1.0f0
            end
        end
        
        # Vorhersage
        prediction = model(phase_field)
        vx = prediction[:, :, 1, 1]
        vz = prediction[:, :, 2, 1]
        
        # Analyse am Kristall
        kristall_mask = phase_field[:, :, 1, 1] .== 1.0f0
        vx_avg = mean(vx[kristall_mask])
        vz_avg = mean(vz[kristall_mask])
        
        direction = vz_avg < -0.01 ? "sinkt" : (vz_avg > 0.01 ? "steigt" : "neutral")
        
        println("  Kristall $name: v_x=$(round(vx_avg, digits=4)), v_z=$(round(vz_avg, digits=4)) → $direction")
        
        push!(results, (name=name, vx=vx_avg, vz=vz_avg, direction=direction))
    end
    
    # Analyse
    vz_signs = [r.vz > 0 ? "+" : "-" for r in results]
    
    if all(s == "+" for s in vz_signs)
        println("\n🔄 KOORDINATENSYSTEM: Alle v_z positiv - Kristalle 'steigen'")
        println("   → Wahrscheinlich Koordinatensystem-Problem oder Training-Artefakt")
        coordinate_issue = true
    elseif all(s == "-" for s in vz_signs)
        println("\n✅ KOORDINATENSYSTEM: Alle v_z negativ - Kristalle sinken korrekt!")
        coordinate_issue = false
    else
        println("\n🔍 KOORDINATENSYSTEM: Gemischte Vorzeichen - komplexe Strömungsmuster")
        coordinate_issue = false
    end
    
    return results, coordinate_issue
end

# ==================== 3. PHYSIKALISCHE PLAUSIBILITÄT ====================

function check_physics(model, target_size=(128, 128))
    println("\n=== 3. PHYSIKALISCHE PLAUSIBILITÄT ===")
    
    h, w = target_size
    
    # Test 1: Symmetrie bei zentriertem Kristall
    println("Test 1: Symmetrie")
    phase_field = zeros(Float32, h, w, 1, 1)
    center_x, center_y = h÷2, w÷2
    radius = 15
    
    for i in 1:h, j in 1:w
        if sqrt((i-center_x)^2 + (j-center_y)^2) < radius
            phase_field[i, j, 1, 1] = 1.0f0
        end
    end
    
    prediction = model(phase_field)
    vx = prediction[:, :, 1, 1]
    
    vx_left = mean(vx[:, 1:w÷2])
    vx_right = mean(vx[:, w÷2+1:end])
    symmetry_error = abs(vx_left + vx_right)
    
    println("  v_x links: $(round(vx_left, digits=4))")
    println("  v_x rechts: $(round(vx_right, digits=4))")
    println("  Symmetrie-Fehler: $(round(symmetry_error, digits=4))")
    
    symmetry_ok = symmetry_error < 0.1
    println("  $(symmetry_ok ? "✓" : "⚠") $(symmetry_ok ? "Gute" : "Schlechte") Symmetrie")
    
    # Test 2: Geschwindigkeits-Hotspot am Kristall
    println("\nTest 2: Geschwindigkeits-Hotspot")
    vz = prediction[:, :, 2, 1]
    velocity_magnitude = sqrt.(vx.^2 .+ vz.^2)
    
    kristall_mask = phase_field[:, :, 1, 1] .== 1.0f0
    max_v_total = maximum(velocity_magnitude)
    max_v_kristall = maximum(velocity_magnitude[kristall_mask])
    velocity_ratio = max_v_kristall / max_v_total
    
    println("  Max |v| gesamt: $(round(max_v_total, digits=4))")
    println("  Max |v| am Kristall: $(round(max_v_kristall, digits=4))")
    println("  Verhältnis: $(round(velocity_ratio, digits=3))")
    
    hotspot_ok = velocity_ratio > 0.7
    println("  $(hotspot_ok ? "✓" : "⚠") $(hotspot_ok ? "Geschwindigkeits-Hotspot am Kristall" : "Hotspot nicht am Kristall")")
    
    return (
        symmetry_error = symmetry_error,
        velocity_ratio = velocity_ratio,
        symmetry_ok = symmetry_ok,
        hotspot_ok = hotspot_ok
    )
end

# ==================== 4. GENAUIGKEITS-BEWERTUNG MIT LAMEM ====================

function evaluate_accuracy_with_lamem(model, target_size=(128, 128))
    println("\n=== 4. GENAUIGKEITS-BEWERTUNG MIT LAMEM ===")
    
    # Test-Konfigurationen
    test_configs = [
        (nx=128, nz=128, η=1e20, Δρ=200, cen_2D=[(0.0, 0.5)], R=[0.05]),
        (nx=128, nz=128, η=1e20, Δρ=300, cen_2D=[(0.3, 0.6)], R=[0.06]),
    ]
    
    results = []
    
    for (i, config) in enumerate(test_configs)
        println("  Test $i: Δρ=$(config.Δρ), Position=$(config.cen_2D[1]), R=$(config.R[1])")
        
        try
            # LaMEM Ground Truth
            x, z, phase_gt, vx_gt, vz_gt, exx, ezz, v_stokes = LaMEM_Single_crystal(;config...)
            
            # Daten vorbereiten
            h, w = target_size
            phase_std = standardize_size(Float32.(phase_gt), h, w)
            vx_gt_norm = standardize_size(Float32.(vx_gt ./ v_stokes), h, w)
            vz_gt_norm = standardize_size(Float32.(vz_gt ./ v_stokes), h, w)
            
            phase_input = reshape(phase_std, h, w, 1, 1)
            
            # UNet Vorhersage
            prediction = model(phase_input)
            vx_pred = prediction[:, :, 1, 1]
            vz_pred = prediction[:, :, 2, 1]
            
            # Metriken
            mse_vx = mean((vx_pred .- vx_gt_norm).^2)
            mse_vz = mean((vz_pred .- vz_gt_norm).^2)
            mse_total = (mse_vx + mse_vz) / 2
            
            # R²
            function r_squared(y_true, y_pred)
                ss_res = sum((y_true .- y_pred).^2)
                ss_tot = sum((y_true .- mean(y_true)).^2)
                return 1 - ss_res/ss_tot
            end
            
            r2_vx = r_squared(vx_gt_norm, vx_pred)
            r2_vz = r_squared(vz_gt_norm, vz_pred)
            
            println("    MSE: $(round(mse_total, digits=5))")
            println("    R² v_x: $(round(r2_vx, digits=3)), R² v_z: $(round(r2_vz, digits=3))")
            
            push!(results, (
                config = config,
                mse_total = mse_total,
                r2_vx = r2_vx,
                r2_vz = r2_vz,
                phase_gt = phase_std,
                vx_gt = vx_gt_norm,
                vz_gt = vz_gt_norm,
                vx_pred = vx_pred,
                vz_pred = vz_pred
            ))
            
        catch e
            println("    ✗ Fehler: $e")
        end
    end
    
    if !isempty(results)
        avg_mse = mean([r.mse_total for r in results])
        avg_r2 = mean([r.r2_vx for r in results] ∪ [r.r2_vz for r in results])
        
        println("\n  ZUSAMMENFASSUNG:")
        println("    Durchschnittliche MSE: $(round(avg_mse, digits=5))")
        println("    Durchschnittliches R²: $(round(avg_r2, digits=3))")
        
        # Qualitätsbewertung
        if avg_mse < 0.01
            println("    ✓ Sehr gute Genauigkeit!")
        elseif avg_mse < 0.05
            println("    ○ Gute Genauigkeit")
        else
            println("    △ Genauigkeit verbesserbar")
        end
        
        return results, avg_mse, avg_r2
    else
        return [], NaN, NaN
    end
end

# ==================== 5. VISUALISIERUNG ====================

function create_comprehensive_visualization(model, lamem_result=nothing, target_size=(128, 128))
    println("\n=== 5. VISUALISIERUNG ===")
    
    h, w = target_size
    
    # Erstelle Testfall
    phase_field = zeros(Float32, h, w, 1, 1)
    center_x, center_y = h÷4, w÷2  # Oben-mitte
    radius = 15
    
    for i in 1:h, j in 1:w
        if sqrt((i-center_x)^2 + (j-center_y)^2) < radius
            phase_field[i, j, 1, 1] = 1.0f0
        end
    end
    
    # Vorhersage
    prediction = model(phase_field)
    vx = prediction[:, :, 1, 1]
    vz = prediction[:, :, 2, 1]
    
    # Plot erstellen
    fig = Figure(resolution=(1400, 1000))
    
    x_coords = 1:w
    z_coords = 1:h
    
    # Row 1: Phasenfeld und UNet Vorhersagen
    ax1 = Axis(fig[1, 1], title="Phasenfeld (Input)", xlabel="x", ylabel="z")
    hm1 = heatmap!(ax1, x_coords, z_coords, phase_field[:, :, 1, 1], colormap=:grays)
    Colorbar(fig[1, 2], hm1, label="Phase")
    
    ax2 = Axis(fig[1, 3], title="UNet: v_x", xlabel="x", ylabel="z")
    hm2 = heatmap!(ax2, x_coords, z_coords, vx, colormap=:RdBu)
    Colorbar(fig[1, 4], hm2, label="v_x")
    contour!(ax2, x_coords, z_coords, phase_field[:, :, 1, 1], levels=[0.5], color=:black, linewidth=2)
    
    ax3 = Axis(fig[2, 1], title="UNet: v_z", xlabel="x", ylabel="z")
    hm3 = heatmap!(ax3, x_coords, z_coords, vz, colormap=:RdBu)
    Colorbar(fig[2, 2], hm3, label="v_z")
    contour!(ax3, x_coords, z_coords, phase_field[:, :, 1, 1], levels=[0.5], color=:black, linewidth=2)
    
    # Geschwindigkeits-Magnitude
    velocity_magnitude = sqrt.(vx.^2 .+ vz.^2)
    ax4 = Axis(fig[2, 3], title="UNet: |v|", xlabel="x", ylabel="z")
    hm4 = heatmap!(ax4, x_coords, z_coords, velocity_magnitude, colormap=:viridis)
    Colorbar(fig[2, 4], hm4, label="|v|")
    contour!(ax4, x_coords, z_coords, phase_field[:, :, 1, 1], levels=[0.5], color=:white, linewidth=2)
    
    # LaMEM Vergleich falls verfügbar
    if lamem_result !== nothing
        ax5 = Axis(fig[3, 1], title="LaMEM: v_z (Ground Truth)", xlabel="x", ylabel="z")
        hm5 = heatmap!(ax5, x_coords, z_coords, lamem_result.vz_gt, colormap=:RdBu)
        Colorbar(fig[3, 2], hm5, label="v_z GT")
        
        diff_vz = abs.(vz .- lamem_result.vz_gt)
        ax6 = Axis(fig[3, 3], title="Differenz |UNet - LaMEM|", xlabel="x", ylabel="z")
        hm6 = heatmap!(ax6, x_coords, z_coords, diff_vz, colormap=:hot)
        Colorbar(fig[3, 4], hm6, label="Fehler")
    end
    
    # Statistiken
    kristall_mask = phase_field[:, :, 1, 1] .== 1.0f0
    vx_kristall = mean(vx[kristall_mask])
    vz_kristall = mean(vz[kristall_mask])
    
    stats_text = """
    UNet Statistiken:
    v_x (Kristall): $(round(vx_kristall, digits=3))
    v_z (Kristall): $(round(vz_kristall, digits=3))
    |v| max: $(round(maximum(velocity_magnitude), digits=3))
    
    Interpretation:
    $(vz_kristall > 0 ? "Kristall steigt ↑" : "Kristall sinkt ↓")
    """
    
    Label(fig[4, 1:4], stats_text, tellwidth=false, fontsize=12)
    
    save_path = "velocity_evaluation_complete.png"
    save(save_path, fig)
    println("Visualisierung gespeichert: $save_path")
    
    display(fig)
    return fig
end

# ==================== HAUPTFUNKTION - KOMPLETTE EVALUIERUNG ====================

function complete_velocity_evaluation(model_path; target_size=(128, 128), run_lamem=true)
    println("=== KOMPLETTE VELOCITY UNET EVALUIERUNG ===")
    println("Modelldatei: $model_path")
    println("Zielgröße: $target_size")
    
    # Modell laden
    model = load_velocity_model(model_path)
    
    # 1. Basis-Test
    basic_ok = test_basic_functionality(model, target_size)
    if !basic_ok
        println("\n❌ ABBRUCH: Basis-Funktionalität fehlgeschlagen!")
        return nothing
    end
    
    # 2. Koordinatensystem
    coord_results, coord_issue = check_coordinate_system(model, target_size)
    
    # 3. Physikalische Tests
    physics_results = check_physics(model, target_size)
    
    # 4. LaMEM Vergleich (optional, da zeitaufwändig)
    if run_lamem
        lamem_results, avg_mse, avg_r2 = evaluate_accuracy_with_lamem(model, target_size)
        best_lamem = !isempty(lamem_results) ? lamem_results[1] : nothing
    else
        println("\n=== 4. LAMEM-VERGLEICH ÜBERSPRUNGEN (run_lamem=false) ===")
        lamem_results, avg_mse, avg_r2, best_lamem = [], NaN, NaN, nothing
    end
    
    # 5. Visualisierung
    fig = create_comprehensive_visualization(model, best_lamem, target_size)
    
    # 6. Gesamtbewertung
    println("\n" * "="^60)
    println("GESAMTBEWERTUNG")
    println("="^60)
    
    println("✓ Basis-Funktionalität: $(basic_ok ? "OK" : "FEHLER")")
    println("$(coord_issue ? "⚠" : "✓") Koordinatensystem: $(coord_issue ? "Möglicherweise invertiert" : "OK")")
    println("$(physics_results.symmetry_ok ? "✓" : "⚠") Symmetrie: $(physics_results.symmetry_ok ? "Gut" : "Verbesserbar")")
    println("$(physics_results.hotspot_ok ? "✓" : "⚠") Geschwindigkeits-Hotspot: $(physics_results.hotspot_ok ? "Am Kristall" : "Nicht am Kristall")")
    
    if run_lamem && !isnan(avg_mse)
        accuracy_rating = avg_mse < 0.01 ? "Sehr gut" : (avg_mse < 0.05 ? "Gut" : "Verbesserbar")
        println("$(avg_mse < 0.05 ? "✓" : "⚠") Genauigkeit vs LaMEM: $accuracy_rating (MSE: $(round(avg_mse, digits=5)))")
    end
    
    # Finale Empfehlung
    println("\n" * "="^60)
    if basic_ok && physics_results.symmetry_ok && physics_results.hotspot_ok && (!run_lamem || avg_mse < 0.1)
        println("🎉 FAZIT: Modell ist BEREIT für den Einsatz!")
        if coord_issue
            println("💡 TIPP: Koordinatensystem prüfen (evtl. v_z * -1)")
        end
    else
        println("🔧 FAZIT: Modell benötigt Verbesserungen")
        println("   - Training mit mehr/besseren Daten")
        println("   - Architektur-Anpassungen")
        println("   - Hyperparameter-Optimierung")
    end
    println("="^60)
    
    return (
        model = model,
        basic_ok = basic_ok,
        coord_results = coord_results,
        coord_issue = coord_issue,
        physics_results = physics_results,
        lamem_results = lamem_results,
        avg_mse = avg_mse,
        avg_r2 = avg_r2,
        visualization = fig
    )
end

# ==================== BEISPIELAUFRUFE ====================

println("\n" * "="^60)
println("VERWENDUNG:")
println("="^60)
println("# Vollständige Evaluierung mit LaMEM:")
println("complete_velocity_evaluation(\"final_velocity_model.bson\")")
println()
println("# Schnelle Evaluierung ohne LaMEM:")
println("complete_velocity_evaluation(\"final_velocity_model.bson\", run_lamem=false)")
println()
println("# Mit anderer Bildgröße:")
println("complete_velocity_evaluation(\"final_velocity_model.bson\", target_size=(256,256))")
println("="^60)

# Automatische Ausführung (auskommentieren falls gewünscht):
results = complete_velocity_evaluation("final_velocity_model.bson", run_lamem=true, target_size=(128, 128))