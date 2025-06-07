using Flux, CUDA, BSON
using GLMakie, Colors
using Statistics
using LaMEM, GeophysicalModelGenerator

# Lade das trainierte Modell
function load_velocity_model(model_path)
    model_dict = BSON.load(model_path)
    
    # Flexibles Laden je nach Schlüssel
    model_cpu = nothing
    for key in [:final_model_cpu, :model_cpu, :model, :trained_model]
        if haskey(model_dict, key)
            model_cpu = model_dict[key]
            break
        end
    end
    
    if model_cpu === nothing
        model_cpu = first(values(model_dict))
    end
    
    return gpu(model_cpu)
end

# Visualisierung der Ergebnisse
function visualize_results(phase, vx_true, vz_true, vx_pred, vz_pred, v_stokes; save_path=nothing)
    fig = Figure(resolution = (1200, 800))
    
    # Grid für Plots erstellen
    x = range(-1, 1, length=size(phase, 1))
    z = range(-1, 1, length=size(phase, 2))
    
    # 1. Phasenfeld (Input)
    ax1 = Axis(fig[1, 1], title="Phase (Input)", xlabel="x (km)", ylabel="z (km)")
    hm1 = heatmap!(ax1, x, z, phase', colormap=:grays)
    contour!(ax1, x, z, phase', levels=[0.5], color=:red, linewidth=2)
    Colorbar(fig[1, 2], hm1)
    
    # 2. v_z Ground Truth  
    ax2 = Axis(fig[1, 3], title="v_z Ground Truth", xlabel="x (km)", ylabel="z (km)")
    hm2 = heatmap!(ax2, x, z, vz_true', colormap=:viridis)
    contour!(ax2, x, z, phase', levels=[0.5], color=:white, linewidth=1)
    Colorbar(fig[1, 4], hm2)
    
    # 3. v_z Vorhersage
    ax3 = Axis(fig[2, 1], title="v_z Vorhersage", xlabel="x (km)", ylabel="z (km)")
    hm3 = heatmap!(ax3, x, z, vz_pred', colormap=:viridis)
    contour!(ax3, x, z, phase', levels=[0.5], color=:white, linewidth=1)
    Colorbar(fig[2, 2], hm3)
    
    # 4. v_z Differenz
    vz_diff = abs.(vz_true .- vz_pred)
    ax4 = Axis(fig[2, 3], title="v_z Differenz (|True - Pred|)", xlabel="x (km)", ylabel="z (km)")
    hm4 = heatmap!(ax4, x, z, vz_diff', colormap=:hot)
    Colorbar(fig[2, 4], hm4)
    
    # Statistiken hinzufügen
    mse_vx = mean((vx_true .- vx_pred).^2)
    mse_vz = mean((vz_true .- vz_pred).^2)
    max_vz_true = maximum(abs.(vz_true))
    max_vz_pred = maximum(abs.(vz_pred))
    
    # Text mit Statistiken
    stats_text = """
    Statistiken:
    MSE v_x: $(round(mse_vx, digits=4))
    MSE v_z: $(round(mse_vz, digits=4))
    Max |v_z| true: $(round(max_vz_true, digits=4))
    Max |v_z| pred: $(round(max_vz_pred, digits=4))
    Stokes v: $(round(v_stokes, digits=2)) cm/year
    """
    
    Label(fig[3, 1:4], stats_text, tellwidth=false, fontsize=12)
    
    if save_path !== nothing
        save(save_path, fig)
        println("Visualisierung gespeichert: $save_path")
    end
    
    return fig
end

# Teste das Modell mit einem neuen Sample
function test_model_comprehensive(model_path; save_plots=true)
    println("Lade trainiertes Modell...")
    model = load_velocity_model(model_path)
    
    println("Generiere Testfall...")
    # Neuen Testfall generieren
    η = 1e20
    Δρ = 300
    cen_2D = [(0.3, 0.6)]
    R = [0.05]
    
    x, z, phase, Vx, Vz, Exx, Ezz, V_stokes = LaMEM_Single_crystal(
        nx=128, nz=128, η=η, Δρ=Δρ, cen_2D=cen_2D, R=R
    )
    
    println("Führe Vorhersage durch...")
    
    # Eingabe vorbereiten (wie im Training)
    STANDARD_HEIGHT, STANDARD_WIDTH = 256, 256
    
    function standardize_size(data)
        h, w = size(data)
        final = zeros(Float32, STANDARD_HEIGHT, STANDARD_WIDTH)
        h_range = 1:min(h, STANDARD_HEIGHT)
        w_range = 1:min(w, STANDARD_WIDTH)
        final[h_range, w_range] .= Float32.(view(data, h_range, w_range))
        return final
    end
    
    phase_input = reshape(standardize_size(phase), STANDARD_HEIGHT, STANDARD_WIDTH, 1, 1)
    
    # Vorhersage
    phase_gpu = gpu(phase_input)
    velocity_pred = cpu(model(phase_gpu))
    
    vx_pred = velocity_pred[:, :, 1, 1]
    vz_pred = velocity_pred[:, :, 2, 1]
    
    # Ground Truth normalisieren und standardisieren
    vx_true_norm = Vx ./ V_stokes
    vz_true_norm = Vz ./ V_stokes
    
    vx_true_std = standardize_size(vx_true_norm)
    vz_true_std = standardize_size(vz_true_norm)
    phase_std = standardize_size(phase)
    
    # Evaluierung
    mse_vx = mean((vx_pred .- vx_true_std).^2)
    mse_vz = mean((vz_pred .- vz_true_std).^2)
    mse_total = (mse_vx + mse_vz) / 2
    
    println("Ergebnisse:")
    println("  MSE v_x: $(round(mse_vx, digits=6))")
    println("  MSE v_z: $(round(mse_vz, digits=6))")
    println("  MSE gesamt: $(round(mse_total, digits=6))")
    println("  Stokes Geschwindigkeit: $(round(V_stokes, digits=2)) cm/year")
    
    # Visualisierung
    if save_plots
        timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
        save_path = "velocity_test_$(timestamp).png"
        fig = visualize_results(phase_std, vx_true_std, vz_true_std, 
                               vx_pred, vz_pred, V_stokes, save_path=save_path)
        display(fig)
    else
        fig = visualize_results(phase_std, vx_true_std, vz_true_std, 
                               vx_pred, vz_pred, V_stokes)
        display(fig)
    end
    
    return (
        mse_total=mse_total,
        vx_pred=vx_pred, vz_pred=vz_pred,
        vx_true=vx_true_std, vz_true=vz_true_std,
        phase=phase_std,
        v_stokes=V_stokes
    )
end

# Batch-Test mit mehreren Konfigurationen
function batch_test_model(model_path, n_tests=10)
    println("Führe Batch-Test mit $n_tests verschiedenen Konfigurationen durch...")
    
    model = load_velocity_model(model_path)
    results = []
    
    for i in 1:n_tests
        println("Test $i/$n_tests...")
        
        # Zufällige Parameter
        η = 10^(rand() * 2 + 19)  # 1e19 bis 1e21
        Δρ = rand(100:50:500)
        x_pos = rand(-0.6:0.1:0.6)
        z_pos = rand(0.2:0.1:0.8)
        radius = rand(0.03:0.01:0.08)
        
        try
            # LaMEM-Simulation
            x, z, phase, Vx, Vz, Exx, Ezz, V_stokes = LaMEM_Single_crystal(
                nx=128, nz=128, η=η, Δρ=Δρ, 
                cen_2D=[(x_pos, z_pos)], R=[radius]
            )
            
            # Vorhersage (wie oben)
            STANDARD_HEIGHT, STANDARD_WIDTH = 256, 256
            
            function standardize_size(data)
                h, w = size(data)
                final = zeros(Float32, STANDARD_HEIGHT, STANDARD_WIDTH)
                h_range = 1:min(h, STANDARD_HEIGHT)
                w_range = 1:min(w, STANDARD_WIDTH)
                final[h_range, w_range] .= Float32.(view(data, h_range, w_range))
                return final
            end
            
            phase_input = reshape(standardize_size(phase), STANDARD_HEIGHT, STANDARD_WIDTH, 1, 1)
            velocity_pred = cpu(model(gpu(phase_input)))
            
            vx_pred = velocity_pred[:, :, 1, 1]
            vz_pred = velocity_pred[:, :, 2, 1]
            
            # Ground Truth
            vx_true = standardize_size(Vx ./ V_stokes)
            vz_true = standardize_size(Vz ./ V_stokes)
            
            # MSE berechnen
            mse_vx = mean((vx_pred .- vx_true).^2)
            mse_vz = mean((vz_pred .- vz_true).^2)
            mse_total = (mse_vx + mse_vz) / 2
            
            push!(results, (
                test_id=i,
                mse_total=mse_total,
                mse_vx=mse_vx,
                mse_vz=mse_vz,
                params=(η=η, Δρ=Δρ, x_pos=x_pos, z_pos=z_pos, radius=radius),
                v_stokes=V_stokes
            ))
            
            println("  MSE gesamt: $(round(mse_total, digits=6))")
            
        catch e
            println("  Fehler bei Test $i: $e")
        end
    end
    
    # Statistiken ausgeben
    if !isempty(results)
        mse_values = [r.mse_total for r in results]
        
        println("\n===== BATCH-TEST ERGEBNISSE =====")
        println("Erfolgreich getestete Konfigurationen: $(length(results))/$n_tests")
        println("MSE Statistiken:")
        println("  Mittelwert: $(round(mean(mse_values), digits=6))")
        println("  Median: $(round(median(mse_values), digits=6))")
        println("  Standardabweichung: $(round(std(mse_values), digits=6))")
        println("  Minimum: $(round(minimum(mse_values), digits=6))")
        println("  Maximum: $(round(maximum(mse_values), digits=6))")
        
        # Beste und schlechteste Ergebnisse
        sorted_results = sort(results, by=r->r.mse_total)
        println("\nBestes Ergebnis (Test $(sorted_results[1].test_id)):")
        println("  MSE: $(round(sorted_results[1].mse_total, digits=6))")
        println("  Parameter: $(sorted_results[1].params)")
        
        println("\nSchlechtestes Ergebnis (Test $(sorted_results[end].test_id)):")
        println("  MSE: $(round(sorted_results[end].mse_total, digits=6))")
        println("  Parameter: $(sorted_results[end].params)")
    end
    
    return results
end

# Vergleiche verschiedene Trainingsmodelle
function compare_velocity_models(model_paths, test_config=nothing)
    println("Vergleiche $(length(model_paths)) Modelle...")
    
    # Standardkonfiguration für Test, falls nicht angegeben
    if test_config === nothing
        test_config = (
            η=1e20, Δρ=250, cen_2D=[(0.0, 0.5)], R=[0.06], nx=128, nz=128
        )
    end
    
    # Testdaten generieren
    println("Generiere Testdaten...")
    x, z, phase, Vx, Vz, Exx, Ezz, V_stokes = LaMEM_Single_crystal(;test_config...)
    
    # Daten standardisieren
    STANDARD_HEIGHT, STANDARD_WIDTH = 256, 256
    
    function standardize_size(data)
        h, w = size(data)
        final = zeros(Float32, STANDARD_HEIGHT, STANDARD_WIDTH)
        h_range = 1:min(h, STANDARD_HEIGHT)
        w_range = 1:min(w, STANDARD_WIDTH)
        final[h_range, w_range] .= Float32.(view(data, h_range, w_range))
        return final
    end
    
    phase_input = reshape(standardize_size(phase), STANDARD_HEIGHT, STANDARD_WIDTH, 1, 1)
    vx_true = standardize_size(Vx ./ V_stokes)
    vz_true = standardize_size(Vz ./ V_stokes)
    
    results = Dict()
    
    for model_path in model_paths
        model_name = basename(model_path)
        println("\nTeste Modell: $model_name")
        
        try
            model = load_velocity_model(model_path)
            velocity_pred = cpu(model(gpu(phase_input)))
            
            vx_pred = velocity_pred[:, :, 1, 1]
            vz_pred = velocity_pred[:, :, 2, 1]
            
            mse_vx = mean((vx_pred .- vx_true).^2)
            mse_vz = mean((vz_pred .- vz_true).^2)
            mse_total = (mse_vx + mse_vz) / 2
            
            results[model_name] = (
                mse_total=mse_total,
                mse_vx=mse_vx,
                mse_vz=mse_vz,
                vx_pred=vx_pred,
                vz_pred=vz_pred
            )
            
            println("  MSE gesamt: $(round(mse_total, digits=6))")
            
        catch e
            println("  FEHLER beim Laden/Testen von $model_name: $e")
        end
    end
    
    # Ergebnisse sortieren und ausgeben
    println("\n===== MODELLVERGLEICH =====")
    sorted_results = sort(collect(results), by=x->x[2].mse_total)
    
    println("Ranking (niedrigste MSE = beste Performance):")
    for (i, (model_name, result)) in enumerate(sorted_results)
        println("$i. $model_name - MSE: $(round(result.mse_total, digits=6))")
    end
    
    return results, (phase_std=standardize_size(phase), vx_true=vx_true, vz_true=vz_true, v_stokes=V_stokes)
end

# Analysiere Modell-Performance vs. Kristallparameter
function analyze_parameter_sensitivity(model_path)
    println("Analysiere Parameter-Sensitivität...")
    
    model = load_velocity_model(model_path)
    
    # Parameter-Bereiche definieren
    viscosity_ratios = [1e3, 1e4, 1e5]  # η_crystal/η_matrix
    density_contrasts = [100, 200, 300, 400, 500]
    radii = [0.03, 0.05, 0.07]
    positions = [(-0.5, 0.7), (0.0, 0.5), (0.5, 0.3)]
    
    results = []
    test_id = 1
    
    for η_ratio in viscosity_ratios
        for Δρ in density_contrasts
            for R in radii
                for (x_pos, z_pos) in positions
                    println("Test $test_id: η_ratio=$η_ratio, Δρ=$Δρ, R=$R, pos=($x_pos,$z_pos)")
                    
                    try
                        # Basis-Viskosität
                        η_matrix = 1e20
                        η_crystal = η_ratio * η_matrix
                        
                        # Simulation anpassen (vereinfacht - du müsstest LaMEM_Single_crystal anpassen)
                        x, z, phase, Vx, Vz, Exx, Ezz, V_stokes = LaMEM_Single_crystal(
                            nx=128, nz=128, η=η_matrix, Δρ=Δρ, 
                            cen_2D=[(x_pos, z_pos)], R=[R]
                        )
                        
                        # Vorhersage
                        STANDARD_HEIGHT, STANDARD_WIDTH = 256, 256
                        function standardize_size(data)
                            h, w = size(data)
                            final = zeros(Float32, STANDARD_HEIGHT, STANDARD_WIDTH)
                            h_range = 1:min(h, STANDARD_HEIGHT)
                            w_range = 1:min(w, STANDARD_WIDTH)
                            final[h_range, w_range] .= Float32.(view(data, h_range, w_range))
                            return final
                        end
                        
                        phase_input = reshape(standardize_size(phase), STANDARD_HEIGHT, STANDARD_WIDTH, 1, 1)
                        velocity_pred = cpu(model(gpu(phase_input)))
                        
                        vx_pred = velocity_pred[:, :, 1, 1]
                        vz_pred = velocity_pred[:, :, 2, 1]
                        
                        vx_true = standardize_size(Vx ./ V_stokes)
                        vz_true = standardize_size(Vz ./ V_stokes)
                        
                        mse_total = (mean((vx_pred .- vx_true).^2) + mean((vz_pred .- vz_true).^2)) / 2
                        
                        push!(results, (
                            test_id=test_id,
                            η_ratio=η_ratio,
                            Δρ=Δρ,
                            R=R,
                            x_pos=x_pos,
                            z_pos=z_pos,
                            mse=mse_total,
                            v_stokes=V_stokes
                        ))
                        
                    catch e
                        println("  Fehler: $e")
                    end
                    
                    test_id += 1
                end
            end
        end
    end
    
    # Analyse der Ergebnisse
    if !isempty(results)
        println("\n===== PARAMETER-SENSITIVITÄTS-ANALYSE =====")
        
        # Gruppierung nach Parametern
        println("Durchschnittliche MSE nach Viskositätsverhältnis:")
        for η_ratio in viscosity_ratios
            subset = filter(r -> r.η_ratio == η_ratio, results)
            if !isempty(subset)
                avg_mse = mean([r.mse for r in subset])
                println("  η_ratio $η_ratio: $(round(avg_mse, digits=6))")
            end
        end
        
        println("\nDurchschnittliche MSE nach Dichtedifferenz:")
        for Δρ in density_contrasts
            subset = filter(r -> r.Δρ == Δρ, results)
            if !isempty(subset)
                avg_mse = mean([r.mse for r in subset])
                println("  Δρ $Δρ kg/m³: $(round(avg_mse, digits=6))")
            end
        end
        
        println("\nDurchschnittliche MSE nach Radius:")
        for R in radii
            subset = filter(r -> r.R == R, results)
            if !isempty(subset)
                avg_mse = mean([r.mse for r in subset])
                println("  R $R km: $(round(avg_mse, digits=6))")
            end
        end
    end
    
    return results
end

# ===== BEISPIELAUFRUFE =====

# Einzeltest:
# test_model_comprehensive("velocity_checkpoints/final_velocity_model.bson")

# Batch-Test:
# batch_results = batch_test_model("velocity_checkpoints/final_velocity_model.bson", 20)

# Modellvergleich (falls mehrere Modelle vorhanden):
# model_paths = [
#     "velocity_checkpoints/velocity_checkpoint_epoch10.bson",
#     "velocity_checkpoints/velocity_checkpoint_epoch20.bson", 
#     "velocity_checkpoints/final_velocity_model.bson"
# ]
# comparison_results = compare_velocity_models(model_paths)

# Parameter-Sensitivitäts-Analyse:
# sensitivity_results = analyze_parameter_sensitivity("velocity_checkpoints/final_velocity_model.bson")

println("Evaluierungsskript geladen. Verfügbare Funktionen:")
println("- test_model_comprehensive(model_path)")
println("- batch_test_model(model_path, n_tests)")
println("- compare_velocity_models(model_paths)")
println("- analyze_parameter_sensitivity(model_path)")