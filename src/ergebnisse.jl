# Einfache Visualisierung OHNE LaMEM-Abhängigkeit
using GLMakie
using BSON: @load
using Statistics

function create_dummy_test_sample(model_type="multi_crystal")
    """
    Erstellt Dummy-Test-Sample ohne LaMEM
    """
    
    if model_type == "multi_crystal"
        # Multi-crystal dummy (128x128)
        h, w = 128, 128
        phase = zeros(h, w)
        
        # 2 Kristalle hinzufügen
        phase[40:60, 50:70] .= 1    # Kristall 1
        phase[80:100, 30:50] .= 2   # Kristall 2
        
        # Dummy velocity fields
        vx = randn(h, w) * 0.1
        vz = randn(h, w) * 0.5 .- 1.0  # Negative für sinking
        
        # Verstärke Geschwindigkeit bei Kristallen
        vz[phase .> 0.5] .*= 3
        
        v_stokes = 1.0
        params = (num_crystals=2, η=1e20, Δρ=200)
        
    else
        # Single crystal dummy (256x256)
        h, w = 256, 256
        phase = zeros(h, w)
        
        # 1 Kristall
        center_x, center_z = 128, 180
        radius = 25
        for i in 1:h, j in 1:w
            if sqrt((i-center_x)^2 + (j-center_z)^2) < radius
                phase[i, j] = 1
            end
        end
        
        # Dummy velocity
        vx = randn(h, w) * 0.05
        vz = randn(h, w) * 0.3 .- 0.8
        
        # Verstärke bei Kristall
        vz[phase .> 0.5] .*= 2
        
        v_stokes = 1.0
        params = (η=1e20, Δρ=200, center=(0.0, 0.5), radius=0.05)
    end
    
    return (
        phase = phase,
        vx = vx,
        vz = vz,
        v_stokes = v_stokes,
        params = params
    )
end

function visualize_model_simple(model_path; model_type="multi_crystal", num_samples=4)
    """
    Einfache Visualisierung mit Dummy-Daten
    """
    
    println("Lade Modell: $model_path")
    
    # Modell laden
    if !isfile(model_path)
        println("WARNUNG: Modell-Datei nicht gefunden. Verwende Dummy-Vorhersagen.")
        model = nothing
    else
        try
            model_dict = BSON.load(model_path)
            model = haskey(model_dict, :final_model_cpu) ? model_dict[:final_model_cpu] : 
                    haskey(model_dict, :model_cpu) ? model_dict[:model_cpu] : 
                    first(values(model_dict))
            println("Modell geladen!")
        catch e
            println("FEHLER beim Laden: $e")
            println("Verwende Dummy-Vorhersagen.")
            model = nothing
        end
    end
    
    # Test-Samples erstellen
    samples = [create_dummy_test_sample(model_type) for _ in 1:num_samples]
    
    # Layout
    rows = 2
    cols = 2
    fig = Figure(resolution=(800, 600))
    
    mse_scores = Float64[]
    
    for (i, sample) in enumerate(samples[1:min(num_samples, 4)])
        # Input vorbereiten
        if model_type == "multi_crystal"
            phase_input = preprocess_multi_crystal_simple(sample.phase)
        else
            phase_input = reshape(Float32.(sample.phase), size(sample.phase)..., 1, 1)
        end
        
        # Vorhersage (oder Dummy)
        if model !== nothing
            try
                prediction = model(phase_input)
                pred_vx = prediction[:, :, 1, 1]
                pred_vz = prediction[:, :, 2, 1]
            catch e
                println("Modell-Fehler: $e")
                pred_vx = sample.vx .+ randn(size(sample.vx)...) * 0.1
                pred_vz = sample.vz .+ randn(size(sample.vz)...) * 0.1
            end
        else
            # Dummy-Vorhersage (GT + Rauschen)
            pred_vx = sample.vx .+ randn(size(sample.vx)...) * 0.1
            pred_vz = sample.vz .+ randn(size(sample.vz)...) * 0.1
        end
        
        # MSE berechnen
        gt_vx = sample.vx ./ sample.v_stokes
        gt_vz = sample.vz ./ sample.v_stokes
        mse_total = (mean((pred_vx .- gt_vx).^2) + mean((pred_vz .- gt_vz).^2)) / 2
        push!(mse_scores, mse_total)
        
        # Plot position
        row = div(i - 1, cols) + 1
        col = mod(i - 1, cols) + 1
        
        ax = Axis(fig[row, col], 
                 title="Sample $i (MSE: $(round(mse_total, digits=3)))",
                 aspect=DataAspect())
        
        # v_z Hintergrund
        heatmap!(ax, pred_vz, colormap=:RdBu, colorrange=(-2, 1), alpha=0.8)
        
        # Kristall-Konturen
        if model_type == "multi_crystal"
            contour!(ax, sample.phase, levels=[0.5, 1.5, 2.5], color=:black, linewidth=2)
        else
            contour!(ax, sample.phase, levels=[0.5], color=:black, linewidth=2)
        end
        
        # Velocity vectors (reduziert)
        step = max(8, size(sample.phase, 1) ÷ 15)
        x_indices = 1:step:size(pred_vx, 1)
        z_indices = 1:step:size(pred_vx, 2)
        
        for ii in x_indices, jj in z_indices
            vx_scaled = pred_vx[ii, jj] * step * 0.8
            vz_scaled = pred_vz[ii, jj] * step * 0.8
            
            if abs(vx_scaled) > 0.1 || abs(vz_scaled) > 0.1
                arrows!(ax, [jj], [ii], [vx_scaled], [vz_scaled], 
                       arrowsize=6, arrowcolor=:white, linewidth=1)
            end
        end
    end
    
    # Statistiken
    avg_mse = mean(mse_scores)
    stats_text = """
    $model_type Model Test ($(model === nothing ? "Dummy" : "Real"))
    Avg MSE: $(round(avg_mse, digits=5))
    """
    
    Label(fig[3, 1:2], stats_text, fontsize=12)
    
    save("model_test_$(model_type).png", fig)
    println("Plot gespeichert: model_test_$(model_type).png")
    
    display(fig)
    
    return fig, mse_scores
end

function preprocess_multi_crystal_simple(phase_field)
    """
    Einfache Multi-Channel Vorverarbeitung
    """
    h, w = size(phase_field)
    max_crystals = 4
    encoded = zeros(Float32, h, w, max_crystals + 1)
    
    # Matrix channel
    encoded[:, :, 1] = Float32.(phase_field .≈ 0)
    
    # Crystal channels  
    for crystal_id in 1:max_crystals
        crystal_mask = Float32.(abs.(phase_field .- crystal_id) .< 0.5)
        encoded[:, :, crystal_id + 1] = crystal_mask
    end
    
    # Normierung
    for i in 1:h, j in 1:w
        channel_sum = sum(encoded[i, j, :])
        if channel_sum > 0
            encoded[i, j, :] ./= channel_sum
        else
            encoded[i, j, 1] = 1.0f0
        end
    end
    
    return reshape(encoded, h, w, max_crystals + 1, 1)
end

# Einfache Verwendung ohne LaMEM:
println("EINFACHE VISUALISIERUNG OHNE LAMEM")
println("Verwendung:")
println("  # Multi-Crystal:")
println("  fig, scores = visualize_model_simple(\"final_gpu_model.bson\", model_type=\"multi_crystal\")")
println("  ")
println("  # Single-Crystal:")  
println("  fig, scores = visualize_model_simple(\"final_model.bson\", model_type=\"single_crystal\")")
println("  ")
println("  # Falls Modell nicht existiert, werden Dummy-Vorhersagen verwendet")