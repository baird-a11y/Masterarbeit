# =============================================================================
# MODEL EVALUATION MODULE - METRIKEN-BERECHNUNG
# =============================================================================
# Speichern als: evaluate_model.jl

using Statistics
using BSON
using Plots

# Module laden (falls nicht bereits geladen)
include("unet_architecture.jl") # Für SimplifiedUNet Definition (WICHTIG!)
include("training.jl")           # Für load_trained_model
include("lamem_interface.jl")    # Für Sample-Generation
include("data_processing.jl")    # Für preprocess_lamem_sample

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

# ... (Rest des Codes bleibt unverändert, wie in deinem Dokument)

println("Model Evaluation Module geladen!")
println("Verfügbare Funktionen:")
println("  - calculate_lamem_fidelity_metrics(model, sample)")
println("  - evaluate_multiple_samples(model, samples)")
println("  - test_lamem_fidelity_evaluation(model_path)")
println("")
println("Zum Testen: test_lamem_fidelity_evaluation()")
