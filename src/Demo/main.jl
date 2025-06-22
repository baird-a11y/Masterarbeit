# =============================================================================
# HAUPTSKRIPT - ADAPTIVE UNET FÜR LAMEM
# =============================================================================
# Speichern als: main.jl

using Flux
using CUDA
using GLMakie
using Statistics
using Random

println("=== ADAPTIVE UNET FÜR LAMEM ===")
println("Lade Module...")

# Module laden (Pfade anpassen falls nötig)
include("lamem_interface.jl")
include("data_processing.jl") 
include("unet_config.jl")
include("unet_architecture.jl")
include("batch_management.jl")

println("Alle Module erfolgreich geladen!")

# =============================================================================
# DEMO-FUNKTIONEN
# =============================================================================

"""
Komplette Demo: Von LaMEM-Daten bis UNet-Training
"""
function demo_complete_pipeline(;
    n_samples = 50,
    target_resolution = 256,
    resolutions = [128, 256],
    verbose = true
)
    println("\n" * "="^80)
    println("KOMPLETTE PIPELINE DEMO")
    println("="^80)
    
    # 1. Datengenerierung
    println("\n1. DATENGENERIERUNG")
    dataset = generate_mixed_resolution_dataset(n_samples, resolutions=resolutions, verbose=verbose)
    
    if length(dataset) == 0
        error("Keine Daten generiert!")
    end
    
    # 2. UNet-Konfiguration
    println("\n2. UNET-KONFIGURATION")
    config = design_adaptive_unet(target_resolution)
    
    # 3. UNet-Erstellung
    println("\n3. UNET-ERSTELLUNG")
    model = create_final_corrected_unet(config)
    
    # 4. Test des UNet
    println("\n4. UNET-TEST")
    success, output_shape = test_corrected_unet(target_resolution, verbose=verbose)
    
    # 5. Batch-Management-Test
    println("\n5. BATCH-MANAGEMENT-TEST")
    println("Erstelle adaptive Batches...")
    
    try
        phase_batch, velocity_batch, successful = create_adaptive_batch(
            dataset[1:min(8, length(dataset))], 
            target_resolution,
            verbose=verbose
        )
        
        println("Batch-Erstellung erfolgreich!")
        println("  Phase Batch: $(size(phase_batch))")
        println("  Velocity Batch: $(size(velocity_batch))")
        println("  Erfolgreiche Samples: $successful")
        
        # 6. Forward-Pass mit echten Daten
        println("\n6. FORWARD-PASS MIT ECHTEN DATEN")
        if CUDA.functional()
            model_gpu = gpu(model)
            phase_gpu = gpu(phase_batch)
            output = cpu(model_gpu(phase_gpu))
        else
            output = model(phase_batch)
        end
        
        println("Forward-Pass erfolgreich!")
        println("  Output: $(size(output))")
        
        # 7. Visualisierung (optional)
        if verbose
            println("\n7. VISUALISIERUNG")
            visualize_sample_prediction(phase_batch, velocity_batch, output)
        end
        
    catch e
        println("Fehler beim Batch-Management: $e")
    end
    
    println("\n" * "="^80)
    println("PIPELINE DEMO ABGESCHLOSSEN")
    println("="^80)
    
    return dataset, model, config
end

"""
Visualisiert eine Vorhersage
"""
function visualize_sample_prediction(phase_batch, velocity_batch, prediction; sample_idx=1)
    try
        # Extrahiere ersten Sample
        phase = phase_batch[:, :, 1, sample_idx]
        vx_true = velocity_batch[:, :, 1, sample_idx]
        vz_true = velocity_batch[:, :, 2, sample_idx]
        vx_pred = prediction[:, :, 1, sample_idx]
        vz_pred = prediction[:, :, 2, sample_idx]
        
        # Erstelle Figure
        fig = Figure(resolution=(1200, 800))
        
        # Plot 1: Phasenfeld
        ax1 = Axis(fig[1, 1], title="Phasenfeld", aspect=DataAspect())
        heatmap!(ax1, phase, colormap=:grays)
        
        # Plot 2: Vz True
        ax2 = Axis(fig[1, 2], title="Vz True", aspect=DataAspect())
        heatmap!(ax2, vz_true, colormap=:RdBu, colorrange=(-3, 1))
        contour!(ax2, phase, levels=[0.5], color=:black, linewidth=2)
        
        # Plot 3: Vz Predicted
        ax3 = Axis(fig[1, 3], title="Vz Predicted", aspect=DataAspect())
        heatmap!(ax3, vz_pred, colormap=:RdBu, colorrange=(-3, 1))
        contour!(ax3, phase, levels=[0.5], color=:black, linewidth=2)
        
        # Plot 4: Vz Error
        ax4 = Axis(fig[1, 4], title="Vz Error", aspect=DataAspect())
        error_vz = abs.(vz_pred .- vz_true)
        heatmap!(ax4, error_vz, colormap=:hot)
        
        # Statistiken
        mse_vz = mean((vz_pred .- vz_true).^2)
        mse_vx = mean((vx_pred .- vx_true).^2)
        
        Label(fig[2, 1:4], "MSE Vz: $(round(mse_vz, digits=6)), MSE Vx: $(round(mse_vx, digits=6))")
        
        save("unet_prediction_demo.png", fig)
        println("Visualisierung gespeichert: unet_prediction_demo.png")
        
        display(fig)
        
    catch e
        println("Visualisierung fehlgeschlagen: $e")
    end
end

"""
Teste alle verfügbaren Auflösungen
"""
function demo_all_resolutions()
    println("\n=== TESTE ALLE AUFLÖSUNGEN ===")
    
    resolutions = [64, 128, 256]
    if CUDA.functional()
        push!(resolutions, 512)
    end
    
    results = Dict{Int, Bool}()
    
    for res in resolutions
        println("\n" * "="^50)
        println("Teste $(res)×$(res)")
        
        try
            # Konfiguration
            config = design_adaptive_unet(res)
            
            # UNet erstellen
            model = create_final_corrected_unet(config)
            
            # Test
            success, output_shape = test_corrected_unet(res, verbose=true)
            results[res] = success
            
        catch e
            println("Fehler bei $(res)×$(res): $e")
            results[res] = false
        end
    end
    
    # Zusammenfassung
    println("\n" * "="^50)
    println("ZUSAMMENFASSUNG ALLER TESTS:")
    successful = sum(values(results))
    total = length(results)
    println("Erfolgreich: $successful/$total")
    
    for (res, success) in sort(collect(results))
        status = success ? "✓" : "✗"
        println("  $(res)×$(res): $status")
    end
    
    return results
end

"""
Schnelle Funktionalitätsprüfung
"""
function quick_test()
    println("\n=== SCHNELLER FUNKTIONALITÄTSTEST ===")
    
    try
        # 1. LaMEM Test
        println("1. Teste LaMEM Interface...")
        x, z, phase, vx, vz, exx, ezz, v_stokes = LaMEM_Multi_crystal(
            resolution=(64, 64),
            n_crystals=1,
            radius_crystal=[0.05],
            cen_2D=[(0.0, 0.5)]
        )
        println("   ✓ LaMEM erfolgreich")
        
        # 2. Preprocessing Test
        println("2. Teste Preprocessing...")
        phase_tensor, velocity_tensor = preprocess_lamem_sample(x, z, phase, vx, vz, v_stokes, target_resolution=128)
        println("   ✓ Preprocessing erfolgreich")
        
        # 3. UNet Test
        println("3. Teste UNet...")
        config = design_adaptive_unet(128)
        model = create_final_corrected_unet(config)
        test_input = randn(Float32, 128, 128, 1, 2)
        output = model(test_input)
        println("   ✓ UNet erfolgreich")
        
        println("\n✓ ALLE TESTS ERFOLGREICH!")
        return true
        
    catch e
        println("\n✗ TEST FEHLGESCHLAGEN: $e")
        return false
    end
end

# =============================================================================
# VERFÜGBARE FUNKTIONEN
# =============================================================================

println("\n" * "="^60)
println("VERFÜGBARE FUNKTIONEN:")
println("="^60)
println()
println("DEMO-FUNKTIONEN:")
println("  quick_test()                    - Schneller Funktionalitätstest")
println("  demo_all_resolutions()          - Teste alle UNet-Auflösungen")
println("  demo_complete_pipeline()        - Komplette Pipeline-Demo")
println()
println("DATENGENERIERUNG:")
println("  LaMEM_Multi_crystal(...)        - Einzelne LaMEM-Simulation")
println("  generate_mixed_resolution_dataset(n) - Generiere n Samples")
println()
println("DATENVERARBEITUNG:")
println("  resize_power_of_2(data, size)   - Größenanpassung")
println("  preprocess_lamem_sample(...)    - Sample für UNet vorbereiten")
println()
println("UNET:")
println("  design_adaptive_unet(resolution) - UNet-Konfiguration erstellen")
println("  create_final_corrected_unet(config) - UNet erstellen")
println("  test_corrected_unet(resolution)  - UNet testen")
println()
println("BATCH-MANAGEMENT:")
println("  create_adaptive_batch(samples, res) - Batch erstellen")
println("  smart_batch_manager(dataset, res)   - Smart Batch Iterator")
println("  get_gpu_memory_info()           - GPU-Speicher-Info")
println()
println("="^60)

# =============================================================================
# AUSFÜHRUNG
# =============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    println("\nHauptskript wird direkt ausgeführt...")
    println("Führe schnellen Test aus...")
    
    if quick_test()
        println("\n🎉 System ist bereit!")
        println("\nNächste Schritte:")
        println("  - demo_complete_pipeline() für vollständige Demo")
        println("  - demo_all_resolutions() für Auflösungstest")
        println("  - Eigene Experimente starten")
    else
        println("\n⚠️  System-Setup unvollständig!")
        println("Bitte prüfen Sie die Fehlermeldungen oben.")
    end
else
    println("\nModule erfolgreich geladen!")
    println("Nutzen Sie quick_test() für einen ersten Test.")
end