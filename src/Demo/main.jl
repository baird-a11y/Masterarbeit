# =============================================================================
# VEREINFACHTES SERVER-SCRIPT - ADAPTIVE UNET FÜR LAMEM
# =============================================================================
# Speichern als: main_server_simple.jl

using Flux
using CUDA
using Statistics
using Random
import Dates
using Serialization
using BSON: @save

println("=== ADAPTIVE UNET FÜR LAMEM - SERVER MODE ===")
println("Zeit: $(Dates.now())")
println("Lade Module...")

# Module laden
include("lamem_interface.jl")
include("data_processing.jl") 
include("unet_config.jl")
include("unet_architecture.jl")
include("batch_management.jl")
include("training.jl")

println("Alle Module erfolgreich geladen!")

# =============================================================================
# SERVER KONFIGURATION
# =============================================================================

const SERVER_CONFIG = (
    # Experimentelle Parameter
    n_samples = 5,                     # Anzahl LaMEM-Simulationen
    resolutions = [128, 256],            # Verschiedene Auflösungen
    target_resolution = 256,             # UNet-Auflösung
    
    # Training-Parameter  
    num_epochs = 1,                     # Training-Epochen
    learning_rate = 0.001f0,             # Lernrate
    batch_size = 1,                      # Batch-Größe
    early_stopping_patience = 15,        # Early stopping
    
    # Output-Parameter
    checkpoint_dir = "server_checkpoints", # Checkpoint-Verzeichnis
    results_dir = "server_results",       # Ergebnisse
    
    # Server-spezifisch
    save_dataset = true,                  # Dataset für spätere Nutzung speichern
    use_gpu = CUDA.functional(),          # GPU falls verfügbar
)

# =============================================================================
# SERVER BATCH-JOB
# =============================================================================

"""
Vereinfachter Server-Batch-Job ohne Logging
"""
function run_server_batch_job()
    # Erstelle Output-Verzeichnisse
    mkpath(SERVER_CONFIG.checkpoint_dir)
    mkpath(SERVER_CONFIG.results_dir)
    
    println("="^80)
    println("STARTE SERVER BATCH-JOB")
    println("="^80)
    
    # Konfiguration ausgeben
    for (key, value) in pairs(SERVER_CONFIG)
        println("$key: $value")
    end
    
    start_time = time()
    
    try
        # 1. SYSTEM-CHECK
        println("\n" * "="^60)
        println("1. SYSTEM-CHECK")
        println("="^60)
        
        # Hardware-Info
        println("Julia Version: $(VERSION)")
        println("CUDA verfügbar: $(CUDA.functional())")
        if CUDA.functional()
            println("GPU: $(CUDA.name(CUDA.device()))")
            mem_info = get_gpu_memory_info()
            println("GPU Memory: $(round(mem_info.total / 1e9, digits=2)) GB")
        end
        
        if !quick_test_server()
            error("System-Test fehlgeschlagen!")
        end
        println("✓ System-Test erfolgreich")
        
        # 2. DATENGENERIERUNG
        println("\n" * "="^60)
        println("2. DATENGENERIERUNG")
        println("="^60)
        
        println("Starte Generierung von $(SERVER_CONFIG.n_samples) Samples...")
        
        dataset = generate_mixed_resolution_dataset(
            SERVER_CONFIG.n_samples,
            resolutions=SERVER_CONFIG.resolutions,
            verbose=false  # Weniger Ausgabe für Server
        )
        
        if length(dataset) == 0
            error("Keine Trainingsdaten generiert!")
        end
        
        println("✓ Dataset erstellt: $(length(dataset)) Samples")
        
        # Dataset speichern
        if SERVER_CONFIG.save_dataset
            
            dataset_path = joinpath(SERVER_CONFIG.results_dir, "dataset.jls")
            serialize(dataset_path, dataset)
            println("✓ Dataset gespeichert: $dataset_path")
        end
        
        # 3. MODELL-ERSTELLUNG
        println("\n" * "="^60)
        println("3. MODELL-ERSTELLUNG")
        println("="^60)
        
        config = design_adaptive_unet(SERVER_CONFIG.target_resolution)
        println("UNet Konfiguration: $(config.depth) Layer, $(config.filter_progression)")
        
        model = create_final_corrected_unet(config)
        println("✓ UNet erstellt")
        
        # Modell-Test
        success, output_shape = test_final_corrected_unet(
            SERVER_CONFIG.target_resolution, 
            verbose=false
        )
        
        if !success
            error("UNet-Test fehlgeschlagen!")
        end
        println("✓ UNet-Test erfolgreich: $(output_shape)")
        
        # 4. TRAINING
        println("\n" * "="^60)
        println("4. TRAINING")
        println("="^60)
        
        # Training-Konfiguration
        train_config = create_training_config(
            learning_rate = SERVER_CONFIG.learning_rate,
            num_epochs = SERVER_CONFIG.num_epochs,
            batch_size = SERVER_CONFIG.batch_size,
            checkpoint_dir = SERVER_CONFIG.checkpoint_dir,
            save_every_n_epochs = 5,  # Häufiger speichern auf Server
            use_gpu = SERVER_CONFIG.use_gpu,
            validation_split = 0.15f0,  # Mehr Validation-Daten
            early_stopping_patience = SERVER_CONFIG.early_stopping_patience
        )
        
        println("Starte Training: $(train_config.num_epochs) Epochen, GPU: $(train_config.use_gpu)")
        
        # Training mit Error-Handling
        trained_model, train_losses, val_losses = train_velocity_unet(
            model, dataset, SERVER_CONFIG.target_resolution,
            config=train_config
        )
        
        println("✓ Training abgeschlossen")
        
        # 5. EVALUIERUNG
        println("\n" * "="^60)
        println("5. EVALUIERUNG")
        println("="^60)
        
        # Training-Statistiken
        final_train_loss = train_losses[end]
        final_val_loss = val_losses[end]
        best_val_loss = minimum(val_losses)
        best_epoch = argmin(val_losses)
        
        println("Training Statistiken:")
        println("  Finale Training Loss: $(round(final_train_loss, digits=6))")
        println("  Finale Validation Loss: $(round(final_val_loss, digits=6))")
        println("  Beste Validation Loss: $(round(best_val_loss, digits=6)) (Epoche $best_epoch)")
        println("  Training Epochen: $(length(train_losses))")
        
        # Speichere Ergebnisse
        
        results_data = Dict(
            "train_losses" => train_losses,
            "val_losses" => val_losses,
            "config" => config,
            "server_config" => SERVER_CONFIG,
            "final_stats" => Dict(
                "final_train_loss" => final_train_loss,
                "final_val_loss" => final_val_loss,
                "best_val_loss" => best_val_loss,
                "best_epoch" => best_epoch
            )
        )
        
        results_path = joinpath(SERVER_CONFIG.results_dir, "training_results.bson")
        @save results_path results_data
        println("✓ Trainingsergebnisse gespeichert: $results_path")
        
        # 6. FINAL TEST
        println("\n" * "="^60)
        println("6. FINAL TEST")
        println("="^60)
        
        if length(dataset) > 0
            # Test mit mehreren Samples
            n_test_samples = min(5, length(dataset))
            total_mse_vx = 0.0
            total_mse_vz = 0.0
            
            for i in 1:n_test_samples
                test_sample = dataset[i]
                x, z, phase, vx, vz, exx, ezz, v_stokes = test_sample
                
                phase_tensor, velocity_tensor = preprocess_lamem_sample(
                    x, z, phase, vx, vz, v_stokes,
                    target_resolution=SERVER_CONFIG.target_resolution
                )
                
                prediction = cpu(trained_model(phase_tensor))
                
                mse_vx = mean((prediction[:,:,1,1] .- velocity_tensor[:,:,1,1]).^2)
                mse_vz = mean((prediction[:,:,2,1] .- velocity_tensor[:,:,2,1]).^2)
                
                total_mse_vx += mse_vx
                total_mse_vz += mse_vz
            end
            
            avg_mse_vx = total_mse_vx / n_test_samples
            avg_mse_vz = total_mse_vz / n_test_samples
            avg_mse_total = (avg_mse_vx + avg_mse_vz) / 2
            
            println("Final Test Ergebnisse ($(n_test_samples) Samples):")
            println("  Durchschnitt MSE Vx: $(round(avg_mse_vx, digits=6))")
            println("  Durchschnitt MSE Vz: $(round(avg_mse_vz, digits=6))")
            println("  Durchschnitt MSE Total: $(round(avg_mse_total, digits=6))")
            
            # Speichere Test-Ergebnisse
            test_results = Dict(
                "avg_mse_vx" => avg_mse_vx,
                "avg_mse_vz" => avg_mse_vz,
                "avg_mse_total" => avg_mse_total,
                "n_test_samples" => n_test_samples
            )
            
            test_path = joinpath(SERVER_CONFIG.results_dir, "test_results.bson")
            @save test_path test_results
            println("✓ Test-Ergebnisse gespeichert: $test_path")
        end
        
        # Erfolgreicher Abschluss
        end_time = time()
        total_time = end_time - start_time
        
        println("\n" * "="^80)
        println("BATCH-JOB ERFOLGREICH ABGESCHLOSSEN")
        println("="^80)
        println("Gesamtzeit: $(round(total_time/3600, digits=2)) Stunden")
        println("Checkpoints: $(SERVER_CONFIG.checkpoint_dir)")
        println("Ergebnisse: $(SERVER_CONFIG.results_dir)")
        
        return true
        
    catch e
        end_time = time()
        total_time = end_time - start_time
        
        println("\n" * "="^80)
        println("BATCH-JOB FEHLGESCHLAGEN")
        println("="^80)
        println("Fehler: $e")
        println("Laufzeit bis Fehler: $(round(total_time/60, digits=1)) Minuten")
        
        # Stacktrace ausgeben für Debugging
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        
        return false
    end
end

"""
Server-optimierter System-Test
"""
function quick_test_server()
    try
        # LaMEM Test
        x, z, phase, vx, vz, exx, ezz, v_stokes = LaMEM_Multi_crystal(
            resolution=(64, 64),
            n_crystals=1,
            radius_crystal=[0.05],
            cen_2D=[(0.0, 0.5)]
        )
        
        # Preprocessing Test
        phase_tensor, velocity_tensor = preprocess_lamem_sample(
            x, z, phase, vx, vz, v_stokes, target_resolution=128
        )
        
        # UNet Test
        config = design_adaptive_unet(128)
        model = create_final_corrected_unet(config)
        test_input = randn(Float32, 128, 128, 1, 2)
        output = model(test_input)
        
        return size(output) == (128, 128, 2, 2)
        
    catch e
        println("System-Test Fehler: $e")
        return false
    end
end

"""
Server-optimierte UNet-Test-Funktion
"""
function test_final_corrected_unet(resolution::Int=256; batch_size::Int=2, verbose::Bool=false)
    config = design_adaptive_unet(resolution)
    model = create_final_corrected_unet(config)
    test_input = randn(Float32, resolution, resolution, 1, batch_size)
    
    try
        output = model(test_input)
        expected_shape = (resolution, resolution, 2, batch_size)
        success = size(output) == expected_shape
        
        return success, size(output)
    catch e
        return false, nothing
    end
end

# =============================================================================
# SERVER BATCH-JOB START
# =============================================================================

println("="^80)
println("SERVER BATCH-JOB INITIALISIERUNG")
println("="^80)

# Führe Batch-Job aus
success = run_server_batch_job()

if success
    println(" SERVER BATCH-JOB ERFOLGREICH ABGESCHLOSSEN! ")
    exit(0)
else
    println(" SERVER BATCH-JOB FEHLGESCHLAGEN! ")
    exit(1)
end