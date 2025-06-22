# =============================================================================
# VEREINFACHTES SERVER-SCRIPT - ADAPTIVE UNET FÜR LAMEM (SICHER)
# =============================================================================
# Speichern als: main_safe.jl

using Flux
using CUDA
using Statistics
using Random
import Dates
using Serialization
using BSON: @save

println("=== ADAPTIVE UNET FÜR LAMEM - SICHERER SERVER MODE ===")
println("Zeit: $(Dates.now())")
println("Lade Module...")

# Module laden
include("lamem_interface.jl")
include("data_processing.jl") 
include("unet_config.jl")

# VERWENDE SICHERE MODULE
println("Lade sichere UNet-Architektur...")
include("unet_architecture.jl")  # Neue sichere Version

println("Lade sicheres Training-Modul...")
include("training.jl")  # Neue sichere Version

include("batch_management.jl")

println("Alle Module erfolgreich geladen!")

# =============================================================================
# SERVER KONFIGURATION - REDUZIERT FÜR STABILITÄT
# =============================================================================

const SERVER_CONFIG = (
    # Experimentelle Parameter - REDUZIERT
    n_samples = 5,                      # Weniger Samples für ersten Test
    resolutions = [128],                # Nur eine Auflösung
    target_resolution = 128,            # Kleinere Auflösung
    
    # Training-Parameter  
    num_epochs = 2,                     # Weniger Epochen für Test
    learning_rate = 0.001f0,            # Lernrate
    batch_size = 2,                     # Kleinere Batch-Größe
    early_stopping_patience = 5,        # Weniger Patience
    
    # Output-Parameter
    checkpoint_dir = "safe_server_checkpoints", # Checkpoint-Verzeichnis
    results_dir = "safe_server_results",       # Ergebnisse
    
    # Server-spezifisch
    save_dataset = true,                # Dataset für spätere Nutzung speichern
    use_gpu = false,                    # CPU für Stabilität
)

# =============================================================================
# SICHERER SERVER BATCH-JOB
# =============================================================================

"""
Sicherer Server-Batch-Job mit reduzierter Komplexität
"""
function run_safe_server_batch_job()
    # Erstelle Output-Verzeichnisse
    mkpath(SERVER_CONFIG.checkpoint_dir)
    mkpath(SERVER_CONFIG.results_dir)
    
    println("="^80)
    println("STARTE SICHEREN SERVER BATCH-JOB")
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
        
        if !quick_test_safe()
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
            verbose=true
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
        println("3. MODELL-ERSTELLUNG (VEREINFACHT)")
        println("="^60)
        
        println("Erstelle vereinfachtes UNet...")
        model = create_simplified_unet(1, 2, 32)  # input_ch, output_ch, base_filters
        
        # Modell-Test
        success, _ = test_simplified_unet(SERVER_CONFIG.target_resolution, batch_size=2)
        
        if !success
            error("UNet-Test fehlgeschlagen!")
        end
        println("✓ Vereinfachtes UNet erfolgreich getestet")
        
        # 4. TRAINING
        println("\n" * "="^60)
        println("4. SICHERES TRAINING")
        println("="^60)
        
        # Training-Konfiguration
        train_config = create_training_config(
            learning_rate = SERVER_CONFIG.learning_rate,
            num_epochs = SERVER_CONFIG.num_epochs,
            batch_size = SERVER_CONFIG.batch_size,
            checkpoint_dir = SERVER_CONFIG.checkpoint_dir,
            save_every_n_epochs = 2,  # Häufiger speichern
            use_gpu = SERVER_CONFIG.use_gpu,
            validation_split = 0.2f0,  # Mehr Validation-Daten
            early_stopping_patience = SERVER_CONFIG.early_stopping_patience
        )
        
        println("Starte sicheres Training: $(train_config.num_epochs) Epochen")
        
        # Training mit sicherer Funktion
        trained_model, train_losses, val_losses = train_velocity_unet_safe(
            model, dataset, SERVER_CONFIG.target_resolution,
            config=train_config
        )
        
        println("✓ Training abgeschlossen")
        
        # 5. EVALUIERUNG
        println("\n" * "="^60)
        println("5. EVALUIERUNG")
        println("="^60)
        
        # Training-Statistiken
        if length(train_losses) > 0
            final_train_loss = train_losses[end]
            println("Finale Training Loss: $(round(final_train_loss, digits=6))")
        else
            println("Keine Training-Losses verfügbar")
        end
        
        if length(val_losses) > 0
            final_val_loss = val_losses[end]
            best_val_loss = minimum(val_losses)
            best_epoch = argmin(val_losses)
            
            println("Finale Validation Loss: $(round(final_val_loss, digits=6))")
            println("Beste Validation Loss: $(round(best_val_loss, digits=6)) (Epoche $best_epoch)")
        else
            println("Keine Validation-Losses verfügbar")
        end
        
        # Speichere Ergebnisse
        results_data = Dict(
            "train_losses" => train_losses,
            "val_losses" => val_losses,
            "server_config" => SERVER_CONFIG,
            "model" => trained_model
        )
        
        results_path = joinpath(SERVER_CONFIG.results_dir, "training_results.bson")
        @save results_path results_data
        println("✓ Trainingsergebnisse gespeichert: $results_path")
        
        # 6. FINAL TEST
        println("\n" * "="^60)
        println("6. FINAL TEST")
        println("="^60)
        
        if length(dataset) > 0
            # Test mit einem Sample
            test_sample = dataset[1]
            x, z, phase, vx, vz, exx, ezz, v_stokes = test_sample
            
            phase_tensor, velocity_tensor = preprocess_lamem_sample(
                x, z, phase, vx, vz, v_stokes,
                target_resolution=SERVER_CONFIG.target_resolution
            )
            
            prediction = cpu(trained_model(phase_tensor))
            
            mse_vx = mean((prediction[:,:,1,1] .- velocity_tensor[:,:,1,1]).^2)
            mse_vz = mean((prediction[:,:,2,1] .- velocity_tensor[:,:,2,1]).^2)
            mse_total = (mse_vx + mse_vz) / 2
            
            println("Final Test Ergebnisse (1 Sample):")
            println("  MSE Vx: $(round(mse_vx, digits=6))")
            println("  MSE Vz: $(round(mse_vz, digits=6))")
            println("  MSE Total: $(round(mse_total, digits=6))")
            
            # Speichere Test-Ergebnisse
            test_results = Dict(
                "mse_vx" => mse_vx,
                "mse_vz" => mse_vz,
                "mse_total" => mse_total
            )
            
            test_path = joinpath(SERVER_CONFIG.results_dir, "test_results.bson")
            @save test_path test_results
            println("✓ Test-Ergebnisse gespeichert: $test_path")
        end
        
        # Erfolgreicher Abschluss
        end_time = time()
        total_time = end_time - start_time
        
        println("\n" * "="^80)
        println("SICHERER BATCH-JOB ERFOLGREICH ABGESCHLOSSEN")
        println("="^80)
        println("Gesamtzeit: $(round(total_time/60, digits=2)) Minuten")
        println("Checkpoints: $(SERVER_CONFIG.checkpoint_dir)")
        println("Ergebnisse: $(SERVER_CONFIG.results_dir)")
        
        return true
        
    catch e
        end_time = time()
        total_time = end_time - start_time
        
        println("\n" * "="^80)
        println("SICHERER BATCH-JOB FEHLGESCHLAGEN")
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
Sicherer System-Test mit vereinfachtem UNet
"""
function quick_test_safe()
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
            x, z, phase, vx, vz, v_stokes, target_resolution=64
        )
        
        # Vereinfachtes UNet Test
        model = create_simplified_unet()
        test_input = randn(Float32, 64, 64, 1, 1)
        output = model(test_input)
        
        return size(output) == (64, 64, 2, 1)
        
    catch e
        println("Sicherer System-Test Fehler: $e")
        return false
    end
end

"""
Demo für sicheres Training
"""
function demo_safe_training(;
    n_samples = 10,
    target_resolution = 128,
    num_epochs = 3
)
    println("=== SICHERES TRAINING DEMO ===")
    
    # 1. Daten generieren
    println("\n1. Generiere wenige Trainingsdaten...")
    dataset = generate_mixed_resolution_dataset(
        n_samples, 
        resolutions=[target_resolution],
        verbose=true
    )
    
    if length(dataset) < 2
        error("Zu wenige Samples generiert!")
    end
    
    # 2. Vereinfachtes UNet erstellen
    println("\n2. Erstelle vereinfachtes UNet...")
    model = create_simplified_unet()
    
    # 3. Training-Konfiguration
    train_config = create_training_config(
        num_epochs = num_epochs,
        learning_rate = 0.001f0,
        batch_size = 2,
        checkpoint_dir = "demo_safe_checkpoints",
        use_gpu = false
    )
    
    # 4. Sicheres Training
    println("\n3. Starte sicheres Training...")
    trained_model, train_losses, val_losses = train_velocity_unet_safe(
        model, dataset, target_resolution, config=train_config
    )
    
    # 5. Evaluierung
    println("\n4. Evaluiere trainiertes Modell...")
    if length(dataset) > 0
        test_sample = dataset[1]
        x, z, phase, vx, vz, exx, ezz, v_stokes = test_sample
        
        phase_tensor, velocity_tensor = preprocess_lamem_sample(
            x, z, phase, vx, vz, v_stokes,
            target_resolution=target_resolution
        )
        
        prediction = cpu(trained_model(phase_tensor))
        
        mse_vx = mean((prediction[:,:,1,1] .- velocity_tensor[:,:,1,1]).^2)
        mse_vz = mean((prediction[:,:,2,1] .- velocity_tensor[:,:,2,1]).^2)
        
        println("Test-Evaluierung:")
        println("  MSE Vx: $(round(mse_vx, digits=6))")
        println("  MSE Vz: $(round(mse_vz, digits=6))")
        println("  MSE Gesamt: $(round((mse_vx + mse_vz)/2, digits=6))")
    end
    
    println("\n✓ Sicheres Training Demo abgeschlossen!")
    return trained_model, train_losses, val_losses
end

# =============================================================================
# SICHERER SERVER BATCH-JOB START
# =============================================================================

println("="^80)
println("SICHERER SERVER BATCH-JOB INITIALISIERUNG")
println("="^80)

# Führe sicheren Batch-Job aus
success = run_safe_server_batch_job()

if success
    println("\n" * "="^80)
    println(" SICHERER SERVER BATCH-JOB ERFOLGREICH ABGESCHLOSSEN!")
    println("="^80)
    exit(0)
else
    println("\n" * "="^80)
    println(" SICHERER SERVER BATCH-JOB FEHLGESCHLAGEN!")
    println("="^80)
    exit(1)
end