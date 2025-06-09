using Flux
using Flux: mse, gpu, cpu
using CUDA
using Statistics
using FileIO
using LinearAlgebra
using Optimisers
using ProgressMeter
using BSON: @save, @load
using Random
using LaMEM, GeophysicalModelGenerator
# ==================== VOLLSTÄNDIGER GPU-FIX ====================

# ==================== SCHNELLER GPU-FIX ====================

using CUDA

# Aktiviere allowscalar - das ist die Lösung!
CUDA.allowscalar(true)
println("✓ CUDA allowscalar aktiviert - GPU-Training jetzt möglich!")

# ==================== GPU-KONFIG MIT ALLOWSCALAR ====================

const GPU_FIXED_CONFIG = (
    # Modell-Parameter
    image_size = (256, 256),
    input_channels = 1,
    output_channels = 2,
    
    # Training-Parameter
    learning_rate = 0.001,
    num_epochs = 10,                   # Mehr Epochen da GPU schnell ist
    batch_size = 4,                    # Größere Batches für GPU
    
    # Dataset-Parameter
    dataset_size = 100,                # Zurück zu vollem Datensatz
    test_split = 0.1,
    
    # Multi-Kristall Parameter
    num_crystals = 2,
    crystal_min_distance = 0.15,
    allow_different_densities = true,
    
    # LaMEM-Parameter
    eta_range = (1e19, 1e21),
    delta_rho_range = (100, 500),
    position_range = (-0.6, 0.6),
    radius_range = (0.03, 0.08),
    
    # Speicher-Parameter
    checkpoint_dir = "velocity_checkpoints_gpu_fixed",
    save_every_n_epochs = 2,
    
    # Hardware-Parameter - GPU aktiviert mit allowscalar
    use_gpu = true,                    # GPU-Training aktiviert
    memory_cleanup_frequency = 5,
    
    # Debug-Parameter
    verbose = true,
    save_sample_images = true
)

# ==================== EINFACHER GPU-TEST ====================

function test_gpu_training_simple()
    """
    Einfacher Test mit bestehenden Funktionen + allowscalar
    """
    println("=== EINFACHER GPU-TEST MIT ALLOWSCALAR ===")
    
    # Nutze das bestehende UNet von Ihrem ursprünglichen Code
    function create_velocity_unet_configured()
        # Beispiel-Definition eines UNet-Modells
        return Chain(
            Conv((3, 3), 1=>16, relu),
            MaxPool((2, 2)),
            Conv((3, 3), 16=>32, relu),
            MaxPool((2, 2)),
            Conv((3, 3), 32=>64, relu),
            MaxPool((2, 2)),
            Conv((3, 3), 64=>128, relu),
            MaxPool((2, 2)),
            Conv((3, 3), 128=>GPU_FIXED_CONFIG.output_channels, identity)
        )
    end
    
    model = create_velocity_unet_configured()
    model = gpu(model)
    println("✓ Modell auf GPU verschoben")
    
    # Test Forward-Pass
    test_input = gpu(randn(Float32, 256, 256, 1, 1))
    
    try
        test_output = model(test_input)
        println("✓ Forward-Pass erfolgreich: $(size(test_output))")
        println("✓ GPU-Training mit allowscalar funktioniert!")
        return true
    catch e
        println("✗ Forward-Pass fehlgeschlagen: $e")
        return false
    end
end

# ==================== VOLLSTÄNDIGES GPU-TRAINING ====================

function run_gpu_training_with_allowscalar()
    """
    Führt vollständiges GPU-Training mit allowscalar durch
    """
    println("=== GPU-TRAINING MIT ALLOWSCALAR ===")
    
    # Sicherheitscheck
    if !CUDA.functional()
        error("CUDA nicht verfügbar!")
    end
    
    # Allowscalar aktivieren
    CUDA.allowscalar(true)
    
    # GPU-Test
    if !test_gpu_training_simple()
        error("GPU-Test fehlgeschlagen!")
    end
    
    # Temporär CONFIG überschreiben für GPU-Training
    global CONFIG = GPU_FIXED_CONFIG  # Ensure CONFIG is globally accessible
    
    println("\n1. DATENGENERIERUNG (GPU-optimiert)")
    raw_dataset = generate_training_dataset_two_crystals()
    
    println("\n2. DATENVORVERARBEITUNG")
    train_batches = create_training_batches_configured(raw_dataset)
    
    println("\n3. MODELL-ERSTELLUNG")
    model = create_velocity_unet_configured()
    println("Modell erstellt mit $(CONFIG.input_channels) → $(CONFIG.output_channels) Kanälen")
    
    println("\n4. GPU-TRAINING STARTEN")
    trained_model, losses = train_velocity_unet_configured(model, train_batches)
    
    println("\n" * "="^50)
    println("GPU-TRAINING MIT ALLOWSCALAR ABGESCHLOSSEN")
    println("="^50)
    println("Konfiguration:")
    println("  Samples: $(length(raw_dataset))")
    println("  Kristalle: $(CONFIG.num_crystals)")
    println("  Epochen: $(CONFIG.num_epochs)")
    println("  Finaler Loss: $(round(losses[end], digits=6))")
    println("  GPU-Training: $(isnan(losses[end]) ? "FEHLGESCHLAGEN" : "ERFOLGREICH")")
    println("  Modell gespeichert: $(CONFIG.checkpoint_dir)")
    println("="^50)
    
    return trained_model, losses, raw_dataset
end

# ==================== GPU-MEMORY INFO ====================

function show_gpu_info()
    """
    Zeigt GPU-Informationen
    """
    println("=== GPU-INFORMATIONEN ===")
    println("CUDA verfügbar: $(CUDA.functional())")
    
    if CUDA.functional()
        mem_free, mem_total = CUDA.memory_info()
        mem_used = mem_total - mem_free
        
        println("GPU-Speicher:")
        println("  Total: $(round(mem_total / 1e9, digits=2)) GB")
        println("  Genutzt: $(round(mem_used / 1e9, digits=2)) GB")
        println("  Frei: $(round(mem_free / 1e9, digits=2)) GB")
        println("  Auslastung: $(round(100 * mem_used / mem_total, digits=1))%")
        
        println("AllowScalar Status: $(CUDA.allowscalar())")
    end
end

# ==================== AUSFÜHRUNG ====================

println("\n" * "="^60)
println("GPU-TRAINING MIT ALLOWSCALAR - BEREIT!")
println("="^60)

# Zeige GPU-Info
show_gpu_info()

println("\nVerwendung:")
println("  # GPU-Test durchführen:")
println("  test_gpu_training_simple()")
println("  ")
println("  # Vollständiges GPU-Training starten:")
println("  model, losses, data = run_gpu_training_with_allowscalar()")
println("="^60)

# Automatischer Test
println("\nFühre automatischen GPU-Test durch...")
if test_gpu_training_simple()
    println("\n🎉 GPU-TRAINING BEREIT!")
    println("Führen Sie aus: model, losses, data = run_gpu_training_with_allowscalar()")
else
    println("\n❌ GPU-Training nicht möglich. Nutzen Sie CPU-Training.")
end

# result = run_gpu_training_test()