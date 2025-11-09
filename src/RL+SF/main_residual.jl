# =============================================================================
# MAIN ENTRY POINT - RESIDUAL LEARNING
# =============================================================================
# Kombiniert alle Module für vollständiges Training

using Dates

println("="^80)
println("RESIDUAL LEARNING + STREAM FUNCTION - MAIN")
println("="^80)
println("Start: $(now())")
println()

# =============================================================================
# 1. LADE ALLE MODULE
# =============================================================================

println(" Lade Module...")

# Core Modules
include("lamem_interface.jl")
include("data_processing.jl")
include("stokes_analytical.jl")
include("stream_function.jl")

# Model Modules
include("gpu_utils.jl")
include("residual_unet.jl")

# Training Modules
include("losses.jl")
include("residual_training.jl")

println(" Alle Module geladen!")
println()

# =============================================================================
# 2. KONFIGURATION
# =============================================================================

"""
Erstelle Custom Config für dein Experiment
"""
function create_experiment_config(;
    # Dataset
    n_samples=100,
    crystal_range=1:10,
    
    # Training
    epochs=50,
    batch_size=4,
    learning_rate=0.001f0,
    
    # Loss
    λ_residual=0.01f0,
    λ_divergence=0.1f0,
    use_stream_function=true,
    
    # Other
    patience=15,
    checkpoint_dir="results/residual_run_$(now())"
)
    
    return TrainingConfig(
        n_samples,
        crystal_range,
        0.8,  # train_split
        
        epochs,
        batch_size,
        learning_rate,
        
        λ_residual,
        λ_divergence,
        use_stream_function,
        
        :adam,
        0.0f0,
        
        patience,
        1e-4,
        
        10,  # save_every
        checkpoint_dir,
        
        false  # use_gpu (wenn verfügbar)
    )
end

# =============================================================================
# 3. EXPERIMENTE
# =============================================================================

"""
Experiment 1: Baseline ohne Stream Function
"""
function run_baseline_experiment()
    println("\n" * "="^80)
    println("EXPERIMENT 1: BASELINE (OHNE STREAM FUNCTION)")
    println("="^80)
    
    config = create_experiment_config(
        n_samples=100,
        epochs=50,
        use_stream_function=true,
        checkpoint_dir="results/baseline"
    )
    
    model, state = train_residual_unet(config=config)
    
    println("\n Baseline Experiment abgeschlossen!")
    return model, state, config
end

"""
Experiment 2: Mit Stream Function
"""
function run_stream_function_experiment()
    println("\n" * "="^80)
    println("EXPERIMENT 2: MIT STREAM FUNCTION")
    println("="^80)
    
    config = create_experiment_config(
        n_samples=100,
        epochs=50,
        use_stream_function=true,
        λ_divergence=0.0f0,  # Nicht nötig mit SF
        checkpoint_dir="results/stream_function"
    )
    
    model, state = train_residual_unet(config=config)
    
    println("\n Stream Function Experiment abgeschlossen!")
    return model, state, config
end

"""
Experiment 3: Kleiner Test (schnell)
"""
function run_quick_test()
    println("\n" * "="^80)
    println("QUICK TEST (20 Samples, 10 Epochen)")
    println("="^80)
    
    config = create_experiment_config(
        n_samples=20,
        crystal_range=1:3,
        epochs=10,
        batch_size=2,
        use_stream_function=false,
        checkpoint_dir="results/quick_test"
    )
    
    model, state = train_residual_unet(config=config)
    
    println("\n Quick Test abgeschlossen!")
    return model, state, config
end

"""
Experiment 4: Lambda-Tuning
"""
function run_lambda_tuning_experiment()
    println("\n" * "="^80)
    println("EXPERIMENT 4: LAMBDA TUNING")
    println("="^80)
    
    λ_values = [0.001f0, 0.01f0, 0.1f0]
    
    results = []
    
    for λ_res in λ_values
        for λ_div in λ_values
            println("\n🔧 Testing λ_residual=$λ_res, λ_divergence=$λ_div")
            
            config = create_experiment_config(
                n_samples=50,
                epochs=20,
                λ_residual=λ_res,
                λ_divergence=λ_div,
                checkpoint_dir="results/lambda_$(λ_res)_$(λ_div)"
            )
            
            model, state = train_residual_unet(config=config)
            
            push!(results, (λ_res, λ_div, state.best_loss))
            
            println("  Best Val Loss: $(state.best_loss)")
        end
    end
    
    # Find Best
    best_idx = argmin([r[3] for r in results])
    best_λ_res, best_λ_div, best_loss = results[best_idx]
    
    println("\n" * "="^80)
    println("BESTE LAMBDA-KOMBINATION:")
    println("  λ_residual = $best_λ_res")
    println("  λ_divergence = $best_λ_div")
    println("  Val Loss = $best_loss")
    println("="^80)
    
    return results
end

# =============================================================================
# 4. STANDARD RUN
# =============================================================================

"""
Standard Training Run (wird beim Include ausgeführt)
"""
function run_standard_training()
    println("\n" * "="^80)
    println("STANDARD TRAINING RUN")
    println("="^80)
    
    # Check GPU
    gpu_available = check_gpu_availability()
    
    # Config
    config = create_experiment_config(
        n_samples=100,
        crystal_range=1:10,
        epochs=50,
        batch_size=4,
        use_stream_function=false
    )
    
    # Train
    model, state = train_residual_unet(config=config)
    
    return model, state, config
end

# =============================================================================
# 5. INTERACTIVE MODE
# =============================================================================

"""
Zeigt interaktive Optionen an
"""
function show_interactive_menu()
    println("\n" * "="^80)
    println("RESIDUAL LEARNING - INTERAKTIVES MENÜ")
    println("="^80)
    println()
    println("Verfügbare Experimente:")
    println()
    println("run_baseline_experiment()")
    println("    → Training ohne Stream Function")
    println()
    println("run_stream_function_experiment()")
    println("    → Training mit Stream Function")
    println()
    println("run_quick_test()")
    println("    → Schneller Test (20 Samples, 10 Epochen)")
    println()
    println("run_lambda_tuning_experiment()")
    println("    → Automatisches Lambda-Tuning")
    println()
    println("run_standard_training()")
    println("    → Standard Setup (100 Samples, 50 Epochen)")
    println()
    println("=" ^80)
    println()
    println("Eigene Konfiguration:")
    println("   julia> config = create_experiment_config(...)")
    println("   julia> model, state = train_residual_unet(config=config)")
    println()
    println("Module testen:")
    println("   julia> test_stokes_single_sphere()")
    println("   julia> test_stream_function_layer()")
    println("   julia> test_residual_unet_shapes()")
    println()
end

# =============================================================================
# 6. AUTO-EXECUTION MODE
# =============================================================================


if abspath(PROGRAM_FILE) == @__FILE__
    println("\n Direkter Aufruf erkannt - Starte Standard Training")
    model, state, config = run_standard_training()
else
    # Interactive Mode
    show_interactive_menu()
end

# =============================================================================
# 7. EXPORT WICHTIGER FUNKTIONEN
# =============================================================================

# Damit sie direkt verfügbar sind
export train_residual_unet
export create_experiment_config
export run_baseline_experiment
export run_stream_function_experiment
export run_quick_test
export run_lambda_tuning_experiment
export run_standard_training

println()
println("="^80)
println("MAIN GELADEN - BEREIT FÜR TRAINING!")
println("="^80)
run_quick_test()