# =============================================================================
# RESIDUAL LEARNING - TRAINING LOOP
# =============================================================================
# Komplette Training-Pipeline für ResidualUNet

using Flux
using Optimisers
using BSON
using Statistics
using Printf
using Dates
using Random  # Für randperm()

println("Residual Training wird geladen...")

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

"""
    TrainingConfig

Zentrale Konfiguration für Training.

Alle Parameter an einem Ort für einfaches Tuning.
"""
struct TrainingConfig
    # Dataset
    n_samples::Int
    crystal_range::UnitRange{Int}
    train_split::Float64
    
    # Training
    epochs::Int
    batch_size::Int
    learning_rate::Float32
    
    # Loss Weights
    λ_residual::Float32
    λ_divergence::Float32
    use_stream_function::Bool
    
    # Optimization
    optimizer_type::Symbol  # :adam, :adamw
    weight_decay::Float32
    
    # Early Stopping
    patience::Int
    min_improvement::Float32
    
    # Checkpointing
    save_every::Int
    checkpoint_dir::String
    
    # GPU
    use_gpu::Bool
end

"""
Default Training Config
"""
function default_training_config()
    return TrainingConfig(
        # Dataset
        100,          # n_samples
        1:10,         # crystal_range
        0.8,          # train_split
        
        # Training
        50,           # epochs
        4,            # batch_size
        0.001f0,      # learning_rate
        
        # Loss
        0.01f0,       # λ_residual
        0.1f0,        # λ_divergence
        false,        # use_stream_function
        
        # Optimizer
        :adam,        # optimizer_type
        0.0f0,        # weight_decay
        
        # Early Stopping
        15,           # patience
        1e-4,         # min_improvement
        
        # Checkpointing
        10,           # save_every
        "checkpoints", # checkpoint_dir
        
        # GPU
        true          # use_gpu
    )
end

# =============================================================================
# TRAINING STATE
# =============================================================================

"""
    TrainingState

Hält Training-Zustand für Monitoring und Checkpointing.
"""
mutable struct TrainingState
    epoch::Int
    best_loss::Float32
    epochs_without_improvement::Int
    train_losses::Vector{Float32}
    val_losses::Vector{Float32}
    learning_rates::Vector{Float32}
    
    # Komponenten-Losses
    velocity_losses::Vector{Float32}
    residual_penalties::Vector{Float32}
    divergence_losses::Vector{Float32}
end

function TrainingState()
    return TrainingState(
        0,              # epoch
        Inf32,          # best_loss
        0,              # epochs_without_improvement
        Float32[],      # train_losses
        Float32[],      # val_losses
        Float32[],      # learning_rates
        Float32[],      # velocity_losses
        Float32[],      # residual_penalties
        Float32[]       # divergence_losses
    )
end

# =============================================================================
# HAUPTTRAINING-FUNKTION
# =============================================================================

"""
    train_residual_unet(; config=default_training_config())

Haupttraining-Funktion für ResidualUNet.

# Workflow
1. Generate Dataset
2. Train/Val Split
3. Setup Model & Optimizer
4. Training Loop mit Early Stopping
5. Checkpointing

# Returns
- `Tuple`: (model, state, losses)
"""
function train_residual_unet(; config::TrainingConfig=default_training_config())
    
    println("\n" * "="^80)
    println("RESIDUAL LEARNING TRAINING")
    println("="^80)
    println("Start: $(now())")
    println()
    
    # Print Config
    print_config(config)
    
    # 1. Dataset Generation
    println("\n DATASET GENERATION")
    println("-"^60)
    
    dataset = generate_dataset(
        config.n_samples,
        crystal_range=config.crystal_range,
        verbose=true
    )
    
    # Train/Val Split
    n_train = floor(Int, length(dataset) * config.train_split)
    train_data = dataset[1:n_train]
    val_data = dataset[n_train+1:end]
    
    println("Train Samples: $(length(train_data))")
    println("Val Samples: $(length(val_data))")
    
    # 2. Model Setup
    println("\n  MODEL SETUP")
    println("-"^60)
    
    # Grid info (von erstem Sample)
    _, _, _, x_vec, z_vec = train_data[1]
    Δx = Float32(x_vec[2] - x_vec[1])
    Δz = Float32(z_vec[2] - z_vec[1])
    
    model = create_residual_unet(
        use_stream_function=config.use_stream_function,
        Δx=Δx,
        Δz=Δz
    )
    
    println("Modell erstellt:")
    println("  Stream Function: $(config.use_stream_function)")
    println("  Parameters: $(sum(length, Flux.params(model)))")
    
    # 3. Optimizer Setup
    println("\n  OPTIMIZER SETUP")
    println("-"^60)
    
    if config.optimizer_type == :adam
        opt = Flux.Adam(config.learning_rate)
    elseif config.optimizer_type == :adamw
        opt = Flux.AdamW(config.learning_rate, (0.9, 0.999), config.weight_decay)
    else
        error("Unbekannter Optimizer: $(config.optimizer_type)")
    end
    
    opt_state = Flux.setup(opt, model)
    
    println("Optimizer: $(config.optimizer_type)")
    println("Learning Rate: $(config.learning_rate)")
    
    # 4. Device Setup
    device = config.use_gpu ? gpu : cpu
    model = device(model)
    
    println("Device: $(config.use_gpu ? "GPU" : "CPU")")
    
    # 5. Training State
    state = TrainingState()
    
    # 6. Checkpointing
    mkpath(config.checkpoint_dir)
    
    # 7. TRAINING LOOP
    println("\n TRAINING START")
    println("="^80)
    
    for epoch in 1:config.epochs
        state.epoch = epoch
        
        println("\n Epoch $epoch/$(config.epochs)")
        println("-"^60)
        
        # Training
        train_loss, train_components = train_epoch!(
            model, train_data, opt_state, config, device
        )
        
        # Validation
        val_loss, val_components = validate_epoch(
            model, val_data, config, device
        )
        
        # Update State
        push!(state.train_losses, train_loss)
        push!(state.val_losses, val_loss)
        push!(state.learning_rates, config.learning_rate)
        push!(state.velocity_losses, train_components[1])
        push!(state.residual_penalties, train_components[2])
        push!(state.divergence_losses, train_components[3])
        
        # Print Progress
        @printf "Train Loss: %.6f | Val Loss: %.6f\n" train_loss val_loss
        @printf "  Velocity: %.6f | Residual: %.6f | Divergence: %.6f\n" train_components...
        
        # Early Stopping Check
        if val_loss < state.best_loss - config.min_improvement
            state.best_loss = val_loss
            state.epochs_without_improvement = 0
            
            # Save Best Model
            save_checkpoint(model, state, config, "best_model.bson")
            println("Neues Best Model gespeichert!")
        else
            state.epochs_without_improvement += 1
            println("Kein Improvement ($(state.epochs_without_improvement)/$(config.patience))")
        end
        
        # Early Stopping
        if state.epochs_without_improvement >= config.patience
            println("\n Early Stopping nach $epoch Epochen")
            break
        end
        
        # Regular Checkpointing
        if epoch % config.save_every == 0
            save_checkpoint(model, state, config, "checkpoint_epoch_$epoch.bson")
            println("Checkpoint gespeichert")
        end
    end
    
    # 8. Final Save
    println("\n" * "="^80)
    println("TRAINING ABGESCHLOSSEN")
    println("="^80)
    
    save_checkpoint(model, state, config, "final_model.bson")
    
    println("Best Val Loss: $(state.best_loss)")
    println("Final Epoch: $(state.epoch)")
    println("Models gespeichert in: $(config.checkpoint_dir)/")
    
    return cpu(model), state
end

# =============================================================================
# TRAINING EPOCH
# =============================================================================

"""
    train_epoch!(model, train_data, opt_state, config, device)

Training für eine Epoche.
"""
function train_epoch!(model, train_data, opt_state, config, device)
    
    model = Flux.trainmode!(model)
    
    epoch_losses = Float32[]
    epoch_vel_losses = Float32[]
    epoch_res_penalties = Float32[]
    epoch_div_losses = Float32[]
    
    # Shuffle Data
    shuffled_indices = randperm(length(train_data))
    
    # Batch Loop
    for i in 1:config.batch_size:length(train_data)
        batch_indices = shuffled_indices[i:min(i+config.batch_size-1, length(train_data))]
        
        if length(batch_indices) < 2
            continue  # Skip kleine Batches
        end
        
        # Create Batch
        batch_samples = [train_data[idx] for idx in batch_indices]
        phase_batch, velocity_batch, crystal_params_batch, stats_batch = create_batch(batch_samples)
        
        # Move to Device
        phase_batch = device(phase_batch)
        velocity_batch = device(velocity_batch)
        
        # Grid Info
        _, _, _, x_vec, z_vec = batch_samples[1]
        
        # Forward & Loss
        loss_val, grads = Flux.withgradient(model) do m
            v_pred, v_stokes, Δv = m(phase_batch, crystal_params_batch, x_vec, z_vec, stats_batch)
            
            total_loss, vel_loss, res_penalty, div_loss = residual_loss(
                v_pred, v_stokes, Δv, velocity_batch;
                λ_residual=config.λ_residual,
                λ_divergence=config.λ_divergence,
                use_stream_function=config.use_stream_function
            )
            
            return total_loss
        end
        
        # Update
        Flux.update!(opt_state, model, grads[1])
        
        # Track total loss
        push!(epoch_losses, loss_val)
        
        # Track Komponenten (außerhalb Gradient-Kontext)
        # Berechne Komponenten außerhalb von withgradient
        v_pred_track, v_stokes_track, Δv_track = model(phase_batch, crystal_params_batch, x_vec, z_vec, stats_batch)
        _, vel, res, div = residual_loss(
            v_pred_track, v_stokes_track, Δv_track, velocity_batch;
            λ_residual=config.λ_residual,
            λ_divergence=config.λ_divergence,
            use_stream_function=config.use_stream_function
        )
        push!(epoch_vel_losses, vel)
        push!(epoch_res_penalties, res)
        push!(epoch_div_losses, div)
    end
    
    avg_loss = mean(epoch_losses)
    avg_components = (
        mean(epoch_vel_losses),
        mean(epoch_res_penalties),
        mean(epoch_div_losses)
    )
    
    return avg_loss, avg_components
end

# =============================================================================
# VALIDATION EPOCH
# =============================================================================

"""
    validate_epoch(model, val_data, config, device)

Validation für eine Epoche.
"""
function validate_epoch(model, val_data, config, device)
    
    model = Flux.testmode!(model)
    
    val_losses = Float32[]
    val_components = [Float32[], Float32[], Float32[]]
    
    for i in 1:config.batch_size:length(val_data)
        batch_indices = i:min(i+config.batch_size-1, length(val_data))
        
        if length(batch_indices) < 2
            continue
        end
        
        batch_samples = [val_data[idx] for idx in batch_indices]
        phase_batch, velocity_batch, crystal_params_batch, stats_batch = create_batch(batch_samples)
        
        phase_batch = device(phase_batch)
        velocity_batch = device(velocity_batch)
        
        _, _, _, x_vec, z_vec = batch_samples[1]
        
        # Forward (no gradients)
        v_pred, v_stokes, Δv = model(phase_batch, crystal_params_batch, x_vec, z_vec, stats_batch)
        
        total_loss, vel_loss, res_penalty, div_loss = residual_loss(
            v_pred, v_stokes, Δv, velocity_batch;
            λ_residual=config.λ_residual,
            λ_divergence=config.λ_divergence,
            use_stream_function=config.use_stream_function
        )
        
        push!(val_losses, total_loss)
        push!(val_components[1], vel_loss)
        push!(val_components[2], res_penalty)
        push!(val_components[3], div_loss)
    end
    
    avg_loss = mean(val_losses)
    avg_components = (mean.(val_components)...,)
    
    return avg_loss, avg_components
end

# =============================================================================
# CHECKPOINTING
# =============================================================================

"""
    save_checkpoint(model, state, config, filename)

Speichert Modell und Training State.
"""
function save_checkpoint(model, state, config, filename)
    path = joinpath(config.checkpoint_dir, filename)
    
    checkpoint = Dict(
        "model_state" => Flux.state(cpu(model)),
        "training_state" => state,
        "config" => config,
        "timestamp" => now()
    )
    
    BSON.@save path checkpoint
end

"""
    load_checkpoint(path)

Lädt Checkpoint.
"""
function load_checkpoint(path)
    BSON.@load path checkpoint
    return checkpoint
end

# =============================================================================
# UTILITIES
# =============================================================================

"""
    print_config(config::TrainingConfig)

Gibt Konfiguration formatiert aus.
"""
function print_config(config)
    println("TRAINING CONFIGURATION")
    println("-"^60)
    println("Dataset:")
    println("  Samples: $(config.n_samples)")
    println("  Crystals: $(config.crystal_range)")
    println("  Train/Val: $(config.train_split*100)% / $((1-config.train_split)*100)%")
    println("")
    println("Training:")
    println("  Epochs: $(config.epochs)")
    println("  Batch Size: $(config.batch_size)")
    println("  Learning Rate: $(config.learning_rate)")
    println("")
    println("Loss Weights:")
    println("  λ_residual: $(config.λ_residual)")
    println("  λ_divergence: $(config.λ_divergence)")
    println("  Stream Function: $(config.use_stream_function)")
    println("")
    println("Optimization:")
    println("  Optimizer: $(config.optimizer_type)")
    println("  Early Stopping Patience: $(config.patience)")
    println("")
end

# =============================================================================
# QUICK TRAINING TEST
# =============================================================================

"""
    quick_training_test(; n_samples=20, epochs=5)

Schneller Test mit wenigen Samples.
"""
function quick_training_test(; n_samples=20, epochs=5)
    config = TrainingConfig(
        n_samples, 1:3, 0.8,
        epochs, 2, 0.001f0,
        0.01f0, 0.1f0, false,
        :adam, 0.0f0,
        5, 1e-4,
        2, "test_checkpoints",
        false  # CPU für schnellen Test
    )
    
    model, state = train_residual_unet(config=config)
    
    println("\n Quick Training Test abgeschlossen!")
    return model, state
end

# =============================================================================
# MODUL-INFO
# =============================================================================

println("Residual Training geladen!")
println("   - Vollständiger Training Loop")
println("   - Early Stopping")
println("   - Checkpointing")
println("   - Train/Val Split")
println("")
println("Wichtige Funktionen:")
println("   - train_residual_unet(config=...) - Hauptfunktion")
println("   - default_training_config() - Standard-Konfiguration")
println("   - quick_training_test() - Schneller Test")
println("")
println("Schnellstart:")
println("   julia> model, state = train_residual_unet()")
println("   julia> quick_training_test()  # Für Tests")