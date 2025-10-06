# =============================================================================
# 3D TRAINING MODULE - ERSETZT DEIN 2D TRAINING.JL
# =============================================================================
# Datei: training.jl (3D Version)

using Flux
using Flux: mse
using CUDA
using Optimisers
using BSON: @save
using Statistics
using Random

include("gpu_utils.jl")  # GPU Utilities laden

# =============================================================================
# 3D TRAINING CONFIGURATION - ERWEITERT VON DEINER VERSION
# =============================================================================

"""
3D Training-Konfiguration (erweitert von deiner TrainingConfig)
"""
struct TrainingConfig3D
    learning_rate::Float32
    num_epochs::Int
    batch_size::Int
    checkpoint_dir::String
    save_every_n_epochs::Int
    use_gpu::Bool
    validation_split::Float32
    early_stopping_patience::Int
    lambda_physics_initial::Float32      
    lambda_physics_final::Float32        
    physics_warmup_epochs::Int
    # NEU FÜR 3D:
    memory_efficient_mode::Bool          # Reduzierte Batch-Größe für 3D
    gradient_accumulation_steps::Int     # Gradient Accumulation für große 3D Modelle
    mixed_precision::Bool                # Float16 für 3D Speicher-Effizienz
end

"""
Erstellt 3D Training-Konfiguration (angepasst von deiner create_training_config)
"""
function create_training_config_3d(;
    learning_rate = 0.0005f0,           # Langsamere Lernrate für 3D
    num_epochs = 30,                    # Weniger Epochen für 3D Tests
    batch_size = 1,                     # Sehr kleine Batches für 3D!
    checkpoint_dir = "checkpoints_3d",
    save_every_n_epochs = 5,            # Häufigere Checkpoints
    use_gpu = false,                    # Konservativer Start
    validation_split = 0.2f0,           # Mehr Validation für 3D
    early_stopping_patience = 8,        # Früher stoppen
    lambda_physics_initial = 0.01f0,
    lambda_physics_final = 0.05f0,      # Kleinerer Physics-Weight für 3D
    physics_warmup_epochs = 5,
    # 3D spezifische Parameter:
    memory_efficient_mode = true,
    gradient_accumulation_steps = 4,    # Simuliere größere Batches
    mixed_precision = false             # Erstmal deaktiviert für Stabilität
)
    return TrainingConfig3D(
        learning_rate, num_epochs, batch_size, checkpoint_dir,
        save_every_n_epochs, use_gpu, validation_split, early_stopping_patience,
        lambda_physics_initial, lambda_physics_final, physics_warmup_epochs,
        memory_efficient_mode, gradient_accumulation_steps, mixed_precision
    )
end

# =============================================================================
# 3D PHYSICS-INFORMED LOSS - ERWEITERT VON DEINER VERSION
# =============================================================================

"""
3D Divergenz-Berechnung (erweitert von deiner 2D compute_divergence)
Berechnet ∂vx/∂x + ∂vy/∂y + ∂vz/∂z = 0 für 3D Kontinuitätsgleichung
"""
function compute_divergence_3d(velocity_pred)
    # velocity_pred hat Shape: (W, H, D, 3, B) für 3D
    vx = velocity_pred[:, :, :, 1, :]
    vy = velocity_pred[:, :, :, 2, :]
    vz = velocity_pred[:, :, :, 3, :]
    
    W, H, D, B = size(vx)
    
    # 3D Finite Differenzen (vereinfacht für GPU-Kompatibilität)
    dvx_dx = vx[2:end, :, :, :] .- vx[1:end-1, :, :, :]
    dvy_dy = vy[:, 2:end, :, :] .- vy[:, 1:end-1, :, :]
    dvz_dz = vz[:, :, 2:end, :] .- vz[:, :, 1:end-1, :]
    
    # Padding für einheitliche Größe
    dvx_dx_padded = cat(zeros(eltype(vx), 1, H, D, B), dvx_dx, dims=1)
    dvy_dy_padded = cat(zeros(eltype(vy), W, 1, D, B), dvy_dy, dims=2)
    dvz_dz_padded = cat(zeros(eltype(vz), W, H, 1, B), dvz_dz, dims=3)
    
    # 3D Divergenz
    divergence = dvx_dx_padded .+ dvy_dy_padded .+ dvz_dz_padded
    
    return divergence
end

"""
3D Physics-Informed Loss (erweitert von deiner physics_informed_loss)
"""
function physics_informed_loss_3d(prediction, velocity_batch; lambda_physics=0.05f0)
    # Standard MSE für Daten-Treue
    data_loss = mse(prediction, velocity_batch)
    
    # 3D Physikalischer Constraint: Divergenz sollte null sein
    divergence = compute_divergence_3d(prediction)
    physics_loss = mean(abs2, divergence)  # L2-Norm der 3D Divergenz
    
    # Kombinierte Verlustfunktion
    total_loss = data_loss + lambda_physics * physics_loss
    
    return total_loss, data_loss, physics_loss
end

# =============================================================================
# 3D DATASET UTILITIES - ERWEITERT VON DEINEM CODE
# =============================================================================

"""
3D Version deiner split_dataset Funktion
"""
function split_dataset_3d(dataset, validation_split=0.2)
    n_total = length(dataset)
    n_val = max(1, Int(round(n_total * validation_split)))
    n_train = n_total - n_val
    
    if n_train < 1
        return dataset, dataset[1:1]
    end
    
    shuffled_indices = randperm(n_total)
    train_indices = shuffled_indices[1:n_train]
    val_indices = shuffled_indices[n_train+1:end]
    
    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]
    
    return train_dataset, val_dataset
end

"""
3D Memory-effiziente Batch-Erstellung
"""
function create_3d_batch_safe(samples, target_resolution; use_gpu=false, verbose=false)
    if isempty(samples)
        error("Keine 3D Samples für Batch!")
    end
    
    batch_size = length(samples)
    W, H, D = target_resolution
    
    # Pre-allocate 3D Tensors
    phase_batch = zeros(Float32, W, H, D, 1, batch_size)
    velocity_batch = zeros(Float32, W, H, D, 3, batch_size)
    
    successful = 0
    
    for (i, sample) in enumerate(samples)
        try
            # Entpacke 3D Sample (erweitert von deinem 2D Format)
            if length(sample) >= 11  # x, y, z, phase, vx, vy, vz, ..., v_stokes
                x, y, z, phase, vx, vy, vz = sample[1:7]
                v_stokes = sample[end]
                
                # 3D Preprocessing (deine neue Funktion)
                phase_tensor, velocity_tensor = preprocess_lamem_sample_3d(
                    x, y, z, phase, vx, vy, vz, v_stokes; 
                    target_resolution=target_resolution
                )
                
                # In Batch einfügen
                phase_batch[:,:,:,:,i] = cpu(phase_tensor)
                velocity_batch[:,:,:,:,i] = cpu(velocity_tensor)
                
                successful += 1
                
            else
                if verbose
                    println("  3D Sample $i hat ungültiges Format: $(length(sample)) Elemente")
                end
            end
            
        catch e
            if verbose
                println("  Fehler bei 3D Sample $i: $e")
            end
            continue
        end
    end
    
    if successful == 0
        error("Keine 3D Samples erfolgreich verarbeitet!")
    end
    
    # Trimme auf erfolgreiche Samples
    if successful < batch_size
        phase_batch = phase_batch[:,:,:,:,1:successful]
        velocity_batch = velocity_batch[:,:,:,:,1:successful]
    end
    
    # GPU Transfer
    if use_gpu && CUDA.functional()
        try
            phase_batch = gpu(phase_batch)
            velocity_batch = gpu(velocity_batch)
        catch gpu_error
            println("  GPU Transfer fehlgeschlagen, verwende CPU: $gpu_error")
            phase_batch = cpu(phase_batch)
            velocity_batch = cpu(velocity_batch)
        end
    end
    
    return phase_batch, velocity_batch, successful
end

# =============================================================================
# 3D EVALUATION - ERWEITERT VON DEINER VERSION
# =============================================================================

"""
3D Version deiner evaluate_model_safe Funktion
"""
function evaluate_model_3d_safe(model, val_dataset, target_resolution; check_physics=true)
    if length(val_dataset) == 0
        return Inf32
    end
    
    model_cpu = cpu(model)
    total_loss = 0.0f0
    total_data_loss = 0.0f0
    total_physics_loss = 0.0f0
    n_batches = 0
    
    # Kleinere Evaluation-Batches für 3D
    max_eval_samples = min(5, length(val_dataset))  # Nur 5 Samples für 3D Evaluation
    
    for i in 1:1:max_eval_samples  # Einzelne Samples für Speicher-Sicherheit
        sample_batch = val_dataset[i:i]
        
        try
            phase_batch, velocity_batch, successful = create_3d_batch_safe(
                sample_batch, target_resolution, use_gpu=false, verbose=false
            )
            
            if successful == 0
                continue
            end
            
            # Evaluation
            pred = model_cpu(phase_batch)
            
            if check_physics
                total_loss_val, data_loss_val, physics_loss_val = physics_informed_loss_3d(
                    pred, velocity_batch, lambda_physics=0.05f0
                )
                total_data_loss += data_loss_val
                total_physics_loss += physics_loss_val
            else
                total_loss_val = mse(pred, velocity_batch)
                total_data_loss += total_loss_val
            end
            
            total_loss += total_loss_val
            n_batches += 1
            
            # Memory cleanup für 3D
            phase_batch = nothing
            velocity_batch = nothing
            pred = nothing
            GC.gc()
            
        catch e
            println("  3D Evaluation-Fehler: $e")
            continue
        end
    end
    
    # Durchschnittliche Losses
    avg_loss = n_batches > 0 ? total_loss / n_batches : Inf32
    
    return avg_loss
end

# =============================================================================
# 3D HAUPTTRAINING-FUNKTION - ERSETZT DEINE 2D VERSION
# =============================================================================

"""
3D Version deiner train_velocity_unet_safe Funktion
"""
function train_velocity_unet_3d_safe(
    model, 
    dataset, 
    target_resolution;
    config = create_training_config_3d()
)
    
    println("=== STARTE 3D PHYSICS-INFORMED UNET TRAINING ===")
    
    # GPU-Check (konservativer für 3D)
    use_gpu = config.use_gpu && check_gpu_availability()
    
    if use_gpu
        println("GPU-Training aktiviert (3D - experimentell)")
        try
            model = safe_gpu(model)
        catch e
            println("GPU-Transfer fehlgeschlagen, verwende CPU: $e")
            use_gpu = false
            model = safe_cpu(model)
        end
    else
        println("CPU-Training aktiviert (empfohlen für 3D)")
        model = safe_cpu(model)
    end
    
    println("3D Konfiguration:")
    println("  Epochen: $(config.num_epochs)")
    println("  Lernrate: $(config.learning_rate)")
    println("  Batch-Größe: $(config.batch_size)")
    println("  Dataset-Größe: $(length(dataset))")
    println("  Target Resolution: $(target_resolution[1])×$(target_resolution[2])×$(target_resolution[3])")
    println("  Physics Lambda: $(config.lambda_physics_initial) → $(config.lambda_physics_final)")
    println("  Memory Efficient Mode: $(config.memory_efficient_mode)")
    
    # Validiere 3D Dataset
    if length(dataset) == 0
        error("3D Dataset ist leer!")
    end
    
    # Checkpoint-Verzeichnis erstellen
    mkpath(config.checkpoint_dir)
    
    # 3D Dataset aufteilen
    train_dataset, val_dataset = split_dataset_3d(dataset, config.validation_split)
    println("  3D Training: $(length(train_dataset)) Samples")
    println("  3D Validation: $(length(val_dataset)) Samples")
    
    # Optimizer setup
    opt_state = Optimisers.setup(Optimisers.Adam(config.learning_rate), model)
    
    # 3D Training-Geschichte
    train_losses = Float32[]
    val_losses = Float32[]
    physics_losses = Float32[]
    best_val_loss = Inf32
    patience_counter = 0
    
    # 3D Training-Loop
    for epoch in 1:config.num_epochs
        println("\n--- 3D Epoche $epoch/$(config.num_epochs) ---")
        
        # Berechne lambda_physics für diese Epoche
        if epoch <= config.physics_warmup_epochs
            lambda_physics = config.lambda_physics_initial + 
                            (config.lambda_physics_final - config.lambda_physics_initial) * 
                            (epoch - 1) / config.physics_warmup_epochs
        else
            lambda_physics = config.lambda_physics_final
        end
        
        println("  Lambda Physics (3D): $(round(lambda_physics, digits=4))")
        
        # 3D Training
        epoch_loss = 0.0f0
        epoch_data_loss = 0.0f0
        epoch_physics_loss = 0.0f0
        n_batches = 0
        
        # Kleine Batches für 3D Memory Management
        max_training_samples = min(10, length(train_dataset))  # Reduziert für 3D
        
        for i in 1:config.batch_size:max_training_samples
            end_idx = min(i + config.batch_size - 1, length(train_dataset))
            
            if i > length(train_dataset)
                break
            end
            
            batch_samples = train_dataset[i:min(end_idx, length(train_dataset))]
            
            try
                # 3D Batch erstellen
                phase_batch, velocity_batch, successful = create_3d_batch_safe(
                    batch_samples, target_resolution, use_gpu=use_gpu, verbose=false
                )
                
                if successful == 0
                    continue
                end
                
                # 3D Loss-Funktion definieren
                function loss_fn_3d(m)
                    pred = m(phase_batch)
                    total_loss, _, _ = physics_informed_loss_3d(
                        pred, velocity_batch, 
                        lambda_physics=lambda_physics
                    )
                    return total_loss
                end
                
                # 3D Gradient-Berechnung
                loss_val = nothing
                grads = nothing
                
                try
                    loss_val, grads = Flux.withgradient(loss_fn_3d, model)
                catch gpu_error
                    if use_gpu && occursin("CUDA", string(gpu_error))
                        println("  3D GPU-Fehler, wechsle zu CPU")
                        model = safe_cpu(model)
                        phase_batch = cpu(phase_batch)
                        velocity_batch = cpu(velocity_batch)
                        opt_state = Optimisers.setup(Optimisers.Adam(config.learning_rate), model)
                        loss_val, grads = Flux.withgradient(loss_fn_3d, model)
                        use_gpu = false
                    else
                        rethrow(gpu_error)
                    end
                end
                
                # Logging für 3D
                pred_for_logging = model(phase_batch)
                _, data_loss_val, physics_loss_val = physics_informed_loss_3d(
                    pred_for_logging, velocity_batch,
                    lambda_physics=lambda_physics
                )
                
                # Parameter-Update
                if grads[1] !== nothing
                    opt_state, model = Optimisers.update!(opt_state, model, grads[1])
                    
                    epoch_loss += loss_val
                    epoch_data_loss += data_loss_val
                    epoch_physics_loss += physics_loss_val
                    n_batches += 1
                end
                
                # 3D Memory Cleanup
                phase_batch = nothing
                velocity_batch = nothing
                pred_for_logging = nothing
                if use_gpu
                    CUDA.reclaim()
                end
                GC.gc()
                
            catch e
                println("  3D Batch-Fehler: $e")
                continue
            end
        end
        
        # Durchschnittliche Epoch-Losses
        avg_train_loss = n_batches > 0 ? epoch_loss / n_batches : Inf32
        avg_data_loss = n_batches > 0 ? epoch_data_loss / n_batches : Inf32
        avg_physics_loss = n_batches > 0 ? epoch_physics_loss / n_batches : Inf32
        
        push!(train_losses, avg_train_loss)
        push!(physics_losses, avg_physics_loss)
        
        # 3D Validation
        println("  3D Validation...")
        val_loss = evaluate_model_3d_safe(model, val_dataset, target_resolution)
        push!(val_losses, val_loss)
        
        # Logging
        println("  3D Training Loss: $(round(avg_train_loss, digits=6))")
        println("  3D Data Loss: $(round(avg_data_loss, digits=6))")
        println("  3D Physics Loss: $(round(avg_physics_loss, digits=6))")
        println("  3D Validation Loss: $(round(val_loss, digits=6))")
        
        # Early Stopping
        if val_loss < best_val_loss
            best_val_loss = val_loss
            patience_counter = 0
            
            # Speichere bestes 3D Modell
            best_model_path = joinpath(config.checkpoint_dir, "best_3d_model.bson")
            @save best_model_path model=cpu(model)
            println("  ✓ Neues bestes 3D Modell gespeichert")
        else
            patience_counter += 1
            println("  Patience: $patience_counter/$(config.early_stopping_patience)")
        end
        
        if patience_counter >= config.early_stopping_patience
            println("  Early Stopping nach $epoch Epochen")
            break
        end
        
        # Checkpoint speichern
        if epoch % config.save_every_n_epochs == 0
            checkpoint_path = joinpath(config.checkpoint_dir, "3d_model_epoch_$epoch.bson")
            @save checkpoint_path model=cpu(model)
            println("  ✓ 3D Checkpoint gespeichert: Epoche $epoch")
        end
        
        # Memory Cleanup nach jeder Epoche
        GC.gc()
        if use_gpu
            CUDA.reclaim()
        end
    end
    
    println("\n=== 3D TRAINING ABGESCHLOSSEN ===")
    println("Beste Validation Loss: $(round(best_val_loss, digits=6))")
    
    # Lade bestes Modell
    best_model_path = joinpath(config.checkpoint_dir, "best_3d_model.bson")
    if isfile(best_model_path)
        BSON.@load best_model_path model
        println("✓ Bestes 3D Modell geladen")
    end
    
    return model, train_losses, val_losses, physics_losses
end

# =============================================================================
# KOMPATIBILITÄTS-WRAPPER FÜR DEIN SYSTEM
# =============================================================================

"""
Wrapper für nahtlose Integration in dein bestehendes System
"""
function train_velocity_unet_safe(model, dataset, target_resolution; config=nothing)
    # Automatische Erkennung ob 2D oder 3D basierend auf target_resolution
    if length(target_resolution) == 3
        # 3D Training
        if config === nothing
            config = create_training_config_3d()
        end
        return train_velocity_unet_3d_safe(model, dataset, target_resolution; config=config)
    else
        error("2D Training nicht mehr unterstützt - verwende 3D Version")
    end
end

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
