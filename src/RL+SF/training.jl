# =============================================================================
# TRAINING MODULE - VOLLSTÄNDIG MIT RESIDUAL LEARNING
# =============================================================================

using Flux
using Flux: mse
using CUDA
using Optimisers
using BSON: @save, @load
using Statistics
using Random
using StatsBase

include("gpu_utils.jl")

# =============================================================================
# KONFIGURATION
# =============================================================================

struct TrainingConfig
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
end

function create_training_config(;
    learning_rate = 0.001f0,
    num_epochs = 50,
    batch_size = 4,
    checkpoint_dir = "checkpoints",
    save_every_n_epochs = 10,
    use_gpu = false,
    validation_split = 0.1f0,
    early_stopping_patience = 10,
    lambda_physics_initial = 0.01f0,
    lambda_physics_final = 0.1f0,
    physics_warmup_epochs = 10
)
    return TrainingConfig(
        learning_rate, num_epochs, batch_size, checkpoint_dir,
        save_every_n_epochs, use_gpu, validation_split, early_stopping_patience,
        lambda_physics_initial, lambda_physics_final, physics_warmup_epochs
    )
end

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

function compute_divergence(velocity_pred)
    vx = velocity_pred[:, :, 1, :]
    vz = velocity_pred[:, :, 2, :]
    H, W, B = size(vx)
    
    dvx_dx = (vx[2:end, :, :] .- vx[1:end-1, :, :])
    dvz_dz = (vz[:, 2:end, :] .- vz[:, 1:end-1, :])
    
    dvx_dx_padded = vcat(zeros(eltype(vx), 1, W, B), dvx_dx)
    dvz_dz_padded = hcat(zeros(eltype(vz), H, 1, B), dvz_dz)
    
    return dvx_dx_padded .+ dvz_dz_padded
end

function split_dataset(dataset, validation_split=0.1)
    n_total = length(dataset)
    n_val = max(1, Int(round(n_total * validation_split)))
    n_train = n_total - n_val
    
    if n_train < 1
        return dataset, dataset[1:1]
    end
    
    shuffled_indices = randperm(n_total)
    train_indices = shuffled_indices[1:n_train]
    val_indices = shuffled_indices[n_train+1:end]
    
    return dataset[train_indices], dataset[val_indices]
end

function evaluate_model_safe(model, val_dataset, target_resolution; check_physics=true)
    if length(val_dataset) == 0
        return Inf32
    end
    
    total_data_loss = 0.0f0
    total_physics_loss = 0.0f0
    n_batches = 0
    
    val_batch_size = min(2, length(val_dataset))
    max_batches = min(5, length(val_dataset))
    
    for i in 1:val_batch_size:max_batches
        end_idx = min(i + val_batch_size - 1, length(val_dataset))
        if i > length(val_dataset)
            break
        end
        
        batch_samples = val_dataset[i:min(end_idx, length(val_dataset))]
        
        try
            phase_batch, velocity_batch, successful = create_adaptive_batch(
                batch_samples, target_resolution, verbose=false
            )
            
            if successful == 0
                continue
            end
            
            phase_batch = cpu(phase_batch)
            velocity_batch = cpu(velocity_batch)
            model = cpu(model)
            
            prediction = model(phase_batch)
            batch_data_loss = mse(prediction, velocity_batch)
            
            if check_physics
                divergence = compute_divergence(prediction)
                batch_physics_loss = mean(abs2, divergence)
            else
                batch_physics_loss = 0.0f0
            end
            
            if isfinite(batch_data_loss)
                total_data_loss += batch_data_loss
                total_physics_loss += batch_physics_loss
                n_batches += 1
            end
        catch e
            println("Validation Batch $i fehlgeschlagen: $e")
            continue
        end
    end
    
    avg_data_loss = n_batches > 0 ? total_data_loss / n_batches : Inf32
    avg_physics_loss = n_batches > 0 ? total_physics_loss / n_batches : Inf32
    
    return avg_data_loss + 0.1f0 * avg_physics_loss
end

# =============================================================================
# STANDARD PHYSICS-INFORMED TRAINING
# =============================================================================

function physics_informed_loss(prediction, velocity_batch; lambda_physics=0.1f0)
    data_loss = mse(prediction, velocity_batch)
    divergence = compute_divergence(prediction)
    physics_loss = mean(abs2, divergence)
    total_loss = data_loss + lambda_physics * physics_loss
    return total_loss, data_loss, physics_loss
end

function train_velocity_unet_safe(
    model, 
    dataset, 
    target_resolution;
    config = create_training_config()
)
    println("=== STARTE PHYSICS-INFORMED UNET TRAINING ===")
    
    use_gpu = config.use_gpu && check_gpu_availability()
    if use_gpu
        println("GPU-Training aktiviert")
        model = safe_gpu(model)
    else
        println("CPU-Training aktiviert")
        model = safe_cpu(model)
    end
    
    println("Konfiguration:")
    println("  Epochen: $(config.num_epochs)")
    println("  Lernrate: $(config.learning_rate)")
    println("  Batch-Größe: $(config.batch_size)")
    println("  Dataset-Größe: $(length(dataset))")
    
    if length(dataset) == 0
        error("Dataset ist leer!")
    end
    
    mkpath(config.checkpoint_dir)
    train_dataset, val_dataset = split_dataset(dataset, config.validation_split)
    println("  Training: $(length(train_dataset)) Samples")
    println("  Validation: $(length(val_dataset)) Samples")
    
    opt_state = Optimisers.setup(Optimisers.Adam(config.learning_rate), model)
    
    train_losses = Float32[]
    val_losses = Float32[]
    physics_losses = Float32[]
    best_val_loss = Inf32
    patience_counter = 0
    
    for epoch in 1:config.num_epochs
        println("\n--- Epoche $epoch/$(config.num_epochs) ---")
        
        if epoch <= config.physics_warmup_epochs
            lambda_physics = config.lambda_physics_initial + 
                            (config.lambda_physics_final - config.lambda_physics_initial) * 
                            (epoch - 1) / config.physics_warmup_epochs
        else
            lambda_physics = config.lambda_physics_final
        end
        
        println("  Lambda Physics: $(round(lambda_physics, digits=4))")
        
        epoch_loss = 0.0f0
        epoch_data_loss = 0.0f0
        epoch_physics_loss = 0.0f0
        n_batches = 0
        
        max_training_batches = min(20, length(train_dataset))
        
        for i in 1:config.batch_size:max_training_batches
            end_idx = min(i + config.batch_size - 1, length(train_dataset))
            if i > length(train_dataset)
                break
            end
            
            batch_samples = train_dataset[i:min(end_idx, length(train_dataset))]
            
            try
                phase_batch, velocity_batch, successful = create_adaptive_batch(
                    batch_samples, target_resolution, verbose=false
                )
                
                if successful == 0
                    continue
                end
                
                if use_gpu
                    phase_batch = safe_gpu(phase_batch)
                    velocity_batch = safe_gpu(velocity_batch)
                else
                    phase_batch = safe_cpu(phase_batch)
                    velocity_batch = safe_cpu(velocity_batch)
                end
                
                function loss_fn(m)
                    pred = m(phase_batch)
                    total_loss, _, _ = physics_informed_loss(
                        pred, velocity_batch, 
                        lambda_physics=lambda_physics
                    )
                    return total_loss
                end
                
                loss_val = nothing
                grads = nothing
                
                try
                    loss_val, grads = Flux.withgradient(loss_fn, model)
                catch gpu_error
                    if use_gpu && occursin("CUDA", string(gpu_error))
                        println("  GPU-Fehler, wechsle zu CPU")
                        model = safe_cpu(model)
                        phase_batch = safe_cpu(phase_batch)
                        velocity_batch = safe_cpu(velocity_batch)
                        opt_state = Optimisers.setup(Optimisers.Adam(config.learning_rate), model)
                        loss_val, grads = Flux.withgradient(loss_fn, model)
                        use_gpu = false
                    else
                        rethrow(gpu_error)
                    end
                end
                
                pred_for_logging = model(phase_batch)
                _, data_loss_val, physics_loss_val = physics_informed_loss(
                    pred_for_logging, velocity_batch,
                    lambda_physics=lambda_physics
                )
                
                if grads[1] !== nothing
                    opt_state, model = Optimisers.update!(opt_state, model, grads[1])
                end
                
                if isfinite(loss_val)
                    epoch_loss += loss_val
                    epoch_data_loss += data_loss_val
                    epoch_physics_loss += physics_loss_val
                    n_batches += 1
                end
                
                if i % (config.batch_size * 5) == 1
                    println("  Batch $i: Loss = $(round(Float32(loss_val), digits=6))")
                end
                
            catch e
                println("  Fehler bei Batch $i: $e")
                continue
            end
            
            if use_gpu && i % 10 == 0
                gpu_memory_cleanup()
            end
        end
        
        if use_gpu
            gpu_memory_cleanup()
        else
            GC.gc()
        end
        
        avg_train_loss = n_batches > 0 ? Float32(epoch_loss / n_batches) : Inf32
        avg_physics_loss = n_batches > 0 ? Float32(epoch_physics_loss / n_batches) : Inf32
        
        push!(train_losses, avg_train_loss)
        push!(physics_losses, avg_physics_loss)
        
        val_loss = evaluate_model_safe(model, val_dataset, target_resolution)
        push!(val_losses, val_loss)
        
        println("Epoche $epoch: Train=$(round(avg_train_loss, digits=6)), Val=$(round(val_loss, digits=6))")
        
        if val_loss < best_val_loss
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = joinpath(config.checkpoint_dir, "best_model.bson")
            @save best_model_path model
            println("  ✓ Bestes Modell gespeichert!")
        else
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience
                println("  Early Stopping!")
                break
            end
        end
        
        if epoch % config.save_every_n_epochs == 0
            checkpoint_path = joinpath(config.checkpoint_dir, "checkpoint_epoch_$(epoch).bson")
            @save checkpoint_path model epoch train_losses val_losses physics_losses
        end
    end
    
    model = safe_cpu(model)
    final_path = joinpath(config.checkpoint_dir, "final_model.bson")
    @save final_path model train_losses val_losses physics_losses
    
    println("\n✓ TRAINING ABGESCHLOSSEN")
    return model, train_losses, val_losses, physics_losses
end

# =============================================================================
# RESIDUAL LEARNING LOSS FUNCTIONS
# =============================================================================

function loss_residual(
    model, 
    phase, 
    velocity_target; 
    lambda_residual = 0.01f0, 
    lambda_sparsity = 0.001f0,
    lambda_physics = 0.1f0
)
    # Forward Pass mit allen Komponenten
    v_pred, v_stokes, Δv = forward_with_components(model, phase)
    
    # 1. Haupt-Loss: Gesamtgeschwindigkeit gegen Ground Truth
    velocity_loss = mse(v_pred, velocity_target)
    
    # 2. Regularisierung: Residuen klein halten
    residual_penalty = lambda_residual * mean(abs2, Δv)
    
    # 3. Sparsity Loss: Fördert spärliche Residuen
    sparsity_loss = lambda_sparsity * mean(abs, Δv)
    
    # 4. Divergenz-Loss (Massenerhaltung als Soft Constraint)
    divergence = compute_divergence(v_pred)
    physics_loss = mean(abs2, divergence)
    
    total_loss = velocity_loss + residual_penalty + sparsity_loss + 
                 lambda_physics * physics_loss
    
    components = Dict(
        "velocity_loss" => velocity_loss,
        "residual_penalty" => residual_penalty,
        "sparsity_loss" => sparsity_loss,
        "physics_loss" => physics_loss,
        "total_loss" => total_loss
    )
    
    return total_loss, components
end

function loss_residual_adaptive(
    model, phase, velocity_target, epoch;
    lambda_residual_initial = 0.001f0,
    lambda_residual_final = 0.01f0,
    lambda_sparsity_initial = 0.0001f0,
    lambda_sparsity_final = 0.001f0,
    lambda_physics = 0.1f0,
    residual_warmup = 10,
    sparsity_warmup = 15
)
    if epoch <= residual_warmup
        lambda_residual = lambda_residual_initial + 
                         (lambda_residual_final - lambda_residual_initial) * 
                         (epoch / residual_warmup)
    else
        lambda_residual = lambda_residual_final
    end
    
    if epoch <= sparsity_warmup
        lambda_sparsity = lambda_sparsity_initial + 
                         (lambda_sparsity_final - lambda_sparsity_initial) * 
                         (epoch / sparsity_warmup)
    else
        lambda_sparsity = lambda_sparsity_final
    end
    
    total_loss, components = loss_residual(
        model, phase, velocity_target;
        lambda_residual = lambda_residual,
        lambda_sparsity = lambda_sparsity,
        lambda_physics = lambda_physics
    )
    
    components["lambda_residual"] = lambda_residual
    components["lambda_sparsity"] = lambda_sparsity
    
    return total_loss, components
end

# =============================================================================
# RESIDUAL HELPER FUNCTIONS
# =============================================================================

function monitor_residual_training(model::ResidualUNet, samples, target_resolution)
    residual_magnitudes = []
    stokes_magnitudes = []
    
    for (phase_file, lamem_file) in samples[1:min(5, length(samples))]
        try
            phase_batch, velocity_batch, successful = create_adaptive_batch(
                [(phase_file, lamem_file)],
                target_resolution,
                verbose=false
            )
            
            if successful == 0
                continue
            end
            
            v_total, v_stokes, Δv = forward_with_components(model, phase_batch)
            
            push!(residual_magnitudes, mean(sqrt.(Δv[:,:,1,:].^2 + Δv[:,:,2,:].^2)))
            push!(stokes_magnitudes, mean(sqrt.(v_stokes[:,:,1,:].^2 + v_stokes[:,:,2,:].^2)))
        catch e
            continue
        end
    end
    
    if !isempty(residual_magnitudes)
        println("    Residuum: $(round(mean(residual_magnitudes), digits=6))")
        println("    Stokes: $(round(mean(stokes_magnitudes), digits=6))")
        if mean(stokes_magnitudes) > 0
            println("    Ratio: $(round(mean(residual_magnitudes) / mean(stokes_magnitudes), digits=4))")
        end
    end
end

function compute_residual_statistics(model::ResidualUNet, samples, target_resolution)
    stats = Dict{String, Float32}()
    residual_mags = []
    sparsity_counts = []
    
    for (phase_file, lamem_file) in samples
        try
            phase_batch, velocity_batch, successful = create_adaptive_batch(
                [(phase_file, lamem_file)],
                target_resolution,
                verbose=false
            )
            
            if successful == 0
                continue
            end
            
            v_total, v_stokes, Δv = forward_with_components(model, phase_batch)
            
            res_mag = mean(sqrt.(Δv[:,:,1,:].^2 + Δv[:,:,2,:].^2))
            push!(residual_mags, res_mag)
            
            threshold = 0.01f0
            sparsity = count(abs.(Δv) .< threshold) / length(Δv)
            push!(sparsity_counts, sparsity)
        catch e
            continue
        end
    end
    
    if !isempty(residual_mags)
        stats["mean_residual_magnitude"] = mean(residual_mags)
        stats["std_residual_magnitude"] = std(residual_mags)
        stats["mean_sparsity"] = mean(sparsity_counts)
        stats["median_residual_magnitude"] = median(residual_mags)
    end
    
    return stats
end

# =============================================================================
# RESIDUAL UNET TRAINING
# =============================================================================

function train_residual_unet(
    model::ResidualUNet,
    dataset,
    target_resolution;
    config = create_training_config(),
    use_adaptive = true,
    monitor_residuals = true
)
    println("\n" * "="^80)
    println("RESIDUAL UNET TRAINING")
    println("="^80)
    println("Modus: $(model.use_stream_function ? "Stream Function" : "Direct Residual")")
    println("Adaptive Loss: $use_adaptive")
    
    use_gpu = config.use_gpu && check_gpu_availability()
    if use_gpu
        println("GPU aktiviert")
        model = safe_gpu(model)
    else
        println("CPU aktiviert")
        model = safe_cpu(model)
    end
    
    train_dataset, val_dataset = split_dataset(dataset, config.validation_split)
    println("Training: $(length(train_dataset)) Samples")
    println("Validation: $(length(val_dataset)) Samples")
    
    if length(train_dataset) == 0
        error("Training dataset leer!")
    end
    
    mkpath(config.checkpoint_dir)
    opt_state = Optimisers.setup(Optimisers.Adam(config.learning_rate), model)
    
    train_losses = Float32[]
    val_losses = Float32[]
    components_history = []
    best_val_loss = Inf32
    patience_counter = 0
    
    println("\n" * "="^80)
    println("START TRAINING")
    println("="^80)
    
    for epoch in 1:config.num_epochs
        println("\n--- Epoche $epoch/$(config.num_epochs) ---")
        
        epoch_loss = 0.0f0
        epoch_components = Dict{String, Float32}()
        n_batches = 0
        
        max_training_batches = min(20, length(train_dataset))
        
        for batch_idx in 1:config.batch_size:max_training_batches
            batch_end = min(batch_idx + config.batch_size - 1, max_training_batches)
            if batch_idx > length(train_dataset)
                break
            end
            
            batch_samples = train_dataset[batch_idx:min(batch_end, length(train_dataset))]
            
            try
                phase_batch, velocity_batch, successful = create_adaptive_batch(
                    batch_samples, 
                    target_resolution,
                    verbose=false
                )
                
                if successful == 0
                    println("  Batch $batch_idx: Keine erfolgreichen Samples")
                    continue
                end
                
                if use_gpu
                    phase_batch = safe_gpu(phase_batch)
                    velocity_batch = safe_gpu(velocity_batch)
                else
                    phase_batch = safe_cpu(phase_batch)
                    velocity_batch = safe_cpu(velocity_batch)
                end
                
                function loss_fn(m)
                    if use_adaptive
                        loss, comps = loss_residual_adaptive(
                            m, phase_batch, velocity_batch, epoch
                        )
                    else
                        loss, comps = loss_residual(m, phase_batch, velocity_batch)
                    end
                    
                    for (key, val) in comps
                        if !haskey(epoch_components, key)
                            epoch_components[key] = 0.0f0
                        end
                        epoch_components[key] += Float32(val)
                    end
                    
                    return loss
                end
                
                loss_val = nothing
                grads = nothing
                
                try
                    loss_val, grads = Flux.withgradient(loss_fn, model)
                catch gpu_error
                    if use_gpu && occursin("CUDA", string(gpu_error))
                        println("  GPU-Fehler → CPU Fallback")
                        model = safe_cpu(model)
                        phase_batch = safe_cpu(phase_batch)
                        velocity_batch = safe_cpu(velocity_batch)
                        opt_state = Optimisers.setup(Optimisers.Adam(config.learning_rate), model)
                        loss_val, grads = Flux.withgradient(loss_fn, model)
                        use_gpu = false
                    else
                        rethrow(gpu_error)
                    end
                end
                
                if !isnothing(grads[1])
                    Optimisers.update!(opt_state, model, grads[1])
                end
                
                if isfinite(loss_val)
                    epoch_loss += loss_val
                    n_batches += 1
                end
                
                if batch_idx % (config.batch_size * 3) == 1
                    println("  Batch $batch_idx: Loss = $(round(Float32(loss_val), digits=6))")
                end
                
            catch e
                println("  Fehler Batch $batch_idx: $e")
                continue
            end
            
            if use_gpu && batch_idx % 5 == 0
                gpu_memory_cleanup()
            end
        end
        
        if use_gpu
            gpu_memory_cleanup()
        else
            GC.gc()
        end
        
        avg_train_loss = n_batches > 0 ? Float32(epoch_loss / n_batches) : Inf32
        push!(train_losses, avg_train_loss)
        
        for key in keys(epoch_components)
            epoch_components[key] /= n_batches
        end
        push!(components_history, epoch_components)
        
        println("  Validation läuft...")
        val_loss = evaluate_model_safe(model, val_dataset, target_resolution, check_physics=false)
        push!(val_losses, val_loss)
        
        println("\nEpoche $epoch:")
        println("  Train: $(round(avg_train_loss, digits=6))")
        println("  Val: $(round(val_loss, digits=6))")
        
        if !isempty(epoch_components)
            println("  Komponenten:")
            for (key, val) in epoch_components
                println("    $key: $(round(val, digits=6))")
            end
        end
        
        if monitor_residuals && epoch % 5 == 0
            println("\n  === RESIDUUM ANALYSE ===")
            monitor_residual_training(model, train_dataset[1:min(10, length(train_dataset))], target_resolution)
        end
        
        if val_loss < best_val_loss
            best_val_loss = val_loss
            patience_counter = 0
            best_model_path = joinpath(config.checkpoint_dir, "best_residual_model.bson")
            @save best_model_path model
            println("  ✓ Bestes Modell gespeichert!")
        else
            patience_counter += 1
            println("  Keine Verbesserung (Patience: $patience_counter/$(config.early_stopping_patience))")
            if patience_counter >= config.early_stopping_patience
                println("\n  Early Stopping!")
                break
            end
        end
        
        if epoch % config.save_every_n_epochs == 0
            checkpoint_path = joinpath(config.checkpoint_dir, "residual_epoch_$epoch.bson")
            @save checkpoint_path model epoch train_losses val_losses components_history
            println("  ✓ Checkpoint gespeichert")
        end
    end
    
    println("\n" * "="^80)
    println("TRAINING ABGESCHLOSSEN")
    println("="^80)
    
    if monitor_residuals
        println("\n=== FINALE STATISTIK ===")
        stats = compute_residual_statistics(
            model, 
            train_dataset[1:min(50, length(train_dataset))],
            target_resolution
        )
        for (key, val) in stats
            println("$key: $(round(val, digits=6))")
        end
    end
    
    final_model_path = joinpath(config.checkpoint_dir, "final_residual_model.bson")
    @save final_model_path model train_losses val_losses components_history
    println("\n✓ Finales Modell: $final_model_path")
    
    return model, train_losses, val_losses, components_history
end

# =============================================================================
# MODEL LOADING
# =============================================================================

function load_trained_model(model_path::String)
    println("Lade Modell: $model_path")
    
    if !isfile(model_path)
        error("Modelldatei nicht gefunden: $model_path")
    end
    
    model_dict = BSON.load(model_path)
    
    for key in [:model, :best_model, :final_model]
        if haskey(model_dict, key)
            println("Modell unter '$key' gefunden")
            return model_dict[key]
        end
    end
    
    for (key, value) in model_dict
        if isa(value, Flux.Chain) || hasproperty(value, :layers) || 
           occursin("UNet", string(typeof(value)))
            println("Modell unter '$key' gefunden")
            return value
        end
    end
    
    error("Kein Modell in BSON gefunden")
end

println("✓ Training Module geladen!")
println("\nVerfügbare Funktionen:")
println("  Standard UNet:")
println("    - train_velocity_unet_safe(model, dataset, resolution)")
println("  Residual UNet:")
println("    - train_residual_unet(model, dataset, resolution)")
println("    - loss_residual(model, phase, velocity)")
println("    - loss_residual_adaptive(model, phase, velocity, epoch)")
println("  Utils:")
println("    - load_trained_model(path)")
println("    - compute_divergence(velocity_field)")