##################################
# Training.jl - Optimized
##################################
module Training

include("Model.jl")
using Flux
using Flux: onehotbatch, logitcrossentropy, gpu, cpu
using Statistics
using LinearAlgebra
using Optimisers
using Optimisers: cpu  # Falls in Ihrer Version verfügbar
using CUDA
using .Model
using BSON: @save, @load
using ProgressMeter

# ===================================================
# Hilfsfunktion: Schichtweises Debugging
# ===================================================
"""
    debug_forward_pass(model, x; verbose=true)

Führt das Eingabetensor `x` Layer-für-Layer durch `model` und gibt nach jedem Layer
min/max/NaN/Inf-Checks aus. Wenn eine Schicht NaN oder Inf liefert, bricht die Funktion ab
und gibt das fehlerhafte `x` zurück. 

Setzen Sie `verbose=false`, wenn Sie die Prints unterdrücken möchten.
"""
function debug_forward_pass(model, x; verbose=true)
    for (layer_idx, layer) in enumerate(model)
        x = layer(x)
        if verbose
            println("  Layer $layer_idx -> min=", minimum(x), 
                    ", max=", maximum(x),
                    ", anyNaN=", any(isnan, x),
                    ", anyInf=", any(isinf, x))
        end
        if any(isnan, x) || any(isinf, x)
            if verbose
                println("   Exploded in layer $layer_idx!")
            end
            return x
        end
    end
    return x
end


# Fixed weighted loss function to avoid scalar indexing
function weighted_loss_fn(model, x, y, class_weights=nothing)
    pred = model(x)
    
    if isnothing(class_weights)
        return logitcrossentropy(pred, y)
    else
        # Apply class weights to loss - avoid scalar indexing
        # Create weight tensor with same shape as y
        weight_shape = size(y)
        weights = CUDA.zeros(Float32, weight_shape)
        
        # Use broadcasting instead of loops and indexing
        # First create weight matrix by class
        class_weights_gpu = gpu(class_weights)
        
        # Apply weights using broadcasting operations
        loss = logitcrossentropy(pred, y; agg=identity)
        
        # Create a temporary tensor for class indices
        class_indices = reshape(1:size(y, 3), 1, 1, :, 1)
        
        # Create a mask for each class (this avoids scalar indexing)
        # For each class c, create a binary mask where y[:,:,c,:] would be 1
        for c in 1:axes(y,3)
            class_mask = y[:,:,c,:] .> 0
            weights = weights .+ class_mask .* class_weights_gpu[c]
        end
        
        # Handle zeros in weights to avoid division by zero
        non_zero_mask = weights .> 0
        weights = weights .+ (1f-5) .* (1.0f0 .- non_zero_mask)
        
        # Apply weights
        weighted_loss = loss .* weights
        
        # Sum and normalize
        total_loss = sum(weighted_loss) 
        total_weights = sum(weights)
        
        return total_loss / total_weights
    end
end

# Basic loss function without weighting
function loss_fn(model, x, y)
    pred = model(x)
    return logitcrossentropy(pred, y)
end

# Learning rate scheduler
function cosine_annealing(initial_lr, epoch, total_epochs)
    # Cosine annealing learning rate schedule
    return initial_lr * (1 + cos(π * epoch / total_epochs)) / 2
end

function constant_lr(initial_lr, epoch, total_epochs)
    return initial_lr
end

# Checkpoint saving function
function save_checkpoint(model, opt_state, epoch, loss, filename)
    model_cpu = cpu(model)
    # opt_state_cpu = cpu(opt_state)  # Nur falls in Ihrer Version möglich; sonst weglassen

    @save filename model=model_cpu epoch=epoch loss=loss
    println("Saved checkpoint (model only) to $filename")
end

# Checkpoint loading function
function load_checkpoint(filename)
    variables = BSON.load(filename)
    model = gpu(variables[:model])
    # opt_state = Optimisers.gpu(variables[:opt_state]) # Nur falls Sie opt_state abgespeichert haben
    epoch = variables[:epoch]
    loss = variables[:loss]
    return model, epoch, loss
end

# Optimized one-hot encoding for batches
function batch_one_hot(batch, num_classes)
    batch_int = Int.(selectdim(batch, 3, 1))
    return gpu(Float32.(permutedims(onehotbatch(batch_int, 0:(num_classes-1)), (2,3,1,4))))
end

# ===================================================
# Standard precision training function
# ===================================================
function train_unet(model, train_data, num_epochs, learning_rate, output_channels; 
                  validation_data=nothing, checkpoint_dir="checkpoints", 
                  checkpoint_freq=5, class_weights=nothing)

    mkpath(checkpoint_dir)
    opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), model)
    
    loss_over_time = Float32[]
    val_loss_over_time = Float32[]
    best_val_loss = Inf32
    
    for epoch in 1:num_epochs
        #current_lr = cosine_annealing(learning_rate, epoch, num_epochs)
        current_lr = constant_lr(learning_rate, epoch, num_epochs)
        println("====== Epoch $epoch (LR: $current_lr) ======")
        
        total_loss = 0f0
        batch_count = 0
        
        p = Progress(length(train_data), 1, "Training epoch $epoch...")
        
        for (batch_idx, (input_batch, mask_batch)) in enumerate(train_data)
            # One-hot encode labels
            mask_batch_oh = batch_one_hot(mask_batch, output_channels)
            # Move data to GPU
            input_batch = gpu(input_batch)
            
            # Compute gradients and loss
            ∇model = gradient(m -> weighted_loss_fn(m, input_batch, mask_batch_oh, class_weights), model)[1]
            batch_loss = weighted_loss_fn(model, input_batch, mask_batch_oh, class_weights)
            
            # Update model parameters
            opt_state, model = Optimisers.update!(opt_state, model, ∇model)
            
            total_loss += batch_loss
            batch_count += 1
            
            next!(p; showvalues = [(:batch, batch_idx), (:loss, batch_loss)])
            CUDA.reclaim()
        end
        
        avg_loss = total_loss / batch_count
        push!(loss_over_time, avg_loss)
        println("Epoch $epoch finished. Average Loss: $avg_loss")
        
        if !isnothing(validation_data)
            val_loss = evaluate_model(model, validation_data, output_channels, class_weights)
            push!(val_loss_over_time, val_loss)
            println("Validation Loss: $val_loss")
            
            if val_loss < best_val_loss
                best_val_loss = val_loss
                save_checkpoint(model, opt_state, epoch, val_loss, 
                                joinpath(checkpoint_dir, "best_model.bson"))
                println("New best model saved!")
            end
        end
        
        if epoch % checkpoint_freq == 0
            save_checkpoint(model, opt_state, epoch, avg_loss, 
                            joinpath(checkpoint_dir, "checkpoint_epoch$(epoch).bson"))
        end
        
        println("--------------------------------------------------")
    end
    
    save_checkpoint(model, opt_state, num_epochs, loss_over_time[end], 
                   joinpath(checkpoint_dir, "final_model.bson"))
    
    return model, loss_over_time, val_loss_over_time
end

# ===================================================
# Evaluation function
# ===================================================
function evaluate_model(model, validation_data, output_channels, class_weights=nothing)
    total_loss = 0f0
    batch_count = 0
    
    model_mode = Flux.trainmode(model, false)
    
    for (input_batch, mask_batch) in validation_data
        mask_batch_oh = batch_one_hot(mask_batch, output_channels)
        
        input_batch = gpu(input_batch)
        
        batch_loss = weighted_loss_fn(model, input_batch, mask_batch_oh, class_weights)
        
        total_loss += batch_loss
        batch_count += 1
        
        CUDA.reclaim()
    end
    
    Flux.trainmode(model, true)
    
    return total_loss / batch_count
end

# ===================================================
# Calculate class weights based on frequency
# ===================================================
function calculate_class_weights(dataset, num_classes)
    class_counts = zeros(Float32, num_classes)
    
    for (_, label) in dataset
        flat_label = reshape(label, :)
        for c in 0:(num_classes-1)
            class_counts[c+1] += count(==(c), flat_label)
        end
    end
    
    total_pixels = sum(class_counts)
    class_frequencies = class_counts ./ total_pixels
    
    class_frequencies = max.(class_frequencies, 1e-6)
    
    weights = 1.0 ./ class_frequencies
    
    weights = weights ./ mean(weights)
    
    return weights
end

# ===================================================
# safe_update!
# ===================================================
function safe_update!(opt_state, model, ∇model)
    has_bad_gradients = false
    for p in Flux.params(∇model)
        if any(isnan, p) || any(isinf, p)
            has_bad_gradients = true
            break
        end
    end
    
    if has_bad_gradients
        println("Warning: NaN or Inf detected in gradients. Skipping update.")
        return opt_state, model
    else
        ∇model_clipped = Flux.clip(∇model, -1.0, 1.0)
        return Optimisers.update!(opt_state, model, ∇model_clipped)
    end
end

end # module Training
