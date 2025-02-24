##################################
# Training.jl - Optimized
##################################
module Training

include("Model.jl")
using Flux
using Flux: onehotbatch, logitcrossentropy, gpu
using Statistics
using LinearAlgebra
using Optimisers
using CUDA
using .Model
using BSON: @save, @load
using ProgressMeter

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
        for c in 1:size(y, 3)
            # Use broadcasting to set weights for this class
            class_mask = y[:,:,c,:] .> 0
            weights = weights + class_mask .* class_weights_gpu[c]
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

# Checkpoint saving function
function save_checkpoint(model, opt_state, epoch, loss, filename)
    model_cpu = cpu(model)
    opt_state_cpu = Optimisers.cpu(opt_state)
    @save filename model=model_cpu opt_state=opt_state_cpu epoch=epoch loss=loss
    println("Saved checkpoint to $filename")
end

# Checkpoint loading function
function load_checkpoint(filename)
    variables = BSON.load(filename)
    model = gpu(variables[:model])
    opt_state = Optimisers.gpu(variables[:opt_state])
    epoch = variables[:epoch]
    loss = variables[:loss]
    return model, opt_state, epoch, loss
end

# Optimized one-hot encoding for batches
function batch_one_hot(batch, num_classes)
    mask_batch_int = Int.(batch[:, :, 1, :])
    return gpu(Float32.(permutedims(onehotbatch(mask_batch_int, 0:(num_classes-1)), (2,3,1,4))))
end

# Standard precision training function
function train_unet(model, train_data, num_epochs, learning_rate, output_channels; 
                  validation_data=nothing, checkpoint_dir="checkpoints", 
                  checkpoint_freq=5, class_weights=nothing)
    
    # Create checkpoint directory if it doesn't exist
    mkpath(checkpoint_dir)
    
    # Setup optimizer
    opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), model)
    
    # Initialize metrics
    loss_over_time = Float32[]
    val_loss_over_time = Float32[]
    best_val_loss = Inf32
    
    for epoch in 1:num_epochs
        # Update learning rate with scheduler
        current_lr = cosine_annealing(learning_rate, epoch, num_epochs)
        println("====== Epoch $epoch (LR: $current_lr) ======")
        
        # Track epoch loss
        total_loss = 0f0
        batch_count = 0
        
        # Create progress meter
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
            
            # Track metrics
            total_loss += batch_loss
            batch_count += 1
            
            # Update progress bar
            next!(p; showvalues = [(:batch, batch_idx), (:loss, batch_loss)])
            
            # Free GPU memory
            CUDA.reclaim()
        end
        
        # Calculate average loss
        avg_loss = total_loss / batch_count
        push!(loss_over_time, avg_loss)
        println("Epoch $epoch finished. Average Loss: $avg_loss")
        
        # Validation if provided
        if !isnothing(validation_data)
            val_loss = evaluate_model(model, validation_data, output_channels, class_weights)
            push!(val_loss_over_time, val_loss)
            println("Validation Loss: $val_loss")
            
            # Save best model
            if val_loss < best_val_loss
                best_val_loss = val_loss
                save_checkpoint(model, opt_state, epoch, val_loss, 
                               joinpath(checkpoint_dir, "best_model.bson"))
                println("New best model saved!")
            end
        end
        
        # Save checkpoint periodically
        if epoch % checkpoint_freq == 0
            save_checkpoint(model, opt_state, epoch, avg_loss, 
                           joinpath(checkpoint_dir, "checkpoint_epoch$(epoch).bson"))
        end
        
        println("--------------------------------------------------")
    end
    
    # Save final model
    save_checkpoint(model, opt_state, num_epochs, loss_over_time[end], 
                   joinpath(checkpoint_dir, "final_model.bson"))
    
    return model, loss_over_time, val_loss_over_time
end

# Mixed precision training function with fixed scalar indexing issues
function train_unet_mixed_precision(model, train_data, num_epochs, learning_rate, output_channels;
    validation_data=nothing, checkpoint_dir="checkpoints", 
    checkpoint_freq=5, class_weights=nothing)
    # Create checkpoint directory if it doesn't exist
    mkpath(checkpoint_dir)

    # Convert model to fp16
    model_fp16 = Model.f16(model)

    # Setup optimizer (keep in fp32 for stability)
    opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), model_fp16)

    # Initialize metrics
    loss_over_time = Float32[]
    val_loss_over_time = Float32[]
    best_val_loss = Inf32

    # Disable scalar indexing for better performance
    CUDA.allowscalar(false)

    # Convert class weights to GPU if provided
    if !isnothing(class_weights)
        class_weights = gpu(Float32.(class_weights))
    end

    for epoch in 1:num_epochs
        # Update learning rate with scheduler
        current_lr = cosine_annealing(learning_rate, epoch, num_epochs)
        println("====== Epoch $epoch (LR: $current_lr, Mixed Precision) ======")

        # Track epoch loss
        total_loss = 0f0
        batch_count = 0

        # Create progress meter
        p = Progress(length(train_data), 1, "Training epoch $epoch...")

        for (batch_idx, (input_batch, mask_batch)) in enumerate(train_data)
            # Debug dimensions before any processing
            println("Batch $batch_idx - Input: $(size(input_batch)), Mask: $(size(mask_batch))")

            # Move data to GPU with fp16
            input_batch_fp16 = gpu(Float16.(input_batch))

            # One-hot encode labels (keep in fp32 for stability)
            mask_batch_oh = batch_one_hot(mask_batch, output_channels)
            println("One-hot encoded mask: $(size(mask_batch_oh))")

            # Check for NaN in forward pass
            output_fp16 = model_fp16(input_batch_fp16)
            if any(isnan, output_fp16) || any(isinf, output_fp16)
                println("Warning: NaN or Inf in model output. Skipping batch.")
                continue
            end

            

            # Check if dimensions match
            if size(output_fp16)[1:2] != size(mask_batch_oh)[1:2]
                println("Warning: Dimension mismatch between output $(size(output_fp16)[1:2]) and target $(size(mask_batch_oh)[1:2])")

                # Handle dimension mismatch by cropping to smaller size
                min_h = min(size(output_fp16)[1], size(mask_batch_oh)[1])
                min_w = min(size(output_fp16)[2], size(mask_batch_oh)[2])

                # Crop both to common dimensions
                output_fp16 = output_fp16[1:min_h, 1:min_w, :, :]
                mask_batch_oh = mask_batch_oh[1:min_h, 1:min_w, :, :]

                println("After cropping - Output: $(size(output_fp16)), Target: $(size(mask_batch_oh))")
            end

            # Compute loss and gradients with custom loss function that avoids scalar indexing
            loss = 0f0
            try
                # Compute loss in fp32 for numerical stability
                # Check loss value
                loss = logitcrossentropy(Float32.(output_fp16), mask_batch_oh)
                if isnan(loss) || isinf(loss)
                    println("Warning: Loss is NaN or Inf. Skipping batch.")
                    continue
                end

                # Compute gradients with mixed precision
                ∇model = gradient(m -> logitcrossentropy(Float32.(m(input_batch_fp16)), mask_batch_oh), 
                    model_fp16)[1]

                # Update model parameters
                opt_state, model_fp16 = Optimisers.update!(opt_state, model_fp16, ∇model)

                # Track metrics
                total_loss += Float32(loss)
                batch_count += 1
            catch e
                println("Error computing loss or gradients: ", e)
                # Skip this batch but continue training
                continue
            end
            
            # Update progress bar
            next!(p; showvalues = [(:batch, batch_idx), (:loss, loss)])

            # Free GPU memory
            CUDA.reclaim()
        end

        # Calculate average loss
        if batch_count > 0
            avg_loss = total_loss / batch_count
            push!(loss_over_time, avg_loss)
            println("Epoch $epoch finished. Average Loss: $avg_loss")
        else
            println("Epoch $epoch finished. No successful batches processed.")
            push!(loss_over_time, NaN32)
        end

        # Validation if provided
        if !isnothing(validation_data)
            # Convert to fp32 for evaluation
            CUDA.allowscalar(true)  # Temporarily allow scalar indexing for evaluation
            model_fp32 = Model.f32(model_fp16)
            val_loss = evaluate_model(model_fp32, validation_data, output_channels, class_weights)
            CUDA.allowscalar(false)  # Disable again

            push!(val_loss_over_time, val_loss)
            println("Validation Loss: $val_loss")

            # Save best model
            if val_loss < best_val_loss
                best_val_loss = val_loss
                CUDA.allowscalar(true)  # Allow scalar indexing for saving
                save_checkpoint(model_fp16, opt_state, epoch, val_loss, 
                    joinpath(checkpoint_dir, "best_model.bson"))
                CUDA.allowscalar(false)  # Disable again
                println("New best model saved!")
            end
        end

        # Save checkpoint periodically
        if epoch % checkpoint_freq == 0
            CUDA.allowscalar(true)  # Allow scalar indexing for saving
            save_checkpoint(model_fp16, opt_state, epoch, loss_over_time[end], 
                joinpath(checkpoint_dir, "checkpoint_epoch$(epoch).bson"))
            CUDA.allowscalar(false)  # Disable again
        end

        println("--------------------------------------------------")
    end

    # Convert back to fp32 for final model
    CUDA.allowscalar(true)  # Allow scalar indexing for final operations
    model_fp32 = Model.f32(model_fp16)

    # Save final model
    save_checkpoint(model_fp32, opt_state, num_epochs, loss_over_time[end], 
        joinpath(checkpoint_dir, "final_model.bson"))

    return model_fp32, loss_over_time, val_loss_over_time
end


# Evaluation function
function evaluate_model(model, validation_data, output_channels, class_weights=nothing)
    total_loss = 0f0
    batch_count = 0
    
    # Set to evaluation mode if model supports it
    model_mode = Flux.trainmode(model, false)
    
    for (input_batch, mask_batch) in validation_data
        # One-hot encode labels
        mask_batch_oh = batch_one_hot(mask_batch, output_channels)
        
        # Move data to GPU
        input_batch = gpu(input_batch)
        
        # Calculate loss
        batch_loss = weighted_loss_fn(model, input_batch, mask_batch_oh, class_weights)
        
        # Track metrics
        total_loss += batch_loss
        batch_count += 1
        
        # Free GPU memory
        CUDA.reclaim()
    end
    
    # Restore model mode
    Flux.trainmode(model, true)
    
    return total_loss / batch_count
end

# Calculate class weights based on frequency
function calculate_class_weights(dataset, num_classes)
    # Count class frequencies
    class_counts = zeros(Float32, num_classes)
    
    for (_, label) in dataset
        # Flatten label
        flat_label = reshape(label, :)
        
        # Count frequencies
        for c in 0:(num_classes-1)
            class_counts[c+1] += count(==(c), flat_label)
        end
    end
    
    # Calculate inverse frequency weights
    total_pixels = sum(class_counts)
    class_frequencies = class_counts ./ total_pixels
    
    # Avoid division by zero
    class_frequencies = max.(class_frequencies, 1e-6)
    
    # Inverse frequency weighting
    weights = 1.0 ./ class_frequencies
    
    # Normalize weights
    weights = weights ./ mean(weights)
    
    return weights
end

function safe_update!(opt_state, model, ∇model)
    # Check for NaN or Inf in gradients
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
        # Apply gradient clipping
        ∇model_clipped = Flux.clip(∇model, -1.0, 1.0)
        # Update model parameters
        return Optimisers.update!(opt_state, model, ∇model_clipped)
    end
end


end # module Training