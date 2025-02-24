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

# Loss function with class weighting option
function weighted_loss_fn(model, x, y, class_weights=nothing)
    pred = model(x)
    
    if isnothing(class_weights)
        return logitcrossentropy(pred, y)
    else
        # Apply class weights to loss
        weights = CUDA.zeros(Float32, size(y))
        for c in 1:size(y, 3)
            weights[:, :, c, :] .= class_weights[c]
        end
        
        # Element-wise weighted loss
        loss = logitcrossentropy(pred, y; agg=identity)
        weighted_loss = loss .* weights
        
        # Sum and normalize
        return sum(weighted_loss) / sum(weights)
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

# Mixed precision training function
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
            # Move data to GPU with fp16
            input_batch_fp16 = gpu(Float16.(input_batch))
            
            # One-hot encode labels (keep in fp32 for stability)
            mask_batch_oh = batch_one_hot(mask_batch, output_channels)
            
            # Forward pass in fp16
            output_fp16 = model_fp16(input_batch_fp16)
            
            # Compute loss in fp32 for numerical stability
            output_fp32 = Float32.(output_fp16)
            loss = weighted_loss_fn(x -> Float32.(model_fp16(Float16.(x))), 
                                  input_batch, mask_batch_oh, class_weights)
            
            # Compute gradients with mixed precision
            ∇model = gradient(m -> weighted_loss_fn(m, input_batch_fp16, mask_batch_oh, class_weights), 
                             model_fp16)[1]
            
            # Update model parameters
            opt_state, model_fp16 = Optimisers.update!(opt_state, model_fp16, ∇model)
            
            # Track metrics
            total_loss += Float32(loss)
            batch_count += 1
            
            # Update progress bar
            next!(p; showvalues = [(:batch, batch_idx), (:loss, loss)])
            
            # Free GPU memory
            CUDA.reclaim()
        end
        
        # Calculate average loss
        avg_loss = total_loss / batch_count
        push!(loss_over_time, avg_loss)
        println("Epoch $epoch finished. Average Loss: $avg_loss")
        
        # Validation if provided
        if !isnothing(validation_data)
            # Convert to fp32 for evaluation
            model_fp32 = Model.f32(model_fp16)
            val_loss = evaluate_model(model_fp32, validation_data, output_channels, class_weights)
            push!(val_loss_over_time, val_loss)
            println("Validation Loss: $val_loss")
            
            # Save best model
            if val_loss < best_val_loss
                best_val_loss = val_loss
                save_checkpoint(model_fp16, opt_state, epoch, val_loss, 
                               joinpath(checkpoint_dir, "best_model.bson"))
                println("New best model saved!")
            end
            
            # Convert back to fp16 for training
            model_fp16 = Model.f16(model_fp32)
        end
        
        # Save checkpoint periodically
        if epoch % checkpoint_freq == 0
            save_checkpoint(model_fp16, opt_state, epoch, avg_loss, 
                           joinpath(checkpoint_dir, "checkpoint_epoch$(epoch).bson"))
        end
        
        println("--------------------------------------------------")
    end
    
    # Convert back to fp32 for final model
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

end # module Training