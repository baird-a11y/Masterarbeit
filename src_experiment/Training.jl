##################################
# Training.jl - Advanced Training Strategies
##################################
module Training

include("Model.jl")
using Flux
using Flux: onehotbatch, logitcrossentropy, gpu, cpu, softmax
using Statistics
using LinearAlgebra
using Optimisers
using Optimisers: cpu
using CUDA
using .Model
using BSON: @save, @load
using ProgressMeter
using Random
using Dates

# ===================================================
# Advanced Learning Rate Schedulers
# ===================================================

"""
    OneCycleLR(max_lr, total_steps; pct_start=0.3, div_factor=25.0, final_div_factor=1e4)

Implements the 1cycle learning rate policy which consists of:
1. First increasing learning rate from initial_lr to max_lr
2. Then decreasing learning rate from max_lr to min_lr
3. Finally, decreasing learning rate even further from min_lr to final_lr

Parameters:
- max_lr: Maximum learning rate
- total_steps: Total number of training iterations (epochs * batches)
- pct_start: Percentage of cycle spent increasing learning rate
- div_factor: Divisor for the initial learning rate (initial_lr = max_lr / div_factor)
- final_div_factor: Divisor for the final learning rate (final_lr = initial_lr / final_div_factor)
"""
struct OneCycleLR
    initial_lr::Float32
    max_lr::Float32
    final_lr::Float32
    total_steps::Int
    pct_start::Float32
    values::Vector{Float32}
end

function OneCycleLR(max_lr, total_steps; pct_start=0.3f0, div_factor=25.0f0, final_div_factor=1.0f4)
    initial_lr = max_lr / div_factor
    final_lr = initial_lr / final_div_factor
    
    # Calculate the step sizes
    step_size_up = floor(Int, total_steps * pct_start)
    step_size_down = total_steps - step_size_up
    
    # Generate the learning rate schedule
    lr_schedule = zeros(Float32, total_steps)
    
    # Increasing phase
    for i in 1:step_size_up
        lr_schedule[i] = initial_lr + (max_lr - initial_lr) * i / step_size_up
    end
    
    # Decreasing phase
    for i in 1:step_size_down
        lr_schedule[step_size_up + i] = max_lr - (max_lr - initial_lr) * i / step_size_down
    end
    
    # Final decrease phase
    final_pct = 0.2f0  # Spend the last 20% of iterations decreasing to final_lr
    final_step_idx = floor(Int, total_steps * (1.0f0 - final_pct))
    
    for i in final_step_idx:total_steps
        t = (i - final_step_idx) / (total_steps - final_step_idx)
        lr_schedule[i] = initial_lr - (initial_lr - final_lr) * t
    end
    
    return OneCycleLR(initial_lr, max_lr, final_lr, total_steps, pct_start, lr_schedule)
end

# Get learning rate for a given step
function get_lr(schedule::OneCycleLR, step::Int)
    if step <= 0
        return schedule.initial_lr
    elseif step > schedule.total_steps
        return schedule.final_lr
    else
        return schedule.values[step]
    end
end

# Cosine annealing with warm restarts
function cosine_annealing_warm_restarts(initial_lr, epoch, total_epochs; T_0=10, T_mult=2)
    # This implementation follows the formula from the paper "SGDR: Stochastic Gradient Descent with Warm Restarts"
    # T_0: Initial cycle length
    # T_mult: Factor to increase cycle length after each restart
    
    # Calculate the current cycle and where we are in the cycle
    cycle = 0
    cycle_length = T_0
    epochs_so_far = 0
    
    while epochs_so_far + cycle_length <= epoch
        epochs_so_far += cycle_length
        cycle += 1
        cycle_length *= T_mult
    end
    
    # Calculate where we are in the current cycle
    t = epoch - epochs_so_far
    
    # Apply cosine annealing within this cycle
    return initial_lr * (1 + cos(π * t / cycle_length)) / 2
end

# ===================================================
# Advanced Loss Functions
# ===================================================

"""
    dice_coefficient(y_pred, y_true, smooth=1.0f0)

Calculate the Dice coefficient between predicted and true segmentation masks.
The Dice coefficient is a measure of overlap between two samples and is defined as
2*|X∩Y|/(|X|+|Y|) where X and Y are the predicted and true masks.

Parameters:
- y_pred: Predicted probabilities (after softmax)
- y_true: One-hot encoded ground truth
- smooth: Smoothing factor to avoid division by zero
"""
function dice_coefficient(y_pred, y_true, smooth=1.0f0)
    # Ensure y_pred has probabilities (apply softmax if needed)
    if size(y_pred, 3) > 1
        y_pred_probs = softmax(y_pred, dims=3)
    else
        y_pred_probs = sigmoid.(y_pred)
    end
    
    # Calculate intersection and union for each class
    dims = (1, 2, 4)  # Spatial and batch dimensions
    intersection = sum(y_pred_probs .* y_true, dims=dims)
    union = sum(y_pred_probs, dims=dims) + sum(y_true, dims=dims)
    
    # Calculate Dice coefficient for each class
    dice = (2.0f0 .* intersection .+ smooth) ./ (union .+ smooth)
    
    # Return mean over classes
    return mean(dice)
end

"""
    dice_loss(y_pred, y_true, smooth=1.0f0)

Calculate the Dice loss as 1 - dice_coefficient.
"""
function dice_loss(y_pred, y_true, smooth=1.0f0)
    return 1.0f0 - dice_coefficient(y_pred, y_true, smooth)
end

"""
    focal_loss(y_pred, y_true; gamma=2.0f0, alpha=0.25f0)

Focal Loss as proposed in "Focal Loss for Dense Object Detection" paper.
It addresses class imbalance by down-weighting easy examples.

Parameters:
- y_pred: Predicted logits
- y_true: One-hot encoded ground truth
- gamma: Focusing parameter for hard examples (higher gamma means more focus on hard examples)
- alpha: Weighting factor for positive class (helps with class imbalance)
"""
function focal_loss(y_pred, y_true; gamma=2.0f0, alpha=0.25f0)
    # Convert logits to probabilities
    probs = softmax(y_pred, dims=3)
    
    # Calculate focal weights for each class
    focal_weights = (1.0f0 .- probs) .^ gamma
    
    # Apply class weights if using alpha
    if alpha != 0.5f0
        weights = y_true .* alpha + (1.0f0 .- y_true) .* (1.0f0 - alpha)
        focal_weights = focal_weights .* weights
    end
    
    # Standard cross-entropy with focal weights
    ce = -y_true .* log.(probs .+ 1f-7)
    
    # Apply focal weights and sum across classes and batch
    weighted_ce = focal_weights .* ce
    
    # Return mean over all dimensions
    return mean(sum(weighted_ce, dims=3))
end

"""
    lovasz_loss(y_pred, y_true)

Lovász-Softmax loss for multi-class semantic segmentation.
Optimizes the Jaccard / IoU score directly.
Simplified implementation.
"""
function lovasz_loss(y_pred, y_true)
    # This is a simplified placeholder - a full implementation would be more complex
    # In practice, you would use a library implementation or implement the full algorithm
    # as described in the paper "The Lovász-Softmax loss: A tractable surrogate for the 
    # optimization of the intersection-over-union measure in neural networks"
    
    # For now, we'll use a weighted combination of Dice and CE which approximates IoU optimization
    return 0.5f0 * dice_loss(y_pred, y_true) + 0.5f0 * logitcrossentropy(y_pred, y_true)
end

"""
    combo_loss(y_pred, y_true; alpha=0.5f0, ce_ratio=0.5f0, focal_gamma=2.0f0)

Combination of different loss functions for better segmentation results.
Combines: Cross-entropy (or Focal Loss) and Dice Loss.

Parameters:
- y_pred: Predicted logits
- y_true: One-hot encoded ground truth
- alpha: Balance between dice and ce/focal
- ce_ratio: Ratio between CE and focal (1.0 = just CE, 0.0 = just focal)
- focal_gamma: Focusing parameter for focal loss
"""
function combo_loss(y_pred, y_true; alpha=0.5f0, ce_ratio=0.5f0, focal_gamma=2.0f0)
    dl = dice_loss(y_pred, y_true)
    
    if ce_ratio == 1.0f0
        ce = logitcrossentropy(y_pred, y_true)
        return alpha * dl + (1.0f0 - alpha) * ce
    elseif ce_ratio == 0.0f0
        fl = focal_loss(y_pred, y_true, gamma=focal_gamma)
        return alpha * dl + (1.0f0 - alpha) * fl
    else
        ce = logitcrossentropy(y_pred, y_true)
        fl = focal_loss(y_pred, y_true, gamma=focal_gamma)
        return alpha * dl + (1.0f0 - alpha) * (ce_ratio * ce + (1.0f0 - ce_ratio) * fl)
    end
end

# ===================================================
# Weighted Loss Functions
# ===================================================

"""
    weighted_combo_loss(model, x, y, class_weights=nothing; combo_params...)

Apply combo loss with class weighting.
"""
function weighted_combo_loss(model, x, y, class_weights=nothing; 
                            alpha=0.5f0, ce_ratio=0.5f0, focal_gamma=2.0f0)
    pred = model(x)
    
    if isnothing(class_weights)
        return combo_loss(pred, y, alpha=alpha, ce_ratio=ce_ratio, focal_gamma=focal_gamma)
    else
        # Apply class weights to loss
        # Create weight tensor with same shape as y
        weight_shape = size(y)
        weights = CUDA.zeros(Float32, weight_shape)
        
        # Apply weights using broadcasting operations
        class_weights_gpu = gpu(class_weights)
        
        # Create a mask for each class
        for c in 1:axes(y,3)
            class_mask = y[:,:,c,:] .> 0
            weights = weights .+ class_mask .* class_weights_gpu[c]
        end
        
        # Handle zeros in weights to avoid division by zero
        non_zero_mask = weights .> 0
        weights = weights .+ (1f-5) .* (1.0f0 .- non_zero_mask)
        
        # Apply weights to the combo loss components
        dl = dice_loss(pred, y)
        
        if ce_ratio == 1.0f0
            # Apply weights to cross-entropy
            ce = logitcrossentropy(pred, y; agg=identity)
            weighted_ce = ce .* weights
            total_ce = sum(weighted_ce) / sum(weights)
            return alpha * dl + (1.0f0 - alpha) * total_ce
        elseif ce_ratio == 0.0f0
            # Apply weights to focal loss
            fl = focal_loss(pred, y, gamma=focal_gamma)
            # Since focal loss already has weighting, we just return the combo
            return alpha * dl + (1.0f0 - alpha) * fl
        else
            # Apply weights to both CE and focal
            ce = logitcrossentropy(pred, y; agg=identity)
            weighted_ce = ce .* weights
            total_ce = sum(weighted_ce) / sum(weights)
            fl = focal_loss(pred, y, gamma=focal_gamma)
            return alpha * dl + (1.0f0 - alpha) * (ce_ratio * total_ce + (1.0f0 - ce_ratio) * fl)
        end
    end
end

# Original weighted loss function (for backward compatibility)
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
        
        # Create a mask for each class
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

# ===================================================
# Early Stopping
# ===================================================

"""
    EarlyStopping(patience=10, min_delta=0.0f0)

Early stopping implementation to prevent overfitting.
Stops training when a monitored metric has stopped improving.

Parameters:
- patience: Number of epochs with no improvement after which training will stop
- min_delta: Minimum change to qualify as an improvement
"""
mutable struct EarlyStopping
    patience::Int
    min_delta::Float32
    best_score::Float32
    counter::Int
    early_stop::Bool
    
    function EarlyStopping(patience=10, min_delta=0.0f0)
        new(patience, min_delta, Inf32, 0, false)
    end
end

function (es::EarlyStopping)(current_score)
    if current_score < (es.best_score - es.min_delta)
        # Score improved
        es.best_score = current_score
        es.counter = 0
        return false
    else
        # Score didn't improve
        es.counter += 1
        if es.counter >= es.patience
            es.early_stop = true
            return true
        end
        return false
    end
end

# ===================================================
# Improved Batching with Class Balancing
# ===================================================

"""
    create_balanced_batches(dataset, batch_size, num_classes; oversampling=true)

Create batches with balanced class representation to address class imbalance.
Either through oversampling or class-balanced sampling.

Parameters:
- dataset: List of (input, label) tuples
- batch_size: Number of samples per batch
- num_classes: Number of segmentation classes
- oversampling: Whether to oversample minority classes
"""
function create_balanced_batches(dataset, batch_size, num_classes; oversampling=true)
    # Count class occurrences across the dataset
    class_counts = zeros(Int, num_classes)
    sample_class_counts = [zeros(Int, num_classes) for _ in 1:length(dataset)]
    
    println("Analyzing class distribution for balanced batching...")
    for (i, (_, label)) in enumerate(dataset)
        flat_label = reshape(label, :)
        for c in 0:(num_classes-1)
            count = sum(flat_label .== c)
            class_counts[c+1] += count
            sample_class_counts[i][c+1] = count
        end
    end
    
    # Calculate class weights inversely proportional to frequency
    total_pixels = sum(class_counts)
    class_frequencies = class_counts ./ total_pixels
    
    if oversampling
        # Oversample rarer classes
        max_count = maximum(class_counts)
        sample_weights = [0.0f0 for _ in 1:length(dataset)]
        
        for i in 1:length(dataset)
            weight = 0.0f0
            for c in 1:(num_classes)
                if class_counts[c] > 0
                    # Weight based on class rarity and presence in this sample
                    class_weight = (max_count / max(class_counts[c], 1))
                    weight += sample_class_counts[i][c] * class_weight
                end
            end
            sample_weights[i] = weight
        end
        
        # Normalize weights to probabilities
        sample_weights ./= sum(sample_weights)
        
        # Create batches with weighted sampling
        n_batches = ceil(Int, length(dataset) / batch_size)
        batched_data = Vector{Tuple}(undef, n_batches)
        
        for i in 1:n_batches
            # Sample batch_size indices based on weights
            indices = sample(1:length(dataset), Weights(sample_weights), batch_size, replace=true)
            
            # Create batch
            imgs = cat([dataset[j][1] for j in indices]...; dims=4)
            labels = cat([dataset[j][2] for j in indices]...; dims=4)
            
            batched_data[i] = (imgs, labels)
        end
        
        return batched_data
    else
        # Just create standard batches but with shuffling
        n_batches = ceil(Int, length(dataset) / batch_size)
        batched_data = Vector{Tuple}(undef, n_batches)
        
        # Shuffle dataset
        indices = randperm(length(dataset))
        
        for i in 1:n_batches
            batch_start = (i-1) * batch_size + 1
            batch_end = min(batch_start + batch_size - 1, length(indices))
            batch_indices = indices[batch_start:batch_end]
            
            # Create batch
            imgs = cat([dataset[j][1] for j in batch_indices]...; dims=4)
            labels = cat([dataset[j][2] for j in batch_indices]...; dims=4)
            
            batched_data[i] = (imgs, labels)
        end
        
        return batched_data
    end
end

# ===================================================
# Optimized one-hot encoding for batches
# ===================================================
function batch_one_hot(batch, num_classes)
    batch_int = Int.(selectdim(batch, 3, 1))
    return gpu(Float32.(permutedims(onehotbatch(batch_int, 0:(num_classes-1)), (2,3,1,4))))
end

# ===================================================
# Calculate class weights based on frequency
# ===================================================
function calculate_class_weights(dataset, num_classes; method="inverse", beta=0.99)
    class_counts = zeros(Float32, num_classes)
    
    for (_, label) in dataset
        flat_label = reshape(label, :)
        for c in 0:(num_classes-1)
            class_counts[c+1] += count(==(c), flat_label)
        end
    end
    
    total_pixels = sum(class_counts)
    class_frequencies = class_counts ./ total_pixels
    
    # Avoid division by zero
    class_frequencies = max.(class_frequencies, 1e-6)
    
    if method == "inverse"
        # Standard inverse frequency weighting
        weights = 1.0 ./ class_frequencies
    elseif method == "balanced"
        # Class-balanced loss weighting from "Class-Balanced Loss Based on Effective Number of Samples"
        effective_num = 1.0 .- beta .^ class_counts
        weights = (1.0 .- beta) ./ effective_num
    elseif method == "sqrt_inverse"
        # Square root of inverse frequency for smoother weighting
        weights = 1.0 ./ sqrt.(class_frequencies)
    else
        throw(ArgumentError("Unknown weighting method: $method"))
    end
    
    # Normalize weights to have mean of 1
    weights = weights ./ mean(weights)
    
    return weights
end

# ===================================================
# Mixed Precision Training
# ===================================================

"""
    train_unet_mixed_precision(model, train_data, num_epochs, learning_rate, output_channels; ...)

Train the UNet model with mixed precision to reduce memory usage and potentially speed up training.
"""
function train_unet_mixed_precision(
    model, train_data, num_epochs, learning_rate, output_channels;
    validation_data=nothing, checkpoint_dir="checkpoints", checkpoint_freq=5,
    class_weights=nothing, loss_type="combo", early_stopping_patience=10,
    lr_scheduler="onecycle"
)
    println("Starting mixed precision training...")
    mkpath(checkpoint_dir)
    
    # Convert model to half precision
    model_f16 = Model.f16(model)
    
    # Parameters for combo loss
    combo_params = (alpha=0.5f0, ce_ratio=0.5f0, focal_gamma=2.0f0)
    
    # Setup optimizer - AdamW with weight decay
    opt_state = Optimisers.setup(Optimisers.AdamW(learning_rate), model)
    
    # Setup learning rate scheduler
    if lr_scheduler == "onecycle"
        total_steps = num_epochs * length(train_data)
        lr_schedule = OneCycleLR(learning_rate, total_steps)
    end
    
    # Setup early stopping
    early_stopper = EarlyStopping(early_stopping_patience)
    
    # Initialize losses
    loss_over_time = Float32[]
    val_loss_over_time = Float32[]
    best_val_loss = Inf32
    
    # Create timestamp for the training run
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
    
    # Main training loop
    step_counter = 0
    for epoch in 1:num_epochs
        # Set appropriate learning rate for this epoch
        if lr_scheduler == "onecycle"
            current_lr = get_lr(lr_schedule, step_counter + 1)
        elseif lr_scheduler == "cosine"
            current_lr = cosine_annealing_warm_restarts(learning_rate, epoch, num_epochs)
        else
            current_lr = learning_rate
        end
        
        println("====== Epoch $epoch (LR: $current_lr) ======")
        
        # Update optimizer learning rate
        opt_state = Optimisers.adjust(opt_state) do opt
            Optimisers.adjust_learning_rate(opt, current_lr)
        end
        
        # Training metrics
        total_loss = 0f0
        batch_count = 0
        
        p = Progress(length(train_data), 1, "Training epoch $epoch...")
        
        # Train on batches
        for (batch_idx, (input_batch, mask_batch)) in enumerate(train_data)
            # Convert to half precision
            input_batch_f16 = Float16.(gpu(input_batch))
            
            # One-hot encode labels (keeping in Float32 for stability)
            mask_batch_oh = batch_one_hot(mask_batch, output_channels)
            
            # Choose loss function based on parameter
            loss_fn_to_use = if loss_type == "combo"
                (m, x, y) -> weighted_combo_loss(m, x, y, class_weights; combo_params...)
            else
                (m, x, y) -> weighted_loss_fn(m, x, y, class_weights)
            end
            
            # Forward pass and compute gradients in f16
            loss_value = 0.0f0
            grads = gradient(model_f16) do m
                # Forward pass
                pred_f16 = m(input_batch_f16)
                
                # Convert predictions back to f32 for stable loss calculation
                pred_f32 = Float32.(pred_f16)
                
                # Compute loss
                loss = loss_fn_to_use(m, input_batch_f16, mask_batch_oh)
                loss_value = loss
                return loss
            end
            
            # Update model parameters
            opt_state, model_f16 = safe_update!(opt_state, model_f16, grads)
            
            # Accumulate loss
            total_loss += loss_value
            batch_count += 1
            
            # Increment step counter for lr scheduler
            step_counter += 1
            
            # Update progress bar
            next!(p; showvalues = [(:batch, batch_idx), (:loss, loss_value)])
            
            # Clear GPU memory
            CUDA.reclaim()
        end
        
        # Calculate average loss for this epoch
        avg_loss = total_loss / batch_count
        push!(loss_over_time, avg_loss)
        println("Epoch $epoch finished. Average Loss: $avg_loss")
        
        # Validation if provided
        if !isnothing(validation_data)
            val_loss = evaluate_model_mixed_precision(model_f16, validation_data, output_channels, class_weights)
            push!(val_loss_over_time, val_loss)
            println("Validation Loss: $val_loss")
            
            # Check for best model
            if val_loss < best_val_loss
                best_val_loss = val_loss
                # Save checkpoint (convert back to f32 for saving)
                model_f32 = Model.f32(model_f16)
                save_checkpoint(model_f32, opt_state, epoch, val_loss, 
                                joinpath(checkpoint_dir, "best_model_$(timestamp).bson"))
                println("New best model saved!")
            end
            
            # Check early stopping
            if early_stopper(val_loss)
                println("Early stopping triggered after $epoch epochs")
                break
            end
        end
        
        # Save regular checkpoint
        if epoch % checkpoint_freq == 0
            # Convert back to f32 for saving
            model_f32 = Model.f32(model_f16)
            save_checkpoint(model_f32, opt_state, epoch, avg_loss, 
                            joinpath(checkpoint_dir, "checkpoint_epoch$(epoch)_$(timestamp).bson"))
        end
        
        println("--------------------------------------------------")
    end
    
    # Save final model (convert back to f32)
    model_f32 = Model.f32(model_f16)
    save_checkpoint(model_f32, opt_state, num_epochs, loss_over_time[end], 
                   joinpath(checkpoint_dir, "final_model_$(timestamp).bson"))
    
    return model_f32, loss_over_time, val_loss_over_time
end

# ===================================================
# Enhanced Standard Precision Training
# ===================================================

"""
    train_unet(model, train_data, num_epochs, learning_rate, output_channels; ...)

Enhanced training function for UNet with advanced features like:
- Various loss functions (CE, Dice, Focal, Combo)
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Logging
"""
function train_unet(
    model, train_data, num_epochs, learning_rate, output_channels;
    validation_data=nothing, checkpoint_dir="checkpoints", checkpoint_freq=5,
    class_weights=nothing, loss_type="combo", early_stopping_patience=10,
    lr_scheduler="onecycle"
)
    println("Starting standard precision training...")
    mkpath(checkpoint_dir)
    
    # Parameters for combo loss
    combo_params = (alpha=0.5f0, ce_ratio=0.5f0, focal_gamma=2.0f0)
    
    # Setup optimizer
    opt_state = Optimisers.setup(Optimisers.AdamW(learning_rate, 0.01), model)
    
    # Setup learning rate scheduler
    if lr_scheduler == "onecycle"
        total_steps = num_epochs * length(train_data)
        lr_schedule = OneCycleLR(learning_rate, total_steps)
    end
    
    # Setup early stopping
    early_stopper = EarlyStopping(early_stopping_patience)
    
    # Initialize losses
    loss_over_time = Float32[]
    val_loss_over_time = Float32[]
    best_val_loss = Inf32
    
    # Create timestamp for the training run
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM")
    
    # Main training loop
    step_counter = 0
    for epoch in 1:num_epochs
        # Set appropriate learning rate for this epoch
        if lr_scheduler == "onecycle"
            current_lr = get_lr(lr_schedule, step_counter + 1)
        elseif lr_scheduler == "cosine"
            current_lr = cosine_annealing_warm_restarts(learning_rate, epoch, num_epochs)
        else
            current_lr = learning_rate
        end
        
        println("====== Epoch $epoch (LR: $current_lr) ======")
        
        # Update optimizer learning rate
        opt_state = Optimisers.adjust(opt_state) do opt
            Optimisers.adjust_learning_rate(opt, current_lr)
        end
        
        # Training metrics
        total_loss = 0f0
        batch_count = 0
        
        p = Progress(length(train_data), 1, "Training epoch $epoch...")
        
        # Train on batches
        for (batch_idx, (input_batch, mask_batch)) in enumerate(train_data)
            # Move data to GPU
            input_batch = gpu(input_batch)
            
            # One-hot encode labels
            mask_batch_oh = batch_one_hot(mask_batch, output_channels)
            
            # Choose loss function based on parameter
            loss_fn_to_use = if loss_type == "combo"
                (m, x, y) -> weighted_combo_loss(m, x, y, class_weights; combo_params...)
            elseif loss_type == "dice"
                (m, x, y) -> dice_loss(m(x), y)
            elseif loss_type == "focal"
                (m, x, y) -> focal_loss(m(x), y)
            elseif loss_type == "lovasz"
                (m, x, y) -> lovasz_loss(m(x), y)
            else
                (m, x, y) -> weighted_loss_fn(m, x, y, class_weights)
            end
            
            # Compute gradients and loss
            ∇model = gradient(m -> loss_fn_to_use(m, input_batch, mask_batch_oh), model)[1]
            batch_loss = loss_fn_to_use(model, input_batch, mask_batch_oh)
            
            # Update model parameters with gradient clipping
            opt_state, model = safe_update!(opt_state, model, ∇model)
            
            # Accumulate loss
            total_loss += batch_loss
            batch_count += 1
            
            # Increment step counter for lr scheduler
            step_counter += 1
            
            # Update progress bar
            next!(p; showvalues = [(:batch, batch_idx), (:loss, batch_loss)])
            
            # Clear GPU memory
            CUDA.reclaim()
        end
        
        # Calculate average loss for this epoch
        avg_loss = total_loss / batch_count
        push!(loss_over_time, avg_loss)
        println("Epoch $epoch finished. Average Loss: $avg_loss")
        
        # Validation if provided
        if !isnothing(validation_data)
            val_loss = evaluate_model(model, validation_data, output_channels, class_weights)
            push!(val_loss_over_time, val_loss)
            println("Validation Loss: $val_loss")
            
            # Check for best model
            if val_loss < best_val_loss
                best_val_loss = val_loss
                save_checkpoint(model, opt_state, epoch, val_loss, 
                                joinpath(checkpoint_dir, "best_model_$(timestamp).bson"))
                println("New best model saved!")
            end
            
            # Check early stopping
            if early_stopper(val_loss)
                println("Early stopping triggered after $epoch epochs")
                break
            end
        end
        
        # Save regular checkpoint
        if epoch % checkpoint_freq == 0
            save_checkpoint(model, opt_state, epoch, avg_loss, 
                            joinpath(checkpoint_dir, "checkpoint_epoch$(epoch)_$(timestamp).bson"))
        end
        
        println("--------------------------------------------------")
    end
    
    # Save final model
    save_checkpoint(model, opt_state, num_epochs, loss_over_time[end], 
                   joinpath(checkpoint_dir, "final_model_$(timestamp).bson"))
    
    return model, loss_over_time, val_loss_over_time
end

# ===================================================
# Evaluation Functions
# ===================================================

function evaluate_model(model, validation_data, output_channels, class_weights=nothing; loss_type="combo")
    total_loss = 0f0
    batch_count = 0
    
    # Switch to evaluation mode
    model_mode = Flux.trainmode(model, false)
    
    # Parameters for combo loss
    combo_params = (alpha=0.5f0, ce_ratio=0.5f0, focal_gamma=2.0f0)
    
    for (input_batch, mask_batch) in validation_data
        # One-hot encode labels
        mask_batch_oh = batch_one_hot(mask_batch, output_channels)
        
        # Move to GPU
        input_batch = gpu(input_batch)
        
        # Choose loss function based on parameter
        if loss_type == "combo"
            batch_loss = weighted_combo_loss(model, input_batch, mask_batch_oh, class_weights; combo_params...)
        else
            batch_loss = weighted_loss_fn(model, input_batch, mask_batch_oh, class_weights)
        end
        
        # Accumulate loss
        total_loss += batch_loss
        batch_count += 1
        
        # Clear GPU memory
        CUDA.reclaim()
    end
    
    # Switch back to training mode
    Flux.trainmode(model, true)
    
    return total_loss / batch_count
end

function evaluate_model_mixed_precision(model_f16, validation_data, output_channels, class_weights=nothing; loss_type="combo")
    total_loss = 0f0
    batch_count = 0
    
    # Switch to evaluation mode
    model_mode = Flux.trainmode(model_f16, false)
    
    # Parameters for combo loss
    combo_params = (alpha=0.5f0, ce_ratio=0.5f0, focal_gamma=2.0f0)
    
    for (input_batch, mask_batch) in validation_data
        # Convert input to half precision
        input_batch_f16 = Float16.(gpu(input_batch))
        
        # One-hot encode labels (keeping in Float32 for stability)
        mask_batch_oh = batch_one_hot(mask_batch, output_channels)
        
        # Forward pass
        pred_f16 = model_f16(input_batch_f16)
        
        # Convert to float32 for loss calculation
        pred_f32 = Float32.(pred_f16)
        
        # Choose loss function based on parameter
        if loss_type == "combo"
            batch_loss = weighted_combo_loss(model_f16, input_batch_f16, mask_batch_oh, class_weights; combo_params...)
        else
            batch_loss = weighted_loss_fn(model_f16, input_batch_f16, mask_batch_oh, class_weights)
        end
        
        # Accumulate loss
        total_loss += batch_loss
        batch_count += 1
        
        # Clear GPU memory
        CUDA.reclaim()
    end
    
    # Switch back to training mode
    Flux.trainmode(model_f16, true)
    
    return total_loss / batch_count
end

# ===================================================
# Safe Update Function
# ===================================================
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
        # Clip gradients to prevent exploding gradients
        ∇model_clipped = Flux.clip(∇model, -1.0f0, 1.0f0)
        return Optimisers.update!(opt_state, model, ∇model_clipped)
    end
end

# ===================================================
# Checkpoint Functions
# ===================================================
function save_checkpoint(model, opt_state, epoch, loss, filename)
    model_cpu = cpu(model)
    # Save additional training info
    @save filename model=model_cpu epoch=epoch loss=loss
    println("Saved checkpoint to $filename")
end

function load_checkpoint(filename)
    variables = BSON.load(filename)
    model = gpu(variables[:model])
    epoch = variables[:epoch]
    loss = variables[:loss]
    return model, epoch, loss
end

# ===================================================
# Debugging Helpers
# ===================================================
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

end # module Training