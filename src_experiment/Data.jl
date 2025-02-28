##################################
# Data.jl - Enhanced with Advanced Augmentation
##################################
module Data

using Images, FileIO, Statistics, Random, ImageTransformations, CoordinateTransformations
using Interpolations, LinearAlgebra, ColorTypes
using Rotations

# Define standard dimensions for the model that work well with UNet
# Dimensions should be multiples of 16 (2^4) for a 4-level UNet
const STANDARD_HEIGHT = 368  # 368 = 23 * 16
const STANDARD_WIDTH = 1232  # 1232 = 77 * 16

# 3D-Daten zurechtschneiden/auffüllen with fixed standard dimensions
function standardize_size(img::AbstractArray{T,3}) where {T}
    h, w, c = size(img)
    
    # Direct allocation of needed size
    final = zeros(T, STANDARD_HEIGHT, STANDARD_WIDTH, c)
    
    # Only copy the relevant area
    h_range = 1:min(h, STANDARD_HEIGHT)
    w_range = 1:min(w, STANDARD_WIDTH)
    final[h_range, w_range, :] .= view(img, h_range, w_range, 1:c)
    
    return final
end

# 2D-Daten (Labels) zurechtschneiden/auffüllen with fixed standard dimensions
function standardize_size_2d(matrix::AbstractMatrix{T}) where {T}
    h, w = size(matrix)
    
    # Direct allocation of needed size
    final = zeros(T, STANDARD_HEIGHT, STANDARD_WIDTH)
    
    # Only copy the relevant area
    h_range = 1:min(h, STANDARD_HEIGHT)
    w_range = 1:min(w, STANDARD_WIDTH)
    final[h_range, w_range] .= view(matrix, h_range, w_range)
    
    return final
end

# Bild laden und normalisieren with fixed dimensions
function load_and_preprocess_image(img_path::String; verbose=true)
    raw_img = load(img_path)
    
    # Record original dimensions for debugging
    original_size = size(raw_img)
    
    # Debug information for image
    if verbose
        println("Image: $(basename(img_path))")
        println("  Original size: $original_size")
    end
    
    # zu Float32 konvertieren
    img_float = Float32.(channelview(raw_img))
    
    # Print image statistics
    if verbose
        println("  Image stats: min=$(minimum(img_float)), max=$(maximum(img_float)), mean=$(mean(img_float))")
    end
    
    # Check for NaN or Inf values
    if any(isnan, img_float) || any(isinf, img_float)
        if verbose
            println("  WARNING: NaN or Inf values found in image!")
        end
        img_float = replace(img_float, NaN => 0.0f0, Inf => 1.0f0, -Inf => 0.0f0)
    end
    
    # Ensure values are normalized
    img_float = clamp.(img_float, 0.0f0, 1.0f0)
    
    img_array = permutedims(img_float, (2, 3, 1))
    
    # Standardize to fixed dimensions
    img_std = standardize_size(img_array)
    return reshape(img_std, size(img_std)..., 1)
end

# Label laden und auf 0..34 skalieren with fixed dimensions
function load_and_preprocess_label(label_path::String; verbose=true)
    raw_label = load(label_path)
    
    # Record original dimensions for debugging
    original_size = size(raw_label)
    
    if verbose
        println("Label: $(basename(label_path))")
        println("  Original size: $original_size")
    end
    
    # Optimierte Skalierung ohne temporäre Zwischenvariablen
    min_val, max_val = extrema(raw_label)
    
    if verbose
        println("  Label raw range: min=$min_val, max=$max_val")
    end
    
    scaled_label = Int.(round.((raw_label .- min_val) .* (34 / (max_val - min_val))))
    
    # Analyze label distribution if verbose
    if verbose
        unique_labels = unique(scaled_label)
        println("  Unique labels after scaling: $unique_labels")
        println("  Number of unique labels: $(length(unique_labels))")
        
        # Count occurrences of each label
        label_counts = Dict{Int, Int}()
        for label in unique_labels
            count = sum(scaled_label .== label)
            label_counts[label] = count
            percentage = 100 * count / length(scaled_label)
            println("    Label $label: $count pixels ($(round(percentage, digits=2))%)")
        end
    end
    
    # Standardize to fixed dimensions
    scaled_label_std = standardize_size_2d(scaled_label)
    max_class = maximum(scaled_label_std)
    
    # Check if any new labels appeared after standardization (shouldn't happen)
    if verbose
        unique_after_std = unique(scaled_label_std)
        if length(unique_after_std) != length(unique(scaled_label))
            println("  WARNING: Label count changed after standardization!")
            println("  Labels after standardization: $unique_after_std")
        end
        
        println("  Final label shape: $(size(scaled_label_std))")
        println("  Max class: $max_class (should be <= 34)")
    end
    
    label = reshape(scaled_label_std, size(scaled_label_std,1), size(scaled_label_std,2), 1, 1)
    
    return label, max_class
end

# Advanced augmentation functions

"""
    rotate_image(img, angle)

Rotate an image by the specified angle in radians.
"""
function rotate_image(img, angle)
    h, w, c, n = size(img)
    
    # Create rotation transform
    rot = recenter(RotMatrix(angle), [h/2, w/2])
    
    # Apply to each channel
    result = similar(img)
    
    for ch in 1:c
        channel = img[:, :, ch, 1]
        # Use linear interpolation for smoother rotation
        result[:, :, ch, 1] = warp(channel, rot, Linear())
    end
    
    return result
end

"""
    rotate_label(label, angle)

Rotate a label by the specified angle in radians.
Use nearest neighbor interpolation to preserve label integers.
"""
function rotate_label(label, angle)
    h, w, c, n = size(label)
    
    # Create rotation transform
    rot = recenter(RotMatrix(angle), [h/2, w/2])
    
    # Apply to each channel with nearest neighbor interpolation to preserve integers
    result = similar(label)
    
    for ch in 1:c
        channel = label[:, :, ch, 1]
        result[:, :, ch, 1] = round.(Int, warp(channel, rot, Constant()))
    end
    
    return result
end

"""
    add_gaussian_noise(img, std_dev)

Add Gaussian noise to the image.
"""
function add_gaussian_noise(img, std_dev)
    noise = randn(Float32, size(img)) .* std_dev
    return clamp.(img .+ noise, 0.0f0, 1.0f0)
end

"""
    adjust_brightness(img, factor)

Adjust the brightness of the image.
"""
function adjust_brightness(img, factor)
    return clamp.(img .* factor, 0.0f0, 1.0f0)
end

"""
    adjust_contrast(img, factor)

Adjust the contrast of the image.
"""
function adjust_contrast(img, factor)
    mean_val = mean(img)
    return clamp.((img .- mean_val) .* factor .+ mean_val, 0.0f0, 1.0f0)
end

"""
    random_crop(img, label, crop_size_h, crop_size_w)

Random crop both image and label with the same crop location.
"""
function random_crop(img, label, crop_size_h, crop_size_w)
    h, w, c, n = size(img)
    
    # Calculate valid ranges for crop starting points
    max_h = h - crop_size_h + 1
    max_w = w - crop_size_w + 1
    
    if max_h < 1 || max_w < 1
        # Image is smaller than crop size, return original
        return img, label
    end
    
    # Random starting points
    start_h = rand(1:max_h)
    start_w = rand(1:max_w)
    
    # Perform crop
    cropped_img = img[start_h:(start_h+crop_size_h-1), start_w:(start_w+crop_size_w-1), :, :]
    cropped_label = label[start_h:(start_h+crop_size_h-1), start_w:(start_w+crop_size_w-1), :, :]
    
    return cropped_img, cropped_label
end

"""
    elastic_transform(img, label, alpha=10.0, sigma=4.0, grid_scale=50)

Apply elastic deformation to both image and label.
This is particularly useful for segmentation tasks.
"""
function elastic_transform(img, label, alpha=10.0, sigma=4.0, grid_scale=50)
    h, w, c, n = size(img)
    
    # Create displacement fields
    x_grid = div(h, grid_scale) + 1
    y_grid = div(w, grid_scale) + 1
    
    # Random displacement field
    dx = randn(Float32, x_grid, y_grid) .* alpha
    dy = randn(Float32, x_grid, y_grid) .* alpha
    
    # Smooth displacement field
    for _ in 1:3  # Simple smoothing by averaging with neighbors
        dx_padded = padarray(dx, Pad(:replicate, 1, 1, 0, 0))
        dy_padded = padarray(dy, Pad(:replicate, 1, 1, 0, 0))
        
        for i in 2:(size(dx_padded,1)-1)
            for j in 2:(size(dx_padded,2)-1)
                dx_padded[i,j] = mean(dx_padded[i-1:i+1, j-1:j+1])
                dy_padded[i,j] = mean(dy_padded[i-1:i+1, j-1:j+1])
            end
        end
        
        dx = dx_padded[2:end-1, 2:end-1]
        dy = dy_padded[2:end-1, 2:end-1]
    end
    
    # Upsample to image size
    dx_full = imresize(dx, (h, w))
    dy_full = imresize(dy, (h, w))
    
    # Create sampling grid
    x_grid = [i for i in 1:h, j in 1:w]
    y_grid = [j for i in 1:h, j in 1:w]
    
    # Apply displacement
    x_indices = clamp.(round.(Int, x_grid + dx_full), 1, h)
    y_indices = clamp.(round.(Int, y_grid + dy_full), 1, w)
    
    # Apply transformation
    result_img = similar(img)
    result_label = similar(label)
    
    for i in 1:h
        for j in 1:w
            src_i = x_indices[i, j]
            src_j = y_indices[i, j]
            
            # Apply to all channels
            for c_idx in 1:c
                result_img[i, j, c_idx, 1] = img[src_i, src_j, c_idx, 1]
                result_label[i, j, c_idx, 1] = label[src_i, src_j, c_idx, 1]
            end
        end
    end
    
    return result_img, result_label
end

"""
    color_jitter(img, brightness=0.2, contrast=0.2, saturation=0.2)

Apply random color jittering to the image.
"""
function color_jitter(img, brightness=0.2, contrast=0.2, saturation=0.2)
    # Apply random brightness adjustment
    if rand() > 0.5
        brightness_factor = 1.0 + (rand() * 2 - 1) * brightness
        img = adjust_brightness(img, brightness_factor)
    end
    
    # Apply random contrast adjustment
    if rand() > 0.5
        contrast_factor = 1.0 + (rand() * 2 - 1) * contrast
        img = adjust_contrast(img, contrast_factor)
    end
    
    # For RGB images, we could also adjust saturation
    # Skipping this for now since it requires converting to HSV
    
    return img
end

# Enhanced data augmentation function
function augment_image(img, label)
    # Create copies to avoid modifying the original
    img_aug = copy(img)
    label_aug = copy(label)
    
    # Random horizontal flip (safe operation that preserves dimensions)
    if rand() > 0.5
        img_aug = reverse(img_aug, dims=2)
        label_aug = reverse(label_aug, dims=2)
    end
    
    # Brightness adjustment (safe operation that preserves dimensions)
    if rand() > 0.5
        brightness_factor = 0.8 + 0.4 * rand()
        img_aug = clamp.(img_aug .* brightness_factor, 0.0f0, 1.0f0)
    end
    
    # Add random noise (safe operation that preserves dimensions)
    if rand() > 0.75
        noise_level = 0.01 * rand()
        noise = randn(Float32, size(img_aug)) .* noise_level
        img_aug = clamp.(img_aug .+ noise, 0.0f0, 1.0f0)
    end
    
    return img_aug, label_aug
end

using Base.Threads

# Threaded dataset loading with fixed dimensions
function load_dataset(image_dir::String, label_dir::String; verbose=true)
    image_files = sort(readdir(image_dir, join=true))
    label_files = sort(readdir(label_dir, join=true))
    
    if verbose
        println("Loading dataset with $(length(image_files)) images")
        println("All images will be standardized to $(STANDARD_HEIGHT)×$(STANDARD_WIDTH)")
    end
    
    return load_dataset(image_files, label_files; verbose=verbose)
end

# Load a subset of data from file lists rather than directories
function load_dataset(image_files::Vector{String}, label_files::Vector{String}; verbose=true)
    if verbose
        println("Loading dataset with $(length(image_files)) images")
        println("All images will be standardized to $(STANDARD_HEIGHT)×$(STANDARD_WIDTH)")
    end
    
    dataset = Vector{Tuple}(undef, length(image_files))
    
    # Track any processing errors
    error_count = Threads.Atomic{Int}(0)
    
    # Track label statistics across the whole dataset
    all_unique_labels = Set{Int}()
    total_label_counts = Dict{Int, Int}()
    
    @threads for i in 1:length(image_files)
        try
            # Process image and label (with reduced verbosity)
            img_data = load_and_preprocess_image(image_files[i], verbose=false)
            label_data, max_class = load_and_preprocess_label(label_files[i], verbose=false)
            
            # Collect label statistics for this sample
            flat_labels = reshape(label_data, :)
            sample_unique_labels = unique(flat_labels)
            
            # Thread-safe updates to global statistics
            lock(ReentrantLock()) do
                union!(all_unique_labels, sample_unique_labels)
                
                for label in sample_unique_labels
                    count = sum(flat_labels .== label)
                    total_label_counts[label] = get(total_label_counts, label, 0) + count
                end
            end
            
            dataset[i] = (img_data, label_data)
        catch e
            Threads.atomic_add!(error_count, 1)
            if verbose
                println("Error processing file pair ($(image_files[i]), $(label_files[i])): $e")
            end
            # Provide empty data as fallback
            dataset[i] = (zeros(Float32, STANDARD_HEIGHT, STANDARD_WIDTH, 3, 1),
                         zeros(Int, STANDARD_HEIGHT, STANDARD_WIDTH, 1, 1))
        end
    end
    
    if error_count[] > 0 && verbose
        println("Warning: Encountered $(error_count[]) errors during dataset loading")
    end
    
    if verbose
        println("\n===== DATASET SUMMARY =====")
        println("Dataset loaded with $(length(dataset)) samples")
        println("All samples standardized to $(STANDARD_HEIGHT)×$(STANDARD_WIDTH)")
        
        # Print overall label statistics
        println("\nLabel distribution across entire dataset:")
        println("Total unique labels: $(length(all_unique_labels))")
        
        total_pixels = sum(values(total_label_counts))
        
        for label in sort(collect(all_unique_labels))
            count = get(total_label_counts, label, 0)
            percentage = 100 * count / total_pixels
            println("  Label $label: $count pixels ($(round(percentage, digits=2))%)")
        end
        
        # Check if we have all expected classes (0-34)
        if maximum(all_unique_labels) <= 34
            println("\nLabel range is within expected range (0-34)")
        else
            println("\nWARNING: Found labels outside expected range (0-34)")
            println("Max label found: $(maximum(all_unique_labels))")
        end
        
        # Check for missing classes
        missing_classes = setdiff(Set(0:34), all_unique_labels)
        if !isempty(missing_classes)
            println("\nWARNING: Some classes are missing in the dataset:")
            println("Missing classes: $missing_classes")
        end
    end
    
    return dataset
end

# Memory-efficient batch creation - loads one batch at a time
function create_batches_lazy(image_files::Vector{String}, label_files::Vector{String}, batch_size::Int)
    n_batches = ceil(Int, length(image_files) / batch_size)
    
    # Return a function that loads batches on demand
    function load_batch(batch_idx::Int)
        if batch_idx > n_batches || batch_idx < 1
            error("Batch index out of range: $batch_idx (max: $n_batches)")
        end
        
        batch_start = (batch_idx-1) * batch_size + 1
        batch_end = min(batch_start + batch_size - 1, length(image_files))
        batch_img_files = image_files[batch_start:batch_end]
        batch_label_files = label_files[batch_start:batch_end]
        
        # Load just this batch
        batch_data = load_dataset(batch_img_files, batch_label_files, verbose=false)
        
        # Combine into tensors
        imgs = cat([item[1] for item in batch_data]...; dims=4)
        labels = cat([item[2] for item in batch_data]...; dims=4)
        
        return (imgs, labels)
    end
    
    return load_batch, n_batches
end

# Shuffle dataset for better training
function shuffle_dataset(dataset)
    return shuffle(dataset)
end

# Create dataset with augmentation
function create_augmented_dataset(dataset, augmentation_factor=2)
    # Calculate total size after augmentation
    total_size = length(dataset) * augmentation_factor
    augmented_dataset = Vector{Tuple}(undef, total_size)
    
    # Copy original data
    for i in 1:length(dataset)
        augmented_dataset[i] = dataset[i]
    end
    
    # Add augmented data
    next_idx = length(dataset) + 1
    for i in 1:length(dataset)
        for j in 1:(augmentation_factor-1)
            if next_idx <= total_size
                img, lbl = augment_image(dataset[i][1], dataset[i][2])
                augmented_dataset[next_idx] = (img, lbl)
                next_idx += 1
            end
        end
    end
    
    println("Created augmented dataset with $(length(augmented_dataset)) samples")
    
    return shuffle(augmented_dataset)
end

# Create dataset with advanced augmentation and class balancing
function create_balanced_augmented_dataset(dataset, num_classes, augmentation_factor=2)
    println("Creating balanced augmented dataset...")
    
    # First, count occurrences of each class
    class_counts = zeros(Int, num_classes)
    sample_class_weights = zeros(Float32, length(dataset))
    
    for (i, (_, label)) in enumerate(dataset)
        flat_label = reshape(label, :)
        
        # Calculate sample weight based on rare classes
        sample_weight = 0.0f0
        
        for c in 0:(num_classes-1)
            count = sum(flat_label .== c)
            class_counts[c+1] += count
            
            if count > 0
                sample_class_weights[i] += count
            end
        end
    end
    
    # Calculate inverse frequency weights
    total_pixels = sum(class_counts)
    class_frequencies = class_counts ./ total_pixels
    class_weights = 1.0f0 ./ max.(class_frequencies, 1e-6)
    class_weights ./= mean(class_weights)  # Normalize to mean 1.0
    
    # Calculate sample importance based on containing rare classes
    # Prioritize samples with rare classes for more augmentation
    for (i, (_, label)) in enumerate(dataset)
        flat_label = reshape(label, :)
        weight = 0.0f0
        
        for c in 0:(num_classes-1)
            count = sum(flat_label .== c)
            if count > 0
                # Higher weight for rarer classes
                weight += count * class_weights[c+1]
            end
        end
        
        sample_class_weights[i] = weight
    end
    
    # Normalize sample weights
    sample_class_weights ./= sum(sample_class_weights)
    
    # Calculate augmentation counts - more augmentations for samples with rare classes
    total_new_samples = (augmentation_factor - 1) * length(dataset)
    augmentation_counts = zeros(Int, length(dataset))
    
    for i in 1:length(dataset)
        augmentation_counts[i] = round(Int, total_new_samples * sample_class_weights[i])
    end
    
    # Adjust to ensure we get the right total
    while sum(augmentation_counts) > total_new_samples
        # Find maximum and reduce by 1
        idx = argmax(augmentation_counts)
        augmentation_counts[idx] -= 1
    end
    
    while sum(augmentation_counts) < total_new_samples
        # Find minimum (among non-zeros) and increase by 1
        non_zero_indices = findall(x -> x > 0, augmentation_counts)
        if isempty(non_zero_indices)
            # If all are zero, pick random
            idx = rand(1:length(augmentation_counts))
        else
            values = [augmentation_counts[i] for i in non_zero_indices]
            min_val_idx = argmin(values)
            idx = non_zero_indices[min_val_idx]
        end
        augmentation_counts[idx] += 1
    end
    
    # Create the augmented dataset
    total_size = length(dataset) + sum(augmentation_counts)
    augmented_dataset = similar(dataset, total_size)
    
    # Copy original data
    augmented_dataset[1:length(dataset)] = dataset
    
    # Add augmented data
    next_idx = length(dataset) + 1
    for i in 1:length(dataset)
        for j in 1:augmentation_counts[i]
            if next_idx <= total_size
                img, lbl = augment_image(dataset[i][1], dataset[i][2])
                augmented_dataset[next_idx] = (img, lbl)
                next_idx += 1
            end
        end
    end
    
    println("Created balanced augmented dataset with $(length(augmented_dataset)) samples")
    println("Original: $(length(dataset)), Augmented: $(length(augmented_dataset) - length(dataset))")
    
    return shuffle(augmented_dataset)
end

# Debug batches to see label distribution
function debug_batch(batch, batch_idx)
    input_batch, label_batch = batch
    
    println("\n===== BATCH $batch_idx DEBUG =====")
    println("Input batch shape: $(size(input_batch))")
    println("Label batch shape: $(size(label_batch))")
    
    # Get overall statistics for this batch
    batch_size = size(input_batch, 4)
    println("Batch size: $batch_size")
    
    # Check each sample in the batch
    for i in 1:batch_size
        sample_input = input_batch[:,:,:,i]
        sample_label = label_batch[:,:,:,i]
        
        println("\nSample $i in batch $batch_idx:")
        println("  Input shape: $(size(sample_input))")
        println("  Input range: min=$(minimum(sample_input)), max=$(maximum(sample_input)), mean=$(mean(sample_input))")
        println("  Label shape: $(size(sample_label))")
        
        # Check label distribution
        flat_labels = reshape(sample_label, :)
        unique_labels = unique(flat_labels)
        
        println("  Unique labels: $unique_labels")
        println("  Number of unique labels: $(length(unique_labels))")
        
        # Count occurrences of each label
        for label in unique_labels
            count = sum(flat_labels .== label)
            percentage = 100 * count / length(flat_labels)
            println("    Label $label: $count pixels ($(round(percentage, digits=2))%)")
        end
    end
    
    return batch
end

# Optimized batching with preallocated vector and debug option
function create_batches(dataset, batch_size; debug=false)
    n_batches = ceil(Int, length(dataset) / batch_size)
    batched_data = Vector{Tuple}(undef, n_batches)
    
    for i in 1:n_batches
        batch_start = (i-1) * batch_size + 1
        batch_end = min(batch_start + batch_size - 1, length(dataset))
        batch_indices = batch_start:batch_end
        
        # Preallocate arrays and then concatenate
        imgs = cat([dataset[j][1] for j in batch_indices]...; dims=4)
        labels = cat([dataset[j][2] for j in batch_indices]...; dims=4)
        
        batch = (imgs, labels)
        
        # Debug this batch if requested
        if debug
            batch = debug_batch(batch, i)
        end
        
        batched_data[i] = batch
    end
    
    println("Created $(n_batches) batches of size up to $(batch_size)")
    
    return batched_data
end

end # module Data