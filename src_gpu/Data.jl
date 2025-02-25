##################################
# Data.jl - With Label Debugging
##################################
module Data

using Images, FileIO, Statistics, Random

# Define standard dimensions for the model that work well with UNet
# Dimensions should be multiples of 16 (2^4) for a 4-level UNet
const STANDARD_HEIGHT = 368  # 368 = 23 * 16
const STANDARD_WIDTH = 1232  # 1232 = 77 * 16

# 3D-Daten zurechtschneiden/auffüllen with fixed standard dimensions
function standardize_size(img::AbstractArray{T,3}) where {T}
    h, w, c = size(img)
    
    # Direkt die notwendige Größe allokieren
    final = zeros(T, STANDARD_HEIGHT, STANDARD_WIDTH, c)
    
    # Nur den relevanten Bereich kopieren
    h_range = 1:min(h, STANDARD_HEIGHT)
    w_range = 1:min(w, STANDARD_WIDTH)
    final[h_range, w_range, :] .= view(img, h_range, w_range, 1:c)
    
    return final
end

# 2D-Daten (Labels) zurechtschneiden/auffüllen with fixed standard dimensions
function standardize_size_2d(matrix::AbstractMatrix{T}) where {T}
    h, w = size(matrix)
    
    # Direkt die notwendige Größe allokieren
    final = zeros(T, STANDARD_HEIGHT, STANDARD_WIDTH)
    
    # Nur den relevanten Bereich kopieren
    h_range = 1:min(h, STANDARD_HEIGHT)
    w_range = 1:min(w, STANDARD_WIDTH)
    final[h_range, w_range] .= view(matrix, h_range, w_range)
    
    return final
end

# Bild laden und normalisieren with fixed dimensions
function load_and_preprocess_image(img_path::String)
    raw_img = load(img_path)
    
    # Record original dimensions for debugging
    original_size = size(raw_img)
    
    # Debug information for image
    println("Image: $(basename(img_path))")
    println("  Original size: $original_size")
    
    # zu Float32 konvertieren
    img_float = Float32.(channelview(raw_img))
    
    # Print image statistics
    println("  Image stats: min=$(minimum(img_float)), max=$(maximum(img_float)), mean=$(mean(img_float))")
    
    # Check for NaN or Inf values
    if any(isnan, img_float) || any(isinf, img_float)
        println("  WARNING: NaN or Inf values found in image!")
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
function load_and_preprocess_label(label_path::String)
    raw_label = load(label_path)
    
    # Record original dimensions for debugging
    original_size = size(raw_label)
    
    println("Label: $(basename(label_path))")
    println("  Original size: $original_size")
    
    # Optimierte Skalierung ohne temporäre Zwischenvariablen
    min_val, max_val = extrema(raw_label)
    println("  Label raw range: min=$min_val, max=$max_val")
    
    scaled_label = Int.(round.((raw_label .- min_val) .* (34 / (max_val - min_val))))
    
    # Analyze label distribution
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
    
    # Standardize to fixed dimensions
    scaled_label_std = standardize_size_2d(scaled_label)
    max_class = maximum(scaled_label_std)
    
    # Check if any new labels appeared after standardization (shouldn't happen)
    unique_after_std = unique(scaled_label_std)
    if length(unique_after_std) != length(unique_labels)
        println("  WARNING: Label count changed after standardization!")
        println("  Labels after standardization: $unique_after_std")
    end
    
    label = reshape(scaled_label_std, size(scaled_label_std,1), size(scaled_label_std,2), 1, 1)
    
    println("  Final label shape: $(size(label))")
    println("  Max class: $max_class (should be <= 34)")
    
    return label, max_class
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
            # println("\nProcessing sample $i: $(basename(image_files[i]))")
            
            # Process image and label
            img_data = load_and_preprocess_image(image_files[i])
            label_data, max_class = load_and_preprocess_label(label_files[i])
            
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
                # println("Error processing file pair ($(image_files[i]), $(label_files[i])): $e")
            end
            # Provide empty data as fallback
            dataset[i] = (zeros(Float32, STANDARD_HEIGHT, STANDARD_WIDTH, 3, 1),
                         zeros(Int, STANDARD_HEIGHT, STANDARD_WIDTH, 1, 1))
        end
    end
    
    if error_count[] > 0
        # println("Warning: Encountered $(error_count[]) errors during dataset loading")
    end
    
    if verbose
        # println("\n===== DATASET SUMMARY =====")
        # println("Dataset loaded with $(length(dataset)) samples")
        # println("All samples standardized to $(STANDARD_HEIGHT)×$(STANDARD_WIDTH)")
        
        # # Print overall label statistics
        # println("\nLabel distribution across entire dataset:")
        # println("Total unique labels: $(length(all_unique_labels))")
        
        total_pixels = sum(values(total_label_counts))
        
        for label in sort(collect(all_unique_labels))
            count = get(total_label_counts, label, 0)
            percentage = 100 * count / total_pixels
            # println("  Label $label: $count pixels ($(round(percentage, digits=2))%)")
        end
        
        # Check if we have all expected classes (0-34)
        if maximum(all_unique_labels) <= 34
            # println("\nLabel range is within expected range (0-34)")
        else
            # println("\nWARNING: Found labels outside expected range (0-34)")
            # println("Max label found: $(maximum(all_unique_labels))")
        end
        
        # Check for missing classes
        missing_classes = setdiff(Set(0:34), all_unique_labels)
        if !isempty(missing_classes)
            # println("\nWARNING: Some classes are missing in the dataset:")
            # println("Missing classes: $missing_classes")
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

# Data augmentation functions
function augment_image(img, label)
    # Random horizontal flip
    if rand() > 0.5
        img = reverse(img, dims=2)
        label = reverse(label, dims=2)
    end
    
    # Random brightness adjustment
    if rand() > 0.5
        brightness_factor = 0.8 + 0.4 * rand()
        img = clamp.(img * brightness_factor, 0, 1)
    end
    
    return img, label
end

# Create dataset with augmentation
function create_augmented_dataset(dataset, augmentation_factor=2)
    augmented_dataset = similar(dataset, length(dataset) * augmentation_factor)
    
    # Copy original data
    augmented_dataset[1:length(dataset)] = dataset
    
    # Add augmented data
    for i in 1:eachindex(dataset)
        for j in 1:(augmentation_factor-1)
            idx = length(dataset) + (i-1)*(augmentation_factor-1) + j
            if idx <= length(augmented_dataset)
                img, lbl = augment_image(dataset[i][1], dataset[i][2])
                augmented_dataset[idx] = (img, lbl)
            end
        end
    end
    
    println("Created augmented dataset with $(length(augmented_dataset)) samples")
    
    return shuffle(augmented_dataset)
end

# Debug batches to see label distribution
function debug_batch(batch, batch_idx)
    input_batch, label_batch = batch
    
    #println("\n===== BATCH $batch_idx DEBUG =====")
    #println("Input batch shape: $(size(input_batch))")
    #println("Label batch shape: $(size(label_batch))")
    
    # Get overall statistics for this batch
    batch_size = size(input_batch, 4)
    #println("Batch size: $batch_size")
    
    # Check each sample in the batch
    for i in 1:batch_size
        sample_input = input_batch[:,:,:,i]
        sample_label = label_batch[:,:,:,i]
        
        # println("\nSample $i in batch $batch_idx:")
        # println("  Input shape: $(size(sample_input))")
        # println("  Input range: min=$(minimum(sample_input)), max=$(maximum(sample_input)), mean=$(mean(sample_input))")
        # println("  Label shape: $(size(sample_label))")
        
        # Check label distribution
        flat_labels = reshape(sample_label, :)
        unique_labels = unique(flat_labels)
        
        # println("  Unique labels: $unique_labels")
        # println("  Number of unique labels: $(length(unique_labels))")
        
        # Count occurrences of each label
        for label in unique_labels
            count = sum(flat_labels .== label)
            percentage = 100 * count / length(flat_labels)
            # println("    Label $label: $count pixels ($(round(percentage, digits=2))%)")
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