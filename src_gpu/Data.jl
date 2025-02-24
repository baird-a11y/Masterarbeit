##################################
# Data.jl - Complete Fixed Version
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
    
    # zu Float32 konvertieren
    img_float = Float32.(channelview(raw_img))
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
    
    # Optimierte Skalierung ohne temporäre Zwischenvariablen
    min_val, max_val = extrema(raw_label)
    scaled_label = Int.(round.((raw_label .- min_val) .* (34 / (max_val - min_val))))
    
    # Standardize to fixed dimensions
    scaled_label_std = standardize_size_2d(scaled_label)
    max_class = maximum(scaled_label_std)
    label = reshape(scaled_label_std, size(scaled_label_std,1), size(scaled_label_std,2), 1, 1)
    
    return label, max_class
end

using Base.Threads

# Load a subset of data from file lists rather than directories
function load_dataset(image_files::Vector{String}, label_files::Vector{String}; verbose=true)
    if verbose
        println("Loading dataset with $(length(image_files)) images")
        println("All images will be standardized to $(STANDARD_HEIGHT)×$(STANDARD_WIDTH)")
    end
    
    dataset = Vector{Tuple}(undef, length(image_files))
    
    # Track any processing errors
    error_count = Threads.Atomic{Int}(0)
    
    @threads for i in 1:length(image_files)
        try
            # Process image and label
            img_data = load_and_preprocess_image(image_files[i])
            label_data, _ = load_and_preprocess_label(label_files[i])
            
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
    
    if error_count[] > 0
        println("Warning: Encountered $(error_count[]) errors during dataset loading")
    end
    
    if verbose
        println("Dataset loaded successfully with $(length(dataset)) samples")
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
    for i in 1:length(dataset)
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

# Optimized batching with preallocated vector
function create_batches(dataset, batch_size)
    n_batches = ceil(Int, length(dataset) / batch_size)
    batched_data = Vector{Tuple}(undef, n_batches)
    
    for i in 1:n_batches
        batch_start = (i-1) * batch_size + 1
        batch_end = min(batch_start + batch_size - 1, length(dataset))
        batch_indices = batch_start:batch_end
        
        # Preallocate arrays and then concatenate
        imgs = cat([dataset[j][1] for j in batch_indices]...; dims=4)
        labels = cat([dataset[j][2] for j in batch_indices]...; dims=4)
        
        batched_data[i] = (imgs, labels)
    end
    
    println("Created $(n_batches) batches of size up to $(batch_size)")
    
    return batched_data
end

end # module Data