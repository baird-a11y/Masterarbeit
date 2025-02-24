##################################
# Data.jl - Optimized
##################################
module Data

using Images, FileIO, Statistics, Random

# 3D-Daten zurechtschneiden/auffüllen
function standardize_size(img::AbstractArray{T,3}, target_h::Int, target_w::Int) where {T}
    h, w, c = size(img)
    # Direkt die notwendige Größe allokieren
    final = zeros(T, target_h, target_w, c)
    # Nur den relevanten Bereich kopieren
    h_range = 1:min(h, target_h)
    w_range = 1:min(w, target_w)
    final[h_range, w_range, :] .= view(img, h_range, w_range, 1:c)
    return final
end

# 2D-Daten (Labels) zurechtschneiden/auffüllen
function standardize_size_2d(matrix::AbstractMatrix{T}, target_h::Int, target_w::Int) where {T}
    h, w = size(matrix)
    # Direkt die notwendige Größe allokieren
    final = zeros(T, target_h, target_w)
    # Nur den relevanten Bereich kopieren
    h_range = 1:min(h, target_h)
    w_range = 1:min(w, target_w)
    final[h_range, w_range] .= view(matrix, h_range, w_range)
    return final
end

# Bild laden und normalisieren
function load_and_preprocess_image(img_path::String; target_h::Int=374, target_w::Int=1238)
    raw_img = load(img_path)
    
    # Konvertierung von RGB{N0f8} zu Float32 - korrigierte Version
    # Zuerst in Array umwandeln, dann channelview, dann zu Float32 konvertieren
    img_float = Float32.(channelview(raw_img))
    
    # channelview gibt uns ein Array mit Dimensionen [channels, height, width]
    # Wir müssen die Dimensionen zu [height, width, channels] umorganisieren
    img_array = permutedims(img_float, (2, 3, 1))
    
    img_std = standardize_size(img_array, target_h, target_w)
    return reshape(img_std, size(img_std)..., 1)
end


# Label laden und auf 0..34 skalieren
function load_and_preprocess_label(label_path::String; target_h::Int=374, target_w::Int=1238)
    raw_label = load(label_path)
    # Optimierte Skalierung ohne temporäre Zwischenvariablen
    min_val, max_val = extrema(raw_label)
    scaled_label = Int.(round.((raw_label .- min_val) .* (34 / (max_val - min_val))))
    scaled_label_std = standardize_size_2d(scaled_label, target_h, target_w)
    max_class = maximum(scaled_label_std)
    label = reshape(scaled_label_std, size(scaled_label_std,1), size(scaled_label_std,2), 1, 1)
    return label, max_class
end

using Base.Threads

# Threaded dataset loading
function load_dataset(image_dir::String, label_dir::String)
    image_files = sort(readdir(image_dir, join=true))
    label_files = sort(readdir(label_dir, join=true))
    
    dataset = Vector{Tuple}(undef, length(image_files))
    
    @threads for i in 1:length(image_files)
        try
            dataset[i] = (
                load_and_preprocess_image(image_files[i]),
                first(load_and_preprocess_label(label_files[i]))
            )
        catch e
            println("Error processing image $(image_files[i]): $e")
            # Provide a default or skip this entry
            # For now, we'll just rethrow to see the specific error
            rethrow(e)
        end
    end
    
    return dataset
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
    
    return batched_data
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
    
    return shuffle(augmented_dataset)
end

end # module Data