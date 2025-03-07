##################################
# Data.jl - Optimiert für einheitliche Bildgrößen
##################################
module Data

using Images, FileIO, Statistics, Random

# Define standard dimensions for the model that work well with UNet
# Dimensions should be multiples of 16 (2^4) for a 4-level UNet
const STANDARD_HEIGHT = 368  # 368 = 23 * 16
const STANDARD_WIDTH = 1232  # 1232 = 77 * 16

# 3D-Daten zurechtschneiden/auffüllen mit festen Standarddimensionen
function standardize_size(img::AbstractArray{T,3}) where {T}
    h, w, c = size(img)
    
    # Allokiere direkt die notwendige Größe
    final = zeros(T, STANDARD_HEIGHT, STANDARD_WIDTH, c)
    
    # Kopiere nur den relevanten Bereich
    h_range = 1:min(h, STANDARD_HEIGHT)
    w_range = 1:min(w, STANDARD_WIDTH)
    final[h_range, w_range, :] .= view(img, h_range, w_range, 1:c)
    
    return final
end

# 2D-Daten (Labels) zurechtschneiden/auffüllen mit festen Standarddimensionen
function standardize_size_2d(matrix::AbstractMatrix{T}) where {T}
    h, w = size(matrix)
    
    # Allokiere direkt die notwendige Größe
    final = zeros(T, STANDARD_HEIGHT, STANDARD_WIDTH)
    
    # Kopiere nur den relevanten Bereich
    h_range = 1:min(h, STANDARD_HEIGHT)
    w_range = 1:min(w, STANDARD_WIDTH)
    final[h_range, w_range] .= view(matrix, h_range, w_range)
    
    return final
end

# Bild laden und normalisieren mit festen Dimensionen
function load_and_preprocess_image(img_path::String; verbose=false)
    raw_img = load(img_path)
    
    # Originalgrößen für Debugging
    original_size = size(raw_img)
    
    if verbose
        println("Image: $(basename(img_path))")
        println("  Original size: $original_size")
    end
    
    # zu Float32 konvertieren
    img_float = Float32.(channelview(raw_img))
    
    if verbose
        println("  Image stats: min=$(minimum(img_float)), max=$(maximum(img_float)), mean=$(mean(img_float))")
    end
    
    # Überprüfung auf NaN oder Inf Werte
    if any(isnan, img_float) || any(isinf, img_float)
        if verbose
            println("  WARNING: NaN or Inf values found in image!")
        end
        img_float = replace(img_float, NaN => 0.0f0, Inf => 1.0f0, -Inf => 0.0f0)
    end
    
    # Werte normalisieren
    img_float = clamp.(img_float, 0.0f0, 1.0f0)
    
    img_array = permutedims(img_float, (2, 3, 1))
    
    # Standardisiere auf feste Dimensionen
    img_std = standardize_size(img_array)
    return reshape(img_std, size(img_std)..., 1)
end

# Label laden und auf 0..34 skalieren mit festen Dimensionen
function load_and_preprocess_label(label_path::String; verbose=false)
    raw_label = load(label_path)
    
    # Originalgrößen für Debugging
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
    
    if verbose
        # Analysiere Label-Verteilung
        unique_labels = unique(scaled_label)
        println("  Unique labels after scaling: $unique_labels")
        println("  Number of unique labels: $(length(unique_labels))")
        
        # Zähle Vorkommen jedes Labels
        for label in unique_labels
            count = sum(scaled_label .== label)
            percentage = 100 * count / length(scaled_label)
            println("    Label $label: $count pixels ($(round(percentage, digits=2))%)")
        end
    end
    
    # Standardisiere auf feste Dimensionen
    scaled_label_std = standardize_size_2d(scaled_label)
    max_class = maximum(scaled_label_std)
    
    if verbose
        # Überprüfe, ob nach der Standardisierung neue Labels erscheinen (sollte nicht passieren)
        unique_after_std = unique(scaled_label_std)
        if length(unique_after_std) != length(unique_labels)
            println("  WARNING: Label count changed after standardization!")
            println("  Labels after standardization: $unique_after_std")
        end
        
        println("  Final label shape: $(size(scaled_label_std))")
        println("  Max class: $max_class (should be <= 34)")
    end
    
    label = reshape(scaled_label_std, size(scaled_label_std,1), size(scaled_label_std,2), 1, 1)
    
    return label, max_class
end

using Base.Threads

# Threaded dataset loading mit festen Dimensionen
function load_dataset(image_dir::String, label_dir::String; verbose=true)
    image_files = sort(readdir(image_dir, join=true))
    label_files = sort(readdir(label_dir, join=true))
    
    if verbose
        println("Loading dataset with $(length(image_files)) images")
        println("All images will be standardized to $(STANDARD_HEIGHT)×$(STANDARD_WIDTH)")
    end
    
    return load_dataset(image_files, label_files; verbose=verbose)
end

# Lade ein Subset von Daten aus Dateilisten anstatt aus Verzeichnissen
function load_dataset(image_files::Vector{String}, label_files::Vector{String}; verbose=true)
    if verbose
        println("Loading dataset with $(length(image_files)) images")
        println("All images will be standardized to $(STANDARD_HEIGHT)×$(STANDARD_WIDTH)")
    end
    
    dataset = Vector{Tuple}(undef, length(image_files))
    
    # Zähle Verarbeitungsfehler
    error_count = Threads.Atomic{Int}(0)
    
    @threads for i in 1:length(image_files)
        try
            # Verarbeite Bild und Label
            img_data = load_and_preprocess_image(image_files[i], verbose=false)
            label_data, max_class = load_and_preprocess_label(label_files[i], verbose=false)
            
            dataset[i] = (img_data, label_data)
        catch e
            Threads.atomic_add!(error_count, 1)
            if verbose
                println("Error processing file pair ($(image_files[i]), $(label_files[i])): $e")
            end
            # Stelle leere Daten als Fallback bereit
            dataset[i] = (zeros(Float32, STANDARD_HEIGHT, STANDARD_WIDTH, 3, 1),
                         zeros(Int, STANDARD_HEIGHT, STANDARD_WIDTH, 1, 1))
        end
    end
    
    if error_count[] > 0 && verbose
        println("Warning: Encountered $(error_count[]) errors during dataset loading")
    end
    
    if verbose
        println("Dataset loaded with $(length(dataset)) samples")
        println("All samples standardized to $(STANDARD_HEIGHT)×$(STANDARD_WIDTH)")
    end
    
    return dataset
end

# Optimierte Batch-Erstellung mit vorallokiertem Vector
function create_batches(dataset, batch_size; debug=false)
    n_batches = ceil(Int, length(dataset) / batch_size)
    batched_data = Vector{Tuple}(undef, n_batches)
    
    for i in 1:n_batches
        batch_start = (i-1) * batch_size + 1
        batch_end = min(batch_start + batch_size - 1, length(dataset))
        batch_indices = batch_start:batch_end
        
        # Vorallokieren und dann Konkatenieren
        imgs = cat([dataset[j][1] for j in batch_indices]...; dims=4)
        labels = cat([dataset[j][2] for j in batch_indices]...; dims=4)
        
        batched_data[i] = (imgs, labels)
    end
    
    println("Created $(n_batches) batches of size up to $(batch_size)")
    
    return batched_data
end

end # module Data