##################################
# Data.jl - Optimiert für RGB-Masken
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

# Neue Funktion für RGB Masken
function load_and_preprocess_rgb_label(label_path::String; verbose=false, cache_colors=nothing)
    raw_label = load(label_path)
    
    # Originalgrößen für Debugging
    original_size = size(raw_label)
    
    if verbose
        println("Label: $(basename(label_path))")
        println("  Original size: $original_size")
    end
    
    # Wenn kein Cache übergeben wurde, erstelle die Farbzuordnung neu
    if isnothing(cache_colors)
        # Extrahiere eindeutige Farben
        unique_colors = unique(raw_label)
        color_to_class = Dict(color => i-1 for (i, color) in enumerate(unique_colors))
        
        if verbose
            println("  Found $(length(unique_colors)) unique colors/classes")
        end
    else
        # Benutze den übergebenen Cache
        color_to_class = cache_colors
    end
    
    # Erstelle Klassenmaske
    class_mask = zeros(Int, size(raw_label))
    for i in CartesianIndices(raw_label)
        color = raw_label[i]
        # Wenn Farbe nicht im Cache, füge sie hinzu oder setze auf 0
        if haskey(color_to_class, color)
            class_mask[i] = color_to_class[color]
        else
            if verbose
                println("  WARNING: Found new color not in mapping: $color")
            end
            class_mask[i] = 0  # Default-Klasse
        end
    end
    
    # Standardisiere auf feste Dimensionen
    class_mask_std = standardize_size_2d(class_mask)
    max_class = maximum(class_mask_std)
    
    if verbose
        # Analysiere Klassenverteilung
        unique_classes = unique(class_mask_std)
        println("  Final classes: $(length(unique_classes))")
        println("  Max class: $max_class")
    end
    
    label = reshape(class_mask_std, size(class_mask_std,1), size(class_mask_std,2), 1, 1)
    
    return label, max_class, color_to_class
end

using Base.Threads

# Threaded dataset loading für RGB-Masken mit Farbzuordnungs-Cache
function load_dataset(image_dir::String, label_dir::String; verbose=true, use_rgb_masks=true)
    # Lese Verzeichnisse
    image_paths = readdir(image_dir, join=true)
    label_paths = readdir(label_dir, join=true)
    
    if verbose
        println("Found $(length(image_paths)) images and $(length(label_paths)) labels")
        println("All images will be standardized to $(STANDARD_HEIGHT)×$(STANDARD_WIDTH)")
        if use_rgb_masks
            println("Processing RGB masks with color-to-class mapping")
        else
            println("Processing grayscale masks with value scaling")
        end
    end
    
    # Sortiere Dateien alphabetisch, um Konsistenz zu gewährleisten
    sorted_image_files = sort(image_paths)
    sorted_label_files = sort(label_paths)
    
    if verbose
        println("Verifying first 5 image-label pairs for correct matching:")
        for i in 1:min(5, length(sorted_image_files))
            println("  Image: $(basename(sorted_image_files[i])) -> Label: $(basename(sorted_label_files[i]))")
        end
    end
    
    # Für RGB-Masken: erstelle erst ein Farbmapping anhand der ersten Maske
    color_to_class_cache = nothing
    if use_rgb_masks
        if verbose
            println("Creating color-to-class mapping from first mask...")
        end
        first_label, _, color_map = load_and_preprocess_rgb_label(
            sorted_label_files[1], verbose=verbose)
        color_to_class_cache = color_map
        
        if verbose
            println("Color mapping created with $(length(color_to_class_cache)) colors")
        end
    end
    
    # Lade Datensatz mit konsistentem Farbmapping
    return load_dataset(sorted_image_files, sorted_label_files; 
                      verbose=verbose, 
                      use_rgb_masks=use_rgb_masks, 
                      color_cache=color_to_class_cache)
end

# Lade ein Subset von Daten aus Dateilisten mit RGB-Unterstützung
function load_dataset(image_files::Vector{String}, label_files::Vector{String}; 
                    verbose=true, use_rgb_masks=true, color_cache=nothing)
    if verbose
        println("Loading dataset with $(length(image_files)) images")
        println("All images will be standardized to $(STANDARD_HEIGHT)×$(STANDARD_WIDTH)")
    end
    
    dataset = Vector{Tuple}(undef, length(image_files))
    
    # Zähle Verarbeitungsfehler
    error_count = Threads.Atomic{Int}(0)
    
    # Für die Fehlerbehandlung als atomare Variable
    max_class_global = Threads.Atomic{Int}(0)
    
    @threads for i in 1:length(image_files)
        try
            # Verarbeite Bild
            img_data = load_and_preprocess_image(image_files[i], verbose=false)
            
            # Verarbeite Label je nach Typ
            local label_data, max_class
            if use_rgb_masks
                label_data, max_class, _ = load_and_preprocess_rgb_label(
                    label_files[i], verbose=false, cache_colors=color_cache)
            else
                # Alte Methode für Grauwertbilder
                label_data, max_class = load_and_preprocess_label(label_files[i], verbose=false)
            end
            
            # Aktualisiere maximale Klasse global
            atomic_max!(max_class_global, max_class)
            
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
        println("Maximum class ID found: $(max_class_global[])")
        println("All samples standardized to $(STANDARD_HEIGHT)×$(STANDARD_WIDTH)")
    end
    
    return dataset
end

# Alte Label-Funktion behalten für Abwärtskompatibilität
function load_and_preprocess_label(label_path::String; verbose=false)
    raw_label = load(label_path)
    
    # Überprüfe, ob es ein RGB-Bild ist
    pixel_type = eltype(raw_label)
    is_rgb = pixel_type <: RGB || pixel_type <: RGBA
    
    if is_rgb
        # Für RGB-Bilder: Verwende die neue Funktion
        label, max_class, _ = load_and_preprocess_rgb_label(label_path, verbose=verbose)
        return label, max_class
    end
    
    # Ab hier: bisherige Implementierung für Grauwertbilder
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
    end
    
    # Standardisiere auf feste Dimensionen
    scaled_label_std = standardize_size_2d(scaled_label)
    max_class = maximum(scaled_label_std)
    
    label = reshape(scaled_label_std, size(scaled_label_std,1), size(scaled_label_std,2), 1, 1)
    
    return label, max_class
end

# Hilfsfunktion für atomare Maximierung
function atomic_max!(atomic_val, new_val)
    old_val = atomic_val[]
    while new_val > old_val
        old_val = Threads.atomic_cas!(atomic_val, old_val, new_val)
        # Wenn CAS erfolgreich war oder neuer Wert kleiner, breche ab
        if old_val >= new_val
            break
        end
    end
end

# Optimierte Batch-Erstellung mit Reihenfolgeerhaltung
function create_batches(dataset, batch_size; debug=false)
    n_batches = ceil(Int, length(dataset) / batch_size)
    batched_data = Vector{Tuple}(undef, n_batches)
    
    for i in 1:n_batches
        batch_start = (i-1) * batch_size + 1
        batch_end = min(batch_start + batch_size - 1, length(dataset))
        batch_indices = batch_start:batch_end
        
        # Hole Bilder und Labels für diesen Batch
        batch_imgs = [dataset[j][1] for j in batch_indices]
        batch_labels = [dataset[j][2] for j in batch_indices]
        
        # Prüfe auf konsistente Dimensionen
        img_dims = [size(img) for img in batch_imgs]
        label_dims = [size(label) for label in batch_labels]
        
        if debug
            println("Batch $i: Indices $batch_start to $batch_end")
            println("  Image dimensions: $img_dims")
            println("  Label dimensions: $label_dims")
        end
        
        # Vorallokieren und dann Konkatenieren
        imgs = cat(batch_imgs...; dims=4)
        labels = cat(batch_labels...; dims=4)
        
        batched_data[i] = (imgs, labels)
        
        if debug && i == 1
            println("First batch created with image shape $(size(imgs)) and label shape $(size(labels))")
        end
    end
    
    println("Created $(n_batches) batches of size up to $(batch_size) in original order")
    
    return batched_data
end

end # module Data