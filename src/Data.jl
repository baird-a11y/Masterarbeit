##############################
# Data.jl
##############################
module Data

using Images, FileIO, Statistics

# Hilfsfunktion für 3D-Daten (Images): auf gewünschte Größe zuschneiden und ggf. auffüllen
function standardize_size(img::AbstractArray{T,3}, target_h::Int, target_w::Int) where {T}
    h, w, c = size(img)
    # 1) Cropping
    cropped = view(img, 1:min(h, target_h), 1:min(w, target_w), 1:c)
    # 2) Zero-Padding
    final = zeros(T, target_h, target_w, c)
    final[1:size(cropped,1), 1:size(cropped,2), :] .= cropped
    return final
end

# Neue Hilfsfunktion für 2D-Daten (Labels): auf gewünschte Größe zuschneiden und ggf. auffüllen
function standardize_size_2d(matrix::AbstractMatrix{T}, target_h::Int, target_w::Int) where {T}
    h, w = size(matrix)
    # 1) Cropping
    cropped = view(matrix, 1:min(h, target_h), 1:min(w, target_w))
    # 2) Zero-Padding
    final = zeros(T, target_h, target_w)
    final[1:size(cropped,1), 1:size(cropped,2)] .= cropped
    return final
end

# ---------------------------
# Bild laden und normalisieren
# ---------------------------
function load_and_preprocess_image(img_path::String; target_h::Int=374, target_w::Int=1238)
    # Bild laden (z.B. Array{RGB{N0f8},2})
    raw_img = load(img_path)
    
    # 1) In Gleitkommadarstellung konvertieren
    img_float = float.(raw_img)  # -> Array{RGB{Float32},2}, [0,1]
    
    # 2) Kanal-Array holen: (Channels, H, W)
    img_chan = channelview(img_float)
    
    # 3) Für Netzwerk-Konvention (H, W, C) permutieren
    img_array = permutedims(img_chan, (2, 3, 1))
    
    # 4) Auf (374, 1238) croppen/zeropadden, ohne Interpolation
    img_std = standardize_size(img_array, target_h, target_w)

    # 5) (H, W, C) -> (H, W, C, 1) für Batch-Dimension
    return reshape(img_std, size(img_std)..., 1)
end

# ----------------------------------
# Label laden, normalisieren, skalieren
# ----------------------------------
function load_and_preprocess_label(label_path::String; target_h::Int=374, target_w::Int=1238)
    raw_label = load(label_path)
    #println("DEBUG: Raw Label Min/Max: ", minimum(raw_label), " / ", maximum(raw_label))

    norm_label = (raw_label .- minimum(raw_label)) ./ (maximum(raw_label) - minimum(raw_label))
    scaled_label = Int.(round.(norm_label .* 34))

    #println("DEBUG: Normalized Label Min/Max: ", minimum(norm_label), " / ", maximum(norm_label))
    #println("DEBUG: Scaled Label Min/Max: ", minimum(scaled_label), " / ", maximum(scaled_label))

    unique_values = unique(scaled_label)
    #println("DEBUG: Unique Scaled Label Values: ", unique_values)
    num_channels = length(unique_values)
    #println("DEBUG: Number of Output Channels (Classes): ", num_channels)
    
    max_class = maximum(unique_values)
    #println("DEBUG: Largest class (maximum value): ", max_class)
    
    # 1) (H, W) auf (374, 1238) croppen/zeropadden
    scaled_label_std = standardize_size_2d(scaled_label, target_h, target_w)

    # 2) (H, W) -> (H, W, 1, 1)
    label = reshape(scaled_label_std, size(scaled_label_std,1), size(scaled_label_std,2), 1, 1)
    return label, max_class
end

function load_dataset(image_dir::String, label_dir::String)
    image_files = sort(readdir(image_dir, join=true))
    label_files = sort(readdir(label_dir, join=true))
    dataset = [
        (
            load_and_preprocess_image(img),
            first(load_and_preprocess_label(lbl))
        )
        for (img, lbl) in zip(image_files, label_files)
    ]
    return dataset
end

# Batches erstellen
function create_batches(dataset, batch_size)
    batched_data = []
    for i in 1:batch_size:length(dataset)
        batch = dataset[i:min(i+batch_size-1, end)]
        imgs = cat([b[1] for b in batch]...; dims=4)
        labels = cat([b[2] for b in batch]...; dims=4)
        push!(batched_data, (imgs, labels))
    end
    return batched_data
end

end # module Data
