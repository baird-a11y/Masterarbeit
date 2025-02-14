##############################
# Data.jl
##############################
module Data

using Images, FileIO, Statistics

# Bild laden und normalisieren
function load_and_preprocess_image(img_path::String)
    img = Float32.(channelview(load(img_path))) / 255.0f0  # Hier als Float32 literal
    img = permutedims(img, (2, 3, 1))
    return reshape(img, size(img)..., 1)
end

# Label laden, normalisieren und skalieren
function load_and_preprocess_label(label_path::String)
    raw_label = load(label_path)
    println("DEBUG: Raw Label Min/Max: ", minimum(raw_label), " / ", maximum(raw_label))
    norm_label = (raw_label .- minimum(raw_label)) ./ (maximum(raw_label) - minimum(raw_label))
    scaled_label = Int.(round.(norm_label .* 34))
    println("DEBUG: Normalized Label Min/Max: ", minimum(norm_label), " / ", maximum(norm_label))
    println("DEBUG: Scaled Label Min/Max: ", minimum(scaled_label), " / ", maximum(scaled_label))
    
    unique_values = unique(scaled_label)
    println("DEBUG: Unique Scaled Label Values: ", unique_values)
    num_channels = length(unique_values)
    println("DEBUG: Number of Output Channels (Classes): ", num_channels)
    
    # Get the largest class (i.e. the maximum of the unique values)
    max_class = maximum(unique_values)
    println("DEBUG: Largest class (maximum value): ", max_class)
    
    # Reshape the label to the desired dimensions
    label = reshape(permutedims(scaled_label, (1, 2)), size(scaled_label, 1), size(scaled_label, 2), 1, 1)
    return label, max_class
end

# Datensatz erstellen
function load_dataset(image_dir::String, label_dir::String)
    image_files = sort(readdir(image_dir, join=true))
    label_files = sort(readdir(label_dir, join=true))
    dataset = [(load_and_preprocess_image(img), load_and_preprocess_label(lbl)) 
               for (img, lbl) in zip(image_files, label_files)]
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
