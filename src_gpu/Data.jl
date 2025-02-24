##################################
# Data.jl
##################################
module Data

using Images, FileIO, Statistics

# 3D-Daten zurechtschneiden/auffüllen
function standardize_size(img::AbstractArray{T,3}, target_h::Int, target_w::Int) where {T}
    h, w, c = size(img)
    cropped = view(img, 1:min(h, target_h), 1:min(w, target_w), 1:c)
    final = zeros(T, target_h, target_w, c)
    final[1:size(cropped,1), 1:size(cropped,2), :] .= cropped
    return final
end

# 2D-Daten (Labels) zurechtschneiden/auffüllen
function standardize_size_2d(matrix::AbstractMatrix{T}, target_h::Int, target_w::Int) where {T}
    h, w = size(matrix)
    cropped = view(matrix, 1:min(h, target_h), 1:min(w, target_w))
    final = zeros(T, target_h, target_w)
    final[1:size(cropped,1), 1:size(cropped,2)] .= cropped
    return final
end

# Bild laden und normalisieren
function load_and_preprocess_image(img_path::String; target_h::Int=374, target_w::Int=1238)
    raw_img = load(img_path)
    # zu Float32 konvertieren
    img_float = float.(raw_img)
    img_chan = channelview(img_float)
    img_array = permutedims(img_chan, (2, 3, 1))
    img_std = standardize_size(img_array, target_h, target_w)
    return reshape(img_std, size(img_std)..., 1)
end

# Label laden und auf 0..34 skalieren
function load_and_preprocess_label(label_path::String; target_h::Int=374, target_w::Int=1238)
    raw_label = load(label_path)
    norm_label = (raw_label .- minimum(raw_label)) ./ (maximum(raw_label) - minimum(raw_label))
    scaled_label = Int.(round.(norm_label .* 34))
    scaled_label_std = standardize_size_2d(scaled_label, target_h, target_w)
    max_class = maximum(scaled_label_std)
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
