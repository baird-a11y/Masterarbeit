module Data

using Images, FileIO, Statistics, Random

########################################################################
# 1) Dein bestehender Code für "reale" Bilder + Masken
########################################################################

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
    
    max_class = maximum(unique_values)
    println("DEBUG: Largest class (maximum value): ", max_class)
    
    label = reshape(permutedims(scaled_label, (1, 2)), size(scaled_label, 1), size(scaled_label, 2), 1, 1)
    return label, max_class
end

function load_dataset(image_dir::String, label_dir::String)
    image_files = sort(readdir(image_dir, join=true))
    label_files = sort(readdir(label_dir, join=true))
    dataset = [(load_and_preprocess_image(img), first(load_and_preprocess_label(lbl))) 
               for (img, lbl) in zip(image_files, label_files)]
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

########################################################################
# 2) Synthetische Datenerzeugung (Proof-of-Concept: bewegter Punkt)
########################################################################

"""
    generate_sample_moving_point(image_size=64, max_speed=3)

Erzeugt ein einzelnes (input_image, target_image)-Paar. 
- input_image hat Shape (H, W, 3):
  1. Kanal: aktuelles Frame (Punkt)
  2. Kanal: vx (als konst. Matrix)
  3. Kanal: vy (als konst. Matrix)
- target_image hat Shape (H, W, 1):
  Punkt an der Position, an der er nach 1 Zeitschritt landet.
"""
function generate_sample_moving_point(image_size=64, max_speed=3)
    # Zufällige Position
    x_pos = rand(1:image_size)
    y_pos = rand(1:image_size)
    # Zufällige Geschwindigkeit
    vx = rand(-max_speed:max_speed)
    vy = rand(-max_speed:max_speed)
    
    # Frame_t
    frame_t = zeros(Float32, image_size, image_size)
    frame_t[x_pos, y_pos] = 1f0
    
    # Nächste Position (geclamped)
    x_next = clamp(x_pos + vx, 1, image_size)
    y_next = clamp(y_pos + vy, 1, image_size)
    
    # Frame_(t+1)
    frame_tplus1 = zeros(Float32, image_size, image_size)
    frame_tplus1[x_next, y_next] = 1f0

    # Velocity-Masks
    vx_mask = fill(Float32(vx), image_size, image_size)
    vy_mask = fill(Float32(vy), image_size, image_size)

    # Input: (H, W, 3)
    input_image = cat(frame_t, vx_mask, vy_mask; dims=3)
    # Target: (H, W, 1)
    target_image = reshape(frame_tplus1, image_size, image_size, 1)

    return input_image, target_image
end

"""
    generate_synthetic_dataset(N; image_size=64, max_speed=3)

Erzeugt N Samples (Input, Target) für das bewegte-Punkt-Szenario.
"""
function generate_synthetic_dataset(N::Int; image_size=64, max_speed=3)
    dataset = Vector{Tuple{Array{Float32,3},Array{Float32,3}}}(undef, N)
    for i in 1:N
        dataset[i] = generate_sample_moving_point(image_size, max_speed)
    end
    return dataset
end

"""
    create_synthetic_batches(dataset, batch_size)

Batcht den synthetischen Datensatz. 
Gibt eine Liste zurück, in der jedes Element = (input_batch, target_batch).
- input_batch hat Shape (H, W, C, N)
- target_batch hat Shape (H, W, C, N)
"""
function create_synthetic_batches(dataset, batch_size)
    idx = shuffle(1:length(dataset))
    batches = []
    i = 1
    while i <= length(dataset)
        upper = min(i+batch_size-1, length(dataset))
        batch_idx = idx[i:upper]
        # Input- und Target-Bilder sammeln
        Xlist = [dataset[j][1] for j in batch_idx]
        Ylist = [dataset[j][2] for j in batch_idx]
        
        X = cat(Xlist...; dims=4)  # (H, W, 3, BatchSize)
        Y = cat(Ylist...; dims=4)  # (H, W, 1, BatchSize)
        
        push!(batches, (X, Y))
        i = i + batch_size
    end
    return batches
end

########################################################################
# 3) Laden bereits erzeugter synthetischer Bilder von Disk (optional)
########################################################################

"""
    load_synthetic_image(path::String)

Lädt ein einzelnes Bild (z. B. PNG) als Float32-Array (H, W, C).
Achtung: Hier wird *kein* Normalisieren auf 0..1 erzwungen, 
außer Du willst das so.
"""
function load_synthetic_image(path::String)
    img = load(path)
    # Wir nehmen an, dass es ein Farbbild oder Graustufenbild ist.
    # In Images.jl kann es sein, dass `channelview` => (Channel, H, W).
    # Je nach Format musst Du ggf. anpassen.
    # Hier z. B.:
    arr = Float32.(channelview(img))       # => (Channel, H, W)
    arr = permutedims(arr, (2, 3, 1))      # => (H, W, Channel)
    return arr
end

"""
    load_synthetic_dataset(input_dir::String, target_dir::String)

Lädt Paare von Bild-Dateien (z. B. .png, .jpg).
Voraussetzung: In input_dir und target_dir liegen passende Dateien 
mit derselben Reihenfolge / Anzahl. 
Dann wird je (input_img, target_img) ein Datensatz-Eintrag erstellt.

Rückgabe: Vector von (input_array, target_array).
"""
function load_synthetic_dataset(input_dir::String, target_dir::String)
    input_files = sort(readdir(input_dir, join=true))
    target_files = sort(readdir(target_dir, join=true))
    
    @assert length(input_files) == length(target_files) "Mismatch in input/target file count!"
    
    dataset = Vector{Tuple{Array{Float32,3},Array{Float32,3}}}(undef, length(input_files))
    for i in 1:length(input_files)
        in_arr = load_synthetic_image(input_files[i]) 
        tg_arr = load_synthetic_image(target_files[i]) 
        # Stelle sicher, dass sie die richtigen Shapes haben (H,W,3) vs. (H,W,1) oder was immer Du brauchst
        dataset[i] = (in_arr, tg_arr)
    end
    return dataset
end

end # module Data
