using Flux
using Flux: @functor
using Statistics
using Random
using Plots
using FileIO  # Für das Laden von Bildern
using Images  # Für Bildverarbeitung
using ImageTransformations
using ColorTypes


input_channels = 3  # Anzahl der Eingabekanäle (z. B. Graustufenbilder)
output_channels = 35  # Anzahl der Ausgabekanäle (z. B. Segmentierungsmasken)

learning_rate = 0.001 # Lernrate
num_epochs = 1 # Anzahl der Epochen
mask_batch = 1 # Anzahl der Masken
target_size = (375, 1242) # Größe der Bilder
batch_size = 4  # Anzahl der Bilder pro Batch


# UNet-Modell definieren
function unet(input_channels::Int, output_channels::Int)
    # Encoder
    encoder = Chain(
        (x -> begin
            println("Eingabegröße Encoder: ", size(x))
            x
        end),

        Conv((3, 3), input_channels => 64, relu, pad = 1),  # Erste Convolution
        (x -> begin
            println("Nach erster Conv: ", size(x))
            x
        end),
        Conv((3, 3), 64 => 64, relu, pad = 1),  # Erste Convolution
        (x -> begin
            println("Nach 1.2: ", size(x))
            x
        end),


        Conv((3, 3), 64 => 128, relu, pad = 1),            # Zweite Convolution
        (x -> begin
            println("Nach zweiten Conv: ", size(x))
            x
        end),
        Conv((3,3), 128 => 128, relu, pad = 1),             # Zweite Convolution
        (x -> begin
            println("Nach 2.2: ", size(x))
            x
        end),
        MaxPool((2, 2), stride=(2, 2), pad = 1),                   # Zweites Downsampling
        (x -> begin
            println("Nach erste MaxPool: ", size(x))
            x
        end),


        Conv((3, 3), 128 => 256, relu, pad = 1),            # Zweite Convolution
        (x -> begin
            println("Nach drittem Conv: ", size(x))
            x
        end),             # Zweite Convolution
        Conv((3, 3), 256 => 256, relu, pad = 1),            # Zweite Convolution
        (x -> begin
            println("Nach 3.2: ", size(x))
            x
        end),
        MaxPool((2, 2), stride=(2, 2), pad = 1),                   # Zweites Downsampling
        (x -> begin
            println("Nach zweiten MaxPool: ", size(x))
            x
        end),


        Conv((3, 3), 256 => 512, relu, pad = 1),            # Zweite Convolution
        (x -> begin
            println("Nach vierten Conv: ", size(x))
            x
        end),             # Zweite Convolution
        Conv((3, 3), 512 => 512, relu, pad = 1),            # Zweite Convolution
        (x -> begin
            println("Nach 4.2: ", size(x))
            x
        end),
        MaxPool((2, 2), stride=(2, 2), pad = 1),                   # Zweites Downsampling
        (x -> begin
            println("Nach dritten MaxPool: ", size(x))
            x
        end)
        
    )
    
    # Flaschenhals
    bottleneck = Chain(
    Conv((3, 3), 512 => 1024, relu, pad=1),  # Erste Convolution
    (x -> begin
        println("Flaschenhals: ", size(x))
        x
    end),
    Conv((3, 3), 1024 => 1024, relu, pad=1),  # Zweite Convolution
    (x -> begin
        println("Nach Conv Bottle: ", size(x))
        x
    end),
    MaxPool((2, 2), stride=(2, 2), pad=1),  # MaxPool als Downsampling
    (x -> begin
        println("Nach MaxPool Bottle: ", size(x))
        x
    end)
    )


        
    # Decoder
    decoder = Chain(
    (x -> begin
        println("Eingabegröße Decoder: ", size(x))
        x
    end),
    
    # Decoder Schritt 4
    ConvTranspose((2, 2), 1024 => 512, stride=(2,2)),
    (x -> begin
        println("Nach erstem Upsampling (1024 => 512): ", size(x))
        x
    end),
    Conv((3, 3), 512 => 512, relu),
    Conv((3, 2), 512 => 512, relu, pad = 1),
    (x -> begin
        println("Nach Upsampling 1.2: ", size(x))
        x
    end),
    
    # Decoder Schritt 3
    ConvTranspose((2, 2), 512 => 256, stride=(2,2)),
    (x -> begin
        println("Nach zweiten Upsampling (512 => 256): ", size(x))
        x
    end),
    Conv((3, 3), 256 => 256, relu),
    Conv((2, 3), 256 => 256, relu, pad = 1),
    (x -> begin
        println("Nach Upsampling 2.2: ", size(x))
        x
    end),
    
    # Decoder Schritt 2
    ConvTranspose((2, 2), 256 => 128, stride=(2,2)),
    (x -> begin
        println("Nach dritten Upsampling (256 => 128): ", size(x))
        x
    end),
    Conv((3, 3), 128 => 128, relu),
    Conv((3, 3), 128 => 128, relu, pad = 1),
    (x -> begin
        println("Nach Upsampling 3.2: ", size(x))
        x
    end),
    
    # Decoder Schritt 1
    ConvTranspose((2, 2), 128 => 64, stride=(2,2)),
    (x -> begin
        println("Nach vierten Upsampling (128 => 64): ", size(x))
        x
    end),
    Conv((3, 3), 64 => 64, relu),
    Conv((3, 3), 64 => 64, relu, pad = 1),
    (x -> begin
        println("Nach Upsampling 4.2: ", size(x))
        x
    end),
    
    # Letzte Schicht
    ConvTranspose((2, 1), 64 => output_channels),
    (x -> begin
        println("Nach letzter Conv (64 => output_channels): ", size(x))
        x
    end)
    )

    return Chain(encoder,bottleneck, decoder)
end

model = unet(input_channels, output_channels)

# One-Hot-Kodierung für Labels
function onehot_labels(labels::Array{Int, 4}, num_classes::Int)
    return Flux.onehotbatch(labels[:, :, 1, :], 0:(num_classes-1))  # (H, W, C, N)
end



function train_unet(model, train_data, num_epochs::Int, learning_rate::Float64)
    # Setze den Optimierer auf
    opt_state = Flux.setup(Adam(learning_rate), model)
    
    # Verlustfunktion: Kreuzentropie für Mehrklassenklassifikation
    loss(m, x, y) = Flux.crossentropy(Flux.softmax(m(x), dims=3), y)
    
    for epoch in 1:num_epochs
        total_loss = 0.0
        for (input_batch, mask_batch) in train_data
            mask_batch = permutedims(onehot_labels(mask_batch, output_channels), (2, 3, 1, 4))
            # Debugging: Dimensionen prüfen
            println("Shape von m(x): ", size(model(input_batch)))
            println("Shape von y: ", size(mask_batch))
            
            # Berechne Gradienten
            grads = Flux.gradient(m -> loss(m, input_batch, mask_batch), model)
            
            # Aktualisiere die Modellparameter
            Flux.update!(opt_state, model, grads[1])
            total_loss += loss(model, input_batch, mask_batch)
            println("Total Loss: ", total_loss)
        end
        println("Epoch $epoch completed")
    end
end


# Funktion: Visualisierung
function visualize_results(model, input_image, ground_truth)
    # Führe Vorhersage durch
    prediction = model(input_image)
    
    # Bilder drehen (spiegelt sie vertikal)
    input_image_flipped = reverse(input_image[:, :, 1, 1], dims=1)
    ground_truth_flipped = reverse(ground_truth[:, :, 1, 1], dims=1)
    prediction_flipped = reverse(prediction[:, :, 1, 1], dims=1)
    
    # Visualisierung mit vertikalem Layout
    plot(
        heatmap(input_image_flipped, title="Input Image", color=:viridis),
        heatmap(ground_truth_flipped, title="Ground Truth Mask", color=:viridis),
        heatmap(prediction_flipped, title="Predicted Mask", color=:viridis),
        layout=(3, 1),  # Vertikales Layout
        size=(600, 900)  # Größere Plotgröße für bessere Darstellung
    )
end




# Funktion zum Laden und Verarbeiten eines einzelnen Bildes
function load_and_preprocess_image(img_path::String)
    # Lade das Bild
    img = load(img_path)

    # Stelle sicher, dass das Bild in Float32 konvertiert wird und normalisiere die Pixelwerte
    img = channelview(img)  # Bringe das Bild ins Format (C, H, W)
    img = Float32.(img) / 255.0  # Normiere die Pixelwerte auf [0, 1]

    # Bringe das Bild ins Flux-Format (H, W, C, N)
    img = permutedims(img, (2, 3, 1))  # Konvertiere zu (H, W, C)
    img = reshape(img, size(img)..., 1)  # Füge Batch-Dimension hinzu (H, W, C, N)

    return img
end

# Funktion für Labels
function load_and_preprocess_label(label_path::String)
    # Lade das Label-Bild
    label = load(label_path)

    # Konvertiere zu Integer-Werten (diskrete Klassen)
    label = round.(Int, label .* 255)

    # Bringe das Label ins Flux-Format (H, W, C, N)
    label = permutedims(label, (1, 2))  # Konvertiere zu (H, W)
    label = reshape(label, size(label)..., 1, 1)  # Füge Kanal- und Batch-Dimension hinzu (H, W, C=1, N=1)

    return label
end

function load_dataset(image_dir::String, label_dir::String)
    # Liste der Bild- und Labeldateien sortieren (wichtig für die richtige Zuordnung)
    image_files = sort(readdir(image_dir, join=true))  # Volle Pfade der Bilder
    label_files = sort(readdir(label_dir, join=true))  # Volle Pfade der Masken
    
    # Überprüfen, ob die Anzahl der Bilder und Labels übereinstimmt
    if length(image_files) != length(label_files)
        error("Die Anzahl der Bilder und Labels stimmt nicht überein!")
    end

    # Lade und preprocess die Bilder und Labels
    dataset = []
    for (img_path, label_path) in zip(image_files, label_files)
        # Bild und Label laden und preprocessen
        img = load_and_preprocess_image(img_path)
        label = load_and_preprocess_label(label_path)
        
        # Füge das Bild-Label-Paar zum Dataset hinzu
        push!(dataset, (img, label))
    end
    
    return dataset
end

function create_batches(dataset, batch_size)
    batched_data = []
    for i in 1:batch_size:length(dataset)
        # Erstelle einen Batch
        batch = dataset[i:min(i+batch_size-1, end)]
        # Separiere Bilder und Labels
        imgs = cat([b[1] for b in batch]...; dims=4)  # Stapeln entlang Batch-Dimension (N)
        labels = cat([b[2] for b in batch]...; dims=4)
        # Kombiniere die Labels entlang der Batch-Dimension
        push!(batched_data, (imgs, labels))
    end
    return batched_data
end

# Ordnerpfade für Bilder und Masken
image_dir = "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Bilder_alle"
label_dir = "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Masken_alle"

# Lade alle Bilder und Masken
train_dataset = load_dataset(image_dir, label_dir)

# Beispiel: Prüfen der Dimensionen eines geladenen Bildes und Labels
for (img, label) in train_dataset
    println("Bild Shape: ", size(img))
    println("Label Shape: ", size(label))
end

# Erstelle Batches
batched_train_data = create_batches(train_dataset, batch_size)
for (imgs, labels) in batched_train_data
    println("Batch Image Shape: ", size(imgs))  # Soll (375, 1242, 3, Batchgröße) sein
    println("Batch Label Shape: ", size(labels))  # Soll (375, 1242, 1, Batchgröße) sein
end

# # Dateipfade für Bild und Label
# img_path = "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Bilder_1/000101_10.png"
# label_path = "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Masken_1/000101_10.png"

# # Lade und preprocess das Bild
# resized_img = load_and_preprocess_image(img_path)
# println("Shape des verarbeiteten Bildes: ", size(resized_img))

# # Optional: Lade und preprocess das Label
# # Lade das Bild
# resized_label = load_and_preprocess_label(label_path)
# println("Shape des verarbeiteten Labels: ", size(resized_label))

# Modell erstellen
model = unet(input_channels, output_channels)

# Training
train_unet(model, batched_train_data, num_epochs, learning_rate)

# Ergebnisse visualisieren
# Beispiel für Visualisierung eines Bildes aus dem Batch
input_image = batched_train_data[1][1][:, :, :, 1]  # Erstes Bild aus Batch
ground_truth = batched_train_data[1][2][:, :, :, 1]  # Erstes Label aus Batch
visualize_results(model, input_image, ground_truth)
