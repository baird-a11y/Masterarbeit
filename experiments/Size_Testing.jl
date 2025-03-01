using Flux
using Flux: @functor
using Statistics
using Random
using Plots
using FileIO  # Für das Laden von Bildern
using Images  # Für Bildverarbeitung
using ImageTransformations
using ColorTypes


input_channels = 1  # Anzahl der Eingabekanäle (z. B. Graustufenbilder)
output_channels = 1  # Anzahl der Ausgabekanäle (z. B. Segmentierungsmasken)

learning_rate = 0.001 # Lernrate
num_epochs = 1 # Anzahl der Epochen
mask_batch = 1 # Anzahl der Masken


# UNet-Modell definieren
function unet(input_channels::Int, output_channels::Int)
    # Encoder
    encoder = Chain(
        (x -> begin
            println("Eingabegröße Encoder: ", size(x))
            x
        end),
        Conv((2, 2), input_channels => 64, relu, pad=1),  # Erste Convolution
        (x -> begin
            println("Nach Conv: ", size(x))
            x
        end),
        MaxPool((2, 2), stride=(2, 2)),                   # Einmaliges Downsampling
        (x -> begin
            println("Nach MaxPool: ", size(x))
            x
        end),
        Conv((2, 2), 64 => 128, relu, pad=1),
        (x -> begin
            println("Nach Conv_2: ", size(x))
            x
        end),             # Zweite Convolution
        MaxPool((2, 2), stride=(2, 2)),                   # Zweites Downsampling
        (x -> begin
            println("Nach zweiter MaxPool: ", size(x))
            x
        end),
        Conv((2, 2), 128 => 256, relu, pad=1),
        (x -> begin
            println("Nach Conv_3: ", size(x))
            x
        end),             # Dritte Convolution
        MaxPool((2, 2), stride=(2, 2)),                   # Drittes Downsampling
        (x -> begin
            println("Nach dritter MaxPool: ", size(x))
            x
        end),
        Conv((2, 2), 256 => 512, relu, pad=1),
        (x -> begin
            println("Nach Conv_4: ", size(x))
            x
        end),             # Vierte Convolution
        MaxPool((2, 2), stride=(2, 2)),                   # Viertes Downsampling
        (x -> begin
            println("Nach viertem MaxPool: ", size(x))
            x
        end),
        Conv((2, 2), 512 => 1024, relu, pad=1),
        (x -> begin
            println("Nach Conv_5: ", size(x))
            x
        end),             # Fünfte Convolution
        MaxPool((2, 2), stride=(2, 2)),                   # Fünftes Downsampling
        (x -> begin
            println("Nach fünftem MaxPool: ", size(x))
            x
        end)
        
    )
    
    # Decoder
    decoder = Chain(
        (x -> begin
            println("Eingabegröße Decoder: ", size(x))
            x
        end),
        ConvTranspose((2, 2), 1024 => 512, stride=(2,2)),   # Erste Upsampling
        (x -> begin
            println("Nach erstem Upsampling: ", size(x))
            x
        end),
        ConvTranspose((2, 2), 512 => 256, stride=(2,2)),   # Zweites Upsampling
        (x -> begin
            println("Nach zweitem Upsampling: ", size(x))
            x
        end),
        ConvTranspose((2, 2), 256 => 128, stride=(2,2)),   #  Upsampling
        (x -> begin
            println("Nach erstem Upsampling: ", size(x))
            x
        end),        
        ConvTranspose((2, 2), 128 => 64, stride=(2,2)),   #  Upsampling
        (x -> begin
            println("Nach zweitem Upsampling: ", size(x))
            x
        end),
        ConvTranspose((2, 2), 64 => output_channels, stride=(2,2)),  # Finales
        (x -> begin
            println("Nach finalem ConvTranspose: ", size(x))
            x
        end)
        
        
    )
    
    return Chain(encoder, decoder)
end

model = unet(input_channels, output_channels)




function train_unet(model, train_data, num_epochs::Int, learning_rate::Float64)
    # Setze den Optimierer auf
    opt_state = Flux.setup(Adam(learning_rate), model)
    
    # Verlustfunktion: Logarithmische Kreuzentropie
    loss(m, x, y) = Flux.logitcrossentropy(m(x), y)
    
    for epoch in 1:num_epochs
        total_loss = 0.0
        for (input_batch, mask_batch) in train_data
            # Konvertiere die Ground-Truth-Daten in Float32
            mask_batch = Array{Float32}(mask_batch)
            mask_batch .= mask_batch ./ maximum(mask_batch)  # Normalisierung
            
            # Debugging: Typ und Wertebereich prüfen
            println("Typ von m(x): ", eltype(model(input_batch)))
            println("Typ von y: ", eltype(mask_batch))
            println("Wertebereich von y: ", minimum(mask_batch), " - ", maximum(mask_batch))
            
            # Berechne Gradienten
            grads = Flux.gradient(m -> loss(m, input_batch, mask_batch), model)
            
            # Aktualisiere die Modellparameter
            Flux.update!(opt_state, model, grads)
        end
        println("Epoch $epoch completed")
    end
end

# # Beispiel-Datensatz vorbereiten
# train_data = [(batched_img_pic, batched_img_mask)]  # Eingabe und Maske
# train_unet(model, train_data, num_epochs, learning_rate)

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

function pad_to_power_of_two(img)
    # Aktuelle Dimensionen
    h, w = size(img, 1), size(img, 2)
    
    # Zielgrößen berechnen (nächste Zweierpotenz)
    target_h = 2^ceil(Int, log2(h))  # Nächste Potenz von 2 für die Höhe
    target_w = 2^ceil(Int, log2(w))  # Nächste Potenz von 2 für die Breite
    
    # Padding berechnen
    pad_h = target_h - h
    pad_w = target_w - w
    pad_top = div(pad_h, 2)
    pad_bottom = pad_h - pad_top
    pad_left = div(pad_w, 2)
    pad_right = pad_w - pad_left

    # Padding hinzufügen
    return padarray(img,Fill(0.5253879,(pad_top, pad_left), (pad_bottom, pad_right)))
end


# Daten laden und vorbereiten
img_path_pic = "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Bilder/000101_10.png"
img_pic = load(img_path_pic)
# img_pic = pad_to_power_of_two(img_pic)
gray_img_pic = Float32.(Gray.(img_pic)) ./ 255.0
batched_img_pic = reshape(gray_img_pic, size(gray_img_pic, 1), size(gray_img_pic, 2), 1, 1)

# Skaliere nur die Höhe und Breite
resized_img_pic = imresize(gray_img_pic, (512, 2048))

# Reshape erneut für die U-Net-kompatiblen Dimensionen (Höhe, Breite, Kanäle, Batchgröße)
batched_img_pic_resized = reshape(resized_img_pic, size(resized_img_pic, 1), size(resized_img_pic, 2), 1, 1)



img_path_mask = "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Masken/000101_10.png"
img_mask = load(img_path_mask)
# img_mask = pad_to_power_of_two(img_mask)
gray_img_mask = Float32.(Gray.(img_mask)) ./ 255.0
batched_img_mask = reshape(gray_img_mask, size(gray_img_mask, 1), size(gray_img_mask, 2), 1, 1)

# Skaliere nur die Höhe und Breite
resized_img_mask = imresize(gray_img_mask, (512, 2048))

# Reshape erneut für die U-Net-kompatiblen Dimensionen (Höhe, Breite, Kanäle, Batchgröße)
batched_img_mask_resized = reshape(resized_img_mask, size(resized_img_pic, 1), size(resized_img_mask, 2), 1, 1)




# Modell erstellen
model = unet(input_channels, output_channels)

# Training
train_data = [(batched_img_pic_resized, batched_img_mask_resized)]
train_unet(model, train_data, num_epochs, learning_rate)

# Ergebnisse visualisieren
visualize_results(model, batched_img_pic, batched_img_mask)
