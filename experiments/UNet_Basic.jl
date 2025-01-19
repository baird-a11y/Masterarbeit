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
num_epochs = 100 # Anzahl der Epochen
mask_batch = 1 # Anzahl der Masken


# UNet-Modell definieren
function unet(input_channels::Int, output_channels::Int)
    # Encoder
    encoder = Chain(
        (x -> begin
            println("Eingabegröße Encoder: ", size(x))
            x
        end),
        Conv((2, 3), input_channels => 64, relu, pad=1),  # Erste Convolution
        (x -> begin
            println("Nach Conv: ", size(x))
            x
        end),
        MaxPool((2, 2), stride=(2, 2)),                   # Einmaliges Downsampling
        (x -> begin
            println("Nach MaxPool: ", size(x))
            x
        end),
        Conv((3, 3), 64 => 128, relu, pad=1),
        (x -> begin
            println("Nach Conv_2: ", size(x))
            x
        end),             # Zweite Convolution
        MaxPool((2, 2), stride=(2, 3)),                   # Zweites Downsampling
        (x -> begin
            println("Nach zweiter MaxPool: ", size(x))
            x
        end)
        
    )
    
    # Decoder
    decoder = Chain(
        (x -> begin
            println("Eingabegröße Decoder: ", size(x))
            x
        end),
        
        ConvTranspose((5, 3), 128 => 64, stride=(2,3),pad=(1,0)),   # Zweites Upsampling
        (x -> begin
            println("Nach erstem Upsampling: ", size(x))
            x
        end),
        ConvTranspose((1, 4), 64 => output_channels, stride=(2,2),pad=(1, 1)),  # Upsampling
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


# Daten laden und vorbereiten
img_path_pic = "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Testdaten/Bilder/000101_10.png"
img_pic = load(img_path_pic)
gray_img_pic = Float32.(Gray.(img_pic)) ./ 255.0
batched_img_pic = reshape(gray_img_pic, size(gray_img_pic, 1), size(gray_img_pic, 2), 1, 1)

img_path_mask = "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Testdaten/Masken/000101_10.png"
img_mask = load(img_path_mask)
gray_img_mask = Float32.(Gray.(img_mask)) ./ 255.0
batched_img_mask = reshape(gray_img_mask, size(gray_img_mask, 1), size(gray_img_mask, 2), 1, 1)

# Modell erstellen
model = unet(input_channels, output_channels)

# Training
train_data = [(batched_img_pic, batched_img_mask)]
train_unet(model, train_data, num_epochs, learning_rate)

# Ergebnisse visualisieren
visualize_results(model, batched_img_pic, batched_img_mask)