using Flux
using Flux: @functor, onehotbatch
using Statistics
using Random
using Plots
using Images
using ImageTransformations
using BSON: @save, @load
using Functors  # Lade Functors.jl

# Modellparameter
input_channels = 3
output_channels = 35
learning_rate = 0.001
num_epochs = 5
batch_size = 4
target_size = (375, 1242)

# U-Net-Definition mit Functors für trainable Parameter
struct UNet
    layers
end
Functors.@functor UNet

function UNet(input_channels::Int, output_channels::Int)
    encoder1 = Chain(Conv((3, 3), input_channels => 64, relu, pad=1),
                     Conv((3, 3), 64 => 64, relu, pad=1))
    encoder2 = Chain(MaxPool((2, 2)),
                     Conv((2, 2), 64 => 128, relu, pad=1),
                     Conv((3, 3), 128 => 128, relu, pad=1))
    encoder3 = Chain(MaxPool((2, 2)),
                     Conv((2, 2), 128 => 256, relu, pad=1),
                     Conv((3, 3), 256 => 256, relu, pad=1))
    encoder4 = Chain(MaxPool((2, 2)),
                     Conv((2, 2), 256 => 512, relu, pad=1),
                     Conv((3, 3), 512 => 512, relu, pad=1))

    bottleneck = Chain(MaxPool((2, 2)),
                       Conv((2, 2), 512 => 1024, relu, pad=1),
                       Conv((3, 3), 1024 => 1024, relu, pad=1))
    decoder4 = Chain(ConvTranspose((2, 3), 1024 => 512, stride=2, pad=1))
    decoder4_1 = Chain(Conv((3, 3), 1024 => 512, relu, pad=1),
                       Conv((3, 3), 512 => 512, relu, pad=1))
    decoder3 = Chain(ConvTranspose((3, 2), 512 => 256, stride=2, pad=1))
    decoder3_1 = Chain(Conv((3, 3), 512 => 256, relu, pad=1),
                       Conv((3, 3), 256 => 256, relu, pad=1))
    decoder2 = Chain(ConvTranspose((2, 2), 256 => 128, stride=2, pad=1))
    decoder2_1 = Chain(Conv((3, 3), 256 => 128, relu, pad=1),
                       Conv((3, 3), 128 => 128, relu, pad=1))
    decoder1 = Chain(ConvTranspose((3, 2), 128 => 64, stride=2, pad=1))
    decoder1_1 = Chain(Conv((3, 3), 128 => 64, relu, pad=1),
                       Conv((3, 3), 64 => 64, relu, pad=1),
                       Conv((1, 1), 64 => output_channels))

    layers = Chain(
        encoder1, encoder2, encoder3, encoder4, bottleneck, 
        decoder4, decoder4_1, decoder3, decoder3_1, decoder2, decoder2_1, 
        decoder1, decoder1_1
    )
    return UNet(layers)
end

# Aufrufmethode für das UNet
function (model::UNet)(x)
    return model.layers(x)
end

# Debug-Funktion, die jeden Layer durchläuft und die Output-Shape ausgibt
function debug_forward(model, x)
    println("DEBUG: Input shape: ", size(x))
    for (i, layer) in enumerate(model.layers)
        x = layer(x)
        println("DEBUG: Nach Layer $(i), shape: ", size(x))
    end
    return x
end

# Modell initialisieren
model = UNet(input_channels, output_channels)
println("DEBUG: Trainable Parameters: ", Flux.trainable(model))

# Daten laden und vorbereiten
function load_and_preprocess_image(img_path::String)
    img = Float32.(channelview(load(img_path))) / 255.0
    img = permutedims(img, (2, 3, 1))
    return reshape(img, size(img)..., 1)
end

function load_and_preprocess_label(label_path::String)
    raw_label = load(label_path)
    println("DEBUG: Raw Label Min/Max: ", minimum(raw_label), " / ", maximum(raw_label))
    norm_label = (raw_label .- minimum(raw_label)) ./ (maximum(raw_label) - minimum(raw_label))
    scaled_label = Int.(round.(norm_label .* 34))
    println("DEBUG: Normalized Label Min/Max: ", minimum(norm_label), " / ", maximum(norm_label))
    println("DEBUG: Scaled Label Min/Max: ", minimum(scaled_label), " / ", maximum(scaled_label))
    println("DEBUG: Unique Scaled Label Values: ", unique(scaled_label))
    label = reshape(permutedims(scaled_label, (1, 2)), size(scaled_label, 1), size(scaled_label, 2), 1, 1)
    return label
end

function load_dataset(image_dir::String, label_dir::String)
    image_files = sort(readdir(image_dir, join=true))
    label_files = sort(readdir(label_dir, join=true))
    dataset = [(load_and_preprocess_image(img), load_and_preprocess_label(lbl)) 
               for (img, lbl) in zip(image_files, label_files)]
    return dataset
end

# Batch-Erstellung
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

# Trainingsfunktion mit erweiterten Debug-Ausgaben
function train_unet(model, train_data, num_epochs, learning_rate, output_channels)
    opt = Adam(learning_rate)
    opt_state = Flux.setup(opt, model)
    trainable = Flux.trainable(model)

    # Loss-Funktion mit Debug-Ausgabe
    function loss_fn(x, y)
        pred = model(x)
        println("DEBUG: Prediction shape: ", size(pred))
        println("DEBUG: Prediction Min/Max: ", minimum(pred), " / ", maximum(pred))
        return Flux.logitcrossentropy(pred, y)
    end

    for epoch in 1:num_epochs
        println("====== Epoch $epoch ======")
        total_loss = 0f0
        for (batch_idx, (input_batch, mask_batch)) in enumerate(train_data)
            println("\n--- Batch $batch_idx ---")
            # One-Hot-Kodierung der Labels
            mask_batch_int = Int.(mask_batch[:, :, 1, :])
            mask_batch_oh = permutedims(onehotbatch(mask_batch_int, 0:(output_channels-1)), (2, 3, 1, 4))
            mask_batch_oh = Float32.(mask_batch_oh)
            println("DEBUG: Mask Batch nach One-Hot, shape: ", size(mask_batch_oh))
            println("DEBUG: Mask Batch Unique Values: ", unique(mask_batch_int))

            # Verschiebe Daten auf die GPU (falls genutzt)
            input_batch, mask_batch_oh = Flux.gpu(input_batch), Flux.gpu(mask_batch_oh)

            # Debug: Ausgabe der Input-Shape
            println("DEBUG: Input Batch shape: ", size(input_batch))

            # Debug: Vorläufige Forward-Pass-Diagnose
            # Nutze debug_forward, um alle Layer zu überprüfen (nur für kleine Batches sinnvoll)
            # debug_forward(model, input_batch)

            # Berechnung des Gradienten
            gs = gradient(Flux.trainable(model)) do
                pred = model(input_batch)
                loss = Flux.logitcrossentropy(pred, mask_batch_oh)
                println("DEBUG: Loss im Gradienten-Block: ", loss)
                return loss
            end

            # Ausgabe der Gradienten-Normen für jeden Parameter
            for (p, g) in zip(Flux.trainable(model), gs)
                println("DEBUG: Gradient norm für Parameter (Größe ", size(p), "): ", norm(g))
            end

            # Debug: Vor Update eines Beispielgewichts
            println("DEBUG: Beispiel-Gewicht vor Update: ", Flux.params(model)[1])

            # Update der Parameter
            Flux.update!(opt_state, trainable, gs)

            # Debug: Nach Update
            println("DEBUG: Beispiel-Gewicht nach Update: ", Flux.params(model)[1])

            # Berechne den Batch-Loss nochmals
            batch_loss = loss_fn(input_batch, mask_batch_oh)
            println("DEBUG: Batch Loss: ", batch_loss)
            total_loss += mean(batch_loss)
        end
        println("Epoch $epoch abgeschlossen. Durchschnittlicher Loss: ", total_loss / length(train_data))
    end
end

# Visualisierungsfunktion (optional)
function visualize_results(model, input_image, ground_truth)
    if ndims(input_image) == 3
        input_image = reshape(input_image, size(input_image)..., 1)
    end
    prediction = model(input_image)
    input_image = input_image[:, :, 1, 1]
    ground_truth = ground_truth[:, :, 1, 1]
    prediction = prediction[:, :, 1, 1]
    input_image = reverse(input_image, dims=1)
    ground_truth = reverse(ground_truth, dims=1)
    prediction = reverse(prediction, dims=1)
    plot(
        heatmap(input_image, title="Input Image", color=:viridis),
        heatmap(ground_truth, title="Ground Truth Mask", color=:viridis),
        heatmap(prediction, title="Predicted Mask", color=:viridis),
        layout=(1, 3),
        size=(900, 300)
    )
end

# Datenpfade (anpassen!)
image_dir = "/local/home/baselt/Datensatz/Training/image_2"
label_dir = "/local/home/baselt/Datensatz/Training/semantic"
# Datensatz laden und vorbereiten
train_dataset = load_dataset(image_dir, label_dir)
batched_train_data = create_batches(train_dataset, batch_size)

# Training starten
train_unet(model, batched_train_data, num_epochs, learning_rate, output_channels)

# Visualisierung der Ergebnisse (Beispiel)
input_image = batched_train_data[1][1][:, :, :, 1]
ground_truth = batched_train_data[1][2][:, :, :, 1]
visualize_results(model, input_image, ground_truth)
