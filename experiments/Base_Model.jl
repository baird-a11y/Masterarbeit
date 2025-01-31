using Flux
using Flux: @functor, onehotbatch
using Statistics
using Random
using Plots
using Images
using ImageTransformations
using BSON: @save, @load

# Modellparameter
input_channels = 3
output_channels = 35
learning_rate = 0.001
num_epochs = 1
batch_size = 4
target_size = (375, 1242)


# function resize_tensor_spatial(tensor, target_h, target_w)
#     h, w, c, b = size(tensor)
#     println("Input Size:", size(tensor))
#     println("Target Size:", target_h, target_w)
#     println("Batch Size:", b)
#     println("Channel Size:", c)
#     println("Resized Size:", target_h," ", target_w, " ", c," ", b) 
#     resized = zeros(Float32, target_h, target_w, c, b)  # Platz für das Ergebnis
#     for i in 1:c, j in 1:b
#         resized[:, :, i, j] = imresize(tensor[:, :, i, j], (target_h, target_w))  # Einzelne Kanäle skalieren
#     end
#     return resized
# end

# UNet mit Skip-Connections
function unet(input_channels::Int, output_channels::Int)
    encoder1 = Chain(Conv((3, 3), input_channels => 64, relu, pad=1),
                     Conv((3, 3), 64 => 64, relu, pad=1))
    encoder2 = Chain(MaxPool((2, 2)),
                     Conv((2,2), 64 => 128, relu, pad=1),
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
    decoder4 = Chain(ConvTranspose((2, 3), 1024 => 512, stride=2, pad = 1))
    decoder4_1 = Chain(Conv((3, 3), 1024 => 512, relu, pad=1),
                       Conv((3, 3), 512 => 512, relu, pad=1))
    decoder3 = Chain(ConvTranspose((3, 2), 512 => 256, stride=2, pad = 1))
    decoder3_1 = Chain(Conv((3, 3), 512 => 256, relu, pad=1),
                       Conv((3, 3), 256 => 256, relu, pad=1))
    decoder2 = Chain(ConvTranspose((2, 2), 256 => 128, stride=2, pad = 1))
    decoder2_1 = Chain(Conv((3, 3), 256 => 128, relu, pad=1),
                     Conv((3, 3), 128 => 128, relu, pad=1))
    decoder1 = Chain(ConvTranspose((3, 2), 128 => 64, stride=2, pad = 1))
    decoder1_1 = Chain(Conv((3, 3), 128 => 64, relu, pad=1),
                     Conv((3, 3), 64 => 64, relu, pad=1),
                     Conv((1, 1), 64 => output_channels))
                     

    function forward(x)
        enc1 = encoder1(x)
        # print("Enc1",size(enc1),"\n")
        enc2 = encoder2(enc1)
        # print("Enc2",size(enc2),"\n")
        enc3 = encoder3(enc2)
        # print("Enc3",size(enc3),"\n")
        enc4 = encoder4(enc3)
        # print("Enc4",size(enc4),"\n")
        bottleneck_out = bottleneck(enc4)
        # print("Bottleneck",size(bottleneck_out),"\n")
        dec4 = decoder4_1(cat(decoder4(bottleneck_out), enc4, dims=3))
        # print("Dec4 combined",size(dec4),"\n")
        dec3 = decoder3_1(cat(decoder3(dec4), enc3, dims=3))
        # print("Dec3",size(dec3),"\n")
        dec2 = decoder2_1(cat(decoder2(dec3), enc2, dims=3))
        # print("Dec2",size(dec2),"\n")
        dec1 = decoder1_1(cat(decoder1(dec2), enc1, dims=3))
        # print("Dec1",size(dec1))
        return dec1
    end

    return Chain(forward)
end

# Modell initialisieren
model = unet(input_channels, output_channels)

# Daten laden und vorbereiten
function load_and_preprocess_image(img_path::String)
    img = Float32.(channelview(load(img_path))) / 255.0
    img = permutedims(img, (2, 3, 1))
    return reshape(img, size(img)..., 1)
end

function load_and_preprocess_label(label_path::String)
    label = Int.(round.(load(label_path) .* 255))
    label = reshape(permutedims(label, (1, 2)), size(label, 1), size(label, 2), 1, 1)
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

# Training
# function train_unet(model, train_data, num_epochs, learning_rate)
#     opt_state = Flux.setup(Adam(learning_rate), model)
#     loss(m, x, y) = Flux.crossentropy(Flux.softmax(m(x), dims=3), y)

#     for epoch in 1:num_epochs
#         total_loss = 0.0
#         for (input_batch, mask_batch) in train_data
#             mask_batch = permutedims(onehotbatch(mask_batch[:, :, 1, :], 0:(output_channels-1)), (2, 3, 1, 4))
#             grads = Flux.gradient(() -> loss(model, input_batch, mask_batch), Flux.params(model))
#             Flux.update!(opt_state, Flux.params(model), grads)
#             total_loss += loss(model, input_batch, mask_batch)
#         end
#         println("Epoch $epoch completed. Loss: ", total_loss / length(train_data))
#     end
# end

function train_unet(model, train_data, num_epochs, learning_rate, output_channels)
    opt = Adam(learning_rate)
    trainable = Flux.params(model)  #  Speichert alle trainierbaren Parameter

    # Loss-Funktion mit Crossentropy für Multi-Klassen-Segmentierung
    function loss_fn(x, y)
        pred = model(x)
        return Flux.crossentropy(Flux.softmax(pred, dims=4), y)
    end

    for epoch in 1:num_epochs
        total_loss = 0f0
        for (input_batch, mask_batch) in train_data
            # Masken in One-Hot-Encoding konvertieren
            mask_batch = permutedims(onehotbatch(mask_batch[:, :, 1, :], 0:(output_channels-1)), (2, 3, 1, 4))

            # Sicherstellen, dass die Daten auf der GPU sind (falls GPU vorhanden)
            input_batch, mask_batch = Flux.gpu(input_batch), Flux.gpu(mask_batch)

            #  Explizite Gradientenberechnung über `trainable`
            gs = gradient(() -> loss_fn(input_batch, mask_batch), trainable)
            Flux.update!(opt, trainable, gs)

            # Loss speichern
            total_loss += mean(loss_fn(input_batch, mask_batch))
        end
        println("Epoch $epoch completed. Loss: ", total_loss / length(train_data))
    end
end





# IoU-Berechnung
function calculate_iou(predictions, labels)
    intersection = sum((predictions .== 1) .& (labels .== 1))
    union = sum((predictions .== 1) .| (labels .== 1))
    return intersection / union
end

# Visualisierung
function visualize_results(model, input_image, ground_truth)
    prediction = model(input_image)
    input_image = input_image[:, :, 1, 1]
    ground_truth = ground_truth[:, :, 1, 1]
    prediction = prediction[:, :, 1, 1]
    plot(
        heatmap(input_image, title="Input Image", color=:viridis),
        heatmap(ground_truth, title="Ground Truth Mask", color=:viridis),
        heatmap(prediction, title="Predicted Mask", color=:viridis),
        layout=(1, 3),
        size=(900, 300)
    )
end

# Datenpfade
image_dir = "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Bilder_1"
label_dir = "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Masken_1"

# Datensatz laden und vorbereiten
train_dataset = load_dataset(image_dir, label_dir)
batched_train_data = create_batches(train_dataset, batch_size)

# Training starten
train_unet(model, batched_train_data, num_epochs, learning_rate)

# Visualisierung der Ergebnisse
input_image = batched_train_data[1][1][:, :, :, 1]
ground_truth = batched_train_data[1][2][:, :, :, 1]
visualize_results(model, input_image, ground_truth)
