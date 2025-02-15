using Flux
using Flux: @functor, onehotbatch
using Statistics
using Random
using Plots
using Images
using ImageTransformations
using BSON: @save, @load
using CUDA  # Falls eine GPU verwendet wird

################################################################################
# 1) Dictionary erstellen: Grauwert -> Klassenindex (falls nötig)
################################################################################
function collect_label_mapping(label_dir::String)
    println("DEBUG: collect_label_mapping => label_dir = ", label_dir)
    label_files = sort(readdir(label_dir, join=true))
    println("DEBUG: Number of label files found: ", length(label_files))

    sample_lbl = load(label_files[1])
    sample_eltype = eltype(sample_lbl)
    println("DEBUG: sample_eltype (from first file) = ", sample_eltype)

    # Erzeuge ein Set vom Elementtyp, nicht vom 'Typ des Typs'
    unique_vals = Set{sample_eltype}()

    for (i, lbl_path) in enumerate(label_files)
        lbl = load(lbl_path)
        vals_in_label = unique(vec(lbl))
        println("DEBUG: label file #$i = $lbl_path => found ", length(vals_in_label), " unique values")
        foreach(v -> push!(unique_vals, v), vals_in_label)
    end

    sorted_vals = sort(collect(unique_vals))
    println("DEBUG: total unique values across all labels = ", length(sorted_vals))

    val_to_idx = Dict(val => i-1 for (i,val) in enumerate(sorted_vals))
    return val_to_idx
end

function mask_to_classindices(label, val_to_idx)
    println("DEBUG: mask_to_classindices => Entering function")

    println("DEBUG: label (raw) -> size=", size(label), ", eltype=", eltype(label))
    label_arr = Array(label)
    println("DEBUG: label_arr -> size=", size(label_arr), ", eltype=", eltype(label_arr))

    class_mask = similar(label_arr, Int)
    for i in eachindex(label_arr)
        class_mask[i] = val_to_idx[label_arr[i]]
    end
    # (H, W) -> (H, W, 1, 1)
    class_mask = reshape(class_mask, size(class_mask,1), size(class_mask,2), 1, 1)
    println("DEBUG: class_mask -> size=", size(class_mask), ", eltype=", eltype(class_mask))
    return class_mask
end

################################################################################
# UNet-Definition
################################################################################
function unet(input_channels::Int, output_channels::Int)
    println("DEBUG: Building UNet => input_channels=$input_channels, output_channels=$output_channels")

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

    function forward(x)
        enc1 = encoder1(x)
        enc2 = encoder2(enc1)
        enc3 = encoder3(enc2)
        enc4 = encoder4(enc3)
        bottleneck_out = bottleneck(enc4)
        dec4 = decoder4_1(cat(decoder4(bottleneck_out), enc4, dims=3))
        dec3 = decoder3_1(cat(decoder3(dec4), enc3, dims=3))
        dec2 = decoder2_1(cat(decoder2(dec3), enc2, dims=3))
        dec1 = decoder1_1(cat(decoder1(dec2), enc1, dims=3))
        return dec1
    end
    return Chain(forward)
end

################################################################################
# 2) Daten laden und in Tensoren überführen
################################################################################
function load_and_preprocess_image(img_path::String, target_size::Tuple{Int,Int})
    println("DEBUG: load_and_preprocess_image => path=", img_path)
    img = load(img_path)
    img_f32 = Float32.(channelview(img)) ./ 255
    img_f32 = permutedims(img_f32, (2, 3, 1))
    resized_img = imresize(img_f32, (target_size[1], target_size[2]))
    if size(resized_img, 3) == 1
        resized_img = cat(resized_img, resized_img, resized_img; dims=3)
    end
    out = reshape(resized_img, size(resized_img)..., 1)
    println("DEBUG: load_and_preprocess_image => shape=", size(out), ", eltype=", eltype(out))
    return out
end

function load_and_preprocess_label(label_path::String,
                                   val_to_idx::Dict,
                                   target_size::Tuple{Int,Int})
    println("DEBUG: load_and_preprocess_label => path=", label_path)
    lbl_raw = load(label_path)
    println("DEBUG: lbl_raw => shape=", size(lbl_raw), ", eltype=", eltype(lbl_raw))

    resized_lbl = imresize(lbl_raw, target_size; interp=:nearest)
    println("DEBUG: resized_lbl => shape=", size(resized_lbl), ", eltype=", eltype(resized_lbl))

    class_mask = mask_to_classindices(resized_lbl, val_to_idx)
    println("DEBUG: load_and_preprocess_label => final class_mask shape=", size(class_mask))
    return class_mask
end

function load_dataset(image_dir::String, label_dir::String,
                      val_to_idx::Dict, target_size::Tuple{Int,Int})
    println("DEBUG: load_dataset => image_dir=", image_dir, ", label_dir=", label_dir)

    image_files = sort(readdir(image_dir, join=true))
    label_files = sort(readdir(label_dir, join=true))
    println("DEBUG: Found ", length(image_files), " image files")
    println("DEBUG: Found ", length(label_files), " label files")

    dataset = [
        (
            load_and_preprocess_image(img_path, target_size),
            load_and_preprocess_label(lbl_path, val_to_idx, target_size)
        )
        for (img_path, lbl_path) in zip(image_files, label_files)
    ]
    return dataset
end

################################################################################
# 3) Batching mit Debug
################################################################################
function create_batches(dataset, batch_size)
    batched_data = []
    i = 1
    batch_index = 1
    while i <= length(dataset)
        start_idx = i
        end_idx   = min(i + batch_size - 1, length(dataset))
        println("DEBUG: create_batches => batch#$batch_index with dataset indices = $start_idx:$end_idx")

        batch = dataset[start_idx:end_idx]
        imgs   = cat([b[1] for b in batch]...; dims=4)
        labels = cat([b[2] for b in batch]...; dims=4)

        println("DEBUG: create_batches => imgs size=", size(imgs), " & labels size=", size(labels))
        push!(batched_data, (imgs, labels))
        i += batch_size
        batch_index += 1
    end
    return batched_data
end

################################################################################
# 4) Training mit Debug-Ausgaben
################################################################################

function train_unet(model, train_data, num_epochs, learning_rate, output_channels)
    opt = Adam(learning_rate)
    opt_state = Flux.setup(opt, model)
    trainable_params = Flux.trainable(model)

    function loss_fn(x, y)
        pred = model(x)
        println("DEBUG: Prediction shape: ", size(pred))
        println("DEBUG: Ground truth shape: ", size(y))

        # Falls y nicht One-Hot ist, dann logitcrossentropy nutzen
        loss = Flux.logitcrossentropy(pred, y)
        println("DEBUG: Loss-Wert: ", loss)
        return loss
    end

    for epoch in 1:num_epochs
        total_loss = 0f0
        for (i, (input_batch, mask_batch)) in enumerate(train_data)
            println("DEBUG: Epoch $epoch - Batch $i")
            println("DEBUG: Input batch size: ", size(input_batch))
            println("DEBUG: Label batch size: ", size(mask_batch))

            # Speicher vor der GPU-Zuordnung checken
            println("DEBUG: GPU Memory before batch: ", CUDA.memory_status())

            xb = gpu(input_batch)
            yb = gpu(mask_batch)

            # Speicher nach der GPU-Zuordnung checken
            println("DEBUG: GPU Memory after batch: ", CUDA.memory_status())

            # Gradientenberechnung mit expliziter Closure
            gs = gradient(trainable_params) do
                loss_fn(xb, yb)
            end

            # Speicher vor dem Update checken
            println("DEBUG: GPU Memory before update: ", CUDA.memory_status())

            # Update
            Flux.update!(opt_state, trainable_params, gs)

            # Speicher nach dem Update checken
            println("DEBUG: GPU Memory after update: ", CUDA.memory_status())

            # Loss berechnen
            batch_loss = loss_fn(xb, yb)
            total_loss += batch_loss

            println("DEBUG: Batch $i abgeschlossen - Loss: ", batch_loss)
            
            # GPU-Speicher freigeben
            CUDA.reclaim()
        end

        println("Epoch $epoch completed. Avg. Loss: ", total_loss / length(train_data))
    end
end


################################################################################
# 5) Visualisierung
################################################################################
function visualize_results(model, input_image, ground_truth)
    println("DEBUG: visualize_results => input_image size=", size(input_image),
            ", ground_truth size=", size(ground_truth))

    if ndims(input_image) == 3
        input_image = reshape(input_image, size(input_image)..., 1)
    end
    x_gpu = gpu(input_image)
    pred_gpu = model(x_gpu)
    pred = Array(pred_gpu)

    println("DEBUG: visualize_results => pred size=", size(pred))

    input_img_2d = input_image[:, :, :, 1]
    gt_mask_2d   = ground_truth[:, :, 1, 1]
    pred_2d      = pred[:, :, :, 1]

    # pred_classes = Flux.onecold(pred_2d, 0:(size(pred_2d,3)-1); dims=3)

    # Reverse(., dims=1) bei Bedarf
    
    plot(
        heatmap(channelview(reverse(input_img_2d, dims=1)), title="Input", color=:viridis),
        heatmap(reverse(gt_mask_2d, dims=1), title="Ground Truth", color=:viridis),
        heatmap(reverse(pred_classes, dims=1), title="Predicted Classes", color=:viridis),
        layout=(1, 3),
        size=(900, 300)
    )
end

################################################################################
# HAUPTPROGRAMM
################################################################################
function main()
    image_dir = "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Bilder_1"
    label_dir = "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Masken_1"

    val_to_idx = collect_label_mapping(label_dir)
    nclasses   = length(val_to_idx)
    println("Gefundene Klassenanzahl: ", nclasses)

    input_channels = 3
    output_channels = nclasses
    learning_rate = 0.001
    num_epochs    = 1
    batch_size    = 4
    target_size   = (375, 1242)

    model = unet(input_channels, output_channels)
    model = gpu(model)
    train_dataset      = load_dataset(image_dir, label_dir, val_to_idx, target_size)
    batched_train_data = create_batches(train_dataset, batch_size)

    train_unet(model, batched_train_data, num_epochs, learning_rate, output_channels)

    input_image  = batched_train_data[1][1][:, :, :, 1]
    ground_truth = batched_train_data[1][2][:, :, :, 1]
    visualize_results(model, input_image, ground_truth)
end

main()
