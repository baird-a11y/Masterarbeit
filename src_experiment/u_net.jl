using Images
using FileIO
using Flux
using Random
using Statistics
using ProgressMeter

# Struktur für die Label-Definitionen
struct Label
    name::String
    id::Int
    trainId::Int
    category::String
    catId::Int
    hasInstances::Bool
    ignoreInEval::Bool
    color::Tuple{Int, Int, Int}
end

# Vollständige Label-Liste gemäß der bereitgestellten Daten
labels = [
    Label("unlabeled", 0, 255, "void", 0, false, true, (0, 0, 0)),
    Label("ego vehicle", 1, 255, "void", 0, false, true, (0, 0, 0)),
    Label("rectification border", 2, 255, "void", 0, false, true, (0, 0, 0)),
    Label("out of roi", 3, 255, "void", 0, false, true, (0, 0, 0)),
    Label("static", 4, 255, "void", 0, false, true, (0, 0, 0)),
    Label("dynamic", 5, 255, "void", 0, false, true, (111, 74, 0)),
    Label("ground", 6, 255, "void", 0, false, true, (81, 0, 81)),
    Label("road", 7, 0, "flat", 1, false, false, (128, 64, 128)),
    Label("sidewalk", 8, 1, "flat", 1, false, false, (244, 35, 232)),
    Label("parking", 9, 255, "flat", 1, false, true, (250, 170, 160)),
    Label("rail track", 10, 255, "flat", 1, false, true, (230, 150, 140)),
    Label("building", 11, 2, "construction", 2, false, false, (70, 70, 70)),
    Label("wall", 12, 3, "construction", 2, false, false, (102, 102, 156)),
    Label("fence", 13, 4, "construction", 2, false, false, (190, 153, 153)),
    Label("guard rail", 14, 255, "construction", 2, false, true, (180, 165, 180)),
    Label("bridge", 15, 255, "construction", 2, false, true, (150, 100, 100)),
    Label("tunnel", 16, 255, "construction", 2, false, true, (150, 120, 90)),
    Label("pole", 17, 5, "object", 3, false, false, (153, 153, 153)),
    Label("polegroup", 18, 255, "object", 3, false, true, (153, 153, 153)),
    Label("traffic light", 19, 6, "object", 3, false, false, (250, 170, 30)),
    Label("traffic sign", 20, 7, "object", 3, false, false, (220, 220, 0)),
    Label("vegetation", 21, 8, "nature", 4, false, false, (107, 142, 35)),
    Label("terrain", 22, 9, "nature", 4, false, false, (152, 251, 152)),
    Label("sky", 23, 10, "sky", 5, false, false, (70, 130, 180)),
    Label("person", 24, 11, "human", 6, true, false, (220, 20, 60)),
    Label("rider", 25, 12, "human", 6, true, false, (255, 0, 0)),
    Label("car", 26, 13, "vehicle", 7, true, false, (0, 0, 142)),
    Label("truck", 27, 14, "vehicle", 7, true, false, (0, 0, 70)),
    Label("bus", 28, 15, "vehicle", 7, true, false, (0, 60, 100)),
    Label("caravan", 29, 255, "vehicle", 7, true, true, (0, 0, 90)),
    Label("trailer", 30, 255, "vehicle", 7, true, true, (0, 0, 110)),
    Label("train", 31, 16, "vehicle", 7, true, false, (0, 80, 100)),
    Label("motorcycle", 32, 17, "vehicle", 7, true, false, (0, 0, 230)),
    Label("bicycle", 33, 18, "vehicle", 7, true, false, (119, 11, 32))
]

# Zählen der gültigen Trainings-IDs (exklusive 255, was als 'zu ignorieren' gilt)
valid_train_ids = [l.trainId for l in labels if l.trainId != 255]
num_classes = length(valid_train_ids) > 0 ? maximum(valid_train_ids) + 1 : 0

println("Anzahl der Trainingsklassen: $num_classes")

# Erstellen einer Farbe-zu-TrainID-Zuordnung für schnellere Verarbeitung
color_to_trainId = Dict{Tuple{Int, Int, Int}, Int}()
for label in labels
    color_to_trainId[label.color] = label.trainId
end

# Bilder und Masken laden
function load_data(images_dir, masks_dir; target_size=(256, 256))
    image_files = filter(f -> endswith(f, ".png") || endswith(f, ".jpg"), readdir(images_dir))
    
    println("Gefundene Bilder: $(length(image_files))")
    
    images = []
    masks = []
    
    @showprogress "Lade Daten..." for img_file in image_files
        # Bildpfad
        img_path = joinpath(images_dir, img_file)
        
        # Gleicher Dateiname für Masken
        mask_path = joinpath(masks_dir, img_file)
        
        if isfile(mask_path)
            img = load(img_path)
            mask = load(mask_path)
            
            # Bilder auf einheitliche Größe bringen
            img_resized = imresize(img, target_size)
            mask_resized = imresize(mask, target_size)
            
            # Bild normalisieren (auf [0,1] skalieren)
            img_norm = Float32.(channelview(img_resized)) / 255.0
            if size(img_norm, 1) == 3  # RGB-Bild
                # Stellen Sie sicher, dass das Format [Kanäle, Höhe, Breite] ist
                if size(img_norm, 3) == 3
                    img_norm = permutedims(img_norm, (3, 1, 2))
                end
            end
            
            push!(images, img_norm)
            push!(masks, mask_resized)
        end
    end
    
    return images, masks
end

# Masken in Trainings-IDs umwandeln
function process_masks(masks)
    processed_masks = []
    
    @showprogress "Verarbeite Masken..." for mask in masks
        height, width = size(mask)
        # One-Hot-Encoding für die Masken erstellen
        one_hot = zeros(Float32, num_classes, height, width)
        
        for i in 1:height
            for j in 1:width
                pixel_color = (Int(red(mask[i, j])*255), Int(green(mask[i, j])*255), Int(blue(mask[i, j])*255))
                
                # Versuche die exakte Farbe zu finden
                train_id = get(color_to_trainId, pixel_color, 255)
                
                # Falls keine exakte Übereinstimmung, suche die nächste Farbe
                if train_id == 255
                    min_dist = Inf
                    for label in labels
                        dist = sum((pixel_color .- label.color).^2)
                        if dist < min_dist
                            min_dist = dist
                            train_id = label.trainId
                        end
                    end
                end
                
                # Setze die Klasse im One-Hot-Encoding, wenn es keine zu ignorierende Klasse ist
                if train_id != 255 && 0 <= train_id < num_classes
                    one_hot[train_id + 1, i, j] = 1.0
                end
            end
        end
        
        push!(processed_masks, one_hot)
    end
    
    return processed_masks
end

# Daten in Trainings- und Validierungssets aufteilen
function split_data(images, masks; split_ratio=0.8)
    n = length(images)
    indices = shuffle(1:n)
    train_size = Int(floor(n * split_ratio))
    
    train_indices = indices[1:train_size]
    val_indices = indices[train_size+1:end]
    
    train_images = images[train_indices]
    train_masks = masks[train_indices]
    val_images = images[val_indices]
    val_masks = masks[val_indices]
    
    println("Trainingsset: $(length(train_images)) Bilder")
    println("Validierungsset: $(length(val_images)) Bilder")
    
    return train_images, train_masks, val_images, val_masks
end

# Sehr vereinfachtes U-Net-Modell
function create_unet(in_channels=3, out_channels=num_classes)
    # Eine flachere, einfachere Version von U-Net mit weniger Ebenen
    # Dies hilft, Dimensionsprobleme zu vermeiden
    
    # Encoder (Downsampling-Pfad)
    encoder_conv1 = Chain(
        Conv((3, 3), in_channels => 64, pad=SamePad(), relu),
        Conv((3, 3), 64 => 64, pad=SamePad(), relu)
    )
    
    encoder_pool1 = MaxPool((2, 2))
    
    encoder_conv2 = Chain(
        Conv((3, 3), 64 => 128, pad=SamePad(), relu),
        Conv((3, 3), 128 => 128, pad=SamePad(), relu)
    )
    
    encoder_pool2 = MaxPool((2, 2))
    
    # Bottleneck
    bottleneck = Chain(
        Conv((3, 3), 128 => 256, pad=SamePad(), relu),
        Conv((3, 3), 256 => 256, pad=SamePad(), relu)
    )
    
    # Decoder (Upsampling-Pfad)
    decoder_up1 = Upsample(2)
    
    decoder_conv1 = Chain(
        Conv((3, 3), 256 + 128 => 128, pad=SamePad(), relu),
        Conv((3, 3), 128 => 128, pad=SamePad(), relu)
    )
    
    decoder_up2 = Upsample(2)
    
    decoder_conv2 = Chain(
        Conv((3, 3), 128 + 64 => 64, pad=SamePad(), relu),
        Conv((3, 3), 64 => 64, pad=SamePad(), relu)
    )
    
    # Output-Schicht
    output_layer = Conv((1, 1), 64 => out_channels)
    
    # Forward-Pass-Funktion
    function forward(x)
        # Encoder
        x1 = encoder_conv1(x)
        x = encoder_pool1(x1)
        
        x2 = encoder_conv2(x)
        x = encoder_pool2(x2)
        
        # Bottleneck
        x = bottleneck(x)
        
        # Decoder mit Skip-Connections
        x = decoder_up1(x)
        x = cat(x, x2, dims=3)  # Skip-Connection
        x = decoder_conv1(x)
        
        x = decoder_up2(x)
        x = cat(x, x1, dims=3)  # Skip-Connection
        x = decoder_conv2(x)
        
        # Output
        return output_layer(x)
    end
    
    return forward
end

# Stabilisierte Verlustfunktion
function crossentropy_loss(y_pred, y_true)
    # Stabilisierung gegen numerische Fehler
    epsilon = 1e-7
    y_pred_stable = clamp.(y_pred, -100.0, 100.0)  # Vermeidet extreme Werte
    
    return Flux.logitcrossentropy(y_pred_stable, y_true)
end

# Trainingsfunktion mit Gradientenclipping
function train_model(model, train_images, train_masks, val_images, val_masks; 
                     epochs=10, learning_rate=0.0001, batch_size=4, 
                     save_model_path=nothing)
    
    # Optimierer mit niedrigerer Lernrate
    opt = ADAM(learning_rate)
    
    # Einfaches Batching
    function create_batch(images, masks, indices)
        batch_images = cat(images[indices]..., dims=4)  # Dimensionen: [C, H, W, B]
        batch_masks = cat(masks[indices]..., dims=4)
        return batch_images, batch_masks
    end
    
    num_train = length(train_images)
    num_batches = ceil(Int, num_train / batch_size)
    
    # Speichere beste Validierungsergebnisse
    best_val_loss = Inf
    best_model_params = nothing
    
    println("Training beginnt...")
    
    for epoch in 1:epochs
        # Indizes für diese Epoche mischen
        batch_indices = shuffle(1:num_train)
        
        total_loss = 0.0
        
        @showprogress "Epoche $epoch/$epochs" for b in 1:num_batches
            # Batch-Indizes
            start_idx = (b-1) * batch_size + 1
            end_idx = min(b * batch_size, num_train)
            batch_idx = batch_indices[start_idx:end_idx]
            
            # Batch erstellen
            x_batch, y_batch = create_batch(train_images, train_masks, batch_idx)
            
            # Gradienten berechnen
            loss, back = Flux.pullback(() -> crossentropy_loss(model(x_batch), y_batch), 
                                      Flux.params(model))
            
            # Gradienten und Update mit explizitem Clipping
            grads = back(1f0)
            for (p, g) in pairs(grads)
                if g !== nothing
                    g_clipped = clamp.(g, -1.0, 1.0)  # Explizites Clipping
                    Flux.Optimise.update!(opt, p, g_clipped)
                end
            end
            
            total_loss += loss
        end
        
        # Durchschnittlicher Verlust für diese Epoche
        avg_loss = total_loss / num_batches
        
        # Validierung
        if !isempty(val_images)
            val_losses = []
            
            for i in 1:batch_size:length(val_images)
                end_idx = min(i + batch_size - 1, length(val_images))
                val_indices = i:end_idx
                
                val_x, val_y = create_batch(val_images, val_masks, val_indices)
                
                # Ohne Gradienten berechnen
                val_loss = crossentropy_loss(model(val_x), val_y)
                push!(val_losses, val_loss)
            end
            
            mean_val_loss = mean(val_losses)
            
            # Speichere das beste Modell
            if mean_val_loss < best_val_loss
                best_val_loss = mean_val_loss
                # Speichere Modellparameter
                if save_model_path !== nothing
                    @save save_model_path model
                    println("Modell gespeichert: $save_model_path (Val Loss: $mean_val_loss)")
                end
            end
            
            println("Epoche $epoch: Train Loss = $avg_loss, Val Loss = $mean_val_loss")
        else
            println("Epoche $epoch: Train Loss = $avg_loss")
        end
    end
    
    return model
end

# Vorhersagefunktion für eine einzelne Eingabe
function predict(model, image)
    # Sicherstellen, dass das Bild die richtige Dimension hat
    if ndims(image) == 3  # [C, H, W]
        image = reshape(image, size(image)..., 1)  # [C, H, W, 1]
    end
    
    # Vorhersage durchführen
    output = model(image)
    
    # Softmax anwenden und Klassenindex mit höchster Wahrscheinlichkeit bestimmen
    probs = softmax(output, dims=1)
    class_indices = Flux.onecold(probs)
    
    return class_indices[:, :, 1]  # Ergebnis zurück zu 2D [H, W]
end

# Funktion zur Visualisierung einer Vorhersage
function visualize_prediction(model, image, mask)
    # Mache eine Vorhersage
    pred_mask = predict(model, image)
    
    # Erstelle ein Bild aus der Vorhersage
    pred_vis = create_color_mask(pred_mask)
    
    # Erstelle ein Bild aus der Grundwahrheit
    true_vis = mask  # Diese sollte bereits ein Bild sein
    
    # Originalbild (umwandeln von [C,H,W] zu [H,W,C] für die Anzeige)
    orig_img = permutedims(image, (2, 3, 1))
    
    # Bilder nebeneinander anzeigen (vereinfachte Version)
    return [orig_img true_vis pred_vis]
end

# Funktion zur Erstellung eines farbigen Maskenbildes aus Klassenindizes
function create_color_mask(mask)
    height, width = size(mask)
    color_mask = zeros(RGB{Float32}, height, width)
    
    for i in 1:height, j in 1:width
        class_id = mask[i, j]
        
        # Finde das Label mit der entsprechenden trainId
        label_idx = findfirst(l -> l.trainId == class_id - 1, labels)  # -1 wegen 1-basierter Indizierung
        
        if label_idx !== nothing
            # Konvertiere RGB-Werte in den Bereich [0, 1]
            r, g, b = labels[label_idx].color
            color_mask[i, j] = RGB(r/255, g/255, b/255)
        else
            # Schwarz für unbekannte Klassen
            color_mask[i, j] = RGB(0, 0, 0)
        end
    end
    
    return color_mask
end

# Hauptfunktion
function main(images_dir, masks_dir; target_size=(256, 256), epochs=10, batch_size=4, learning_rate=0.0001, save_path=nothing)
    # Daten laden und vorbereiten
    println("Lade Daten...")
    images, masks = load_data(images_dir, masks_dir, target_size=target_size)
    
    println("Verarbeite Masken...")
    processed_masks = process_masks(masks)
    
    println("Teile Daten...")
    train_images, train_masks, val_images, val_masks = split_data(images, processed_masks)
    
    println("Erstelle Modell...")
    model = create_unet()
    
    println("Starte Training...")
    trained_model = train_model(model, train_images, train_masks, val_images, val_masks, 
                              epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                              save_model_path=save_path)
    
    return trained_model, train_images, train_masks, val_images, val_masks
end

# Diese Funktion kann folgendermaßen aufgerufen werden:
model, train_images, train_masks, val_images, val_masks = main(
    "S:/Masterarbeit/Datensatz/Training/images", 
    "S:/Masterarbeit/Datensatz/Training/semantic_rgb/", 
    target_size=(256, 256),  # Kleinere Bildgröße für schnelleres Training
    epochs=15,               # Anzahl der Trainingsepochen
    batch_size=2,            # Kleine Batch-Größe für Stabilität
    learning_rate=0.0001,    # Niedrige Lernrate für Stabilität
    save_path="mein_unet_modell.bson"  # Pfad zum Speichern des besten Modells
)