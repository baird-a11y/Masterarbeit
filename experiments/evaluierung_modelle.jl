using Flux
using BSON: @load, @save  # Import der Makros für Modellspeicherung
using BSON  # Vollständiger Import der BSON-Bibliothek
using Images
using Statistics
using CUDA
using FileIO
using ColorSchemes
using Plots
using Dates  # Für Zeitstempel in der Zusammenfassung

# Struktur für U-Net-Modell (muss mit der Trainingsstruktur übereinstimmen)
struct UNet
    encoder1
    encoder2
    encoder3
    encoder4
    bottleneck
    decoder4
    decoder4_1
    decoder3
    decoder3_1
    decoder2
    decoder2_1
    decoder1
    decoder1_1
end

# Ermöglicht Parameterupdates durch Flux (modernere Syntax)
Flux.@layer UNet

# Crop and Concat Funktion (identisch wie im Trainingscode)
function crop_and_concat(x, skip, dims=3)
    x_size = size(x)
    skip_size = size(skip)
    
    height_diff = skip_size[1] - x_size[1]
    width_diff  = skip_size[2] - x_size[2]
    
    if height_diff < 0 || width_diff < 0
        # Skip-Verbindung ist kleiner, daher auffüllen
        padded_skip = zeros(eltype(skip),
                            max(x_size[1], skip_size[1]),
                            max(x_size[2], skip_size[2]),
                            skip_size[3], skip_size[4])
        
        h_start = abs(min(0, height_diff)) ÷ 2 + 1
        w_start = abs(min(0, width_diff)) ÷ 2 + 1
        
        padded_skip[h_start:h_start+skip_size[1]-1,
                    w_start:w_start+skip_size[2]-1, :, :] .= skip
        
        return cat(x, padded_skip, dims=dims)
    else
        # Skip-Verbindung ist größer, daher zuschneiden
        h_start = height_diff ÷ 2 + 1
        w_start = width_diff  ÷ 2 + 1
        
        cropped_skip = skip[h_start:h_start+x_size[1]-1,
                            w_start:w_start+x_size[2]-1, :, :]
        
        return cat(x, cropped_skip, dims=dims)
    end
end

# Forward-Pass durch das UNet-Modell (identisch wie im Trainingscode)
function (model::UNet)(x)
    # Encoder-Pfad
    e1 = model.encoder1(x)
    e2 = model.encoder2(e1)
    e3 = model.encoder3(e2)
    e4 = model.encoder4(e3)
    b  = model.bottleneck(e4)

    # Decoder-Pfad mit Skip-Connections
    d4 = model.decoder4(b)
    d4 = model.decoder4_1(crop_and_concat(d4, e4))  # Skip-Connection von e4

    d3 = model.decoder3(d4)
    d3 = model.decoder3_1(crop_and_concat(d3, e3))  # Skip-Connection von e3

    d2 = model.decoder2(d3)
    d2 = model.decoder2_1(crop_and_concat(d2, e2))  # Skip-Connection von e2

    d1 = model.decoder1(d2)
    d1 = model.decoder1_1(crop_and_concat(d1, e1))  # Skip-Connection von e1

    return d1
end

# Konstanten für Bildgröße (müssen mit Trainingswerten übereinstimmen)
STANDARD_HEIGHT = 368
STANDARD_WIDTH = 1232

# Label-Struktur definieren (wie im Trainingscode)
struct SemanticLabel
    name::String
    id::Int
    trainId::Int
    category::String
    categoryId::Int
    hasInstances::Bool
    ignoreInEval::Bool
    color::Tuple{Int,Int,Int}
end

# Alternative einfachere Implementierung der Farblegendenfunktion ohne ImageDraw
function create_color_legend(class_colors, output_dir="S:/Masterarbeit/Auswertung")
    # Liste aller Labels für Cityscapes
    labels = [
        SemanticLabel("unlabeled", 0, 255, "void", 0, false, true, (0, 0, 0)),
        SemanticLabel("ego vehicle", 1, 255, "void", 0, false, true, (0, 0, 0)),
        SemanticLabel("rectification border", 2, 255, "void", 0, false, true, (0, 0, 0)),
        SemanticLabel("out of roi", 3, 255, "void", 0, false, true, (0, 0, 0)),
        SemanticLabel("static", 4, 255, "void", 0, false, true, (0, 0, 0)),
        SemanticLabel("dynamic", 5, 255, "void", 0, false, true, (111, 74, 0)),
        SemanticLabel("ground", 6, 255, "void", 0, false, true, (81, 0, 81)),
        SemanticLabel("road", 7, 0, "flat", 1, false, false, (128, 64, 128)),
        SemanticLabel("sidewalk", 8, 1, "flat", 1, false, false, (244, 35, 232)),
        SemanticLabel("parking", 9, 255, "flat", 1, false, true, (250, 170, 160)),
        SemanticLabel("rail track", 10, 255, "flat", 1, false, true, (230, 150, 140)),
        SemanticLabel("building", 11, 2, "construction", 2, false, false, (70, 70, 70)),
        SemanticLabel("wall", 12, 3, "construction", 2, false, false, (102, 102, 156)),
        SemanticLabel("fence", 13, 4, "construction", 2, false, false, (190, 153, 153)),
        SemanticLabel("guard rail", 14, 255, "construction", 2, false, true, (180, 165, 180)),
        SemanticLabel("bridge", 15, 255, "construction", 2, false, true, (150, 100, 100)),
        SemanticLabel("tunnel", 16, 255, "construction", 2, false, true, (150, 120, 90)),
        SemanticLabel("pole", 17, 5, "object", 3, false, false, (153, 153, 153)),
        SemanticLabel("polegroup", 18, 255, "object", 3, false, true, (153, 153, 153)),
        SemanticLabel("traffic light", 19, 6, "object", 3, false, false, (250, 170, 30)),
        SemanticLabel("traffic sign", 20, 7, "object", 3, false, false, (220, 220, 0)),
        SemanticLabel("vegetation", 21, 8, "nature", 4, false, false, (107, 142, 35)),
        SemanticLabel("terrain", 22, 9, "nature", 4, false, false, (152, 251, 152)),
        SemanticLabel("sky", 23, 10, "sky", 5, false, false, (70, 130, 180)),
        SemanticLabel("person", 24, 11, "human", 6, true, false, (220, 20, 60)),
        SemanticLabel("rider", 25, 12, "human", 6, true, false, (255, 0, 0)),
        SemanticLabel("car", 26, 13, "vehicle", 7, true, false, (0, 0, 142)),
        SemanticLabel("truck", 27, 14, "vehicle", 7, true, false, (0, 0, 70)),
        SemanticLabel("bus", 28, 15, "vehicle", 7, true, false, (0, 60, 100)),
        SemanticLabel("caravan", 29, 255, "vehicle", 7, true, true, (0, 0, 90)),
        SemanticLabel("trailer", 30, 255, "vehicle", 7, true, true, (0, 0, 110)),
        SemanticLabel("train", 31, 16, "vehicle", 7, true, false, (0, 80, 100)),
        SemanticLabel("motorcycle", 32, 17, "vehicle", 7, true, false, (0, 0, 230)),
        SemanticLabel("bicycle", 33, 18, "vehicle", 7, true, false, (119, 11, 32))
    ]
    
    # Nur trainierbare Klassen (trainId != 255) filtern und nach trainId sortieren
    trainable_labels = filter(l -> l.trainId != 255, labels)
    sort!(trainable_labels, by = l -> l.trainId)
    
    # Erzeuge Texttabelle mit Klassenliste und Farben
    legend_text = "# Cityscapes Klassenlegende\n\n"
    legend_text *= "| trainId | Name | RGB-Farbe |\n"
    legend_text *= "|---------|------|------------|\n"
    
    for label in trainable_labels
        r, g, b = label.color
        legend_text *= "| $(label.trainId) | $(label.name) | ($(r), $(g), $(b)) |\n"
    end
    
    # Speichere als Textdatei
    legend_path = joinpath(output_dir, "cityscapes_legende.txt")
    open(legend_path, "w") do io
        write(io, legend_text)
    end
    
    # Erzeuge ein Farbgitter als Bild
    patch_size = 50
    num_classes = length(trainable_labels)
    
    # Klassen in einem vertikal ausgerichteten Bild anzeigen
    legend_img = zeros(RGB{N0f8}, num_classes * patch_size, 300)
    
    # Zeichne für jede Klasse ein Farbpatch und seinen Namen
    for (i, label) in enumerate(trainable_labels)
        # Position für diesen Patch
        y_start = (i-1) * patch_size + 1
        y_end = y_start + patch_size - 5
        
        # Farbpatch
        r, g, b = label.color
        color = RGB{N0f8}(r/255, g/255, b/255)
        
        # Fülle den linken Teil mit der Klassenfarbe
        legend_img[y_start:y_end, 1:100] .= color
        
        # Fülle den rechten Teil mit weißem Hintergrund (für Text)
        legend_img[y_start:y_end, 101:300] .= RGB{N0f8}(1, 1, 1)
    end
    
    # Speichere die Farblegende
    colors_path = joinpath(output_dir, "cityscapes_farben.png")
    save(colors_path, legend_img)
    
    println("Farblegende gespeichert als:")
    println("  - Textdatei: $legend_path")
    println("  - Farbbild: $colors_path")
    
    return legend_text, legend_img
end

# Farbzuordnung für Visualisierung erstellen
function create_color_mapping()
    labels = [
        SemanticLabel("unlabeled", 0, 255, "void", 0, false, true, (0, 0, 0)),
        SemanticLabel("ego vehicle", 1, 255, "void", 0, false, true, (0, 0, 0)),
        SemanticLabel("rectification border", 2, 255, "void", 0, false, true, (0, 0, 0)),
        SemanticLabel("out of roi", 3, 255, "void", 0, false, true, (0, 0, 0)),
        SemanticLabel("static", 4, 255, "void", 0, false, true, (0, 0, 0)),
        SemanticLabel("dynamic", 5, 255, "void", 0, false, true, (111, 74, 0)),
        SemanticLabel("ground", 6, 255, "void", 0, false, true, (81, 0, 81)),
        SemanticLabel("road", 7, 0, "flat", 1, false, false, (128, 64, 128)),
        SemanticLabel("sidewalk", 8, 1, "flat", 1, false, false, (244, 35, 232)),
        SemanticLabel("parking", 9, 255, "flat", 1, false, true, (250, 170, 160)),
        SemanticLabel("rail track", 10, 255, "flat", 1, false, true, (230, 150, 140)),
        SemanticLabel("building", 11, 2, "construction", 2, false, false, (70, 70, 70)),
        SemanticLabel("wall", 12, 3, "construction", 2, false, false, (102, 102, 156)),
        SemanticLabel("fence", 13, 4, "construction", 2, false, false, (190, 153, 153)),
        SemanticLabel("guard rail", 14, 255, "construction", 2, false, true, (180, 165, 180)),
        SemanticLabel("bridge", 15, 255, "construction", 2, false, true, (150, 100, 100)),
        SemanticLabel("tunnel", 16, 255, "construction", 2, false, true, (150, 120, 90)),
        SemanticLabel("pole", 17, 5, "object", 3, false, false, (153, 153, 153)),
        SemanticLabel("polegroup", 18, 255, "object", 3, false, true, (153, 153, 153)),
        SemanticLabel("traffic light", 19, 6, "object", 3, false, false, (250, 170, 30)),
        SemanticLabel("traffic sign", 20, 7, "object", 3, false, false, (220, 220, 0)),
        SemanticLabel("vegetation", 21, 8, "nature", 4, false, false, (107, 142, 35)),
        SemanticLabel("terrain", 22, 9, "nature", 4, false, false, (152, 251, 152)),
        SemanticLabel("sky", 23, 10, "sky", 5, false, false, (70, 130, 180)),
        SemanticLabel("person", 24, 11, "human", 6, true, false, (220, 20, 60)),
        SemanticLabel("rider", 25, 12, "human", 6, true, false, (255, 0, 0)),
        SemanticLabel("car", 26, 13, "vehicle", 7, true, false, (0, 0, 142)),
        SemanticLabel("truck", 27, 14, "vehicle", 7, true, false, (0, 0, 70)),
        SemanticLabel("bus", 28, 15, "vehicle", 7, true, false, (0, 60, 100)),
        SemanticLabel("caravan", 29, 255, "vehicle", 7, true, true, (0, 0, 90)),
        SemanticLabel("trailer", 30, 255, "vehicle", 7, true, true, (0, 0, 110)),
        SemanticLabel("train", 31, 16, "vehicle", 7, true, false, (0, 80, 100)),
        SemanticLabel("motorcycle", 32, 17, "vehicle", 7, true, false, (0, 0, 230)),
        SemanticLabel("bicycle", 33, 18, "vehicle", 7, true, false, (119, 11, 32))
    ]
    
    # Erstelle Mapping von trainId zu RGB-Farbe für Visualisierung
    trainId_to_color = Dict{Int, RGB{N0f8}}()
    
    for label in labels
        if label.trainId != 255
            r, g, b = label.color
            color = RGB{N0f8}(r/255, g/255, b/255)
            trainId_to_color[label.trainId] = color
        end
    end
    
    # Speziell für ignorierte Klassen
    trainId_to_color[255] = RGB{N0f8}(0, 0, 0)  # Schwarz für ignorierte Klassen
    
    return trainId_to_color
end

# Bildvorverarbeitung für Inferenz
function preprocess_image(img_path)
    raw_img = load(img_path)
    println("Bild geladen: $(basename(img_path)), Größe: $(size(raw_img))")
    
    # In Float32-Array konvertieren
    img_float = Float32.(channelview(raw_img))
    
    # NaN/Inf-Werte behandeln
    if any(isnan, img_float) || any(isinf, img_float)
        println("  WARNUNG: NaN oder Inf-Werte im Bild gefunden, werden ersetzt")
        img_float = replace(img_float, NaN => 0.0f0, Inf => 1.0f0, -Inf => 0.0f0)
    end
    
    # Werte auf [0,1] begrenzen
    img_float = clamp.(img_float, 0.0f0, 1.0f0)
    
    # Dimensionen umordnen für CNN (Höhe, Breite, Kanäle)
    img_array = permutedims(img_float, (2, 3, 1))
    
    # Auf Standardgröße bringen
    h, w, c = size(img_array)
    final = zeros(Float32, STANDARD_HEIGHT, STANDARD_WIDTH, c)
    
    h_range = 1:min(h, STANDARD_HEIGHT)
    w_range = 1:min(w, STANDARD_WIDTH)
    final[h_range, w_range, :] .= view(img_array, h_range, w_range, 1:c)
    
    # Batch-Dimension hinzufügen
    return reshape(final, size(final)..., 1)
end

# Label-Bild vorverarbeiten
function preprocess_label(label_path, color_to_class)
    raw_label = load(label_path)
    println("Label geladen: $(basename(label_path)), Größe: $(size(raw_label))")
    
    # RGB zu Klassenindizes konvertieren
    h, w = size(raw_label)
    class_indices = zeros(Int, h, w)
    
    for i in 1:h
        for j in 1:w
            color = raw_label[i, j]
            if haskey(color_to_class, color)
                class_indices[i, j] = color_to_class[color]
            else
                # Für unbekannte Farben, Standardklasse zuweisen
                class_indices[i, j] = 255  # Ignorierte Klasse
            end
        end
    end
    
    # Auf Standardgröße bringen
    scaled_label = zeros(Int, STANDARD_HEIGHT, STANDARD_WIDTH)
    
    h_range = 1:min(h, STANDARD_HEIGHT)
    w_range = 1:min(w, STANDARD_WIDTH)
    scaled_label[h_range, w_range] .= view(class_indices, h_range, w_range)
    
    return scaled_label
end

# Farbzuordnung für Label zu trainId erstellen
function create_global_color_to_class_mapping()
    labels = [
        SemanticLabel("unlabeled", 0, 255, "void", 0, false, true, (0, 0, 0)),
        SemanticLabel("ego vehicle", 1, 255, "void", 0, false, true, (0, 0, 0)),
        SemanticLabel("rectification border", 2, 255, "void", 0, false, true, (0, 0, 0)),
        SemanticLabel("out of roi", 3, 255, "void", 0, false, true, (0, 0, 0)),
        SemanticLabel("static", 4, 255, "void", 0, false, true, (0, 0, 0)),
        SemanticLabel("dynamic", 5, 255, "void", 0, false, true, (111, 74, 0)),
        SemanticLabel("ground", 6, 255, "void", 0, false, true, (81, 0, 81)),
        SemanticLabel("road", 7, 0, "flat", 1, false, false, (128, 64, 128)),
        SemanticLabel("sidewalk", 8, 1, "flat", 1, false, false, (244, 35, 232)),
        SemanticLabel("parking", 9, 255, "flat", 1, false, true, (250, 170, 160)),
        SemanticLabel("rail track", 10, 255, "flat", 1, false, true, (230, 150, 140)),
        SemanticLabel("building", 11, 2, "construction", 2, false, false, (70, 70, 70)),
        SemanticLabel("wall", 12, 3, "construction", 2, false, false, (102, 102, 156)),
        SemanticLabel("fence", 13, 4, "construction", 2, false, false, (190, 153, 153)),
        SemanticLabel("guard rail", 14, 255, "construction", 2, false, true, (180, 165, 180)),
        SemanticLabel("bridge", 15, 255, "construction", 2, false, true, (150, 100, 100)),
        SemanticLabel("tunnel", 16, 255, "construction", 2, false, true, (150, 120, 90)),
        SemanticLabel("pole", 17, 5, "object", 3, false, false, (153, 153, 153)),
        SemanticLabel("polegroup", 18, 255, "object", 3, false, true, (153, 153, 153)),
        SemanticLabel("traffic light", 19, 6, "object", 3, false, false, (250, 170, 30)),
        SemanticLabel("traffic sign", 20, 7, "object", 3, false, false, (220, 220, 0)),
        SemanticLabel("vegetation", 21, 8, "nature", 4, false, false, (107, 142, 35)),
        SemanticLabel("terrain", 22, 9, "nature", 4, false, false, (152, 251, 152)),
        SemanticLabel("sky", 23, 10, "sky", 5, false, false, (70, 130, 180)),
        SemanticLabel("person", 24, 11, "human", 6, true, false, (220, 20, 60)),
        SemanticLabel("rider", 25, 12, "human", 6, true, false, (255, 0, 0)),
        SemanticLabel("car", 26, 13, "vehicle", 7, true, false, (0, 0, 142)),
        SemanticLabel("truck", 27, 14, "vehicle", 7, true, false, (0, 0, 70)),
        SemanticLabel("bus", 28, 15, "vehicle", 7, true, false, (0, 60, 100)),
        SemanticLabel("caravan", 29, 255, "vehicle", 7, true, true, (0, 0, 90)),
        SemanticLabel("trailer", 30, 255, "vehicle", 7, true, true, (0, 0, 110)),
        SemanticLabel("train", 31, 16, "vehicle", 7, true, false, (0, 80, 100)),
        SemanticLabel("motorcycle", 32, 17, "vehicle", 7, true, false, (0, 0, 230)),
        SemanticLabel("bicycle", 33, 18, "vehicle", 7, true, false, (119, 11, 32))
    ]
    
    color_to_class = Dict{RGB{N0f8}, Int}()
    
    for label in labels
        r, g, b = label.color
        color = RGB{N0f8}(r/255, g/255, b/255)
        
        class_idx = label.trainId
        color_to_class[color] = class_idx
    end
    
    return color_to_class
end

# Inferenz mit dem Modell durchführen
function predict_segmentation(model, input_img)
    # Daten auf GPU verschieben
    input_img_gpu = gpu(input_img)
    
    # Vorhersage machen
    output = model(input_img_gpu)
    
    # Argmax über Kanal-Dimension (Klassenzuweisung pro Pixel)
    _, indices = findmax(cpu(output), dims=3)
    class_indices = [idx[3] - 1 for idx in indices[:,:,1,:]]
    
    return class_indices
end

# IoU berechnen (Intersection over Union)
function calculate_iou(predictions, ground_truth, num_classes)
    ious = zeros(Float32, num_classes)
    valid_classes = Set{Int}()
    
    # Ignorierte Klassen (255) ausschließen aus ground_truth
    valid_mask = ground_truth .!= 255
    
    # Für jede Klasse IoU berechnen
    for class_idx in 0:(num_classes-1)
        # Nur auswerten, wenn die Klasse im Ground Truth vorhanden ist
        if any(ground_truth .== class_idx)
            push!(valid_classes, class_idx)
            
            pred_mask = predictions .== class_idx
            gt_mask = ground_truth .== class_idx
            
            # Bei der Berechnung nur gültige Pixel berücksichtigen
            pred_mask = pred_mask .& valid_mask
            gt_mask = gt_mask .& valid_mask
            
            # Schnittmenge und Vereinigungsmenge berechnen
            intersection = sum(pred_mask .& gt_mask)
            union = sum(pred_mask .| gt_mask)
            
            # IoU = Schnittmenge / Vereinigungsmenge
            if union > 0
                ious[class_idx+1] = intersection / union
            end
        end
    end
    
    # Klassen-IoUs ausgeben
    for class_idx in valid_classes
        println("IoU für Klasse $class_idx: $(round(ious[class_idx+1], digits=4))")
    end
    
    # Mean IoU nur über vorhandene Klassen
    valid_ious = [ious[i+1] for i in valid_classes]
    mean_iou = isempty(valid_ious) ? 0.0f0 : mean(valid_ious)
    
    return ious, mean_iou
end

# Segmentierungsvorhersage visualisieren
function visualize_segmentation(prediction, class_colors)
    h, w = size(prediction)
    result = Array{RGB{N0f8}}(undef, h, w)
    
    for i in 1:h
        for j in 1:w
            class_idx = prediction[i, j]
            if haskey(class_colors, class_idx)
                result[i, j] = class_colors[class_idx]
            else
                # Schwarze Farbe für unbekannte Klassen
                result[i, j] = RGB{N0f8}(0, 0, 0)
            end
        end
    end
    
    return result
end

# Funktion zum Laden eines Modells und Vergleich mit Ground Truth
function evaluate_model(model_path, image_path, label_path, output_dir="S:/Masterarbeit/Auswertung")
    println("Lade Modell: $model_path")
    
    # Sicherstellen, dass das Ausgabeverzeichnis existiert
    mkpath(output_dir)
    
    # Modell laden - flexibler mit verschiedenen Modellnamen umgehen
    model_dict = BSON.load(model_path)
    
    # Überprüfen, welche Schlüssel in der BSON-Datei vorhanden sind
    println("Verfügbare Schlüssel in der BSON-Datei: ", keys(model_dict))
    
    # Versuchen, das Modell unter verschiedenen möglichen Schlüsseln zu finden
    model_cpu = nothing
    for key_option in [:model_cpu, :model, :trained_model, :final_model, :final_model_cpu]
        if haskey(model_dict, key_option)
            model_cpu = model_dict[key_option]
            println("Modell unter Schlüssel '$key_option' gefunden")
            break
        end
    end
    
    if model_cpu === nothing
        # Falls kein bekannter Schlüssel gefunden wurde, nehmen wir den ersten Wert
        model_cpu = first(values(model_dict))
        println("Kein bekannter Modellschlüssel gefunden, verwende ersten Wert")
    end
    
    model = gpu(model_cpu)  # Auf GPU verschieben für Inferenz
    
    # Farbzuordnungen erstellen
    color_to_class = create_global_color_to_class_mapping()
    class_colors = create_color_mapping()
    
    # Erstelle auch die Legende
    create_color_legend(class_colors, output_dir)
    
    # Anzahl der Klassen (für IoU-Berechnung)
    num_classes = length(Set([v for (k,v) in color_to_class if v != 255]))
    println("Anzahl der Klassen: $num_classes")
    
    # Bild laden und vorverarbeiten
    input_img = preprocess_image(image_path)
    
    # Ground Truth Label laden
    gt_labels = preprocess_label(label_path, color_to_class)
    
    # Segmentierung vorhersagen
    println("Führe Segmentierung durch...")
    predicted_segmentation = predict_segmentation(model, input_img)
    
    # IoU berechnen
    println("\nBerechne IoU-Metriken:")
    _, mean_iou = calculate_iou(predicted_segmentation, gt_labels, num_classes)
    println("Mean IoU: $(round(mean_iou, digits=4))")
    
    # Originalbild laden für Visualisierung
    original_img = load(image_path)
    
    # Ground Truth und Vorhersage visualisieren
    gt_vis = visualize_segmentation(gt_labels, class_colors)
    pred_vis = visualize_segmentation(predicted_segmentation, class_colors)
    
    # Ergebnisse anzeigen und speichern
    result_image = vcat(
        hcat(imresize(original_img, (STANDARD_HEIGHT, STANDARD_WIDTH)), gt_vis),
        hcat(pred_vis, 0.5 .* pred_vis .+ 0.5 .* imresize(original_img, (STANDARD_HEIGHT, STANDARD_WIDTH)))
    )
    
    # Dateiname für Ausgabe
    output_filename = joinpath(output_dir, "segmentation_result_$(splitext(basename(model_path))[1]).png")
    save(output_filename, result_image)
    
    println("\nErgebnis gespeichert als: $output_filename")
    println("Obere Zeile: Original (links) und Ground Truth (rechts)")
    println("Untere Zeile: Vorhersage (links) und Vorhersage-Überlagerung (rechts)")
    
    return mean_iou, original_img, gt_vis, pred_vis, result_image
end

# Funktion zum Vergleich mehrerer Modelle
function compare_models(model_paths, image_path, label_path, output_dir="S:/Masterarbeit/Auswertung")
    # Sicherstellen, dass das Ausgabeverzeichnis existiert
    mkpath(output_dir)
    
    # Farbzuordnungen für Klassen erstellen (für Visualisierung und Legende)
    class_colors = create_color_mapping()
    
    # Farblegende erstellen
    create_color_legend(class_colors, output_dir)
    
    results = Dict()
    
    for model_path in model_paths
        println("\n==== Evaluiere Modell: $(basename(model_path)) ====")
        mean_iou, _, _, _, _ = evaluate_model(model_path, image_path, label_path, output_dir)
        results[basename(model_path)] = mean_iou
    end
    
    # Ergebnisse sortieren und ausgeben
    println("\n==== Zusammenfassung ====")
    println("Modell | Mean IoU")
    println("-------|--------")
    
    sorted_results = sort(collect(results), by=x->x[2], rev=true)
    
    for (model_name, iou) in sorted_results
        println("$model_name | $(round(iou, digits=4))")
    end
    
    # Zusammenfassungsdatei erstellen
    summary_file = joinpath(output_dir, "vergleich_zusammenfassung.txt")
    open(summary_file, "w") do io
        println(io, "===== MODELLVERGLEICH =====")
        println(io, "Testbild: $(basename(image_path))")
        println(io, "Datum: $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM"))")
        println(io, "\nModell | Mean IoU")
        println(io, "-------|--------")
        
        for (model_name, iou) in sorted_results
            println(io, "$model_name | $(round(iou, digits=4))")
        end
    end
    
    println("\nZusammenfassung gespeichert in: $summary_file")
    
    return results
end

# Beispielaufruf:
# 1. Einzelner Modellvergleich:
#evaluate_model("S:/Masterarbeit/Modelle/final_model_0001_1.bson", "S:/Masterarbeit/Datensatz/Training/image_2/000024_10.png", "S:/Masterarbeit/Datensatz/Training/semantic_rgb/000024_10.png")

# 2. Vergleich mehrerer Modelle:
 model_paths = [
     "S:/Masterarbeit/Modelle/final_model_000001.bson",
     "S:/Masterarbeit/Modelle/final_model_00001.bson",
     "S:/Masterarbeit/Modelle/final_model_0001.bson",
     "S:/Masterarbeit/Modelle/final_model_001.bson",
     "S:/Masterarbeit/Modelle/final_model_0001_70.bson"
 ]
compare_models(model_paths, "S:/Masterarbeit/Datensatz/Training/image_2/000024_10.png", "S:/Masterarbeit/Datensatz/Training/semantic_rgb/000024_10.png")