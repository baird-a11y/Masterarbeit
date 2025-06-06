using Flux          # Neuronale Netzwerk-Bibliothek
using Flux: onehotbatch, logitcrossentropy, gpu, cpu
using CUDA          # GPU-Unterstützung
using Statistics    # Für statistische Funktionen (mean, etc.)
using FileIO        # Datei-Ein/Ausgabe
using Images        # Bildverarbeitung
using LinearAlgebra # Für mathematische Operationen
using Optimisers    # Optimierungsalgorithmen
using ProgressMeter # Fortschrittsanzeige
using BSON: @save, @load  # Modellspeicherung
using Random        # Zufallszahlengenerator

# ==================== Label-Struktur und Klassenzuordnung ====================

# Label-Struktur definieren
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

# Globale Farbzuordnung erstellen (einmalig, nicht für jedes Bild)
function create_global_color_mapping()
    # Label-Liste aus dem Cityscapes-Dataset
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
    
    # Erstelle eine konsistente Zuordnung von RGB-Farben zu trainId-Klassen
    color_to_class = Dict{RGB{N0f8}, Int}()
    used_train_ids = Set{Int}()
    
    for label in labels
        r, g, b = label.color
        # RGB-Werte von 0-255 in 0-1 für RGB{N0f8} umwandeln
        color = RGB{N0f8}(r/255, g/255, b/255)
        
        # trainId nutzen (255 ist für "ignore" Klassen)
        class_idx = label.trainId
        if class_idx != 255
            color_to_class[color] = class_idx
            push!(used_train_ids, class_idx)
        end
    end
    
    # Anzahl gültiger Klassen (für One-Hot-Encoding)
    num_classes = length(used_train_ids)
    
    println("Globale Farbzuordnung erstellt:")
    println("  $(length(color_to_class)) gültige Farbzuordnungen")
    println("  $num_classes trainierbare Klassen (trainId 0-$(num_classes-1))")
    
    return color_to_class, num_classes
end

# Globale Farb-zu-Klassen-Zuordnung und Anzahl gültiger Klassen
const GLOBAL_COLOR_TO_CLASS, NUM_VALID_CLASSES = create_global_color_mapping()

# ==================== KONFIGURATION ====================

# Bildgrößen müssen für UNet-Architektur angepasst werden
STANDARD_HEIGHT = 368  # Höhe der verarbeiteten Bilder
STANDARD_WIDTH = 1232  # Breite der verarbeiteten Bilder
NUM_EPOCHS = 2         # Anzahl der Trainingsdurchläufe
LEARNING_RATE = 0.0001  # Lernrate für Optimierer
INPUT_CHANNELS = 3     # RGB-Eingangskanäle
OUTPUT_CHANNELS = NUM_VALID_CLASSES  # Anzahl der gültigen Klassen (0-18)
BATCH_SIZE = 2         # Bilder pro Batch
CHECKPOINT_DIR = "checkpoints"  # Speicherort für Modell-Checkpoints
SUBSET_SIZE = 200       # Teilmenge der Bilder für schnelleres Testen

# Dateipfade - an eigene Pfade anpassen
IMG_DIR = "/local/home/baselt/Datensatz/Training/image_2"        # Eingabebilder
MASK_DIR = "/local/home/baselt/Datensatz/Training/semantic_rgb"  # Semantische Masken

# ==================== HILFSFUNKTIONEN ====================

# GPU-Speicher freigeben (wichtig für längere Trainingsläufe)
function clear_gpu_memory()
    GC.gc()           # Julia-Garbage-Collection aufrufen
    CUDA.reclaim()    # CUDA-Speicher freigeben
    println("GPU-Speicher freigegeben")
end

# Überprüft, ob Bild- und Maskendateien korrekt zugeordnet sind
function check_image_label_order(image_files, label_files)
    println("\n===== ÜBERPRÜFUNG DER BILD-LABEL-REIHENFOLGE =====")
    
    # Prüfen, ob gleiche Anzahl von Bildern und Labels vorhanden ist
    if length(image_files) != length(label_files)
        println("WARNUNG: Unterschiedliche Anzahl von Bildern ($(length(image_files))) und Labels ($(length(label_files)))!")
    end
    
    # Anzahl der zu prüfenden Paare festlegen
    max_check = min(length(image_files), length(label_files), SUBSET_SIZE)
    
    println("Prüfe die ersten $max_check Paare:")
    all_matching = true
    
    # Prüfe jeden Dateinamen
    for i in 1:max_check
        img_name = basename(image_files[i])
        lbl_name = basename(label_files[i])
        
        match_status = img_name == lbl_name ? "✓" : "✗"
        
        if img_name != lbl_name
            all_matching = false
        end
        
        println("  $i: Bild=$img_name, Label=$lbl_name  $match_status")
    end
    
    # Ergebnis ausgeben
    if all_matching
        println("\nALLE PAARE STIMMEN ÜBEREIN! Die Reihenfolge ist korrekt.")
    else
        println("\nACHTUNG! Einige Paare stimmen nicht überein. Die Reihenfolge könnte falsch sein.")
    end
    
    return all_matching
end

# Hilfsfunktion zum Finden von Klassenname basierend auf TrainID
function get_class_name_by_train_id(train_id)
    # Erstelle Label-Liste erneut
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
    
    for label in labels
        if label.trainId == train_id
            return label.name
        end
    end
    
    return "unknown"
end

# Verbesserte Ausgabe der Klassenzuordnung - einmal beim Programmstart ausführen
function print_class_mapping()
    println("\n===== KLASSENZUORDNUNG FÜR TRAINING =====")
    println("Insgesamt $(NUM_VALID_CLASSES) trainierbare Klassen:")
    
    # Klassenliste erstellen
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
    
    # Nur trainierbare Klassen auflisten (trainId != 255)
    trainable_labels = filter(l -> l.trainId != 255, labels)
    
    # Nach trainId sortieren
    sort!(trainable_labels, by = l -> l.trainId)
    
    # Tabelle mit Klassenzuordnung ausgeben
    println("Index | trainId | Name")
    println("------|---------|-----------------")
    for label in trainable_labels
        r, g, b = label.color
        println("  $(label.trainId) | $(label.id) | $(label.name) (RGB: $r, $g, $b)")
    end
    
    println("\nHinweis: Die trainId wird als Index im Modell verwendet, nicht die id!")
end

# ==================== DATENVORBEREITUNG ====================

# Standardisiert die Bildgröße auf feste Dimensionen
function standardize_size(img::AbstractArray{T,3}) where {T}
    h, w, c = size(img)
    
    # Neues Array mit Standardgröße erstellen
    final = zeros(T, STANDARD_HEIGHT, STANDARD_WIDTH, c)
    
    # Bildbereich kopieren (so viel wie möglich)
    h_range = 1:min(h, STANDARD_HEIGHT)
    w_range = 1:min(w, STANDARD_WIDTH)
    final[h_range, w_range, :] .= view(img, h_range, w_range, 1:c)
    
    return final
end

# Standardisiert 2D-Daten (Labels/Masken)
function standardize_size_2d(matrix::AbstractMatrix{T}) where {T}
    h, w = size(matrix)
    
    final = zeros(T, STANDARD_HEIGHT, STANDARD_WIDTH)
    
    h_range = 1:min(h, STANDARD_HEIGHT)
    w_range = 1:min(w, STANDARD_WIDTH)
    final[h_range, w_range] .= view(matrix, h_range, w_range)
    
    return final
end

# Lädt und verarbeitet ein Eingabebild
function load_and_preprocess_image(img_path::String)
    # Bild laden
    raw_img = load(img_path)
    println("Bild: $(basename(img_path))")
    println("  Originalgröße: $(size(raw_img))")
    
    # In Float32-Array konvertieren
    img_float = Float32.(channelview(raw_img))
    
    # NaN/Inf-Werte behandeln
    if any(isnan, img_float) || any(isinf, img_float)
        println("  WARNUNG: NaN oder Inf-Werte im Bild gefunden!")
        img_float = replace(img_float, NaN => 0.0f0, Inf => 1.0f0, -Inf => 0.0f0)
    end
    
    # Werte auf [0,1] begrenzen
    img_float = clamp.(img_float, 0.0f0, 1.0f0)
    
    # Dimensionen umordnen für CNN (Höhe, Breite, Kanäle)
    img_array = permutedims(img_float, (2, 3, 1))
    
    # Auf Standardgröße bringen
    img_std = standardize_size(img_array)
    return reshape(img_std, size(img_std)..., 1)  # Batch-Dimension hinzufügen
end

# Korrigierte Version der Labelfunktion - Fehler bei der Farbdarstellung behoben
function load_and_preprocess_label(label_path::String)
    raw_label = load(label_path)
    
    println("Label: $(basename(label_path))")
    println("  Originalgröße: $(size(raw_label))")
    
    # RGB zu Klassenindizes mit globaler Zuordnung konvertieren
    h, w = size(raw_label)
    class_indices = zeros(Int, h, w)
    
    # Zähler für unbekannte Farben
    unknown_colors = Dict{RGB{N0f8}, Int}()
    class_counts = zeros(Int, OUTPUT_CHANNELS)
    
    for i in 1:h
        for j in 1:w
            color = raw_label[i, j]
            
            if haskey(GLOBAL_COLOR_TO_CLASS, color)
                class_idx = GLOBAL_COLOR_TO_CLASS[color]
                class_indices[i, j] = class_idx
                class_counts[class_idx + 1] += 1  # +1 weil Julia-Arrays bei 1 beginnen
            else
                # Für unbekannte Farben, verfolgen und Standardklasse zuweisen
                if !haskey(unknown_colors, color)
                    unknown_colors[color] = 0
                end
                unknown_colors[color] += 1
                class_indices[i, j] = 0  # Default zur Klasse "unlabeled"
            end
        end
    end
    
    # Ausgabe der Statistiken für dieses Bild
    if !isempty(unknown_colors)
        println("  WARNUNG: $(length(unknown_colors)) Farben nicht in der vordefinierten Zuordnung gefunden")
        # Top 5 häufigste unbekannte Farben anzeigen
        for (color, count) in sort(collect(unknown_colors), by=x->x[2], rev=true)[1:min(5, length(unknown_colors))]
            # Direkter Zugriff auf die RGB-Komponenten, um den channelview-Fehler zu vermeiden
            r = Float64(color.r) * 255
            g = Float64(color.g) * 255
            b = Float64(color.b) * 255
            println("    RGB($(round(Int, r)), $(round(Int, g)), $(round(Int, b))): $(count) Pixel")
        end
        if length(unknown_colors) > 5
            println("    (und $(length(unknown_colors)-5) weitere...)")
        end
    end
    
    # Gefundene Klassen ausgeben
    present_classes = [i-1 for i in 1:length(class_counts) if class_counts[i] > 0]
    println("  Gefundene Klassen in diesem Bild: $(length(present_classes))")
    for class_idx in present_classes
        class_count = class_counts[class_idx + 1]
        class_percent = round(class_count / (h * w) * 100, digits=2)
        println("    Klasse $class_idx: $(class_count) Pixel ($(class_percent)%)")
    end
    
    # Auf Standardgröße bringen
    scaled_label_std = standardize_size_2d(class_indices)
    
    label = reshape(scaled_label_std, size(scaled_label_std,1), size(scaled_label_std,2), 1, 1)
    
    return label
end

# Lädt einen kompletten Datensatz aus Bildern und Labels
function load_dataset(image_files::Vector{String}, label_files::Vector{String})
    println("Lade Datensatz mit $(length(image_files)) Bildern")
    println("Alle Bilder werden auf $(STANDARD_HEIGHT)×$(STANDARD_WIDTH) standardisiert")
    
    dataset = []
    
    for i in 1:length(image_files)
        try
            println("Verarbeite Sample $i: $(basename(image_files[i]))")
            
            # Bild und zugehörige Maske verarbeiten
            img_data = load_and_preprocess_image(image_files[i])
            label_data = load_and_preprocess_label(label_files[i])
            
            push!(dataset, (img_data, label_data))
        catch e
            println("Fehler bei Verarbeitung von ($(image_files[i]), $(label_files[i])): $e")
            println("  Fehlermeldung: ", sprint(showerror, e))
            println("  Stack-Trace: ", sprint(Base.show_backtrace, catch_backtrace()))
        end
    end
    
    return dataset
end

# Erstellt Batches aus dem Datensatz für effizientes Training
function create_batches(dataset, batch_size)
    if isempty(dataset)
        println("Warnung: Datensatz ist leer, gebe leere Batch-Liste zurück")
        return []
    end
    
    # Anzahl der Batches berechnen
    n_batches = ceil(Int, length(dataset) / batch_size)
    batched_data = []
    
    for i in 1:n_batches
        batch_start = (i-1) * batch_size + 1
        batch_end = min(batch_start + batch_size - 1, length(dataset))
        batch_indices = batch_start:batch_end
        
        # Bilder und Labels zusammenfügen
        imgs = cat([dataset[j][1] for j in batch_indices]...; dims=4)
        labels = cat([dataset[j][2] for j in batch_indices]...; dims=4)
        
        push!(batched_data, (imgs, labels))
    end
    
    println("$(n_batches) Batches mit Größe bis zu $(batch_size) erstellt")
    
    return batched_data
end

# ==================== MODELL-DEFINITION ====================

# UNet-Modellstruktur
struct UNet
    # Encoder-Pfad (Downsampling)
    encoder1
    encoder2
    encoder3
    encoder4
    bottleneck
    # Decoder-Pfad (Upsampling)
    decoder4
    decoder4_1
    decoder3
    decoder3_1
    decoder2
    decoder2_1
    decoder1
    decoder1_1
end

# Ermöglicht Parameterupdates durch Flux
Flux.@functor UNet

# Verknüpft Skip-Connections für U-Net
# Schneidet oder padded Features, damit sie zum Decoder passen
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

# Erstellt ein UNet-Modell mit gegebener Konfiguration
function create_unet(input_channels, output_channels)
    # Encoder-Stufe 1: Originalgröße
    encoder1 = Chain(
        Conv((3, 3), input_channels => 48, relu, pad=SamePad()),
        BatchNorm(48),
        Conv((3, 3), 48 => 48, relu, pad=SamePad()),
        BatchNorm(48)
    )
    
    # Encoder-Stufe 2: 1/2 Größe
    encoder2 = Chain(
        MaxPool((2,2)),
        Conv((3, 3), 48 => 96, relu, pad=SamePad()),
        BatchNorm(96),
        Conv((3, 3), 96 => 96, relu, pad=SamePad()),
        BatchNorm(96)
    )
    
    # Encoder-Stufe 3: 1/4 Größe
    encoder3 = Chain(
        MaxPool((2,2)),
        Conv((3, 3), 96 => 192, relu, pad=SamePad()),
        BatchNorm(192),
        Conv((3, 3), 192 => 192, relu, pad=SamePad()),
        BatchNorm(192)
    )
    
    # Encoder-Stufe 4: 1/8 Größe
    encoder4 = Chain(
        MaxPool((2,2)),
        Conv((3, 3), 192 => 384, relu, pad=SamePad()),
        BatchNorm(384),
        Conv((3, 3), 384 => 384, relu, pad=SamePad()),
        BatchNorm(384)
    )
    
    # Bottleneck: 1/16 Größe (tiefster Punkt)
    bottleneck = Chain(
        MaxPool((2,2)),
        Conv((3,3), 384 => 768, relu, pad=SamePad()),
        BatchNorm(768),
        Dropout(0.5),  # Verhindert Overfitting
        Conv((3,3), 768 => 768, relu, pad=SamePad()),
        BatchNorm(768)
    )
    
    # Decoder-Stufe 4: Upsampling auf 1/8 Größe
    decoder4 = ConvTranspose((2, 2), 768 => 384, stride=2)  # Transponierte Faltung zum Upsampling
    decoder4_1 = Chain(
        Conv((3,3), 768 => 384, relu, pad=SamePad()),  # 768 wegen Konkatenation
        BatchNorm(384),
        Conv((3,3), 384 => 384, relu, pad=SamePad()),
        BatchNorm(384)
    )
    
    # Decoder-Stufe 3: Upsampling auf 1/4 Größe
    decoder3 = ConvTranspose((2,2), 384 => 192, stride=2)
    decoder3_1 = Chain(
        Conv((3,3), 384 => 192, relu, pad=SamePad()),  # 384 wegen Konkatenation
        BatchNorm(192),
        Conv((3,3), 192 => 192, relu, pad=SamePad()),
        BatchNorm(192)
    )
    
    # Decoder-Stufe 2: Upsampling auf 1/2 Größe
    decoder2 = ConvTranspose((2,2), 192 => 96, stride=2)
    decoder2_1 = Chain(
        Conv((3,3), 192 => 96, relu, pad=SamePad()),  # 192 wegen Konkatenation
        BatchNorm(96),
        Conv((3,3), 96 => 96, relu, pad=SamePad()),
        BatchNorm(96)
    )
    
    # Decoder-Stufe 1: Upsampling auf Originalgröße
    decoder1 = ConvTranspose((2,2), 96 => 48, stride=2)
    decoder1_1 = Chain(
        Conv((3,3), 96 => 48, relu, pad=SamePad()),  # 96 wegen Konkatenation
        BatchNorm(48),
        Conv((3,3), 48 => 48, relu, pad=SamePad()),
        BatchNorm(48),
        Dropout(0.25),
        Conv((1,1), 48 => output_channels)  # 1x1 Faltung für finale Klassifikation
    )
    
    # UNet-Modell zusammensetzen
    model = UNet(encoder1, encoder2, encoder3, encoder4, bottleneck,
                decoder4, decoder4_1, decoder3, decoder3_1,
                decoder2, decoder2_1, decoder1, decoder1_1)
    
    return model
end

# Forward-Pass durch das UNet-Modell
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

    return d1  # Ausgabe sind die Logits für jede Klasse
end

# ==================== TRAININGS-FUNKTIONEN ====================

# One-Hot-Kodierung für Batch-Labels
function batch_one_hot(batch, num_classes)
    # Extrahiert Integer-Labels aus dem Batch
    batch_int = Int.(selectdim(batch, 3, 1))
    # Konvertiert zu One-Hot und verschiebt auf GPU
    return gpu(Float32.(permutedims(onehotbatch(batch_int, 0:(num_classes-1)), (2,3,1,4))))
end

# Verlustfunktion (Categorical Cross-Entropy)
function loss_fn(model, x, y)
    pred = model(x)
    return logitcrossentropy(pred, y)
end

# Haupt-Trainingsfunktion
function train_unet(model, train_data, num_epochs, learning_rate, output_channels; 
    checkpoint_dir="checkpoints", checkpoint_freq=1)

    # Checkpoint-Verzeichnis erstellen, falls nicht vorhanden
    mkpath(checkpoint_dir)
    
    # Optimierer konfigurieren (Adam)
    opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), model)

    # Array für Verlust pro Epoche
    losses = Float32[]

    # Prüfen, ob Trainingsdaten vorhanden sind
    if isempty(train_data)
        println("FEHLER: Keine Trainingsdaten verfügbar. Training wird übersprungen.")
        return model, losses
    end

    # Training über mehrere Epochen
    for epoch in 1:num_epochs
        println("====== Epoche $epoch/$num_epochs ======")

        total_loss = 0f0
        batch_count = 0
        batch_losses = Float32[]  # Speichert Verluste für jeden Batch
        
        println("  Verarbeite $(length(train_data)) Batches...")

        # Iteration über alle Batches
        for (batch_idx, (input_batch, mask_batch)) in enumerate(train_data)
            try
                # Labels in One-Hot-Kodierung umwandeln
                mask_batch_oh = batch_one_hot(mask_batch, output_channels)
                
                # Daten auf GPU verschieben
                input_batch = gpu(input_batch)
                
                # Gradienten und Verlust berechnen
                ∇model = gradient(m -> loss_fn(m, input_batch, mask_batch_oh), model)[1]
                batch_loss = loss_fn(model, input_batch, mask_batch_oh)
                
                # Batch-Verlust speichern
                push!(batch_losses, batch_loss)
                
                # Modellparameter aktualisieren
                opt_state, model = Optimisers.update!(opt_state, model, ∇model)
                
                total_loss += batch_loss
                batch_count += 1
                
                # Status ausgeben
                println("  Batch $batch_idx: Verlust = $batch_loss")
            
            catch e
                println("  FEHLER während des Trainings von Batch $batch_idx: $e")
                # Mit nächstem Batch fortfahren
            end

            # GPU-Speicher freigeben nach jedem Batch
            CUDA.reclaim()
        end

        # Falls keine Batches erfolgreich verarbeitet wurden
        if batch_count == 0
            println("WARNUNG: Keine Batches wurden in dieser Epoche erfolgreich verarbeitet")
            avg_loss = NaN32
        else
            avg_loss = total_loss / batch_count
        end

        push!(losses, avg_loss)

        # Statistiken für diese Epoche
        min_loss = isempty(batch_losses) ? NaN32 : minimum(batch_losses)
        max_loss = isempty(batch_losses) ? NaN32 : maximum(batch_losses)

        println("\nEpoche $epoch abgeschlossen:")
        println("  Durchschnittsverlust: $avg_loss")
        println("  Min Batch-Verlust: $min_loss")
        println("  Max Batch-Verlust: $max_loss")
        println("  Erfolgreiche Batches: $batch_count / $(length(train_data))")

        # Checkpoint speichern
        if epoch % checkpoint_freq == 0
            model_cpu = cpu(model)  # Modell auf CPU verschieben für Speicherung
            @save joinpath(checkpoint_dir, "checkpoint_epoch$(epoch).bson") model_cpu
            println("  Checkpoint für Epoche $epoch gespeichert")
        end
        
        # GPU-Speicher nach jeder Epoche freigeben
        clear_gpu_memory()
    end

    return model, losses
end

# ==================== EVALUIERUNG ====================

# Vorhersagefunktion für Inferenz
function predict(model, input_batch)
    # Daten auf GPU verschieben
    input_batch_gpu = gpu(input_batch)
    
    # Rohe Vorhersagen vom Modell erhalten
    predictions = model(input_batch_gpu)
    
    # Konvertierung in Klassenindizes (argmax entlang der Kanal-Dimension)
    _, indices = findmax(cpu(predictions), dims=3)
    class_indices = [idx[3] - 1 for idx in indices[:,:,1,:]]
    
    return class_indices
end

# Berechnet IoU (Intersection over Union) für Evaluierung
function calculate_iou(predictions, ground_truth, num_classes)
    ious = zeros(Float32, num_classes)
    
    # Für jede Klasse IoU berechnen
    for class_idx in 0:(num_classes-1)
        pred_mask = predictions .== class_idx
        gt_mask = ground_truth .== class_idx
        
        # Schnittmenge und Vereinigungsmenge berechnen
        intersection = sum(pred_mask .& gt_mask)
        union = sum(pred_mask .| gt_mask)
        
        # IoU = Schnittmenge / Vereinigungsmenge
        if union > 0
            ious[class_idx+1] = intersection / union
        end
    end
    
    # Rückgabe: IoUs pro Klasse und Mean IoU
    return ious, mean(filter(!isnan, ious))
end

# ==================== HAUPTPROGRAMM ====================

# Klassenzuordnung ausgeben
print_class_mapping()

# Lade Dateilisten
println("Lade Dateilisten...")
image_files = sort(readdir(IMG_DIR, join=true))
label_files = sort(readdir(MASK_DIR, join=true))

# Verwende nur eine Teilmenge der Bilder für schnelleres Training/Testing
subset_size = min(SUBSET_SIZE, length(image_files))
println("Verwende eine Teilmenge von $subset_size Bildern (von insgesamt $(length(image_files)))")

subset_img_files = image_files[1:subset_size]
subset_label_files = label_files[1:subset_size]

# Überprüfe die Reihenfolge der Bild-Label-Paare
is_order_correct = check_image_label_order(subset_img_files, subset_label_files)

# Datensatz laden
println("\n===== LADE DATENSATZ =====")
dataset = load_dataset(subset_img_files, subset_label_files)
println("$(length(dataset)) Samples geladen")

# Batches erstellen
println("\n===== ERSTELLE BATCHES =====")
batched_data = create_batches(dataset, BATCH_SIZE)

# Modell erstellen und initialisieren
println("\n===== INITIALISIERE MODELL =====")
model = create_unet(INPUT_CHANNELS, OUTPUT_CHANNELS)
model = gpu(model)  # Modell auf GPU verschieben

# Modell trainieren
println("\n===== STARTE TRAINING =====")
trained_model, training_losses = train_unet(
    model,
    batched_data,
    NUM_EPOCHS,
    LEARNING_RATE,
    OUTPUT_CHANNELS,
    checkpoint_dir=CHECKPOINT_DIR
)

# Finales Modell speichern
final_model_cpu = cpu(trained_model)
@save joinpath(CHECKPOINT_DIR, "final_model.bson") final_model_cpu
println("\nTraining abgeschlossen. Finales Modell gespeichert.")

# Trainingsverluste ausgeben
println("\nTrainingsverluste pro Epoche:")
for (i, loss) in enumerate(training_losses)
    println("  Epoche $i: $loss")
end

# Optional: Auf Validierungsset evaluieren, falls verfügbar
if isdir(joinpath(dirname(IMG_DIR), "Validation"))
    println("\n===== EVALUIERE MODELL =====")
    val_img_dir = joinpath(dirname(IMG_DIR), "Validation", "image_2")
    val_mask_dir = joinpath(dirname(IMG_DIR), "Validation", "semantic_rgb")
    
    # Validierungsdateien laden
    val_image_files = sort(readdir(val_img_dir, join=true))[1:min(SUBSET_SIZE, length(readdir(val_img_dir)))]
    val_label_files = sort(readdir(val_mask_dir, join=true))[1:min(SUBSET_SIZE, length(readdir(val_mask_dir)))]
    
    # Validierungsdatensatz erstellen
    val_dataset = load_dataset(val_image_files, val_label_files)
    val_batched_data = create_batches(val_dataset, BATCH_SIZE)
    
    mean_ious = []
    
    # Jeder Batch evaluieren
    for (i, (input_batch, mask_batch)) in enumerate(val_batched_data)
        try
            println("Evaluiere Batch $i...")
            predictions = predict(trained_model, input_batch)
            
            # Ground-Truth-Indizes extrahieren
            gt_indices = Int.(selectdim(cpu(mask_batch), 3, 1))
            
            # IoU berechnen
            _, mean_iou = calculate_iou(predictions, gt_indices, OUTPUT_CHANNELS)
            println("  Batch $i Mean IoU: $mean_iou")
            
            push!(mean_ious, mean_iou)
        catch e
            println("  FEHLER während der Evaluierung von Batch $i: $e")
        end
    end
    
    # Gesamtergebnis ausgeben
    if !isempty(mean_ious)
        println("\nValidierungs-Mean-IoU: $(mean(mean_ious))")
    else
        println("\nKeine gültigen Evaluierungsergebnisse verfügbar")
    end
end

println("\n===== PROGRAMM ABGESCHLOSSEN =====")