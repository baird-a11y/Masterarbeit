##################################
# Main.jl - Vereinfachte Test-Version
##################################

# Einbinden der anderen Dateien
include("Data.jl")
include("Model.jl")
include("Training.jl")
include("Visualization.jl")

using Flux
using CUDA
using .Model
using .Data
using .Training
using .Visualization
using Dates
import Base.GC
using Statistics
using FileIO, Images

# Hyperparameter
num_epochs = 10
learning_rate = 0.001
input_channels = 3
batch_size = 2
output_channels = 35
checkpoint_dir = "/local/home/baselt/checkpoints"

# GPU-Speicherverwaltung
function clear_gpu_memory()
    GC.gc()
    CUDA.reclaim()
    println("GPU memory cleared")
end

# Verzeichnisse
img_dir = "/local/home/baselt/Datensatz/Training/image_2"
mask_dir = "/local/home/baselt/Datensatz/Training/semantic"

# ------------ TESTFUNKTIONEN ------------

# 1. Test der Label-Erstellung
function test_label_creation(label_files)
    println("\n===== TEST 1: LABEL CREATION =====")
    
    for (i, label_file) in enumerate(label_files)
        println("\n---- Testing label file $i: $(basename(label_file)) ----")
        
        # Lade das ursprüngliche Label-Bild
        raw_label = load(label_file)
        println("Original size: $(size(raw_label))")
        
        # Wertebereiche des Originalbildes
        min_val, max_val = extrema(raw_label)
        println("Original range: min=$min_val, max=$max_val")
        
        # Einzigartige Werte
        unique_values = unique(raw_label)
        println("Unique values in original: $(length(unique_values)) values")
        if length(unique_values) < 10
            println("All unique values: $unique_values")
        else
            println("First 5 unique values: $(unique_values[1:5])")
        end
        
        # Teste die Skalierung auf 0-34 Bereich
        scaled_label = Int.(round.((raw_label .- min_val) .* (34 / (max_val - min_val))))
        
        # Überprüfe Skalierungsergebnis
        min_scaled, max_scaled = extrema(scaled_label)
        unique_scaled = unique(scaled_label)
        
        println("After scaling: min=$min_scaled, max=$max_scaled")
        println("Unique values after scaling: $(length(unique_scaled)) values")
        if length(unique_scaled) < 50
            println("All scaled values: $unique_scaled")
        else
            println("Too many unique values to display")
        end
        
        # Zeige Verteilung der Top-5 Klassen
        counts = Dict{Int, Int}()
        for val in scaled_label
            counts[val] = get(counts, val, 0) + 1
        end
        
        println("\nTop 5 classes by frequency:")
        sorted_counts = sort(collect(pairs(counts)), by=x->x[2], rev=true)
        for (k, (cls, count)) in enumerate(sorted_counts[1:min(5, length(sorted_counts))])
            percentage = 100 * count / length(scaled_label)
            println("  #$k: Class $cls - $count pixels ($(round(percentage, digits=2))%)")
        end
        
        # Teste die offizielle Funktion
        println("\nUsing Data.load_and_preprocess_label function:")
        label_data, max_class = Data.load_and_preprocess_label(label_file, verbose=false)
        println("Resulting label shape: $(size(label_data))")
        println("Max class detected: $max_class")
        
        # Vergleiche Ergebnisse
        reshaped_label = reshape(label_data, :)
        unique_official = unique(reshaped_label)
        println("Unique values in official result: $(length(unique_official))")
        println("Range in official result: min=$(minimum(reshaped_label)), max=$(maximum(reshaped_label))")
        
        # Überpüfe ob alles im gültigen Bereich liegt
        if any(reshaped_label .< 0) || any(reshaped_label .> 34)
            println("WARNING: Label values outside valid range (0-34)!")
        else
            println("All labels within valid range (0-34) ✓")
        end
        
        println()
    end
end

# 2. Test der One-Hot-Kodierung
function test_one_hot_encoding(batch, output_channels)
    println("\n===== TEST 2: ONE-HOT ENCODING =====")
    
    x, y = batch
    println("Label shape before one-hot: $(size(y))")
    
    # Konvertiere zu One-Hot
    y_oh = Training.batch_one_hot(y, output_channels)
    println("Label shape after one-hot: $(size(y_oh))")
    
    # Überprüfe die Konsistenz der One-Hot-Kodierung
    flat_y = reshape(y, :)
    
    # Stichproben-Überprüfung für 5 zufällige Pixel
    println("\nSample checks (original label → one-hot):")
    
    # Nimm immer gleiche Positionen für Reproduzierbarkeit
    test_positions = [(100, 100), (200, 300), (150, 400), (50, 200), (300, 600)]
    
    for (i, (row, col)) in enumerate(test_positions)
        if row <= size(y, 1) && col <= size(y, 2)
            orig_label = y[row, col, 1, 1]
            one_hot_vec = y_oh[row, col, :, 1]
            
            # Extrahiere Daten von der GPU, falls nötig
            if typeof(one_hot_vec) <: CuArray
                one_hot_vec = Array(one_hot_vec)
            end
            
            # Finde den Index des maximalen Wertes im One-Hot-Vektor
            one_hot_idx = argmax(one_hot_vec) - 1  # -1 weil Julia-Indizes bei 1 beginnen
            
            # Überprüfe, ob die Indizes übereinstimmen
            is_match = (orig_label == one_hot_idx)
            
            println("  Sample $i at ($row, $col): Original=$orig_label, One-hot max=$one_hot_idx ($(is_match ? "Match ✓" : "MISMATCH ❌"))")
            
            # Zeige One-Hot-Vektor für Fehler
            if !is_match
                println("    One-hot vector at this position: $one_hot_vec")
                println("    Indices with non-zero values: $(findall(one_hot_vec .> 0))")
            end
        end
    end
    
    # Überprüfe, ob jeder One-Hot-Vektor genau einen Eintrag mit 1.0 hat
    println("\nChecking one-hot encoding validity:")
    
    # Reshape zu 2D: (pixel, class)
    reshaped_oh = reshape(cpu(y_oh), :, output_channels)
    
    # Überprüfe Summen
    row_sums = sum(reshaped_oh, dims=2)
    all_ones = all(row_sums .≈ 1.0)
    println("  All rows sum to 1.0: $(all_ones ? "Yes ✓" : "No ❌")")
    
    if !all_ones
        # Finde problematische Zeilen
        problem_rows = findall(vec(abs.(row_sums .- 1.0) .> 1e-5))
        println("  Number of problem rows: $(length(problem_rows))")
        
        if !isempty(problem_rows)
            for i in 1:min(5, length(problem_rows))
                row_idx = problem_rows[i]
                row_sum = row_sums[row_idx]
                println("    Row $row_idx: Sum = $row_sum")
                
                # Finde Original-Pixel-Position
                h, w = size(y, 1), size(y, 2)
                pixel_idx = row_idx[1]  # row_idx ist ein CartesianIndex
                y_pos, x_pos = mod1(pixel_idx, h), div(pixel_idx-1, h)+1
                
                if y_pos <= h && x_pos <= w
                    orig_label = y[y_pos, x_pos, 1, 1]
                    println("    Original label at this position: $orig_label")
                end
            end
        end
    end
    
    return y_oh
end

# ------------ HAUPTPROGRAMM ------------

println("Number of images in img_dir: ", length(readdir(img_dir)))
println("Number of images in mask_dir: ", length(readdir(mask_dir)))

# Lade einige Label-Dateien für Tests
label_files = readdir(mask_dir, join=true)[1:3]

# Lade Datensatz
println("\nLoading dataset sample...")
dataset = Data.load_dataset(img_dir, mask_dir, verbose=false)
println("Dataset loaded with $(length(dataset)) samples")

# Erstelle Batches
train_data = Data.create_batches(dataset[1:5], batch_size, debug=false)
println("Created $(length(train_data)) batches for testing")

# Extrahiere ersten Batch für Tests
first_batch = first(train_data)

clear_gpu_memory()

# Führe nur die ersten beiden Tests aus
println("\n========= RUNNING DIAGNOSTIC TESTS =========")

# Test 1: Label-Erstellung
test_label_creation(label_files)
clear_gpu_memory()

# Test 2: One-Hot-Encoding
test_one_hot_encoding(first_batch, output_channels)
clear_gpu_memory()

println("\n========= DIAGNOSTIC TESTS COMPLETED =========")
println("\nNächste Schritte zur Fehlerbehebung:")
println("1. Überprüfe die Skalierung der Labels in Data.jl")
println("2. Stelle sicher, dass die One-Hot-Kodierung korrekt funktioniert")
println("3. Wenn beides korrekt ist, dann prüfe:")
println("   - Lernrate (versuche 0.01 oder 0.001)")
println("   - Loss-Funktion (implementiere Klassengewichtung)")
println("   - Modellinitialisierung (Anzahl der Kanäle, etc.)")

println("\nDiagnose abgeschlossen!")