using ImageTransformations
using CUDA
using ThreadsX

include("UNetFramework.jl")
include("DataLoader.jl")
include("Model.jl")
include("train.jl")

# Modul importieren
import .UNetFramework
import .DataLoader
import .Model
import .Train

# Parameter
epochs = 10
batch_size = 2
learning_rate = 0.001
over_size = (512, 2048)

# CUDA aktivieren
use_cuda = true

# Überprüfen der verfügbaren GPU-Ressourcen
function check_cuda_resources()
    if CUDA.has_cuda()
        device_count = CUDA.ndevices()  # Anzahl der CUDA-Geräte
        println("Anzahl der verfügbaren CUDA-Geräte: $device_count")
        for i in 0:(device_count-1)
            CUDA.device!(i)  # Aktiviere das Gerät mit Index i
            memory_status = CUDA.memory_status()  # Speicherstatus des aktuellen Geräts
            if memory_status !== nothing
                free_memory, total_memory = memory_status
                println("Gerät $i: Freier Speicher = $(free_memory / 1e9) GB, Gesamtspeicher = $(total_memory / 1e9) GB")
            else
                println("Gerät $i: Speicherstatus konnte nicht abgerufen werden.")
            end
        end
        return true
    else
        println("Keine CUDA-fähigen Geräte gefunden.")
        return false
    end
end

use_cuda = check_cuda_resources()


function load_and_preprocess_data(image_path, label_path, batch_size, over_size)
    println("Lade Daten...")
    images, labels = DataLoader.load_data(image_path, label_path)
    println("Daten vorverarbeiten...")
    train_data = DataLoader.preprocess_and_batch(images, labels, batch_size, over_size)
    return train_data
end

if use_cuda
    # Daten laden und vorbereiten
    train_data = load_and_preprocess_data(
        "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Bilder",
        "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Masken",
        batch_size, over_size
    )

    # Daten auf die GPU verschieben
    println("Daten auf die GPU verschieben...")
    train_data_gpu = ThreadsX.map(x -> (cu(x[1]), cu(x[2])), train_data)

    # U-Net Modell erstellen und auf die GPU verschieben
    println("Erstelle U-Net-Modell...")
    model = Model.create_unet(1, 1)  # Single-Channel Input/Output
    trained_model_gpu = cu(model)

    # Training starten
    println("Starte Training...")
    trained_model = Train.train_model_cuda(trained_model_gpu, train_data_gpu, train_data_gpu; epochs=epochs, learning_rate=learning_rate)

    # Ergebnisse evaluieren
    println("Evaluierung der Ergebnisse...")
    UNetFramework.evaluate_model(trained_model_gpu, train_data_gpu)

    # Optional: Visualisierung eines Beispiels
    println("Visualisiere Ergebnisse...")
    test_image, test_label = train_data[1]  # Erstes Beispiel
    test_image_gpu = cu(test_image)
    test_label_gpu = cu(test_label)
    UNetFramework.visualize_results(trained_model_gpu, test_image_gpu, test_label_gpu)

else
    # Daten laden und vorbereiten
    train_data = load_and_preprocess_data(
        "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Bilder",
        "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Masken",
        batch_size, over_size
    )

    # U-Net Modell erstellen
    println("Erstelle U-Net-Modell...")
    model = Model.create_unet(1, 1)  # Single-Channel Input/Output

    # Training starten
    println("Starte Training ohne CUDA...")
    trained_model = Train.train_model(model, train_data, train_data; epochs=epochs, learning_rate=learning_rate)

    # Ergebnisse evaluieren
    println("Evaluierung der Ergebnisse...")
    UNetFramework.evaluate_model(trained_model, train_data)

    # Optional: Visualisierung eines Beispiels
    println("Visualisiere Ergebnisse...")
    test_image, test_label = train_data[1]  # Erstes Beispiel
    UNetFramework.visualize_results(trained_model, test_image, test_label)
end
