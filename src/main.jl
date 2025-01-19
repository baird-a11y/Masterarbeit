
include("UNetFramework.jl")
include("DataLoader.jl")
include("Model.jl")
include("train.jl")
# Modul importieren
import .UNetFramework
import .DataLoader
import .Model
import .Train
using ImageTransformations
using CUDA

# Parameter
epochs = 10

# # Dynamische Batchgröße
# available_memory = 8.0  # Beispiel: 8 GB GPU-Speicher
# over_size = (512, 2048)  # Bildgröße
# batch_size = Int(floor(available_memory / (prod(over_size) * 4 * 1e-6)))  # Schätzung
# println("Empfohlene Batchgröße: ", batch_size)

# # Fallback, falls zu große Werte geschätzt werden
# batch_size = min(batch_size, 32)  # Maximal Batchgröße von 32

batch_size = 4
learning_rate = 0.001
over_size = (512, 2048)

# CUDA aktivieren
use_cude = false

if use_cude==true
    # Daten laden und vorbereiten
    println("Lade Daten...")
    images, labels = DataLoader.load_data(
        "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Bilder",
        "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Masken"
    )

    # Daten vorverarbeiten und auf die GPU verschieben
    train_data = DataLoader.preprocess_and_batch(images, labels, batch_size, over_size)
    train_data_gpu = [(cu(x), cu(y)) for (x, y) in train_data]

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
    # Hole ein Testbeispiel aus den Trainingsdaten
    test_image, test_label = train_data[1]  # Erstes Beispiel

    # Testbild und Label auf die GPU verschieben
    test_image_gpu = CUDA.array(test_image)
    test_label_gpu = CUDA.array(test_label)

    # Visualisierung der Ergebnisse mit Ground-Truth
    UNetFramework.visualize_results(trained_model_gpu, test_image, test_label)

else
  
    # Code without CUDA #

    # Daten laden und vorbereiten
    println("Lade Daten...")
    images, labels = DataLoader.load_data(
        "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Bilder",
        "G:/Meine Ablage/Geowissenschaften/Masterarbeit/Masterarbeit/Datensatz/Training/Masken"
    )

    # images = DataLoader.preprocess_image(images, over_size)
    # labels = DataLoader.preprocess_image(labels, over_size)
    train_data = DataLoader.preprocess_and_batch(images, labels, batch_size, over_size)

    # U-Net Modell erstellen
    println("Erstelle U-Net-Modell...")
    model = Model.create_unet(1, 1)  # Single-Channel Input/Output

    # Training starten
    println("Starte Training ohne Cuda...")
    trained_model = Train.train_model(model, train_data, train_data; epochs=epochs, learning_rate=learning_rate)

    # Ergebnisse evaluieren
    println("Evaluierung der Ergebnisse...")
    UNetFramework.evaluate_model(trained_model, train_data)

    # Optional: Visualisierung eines Beispiels
    println("Visualisiere Ergebnisse...")
    # Hole ein Testbeispiel aus den Trainingsdaten
    test_image, test_label = train_data[1]  # Erstes Beispiel
    # Visualisierung der Ergebnisse mit Ground-Truth
    UNetFramework.visualize_results(trained_model, test_image, test_label)

end