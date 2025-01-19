
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

# Parameter
epochs = 1
batch_size = 1
learning_rate = 0.001
over_size =(512, 2048)
target_size = (16, 64)

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
println("Starte Training...")
trained_model = Train.train_model(model, train_data, train_data; epochs=epochs, learning_rate=learning_rate)





# Ergebnisse evaluieren
println("Evaluierung der Ergebnisse...")
evaluate_model(trained_model, train_data)

# Optional: Visualisierung eines Beispiels
println("Visualisiere Ergebnisse...")
test_image, test_label = train_data[1]  # Erstes Beispiel
prediction = trained_model(test_image)
visualize_results(test_image, prediction)
