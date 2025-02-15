###########################
# Main.jl (nur synthetische Daten)
###########################
include("Data.jl")
include("Model.jl")
include("Training.jl")
include("Visualization.jl")

using .Data
using .Model
using .Training
using .Visualization

# Hyperparameter
const NUM_EPOCHS    = 10
const LEARNING_RATE = 0.001
const BATCH_SIZE    = 4
const DATASET_SIZE  = 10  # Gesamtanzahl synthetischer Beispiele
const IMAGE_SIZE    = 64
const MAX_SPEED     = 3

# 1) Erstelle synthetischen Datensatz (Bewegter Punkt + vx/vy)
dataset = Data.generate_synthetic_dataset(DATASET_SIZE; 
    image_size=IMAGE_SIZE, 
    max_speed=MAX_SPEED
)

println("Erzeugtes Dataset: ", length(dataset), " Samples.")

# 2) Erzeuge Batches
train_data = Data.create_synthetic_batches(dataset, BATCH_SIZE)
println("Anzahl Trainingsbatches: ", length(train_data))

# 3) Erzeuge ein U-Net: 3 Input-Kanäle (Frame, vx, vy), 1 Output-Kanal (nächstes Frame)
model = Model.UNet(3, 1)

# 4) Training auf synthetischen Daten
losses = Training.train_unet_synthetic(model, train_data, NUM_EPOCHS, LEARNING_RATE)

# 5) Visualisiere eine Vorhersage anhand des ersten Samples
Visualization.visualize_synthetic_results(model, dataset[1])

println("Training abgeschlossen.")
