##################################
# Main.jl - Optimiert für Standard Precision
##################################

# Einbinden der anderen Dateien
include("Data.jl")
include("Model.jl")
include("Training.jl")
include("Visualization.jl")

using Flux
using Flux: gpu
using CUDA
using .Model
using .Data
using .Training
using .Visualization
using Dates
import Base.GC

# Hyperparameter
num_epochs = 10       # Anzahl Trainings-Epochen
learning_rate = 0.0001
input_channels = 3    # RGB
batch_size = 2        # Batch-Größe
output_channels = 35  # Klassenanzahl (35 Klassen: 0-34)
checkpoint_dir = "/local/home/baselt/checkpoints"  # Verzeichnis zum Speichern von Checkpoints
checkpoint_freq = 1   # Speichere Checkpoint nach N Epochen

# GPU-Speicherverwaltung
function clear_gpu_memory()
    GC.gc()
    CUDA.reclaim()
    println("GPU memory cleared")
end

# Verzeichnisse
img_dir = "/local/home/baselt/Datensatz/Training/image_2"
mask_dir = "/local/home/baselt/Datensatz/Training/semantic"

# Überprüfe die Anzahl der Bilder in den Verzeichnissen
println("Number of images in img_dir: ", length(readdir(img_dir)))
println("Number of images in mask_dir: ", length(readdir(mask_dir)))

println("Loading dataset...")

# Lade den vollständigen Datensatz mit einheitlicher Bildgröße
dataset = Data.load_dataset(img_dir, mask_dir)
println("Number of samples in dataset: ", length(dataset))

clear_gpu_memory()

# Batches erstellen - kein Shuffling
train_data = Data.create_batches(dataset, batch_size)
println("Number of batches: ", length(train_data))

# UNet-Modell erstellen - Memory-efficient
model = Model.UNet(input_channels, output_channels, memory_efficient=true)
println("UNet model created")

# Modell auf die GPU verschieben
model = gpu(model)
println("Model moved to GPU")

clear_gpu_memory()

# Checkpoint-Verzeichnis erstellen
mkpath(checkpoint_dir)

start_time = now()
println("Starting training at ", start_time)

# Training - Standard Precision
model, losses = Training.train_unet(
    model, train_data, num_epochs, learning_rate, output_channels;
    checkpoint_dir=checkpoint_dir, checkpoint_freq=checkpoint_freq
)

end_time = now()
duration = end_time - start_time
println("Training took ", duration)
println("Training finished with final loss: ", losses[end])

clear_gpu_memory()

println("Done!")
