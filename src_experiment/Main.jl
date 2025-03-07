##################################
# Main.jl - Einfaches Training mit CPU Fallback
##################################

include("Data.jl")
include("Model.jl")
include("Training.jl")
include("Visualization.jl")

using Flux
using Flux: logitcrossentropy
using CUDA
using .Model
using .Data
using .Training
using .Visualization
using Dates
import Base.GC
using Statistics
using BSON: @save

# Hyperparameter
num_epochs = 10
learning_rate = 0.01  # Höhere Lernrate
input_channels = 3    
batch_size = 2        
output_channels = 35  
checkpoint_dir = "/local/home/baselt/checkpoints"

# GPU/CPU Management
function use_gpu()
    if CUDA.functional()
        return true
    else
        println("GPU nicht verfügbar, nutze CPU")
        return false
    end
end

USE_GPU = use_gpu()

# Wrapper-Funktionen für GPU/CPU
function to_device(x)
    return USE_GPU ? gpu(x) : x
end

function from_device(x)
    return USE_GPU ? cpu(x) : x
end

# GPU-Speicherverwaltung
function clear_memory()
    GC.gc()
    if USE_GPU
        CUDA.reclaim()
    end
    println("Memory cleared")
end

# Verzeichnisse
img_dir = "/local/home/baselt/Datensatz/Training/image_2"
mask_dir = "/local/home/baselt/Datensatz/Training/semantic"

# Lade Datensatz
println("Loading dataset...")
dataset = Data.load_dataset(img_dir, mask_dir, verbose=false)
println("Number of samples in dataset: ", length(dataset))

# Erstelle Batches
train_data = Data.create_batches(dataset, batch_size)
println("Number of batches: ", length(train_data))

clear_memory()

# Erstelle Modell
model = Model.UNet(input_channels, output_channels, memory_efficient=true)
model = to_device(model)
println("Model created and moved to ", USE_GPU ? "GPU" : "CPU")

# Verbesserte One-Hot-Kodierung direkt hier implementiert
function manual_one_hot(batch, num_classes)
    # Extrahiere Labels
    batch_int = Int.(selectdim(batch, 3, 1))
    
    # Prüfe auf ungültige Werte
    if any(batch_int .< 0) || any(batch_int .>= num_classes)
        println("WARNING: Labels außerhalb des gültigen Bereichs gefunden!")
        batch_int = clamp.(batch_int, 0, num_classes-1)
    end
    
    # Erstelle One-Hot Tensor auf CPU
    batch_size = size(batch, 4)
    height, width = size(batch, 1), size(batch, 2)
    one_hot = zeros(Float32, height, width, num_classes, batch_size)
    
    # Fülle One-Hot Tensor manuell
    for b in 1:batch_size
        for j in 1:width
            for i in 1:height
                label = batch_int[i, j, b]
                one_hot[i, j, label+1, b] = 1.0f0
            end
        end
    end
    
    # Auf GPU verschieben falls nötig
    return to_device(one_hot)
end

# Vereinfachte Loss-Funktion
function simple_loss_fn(model, x, y)
    pred = model(x)
    return logitcrossentropy(pred, y)
end

# Trainingsschleife
function train_simple(model, train_data, num_epochs, learning_rate, output_channels;
                    checkpoint_dir="checkpoints", checkpoint_freq=1)
    mkpath(checkpoint_dir)
    opt = Optimisers.Adam(learning_rate)
    opt_state = Optimisers.setup(opt, model)
    
    losses = Float32[]
    
    # Berechne initialen Loss
    x, y = first(train_data)
    x_device = to_device(x)
    y_oh = manual_one_hot(y, output_channels)
    initial_loss = simple_loss_fn(model, x_device, y_oh)
    println("Initial loss: $initial_loss")
    
    for epoch in 1:num_epochs
        println("====== Epoch $epoch / $num_epochs ======")
        
        total_loss = 0f0
        batch_count = 0
        
        for (batch_idx, (input_batch, mask_batch)) in enumerate(train_data)
            # Auf GPU/CPU verschieben
            input_device = to_device(input_batch)
            
            # One-Hot Encoding
            mask_oh = manual_one_hot(mask_batch, output_channels)
            
            # Trainingsprozess in try-catch Block
            try
                # Gradienten und Loss berechnen
                gs = gradient(() -> simple_loss_fn(model, input_device, mask_oh), Flux.params(model))
                batch_loss = simple_loss_fn(model, input_device, mask_oh)
                
                # Modell aktualisieren
                Flux.update!(opt, Flux.params(model), gs)
                
                total_loss += batch_loss
                batch_count += 1
                
                # Ausgabe
                if batch_idx % 10 == 0
                    println("  Batch $batch_idx/$(length(train_data)) - Loss: $batch_loss")
                end
            catch e
                println("Error in batch $batch_idx: $e")
            end
            
            # Speicher freigeben
            if batch_idx % 5 == 0
                clear_memory()
            end
        end
        
        # Berechne durchschnittlichen Loss
        avg_loss = total_loss / batch_count
        push!(losses, avg_loss)
        println("Epoch $epoch completed. Average Loss: $avg_loss")
        
        # Speichere Checkpoint
        if epoch % checkpoint_freq == 0 || epoch == num_epochs
            checkpoint_path = joinpath(checkpoint_dir, "checkpoint_epoch$(epoch).bson")
            model_cpu = cpu(model)
            @save checkpoint_path model=model_cpu epoch=epoch loss=avg_loss
            println("Saved checkpoint to $checkpoint_path")
        end
    end
    
    return model, losses
end

# Checkpoint-Verzeichnis erstellen
mkpath(checkpoint_dir)

# Trainieren
start_time = now()
println("Starting training at $start_time")

model, losses = train_simple(model, train_data, num_epochs, learning_rate, output_channels;
                         checkpoint_dir=checkpoint_dir)

end_time = now()
duration = end_time - start_time
println("Training took $duration")

# Speichere Ergebnisse
println("Final loss: $(losses[end])")
println("Loss progression:", losses)

# Visualisiere Loss-Kurve
using Plots
p = plot(losses, title="Training Loss", xlabel="Epoch", ylabel="Loss", legend=false, marker=:circle)
savefig(p, joinpath(checkpoint_dir, "loss_curve.png"))

println("Training completed!")