##################################
# Training.jl - Optimiert für Standard Precision
##################################
module Training

include("Model.jl")
include("Data.jl")
using Flux
using Flux: onehotbatch, logitcrossentropy, gpu, cpu
using Statistics
using LinearAlgebra
using Optimisers
using CUDA
using .Model
using .Data
using BSON: @save, @load
using ProgressMeter

# Optimierte One-Hot-Kodierung für Batches
# Korrigierte Version
# Füge diese korrigierte Funktion in deine Training.jl Datei ein

function batch_one_hot(batch, num_classes)
    # Extrahiere die Labels als Integer
    batch_int = Int.(selectdim(batch, 3, 1))
    
    # Prüfe auf ungültige Labelwerte
    if any(batch_int .< 0) || any(batch_int .>= num_classes)
        println("WARNING: Label values outside valid range found!")
        println("Range: ", minimum(batch_int), " to ", maximum(batch_int))
        # Begrenze die Werte auf den gültigen Bereich
        batch_int = clamp.(batch_int, 0, num_classes-1)
    end
    
    # Manuelle One-Hot-Kodierung erstellen
    batch_size = size(batch, 4)
    height, width = size(batch, 1), size(batch, 2)
    one_hot = zeros(Float32, height, width, num_classes, batch_size)
    
    for b in 1:batch_size
        for j in 1:width
            for i in 1:height
                label = batch_int[i, j, b]
                if 0 <= label < num_classes
                    one_hot[i, j, label+1, b] = 1.0f0
                end
            end
        end
    end
    
    return gpu(one_hot)
end

function debug_gradients(model, x, y, y_oh)
    # Berechne Gradienten
    grads = gradient(model -> loss_fn(model, x, y_oh), model)[1]
    
    # Überprüfe Gradienten für einige Schichten
    total_grad_norm = 0.0
    for layer_name in [:encoder1, :bottleneck, :decoder1_1]
        layer = getproperty(model, layer_name)
        if isa(layer, Chain) && !isempty(layer)
            first_layer = layer[1]
            if hasproperty(first_layer, :weight)
                param = first_layer.weight
                param_grads = grads.params[layer][1].weight
                
                param_cpu = cpu(param_grads)
                grad_norm = norm(param_cpu)
                total_grad_norm += grad_norm
                
                println("Gradients for $layer_name - norm: $grad_norm")
                println("  mean: $(mean(param_cpu)), std: $(std(param_cpu))")
                println("  min: $(minimum(param_cpu)), max: $(maximum(param_cpu))")
                
                # Prüfe auf Vanishing/Exploding Gradients
                if maximum(abs.(param_cpu)) < 1e-6
                    println("  WARNING: Possible vanishing gradients")
                elseif maximum(abs.(param_cpu)) > 1.0
                    println("  WARNING: Possible exploding gradients")
                end
            end
        end
    end
    println("Total gradient norm: $total_grad_norm")
    return grads
end

# Einfache Loss-Funktion für semantische Segmentierung
function loss_fn(model, x, y)
    pred = model(x)
    return logitcrossentropy(pred, y)
end

# Checkpoint-Speicherfunktion
function save_checkpoint(model, epoch, loss, filename)
    model_cpu = cpu(model)
    @save filename model=model_cpu epoch=epoch loss=loss
    println("Saved checkpoint to $filename")
end

# ===================================================
# Standard Precision Training Funktion mit Batch-Verifizierung
# ===================================================
function train_unet(model, train_data, num_epochs, learning_rate, output_channels; 
    checkpoint_dir="checkpoints", checkpoint_freq=5)

mkpath(checkpoint_dir)
opt = Optimisers.Adam(learning_rate)
opt_state = Optimisers.setup(opt, model)

losses = Float32[]

# Initialen Loss berechnen
x_batch, y_batch = first(train_data)
x_gpu = gpu(x_batch)
y_oh = batch_one_hot(y_batch, output_channels)
initial_loss = loss_fn(model, x_gpu, y_oh)
println("Initial loss before training: $initial_loss")

# Debug Gradienten
println("\nInitial gradient check:")
debug_gradients(model, x_gpu, y_batch, y_oh)

for epoch in 1:num_epochs
println("====== Epoch $epoch / $num_epochs ======")

total_loss = 0f0
batch_count = 0

# Speichere Modell-Parameter vor der Epoche
if epoch == 1
first_param_sample = Dict()
for layer_name in [:encoder1, :bottleneck, :decoder1_1]
layer = getproperty(model, layer_name)
if isa(layer, Chain) && !isempty(layer) && hasproperty(layer[1], :weight)
    w_sample = cpu(layer[1].weight)[1:5, 1:5, 1, 1]
    first_param_sample[layer_name] = copy(w_sample)
end
end
end

p = Progress(length(train_data), 1, "Training epoch $epoch...")

for (batch_idx, (input_batch, mask_batch)) in enumerate(train_data)
# One-Hot-Kodierung der Labels
mask_batch_oh = batch_one_hot(mask_batch, output_channels)

# Daten auf die GPU verschieben
input_batch = gpu(input_batch)

# Debug für ersten Batch der ersten Epoche
if epoch == 1 && batch_idx == 1
println("\nChecking first batch encoding:")
flat_oh = reshape(cpu(mask_batch_oh), :, output_channels)
row_sums = vec(sum(flat_oh, dims=2))
num_zeros = count(row_sums .< 0.5)
println("One-hot vectors with zero sum: $num_zeros out of $(length(row_sums))")

# Speichere das One-Hot-Encoding für spätere Überprüfung
first_oh_sample = mask_batch_oh[100:105, 100:105, :, 1]
end

# Gradienten und Loss berechnen
grads = gradient(model -> loss_fn(model, input_batch, mask_batch_oh), model)[1]
batch_loss = loss_fn(model, input_batch, mask_batch_oh)

# Debug für ersten Batch der ersten Epoche
if epoch == 1 && batch_idx == 1
println("\nFirst batch gradient analysis:")
debug_gradients(model, input_batch, mask_batch, mask_batch_oh)
end

# Modellparameter aktualisieren
opt_state, model = Optimisers.update!(opt_state, model, grads)

total_loss += batch_loss
batch_count += 1

next!(p; showvalues = [(:batch, batch_idx), (:loss, batch_loss)])

# GPU-Speicher freigeben
CUDA.reclaim()
end

avg_loss = total_loss / batch_count
push!(losses, avg_loss)
println("Epoch $epoch finished. Average Loss: $avg_loss")

# Überprüfe, ob sich die Modellparameter geändert haben
if epoch == 1
println("\nChecking if parameters changed after one epoch:")
for layer_name in [:encoder1, :bottleneck, :decoder1_1]
layer = getproperty(model, layer_name)
if isa(layer, Chain) && !isempty(layer) && hasproperty(layer[1], :weight) && haskey(first_param_sample, layer_name)
    w_sample_now = cpu(layer[1].weight)[1:5, 1:5, 1, 1]
    w_sample_before = first_param_sample[layer_name]
    
    max_change = maximum(abs.(w_sample_now - w_sample_before))
    println("$layer_name - Max weight change: $max_change")
    
    if max_change < 1e-10
        println("  WARNING: Weights barely changed!")
    else
        println("  Weights updated successfully")
    end
end
end
end

# Checkpoint speichern
if epoch % checkpoint_freq == 0 || epoch == num_epochs
save_checkpoint(model, epoch, avg_loss, 
           joinpath(checkpoint_dir, "checkpoint_epoch$(epoch).bson"))
end

println("--------------------------------------------------")
end

# Finales Modell speichern
save_checkpoint(model, num_epochs, losses[end], 
   joinpath(checkpoint_dir, "final_model.bson"))

return model, losses
end

end # module Training