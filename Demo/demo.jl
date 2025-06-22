using CUDA, Flux, BSON, LaMEM, GeophysicalModelGenerator, Random, Statistics
using Flux: mse, gpu, cpu
using Functors
using Optimisers
using ProgressMeter
using BSON: @save, @load
import Dates

# Konfigurationsstruktur
Base.@kwdef mutable struct TrainingConfig
    image_size::Tuple{Int, Int} = (256, 256)
    num_crystals::Int = 3
    batch_size::Int = 4
    learning_rate::Float32 = 0.001f0
    epochs::Int = 50
    save_path::String = "./models"
    nx::Int = 255
    nz::Int = 255
    η::Float64 = 1e20
    Δρ::Float64 = 200.0
    R::Float64 = 0.05
    use_gpu::Bool = false
    dataset_size::Int = 100
    validation_split::Float32 = 0.2f0
end

# Hilfsfunktionen
function get_timestamp()
    return Dates.format(Dates.now(), "yyyy-mm-dd_HH-MM-SS")
end

function validate_config(config::TrainingConfig)
    @assert config.image_size[1] == config.image_size[2] "Image size must be square"
    @assert config.num_crystals >= 1 && config.num_crystals <= 5 "Number of crystals must be between 1 and 5"
    @assert config.batch_size > 0 "Batch size must be positive"
    @assert config.learning_rate > 0 "Learning rate must be positive"
    @assert config.epochs > 0 "Number of epochs must be positive"
    @assert config.nx == config.image_size[1] - 1 "nx should be image_size - 1 for LaMEM"
    @assert config.nz == config.image_size[2] - 1 "nz should be image_size - 1 for LaMEM"
end

# Kristall-Platzierung mit Mindestabstand
function place_crystals(config::TrainingConfig)
    crystal_positions = []
    min_distance = 2.5 * config.R
    max_attempts = 1000
    
    for i in 1:config.num_crystals
        placed = false
        attempts = 0
        
        while !placed && attempts < max_attempts
            margin = config.R + 0.1
            x = rand() * (2 - 2*margin) - (1 - margin)
            z = rand() * (1.6 - 2*margin) + (margin - 0.8)
            pos = (x, z)
            
            if isempty(crystal_positions) || 
               all(sqrt((pos[1] - p[1])^2 + (pos[2] - p[2])^2) >= min_distance 
                   for p in crystal_positions)
                push!(crystal_positions, pos)
                placed = true
            end
            attempts += 1
        end
        
        if !placed
            @warn "Could not place crystal $i after $max_attempts attempts"
        end
    end
    
    return crystal_positions
end

# LaMEM Datengenerierung
function generate_multi_crystal_data(config::TrainingConfig, max_retries::Int=3)
    for attempt in 1:max_retries
        try
            crystal_positions = place_crystals(config)
            
            if length(crystal_positions) < config.num_crystals
                continue
            end
            
            η_crystal = 1e4 * config.η
            ρ_magma = 2700.0
            
            model = Model(
                Grid(nel=(config.nx, config.nz), x=[-1, 1], z=[-1, 1]), 
                LaMEM.Time(nstep_max=1),
                Output(out_strain_rate=1)
            )
            
            matrix = Phase(ID=0, Name="matrix", eta=config.η, rho=ρ_magma)
            crystal = Phase(ID=1, Name="crystal", eta=η_crystal, rho=ρ_magma + config.Δρ)
            add_phase!(model, crystal, matrix)
            
            for pos in crystal_positions
                add_sphere!(model, cen=(pos[1], 0.0, pos[2]), radius=config.R, phase=ConstantPhase(1))
            end
            
            run_lamem(model, 1)
            data, _ = read_LaMEM_timestep(model, 1)
            
            phase = data.fields.phase[:, 1, :]
            Vx = data.fields.velocity[1][:, 1, :]
            Vz = data.fields.velocity[3][:, 1, :]
            
            if any(isnan.(phase)) || any(isnan.(Vx)) || any(isnan.(Vz))
                continue
            end
            
            target_h, target_w = config.image_size
            actual_h, actual_w = size(phase)
            
            if (actual_h, actual_w) != (target_h, target_w)
                phase_resized = zeros(Float32, target_h, target_w)
                Vx_resized = zeros(Float32, target_h, target_w)
                Vz_resized = zeros(Float32, target_h, target_w)
                
                end_h = min(actual_h, target_h)
                end_w = min(actual_w, target_w)
                
                phase_resized[1:end_h, 1:end_w] = phase[1:end_h, 1:end_w]
                Vx_resized[1:end_h, 1:end_w] = Vx[1:end_h, 1:end_w]
                Vz_resized[1:end_h, 1:end_w] = Vz[1:end_h, 1:end_w]
                
                phase, Vx, Vz = phase_resized, Vx_resized, Vz_resized
            end
            
            V_stokes = 2/9 * config.Δρ * 9.81 * (config.R * 1000)^2 / config.η
            V_stokes_cm_year = V_stokes * 100 * (3600 * 24 * 365.25)
            
            return phase, Vx, Vz, V_stokes_cm_year, crystal_positions
            
        catch e
            if attempt == max_retries
                rethrow(e)
            end
            sleep(0.1)
        end
    end
end

# Phasenfeld-Kodierung
function encode_multi_phase_field(phase_field::Matrix{Float32})
    h, w = size(phase_field)
    encoded = zeros(Float32, h, w, 2)
    
    encoded[:, :, 1] = (phase_field .≈ 0.0)  # Matrix-Kanal
    encoded[:, :, 2] = (phase_field .≈ 1.0)  # Kristall-Kanal
    
    return encoded
end

# Datenvorverarbeitung
function preprocess_sample(phase, vx, vz, v_stokes, config::TrainingConfig)
    phase_encoded = encode_multi_phase_field(Float32.(phase))
    
    vx_norm = Float32.(vx ./ v_stokes)
    vz_norm = Float32.(vz ./ v_stokes)
    
    input_tensor = reshape(phase_encoded, config.image_size..., 2, 1)
    
    target_tensor = zeros(Float32, config.image_size..., 2, 1)
    target_tensor[:, :, 1, 1] = vx_norm
    target_tensor[:, :, 2, 1] = vz_norm
    
    return input_tensor, target_tensor
end

# Dataset-Generierung
function create_dataset(config::TrainingConfig)
    println("Generiere $(config.dataset_size) Samples...")
    
    inputs = []
    targets = []
    metadata = []
    failed_samples = 0
    
    progress = Progress(config.dataset_size, desc="Generating samples: ")
    
    for i in 1:config.dataset_size
        try
            phase, vx, vz, v_stokes, positions = generate_multi_crystal_data(config)
            input_tensor, target_tensor = preprocess_sample(phase, vx, vz, v_stokes, config)
            
            if any(isnan.(input_tensor)) || any(isnan.(target_tensor))
                failed_samples += 1
                continue
            end
            
            push!(inputs, input_tensor)
            push!(targets, target_tensor)
            push!(metadata, (v_stokes=v_stokes, positions=positions, sample_id=i))
            
        catch e
            failed_samples += 1
        end
        
        next!(progress)
        
        if i % 20 == 0
            GC.gc()
        end
    end
    
    finish!(progress)
    
    success_rate = round(100 * length(inputs) / config.dataset_size, digits=1)
    println("Dataset-Generierung abgeschlossen: $(length(inputs))/$(config.dataset_size) ($success_rate%)")
    
    if length(inputs) < config.dataset_size * 0.5
        @warn "Weniger als 50% der Samples erfolgreich generiert"
    end
    
    return inputs, targets, metadata
end

# UNet-Architektur
struct MultiCrystalUNet
    encoder1; encoder2; encoder3; encoder4; bottleneck
    decoder4; decoder4_conv; decoder3; decoder3_conv
    decoder2; decoder2_conv; decoder1; final_conv
end

@functor MultiCrystalUNet

# Einfache Skip-Connections für feste 256x256 Dimensionen
function simple_concat(upsampled, skip_connection)
    # Bei festen Dimensionen sind die Größen immer kompatibel
    return cat(upsampled, skip_connection, dims=3)
end

# UNet-Konstruktor
function create_multi_crystal_unet(config::TrainingConfig)
    encoder1 = Chain(
        Conv((3, 3), 2 => 32, relu, pad=SamePad()),
        BatchNorm(32),
        Conv((3, 3), 32 => 32, relu, pad=SamePad()),
        BatchNorm(32)
    )
    
    encoder2 = Chain(
        MaxPool((2, 2)),
        Conv((3, 3), 32 => 64, relu, pad=SamePad()),
        BatchNorm(64),
        Conv((3, 3), 64 => 64, relu, pad=SamePad()),
        BatchNorm(64)
    )
    
    encoder3 = Chain(
        MaxPool((2, 2)),
        Conv((3, 3), 64 => 128, relu, pad=SamePad()),
        BatchNorm(128),
        Conv((3, 3), 128 => 128, relu, pad=SamePad()),
        BatchNorm(128)
    )
    
    encoder4 = Chain(
        MaxPool((2, 2)),
        Conv((3, 3), 128 => 256, relu, pad=SamePad()),
        BatchNorm(256),
        Conv((3, 3), 256 => 256, relu, pad=SamePad()),
        BatchNorm(256)
    )
    
    bottleneck = Chain(
        MaxPool((2, 2)),
        Conv((3, 3), 256 => 512, relu, pad=SamePad()),
        BatchNorm(512),
        Dropout(0.5),
        Conv((3, 3), 512 => 512, relu, pad=SamePad()),
        BatchNorm(512)
    )
    
    decoder4 = ConvTranspose((2, 2), 512 => 256, stride=2)
    decoder4_conv = Chain(
        Conv((3, 3), 512 => 256, relu, pad=SamePad()),
        BatchNorm(256),
        Conv((3, 3), 256 => 256, relu, pad=SamePad()),
        BatchNorm(256)
    )
    
    decoder3 = ConvTranspose((2, 2), 256 => 128, stride=2)
    decoder3_conv = Chain(
        Conv((3, 3), 256 => 128, relu, pad=SamePad()),
        BatchNorm(128),
        Conv((3, 3), 128 => 128, relu, pad=SamePad()),
        BatchNorm(128)
    )
    
    decoder2 = ConvTranspose((2, 2), 128 => 64, stride=2)
    decoder2_conv = Chain(
        Conv((3, 3), 128 => 64, relu, pad=SamePad()),
        BatchNorm(64),
        Conv((3, 3), 64 => 64, relu, pad=SamePad()),
        BatchNorm(64)
    )
    
    decoder1 = ConvTranspose((2, 2), 64 => 32, stride=2)
    final_conv = Chain(
        Conv((3, 3), 64 => 32, relu, pad=SamePad()),
        BatchNorm(32),
        Conv((3, 3), 32 => 32, relu, pad=SamePad()),
        BatchNorm(32),
        Conv((1, 1), 32 => 2)
    )
    
    return MultiCrystalUNet(
        encoder1, encoder2, encoder3, encoder4, bottleneck,
        decoder4, decoder4_conv, decoder3, decoder3_conv,
        decoder2, decoder2_conv, decoder1, final_conv
    )
end

# Forward-Pass mit vereinfachten Skip-Connections
function (model::MultiCrystalUNet)(x)
    e1 = model.encoder1(x)
    e2 = model.encoder2(e1)
    e3 = model.encoder3(e2)
    e4 = model.encoder4(e3)
    
    b = model.bottleneck(e4)
    
    d4 = model.decoder4(b)
    d4 = model.decoder4_conv(simple_concat(d4, e4))
    
    d3 = model.decoder3(d4)
    d3 = model.decoder3_conv(simple_concat(d3, e3))
    
    d2 = model.decoder2(d3)
    d2 = model.decoder2_conv(simple_concat(d2, e2))
    
    d1 = model.decoder1(d2)
    output = model.final_conv(simple_concat(d1, e1))
    
    return output
end

# Batch-Erstellung
function create_batches(inputs, targets, batch_size)
    n_samples = length(inputs)
    batches = []
    
    for i in 1:batch_size:n_samples
        batch_end = min(i + batch_size - 1, n_samples)
        batch_indices = i:batch_end
        
        try
            input_batch = cat(inputs[batch_indices]..., dims=4)
            target_batch = cat(targets[batch_indices]..., dims=4)
            push!(batches, (input_batch, target_batch))
        catch e
            @warn "Fehler bei Batch-Erstellung für Indices $batch_indices: $e"
        end
    end
    
    return batches
end

# Training-Loop - GPU-Probleme behoben
function train_model(model, train_batches, config::TrainingConfig)
    println("Training startet: $(length(train_batches)) Batches, $(config.epochs) Epochen")
    
    opt_state = Optimisers.setup(Optimisers.Adam(config.learning_rate), model)
    
    # GPU nur wenn explizit aktiviert und funktional
    use_gpu_training = config.use_gpu && CUDA.functional()
    if use_gpu_training
        model = gpu(model)
        println("Modell auf GPU verschoben")
    else
        println("Training auf CPU")
    end
    
    losses = Float32[]
    
    for epoch in 1:config.epochs
        epoch_loss = 0.0f0
        batch_count = 0
        
        for (batch_idx, (input_batch, target_batch)) in enumerate(train_batches)
            try
                # Nur auf GPU verschieben wenn GPU-Training aktiviert
                if use_gpu_training
                    input_batch = gpu(input_batch)
                    target_batch = gpu(target_batch)
                end
                
                # Gradient berechnung
                loss_fn(m) = mse(m(input_batch), target_batch)
                ∇model = gradient(loss_fn, model)[1]
                batch_loss = loss_fn(model)
                
                # Parameter-Update
                opt_state, model = Optimisers.update!(opt_state, model, ∇model)
                
                epoch_loss += batch_loss
                batch_count += 1
                
            catch e
                println("Fehler bei Batch $batch_idx: $e")
                # Bei GPU-Fehlern auf CPU wechseln
                if use_gpu_training && occursin("GPU", string(e))
                    println("GPU-Fehler erkannt, wechsle zu CPU")
                    model = cpu(model)
                    use_gpu_training = false
                end
            end
            
            # Memory cleanup nur bei GPU
            if use_gpu_training && CUDA.functional()
                CUDA.reclaim()
            end
        end
        
        avg_loss = batch_count > 0 ? epoch_loss / batch_count : NaN32
        push!(losses, avg_loss)
        
        if epoch % 10 == 0
            println("Epoche $epoch: Loss = $(round(avg_loss, digits=6))")
            
            model_cpu = cpu(model)
            checkpoint_path = joinpath(config.save_path, "checkpoint_epoch_$(epoch).bson")
            mkpath(dirname(checkpoint_path))
            @save checkpoint_path model_cpu
        end
        
        GC.gc()
        if use_gpu_training && CUDA.functional()
            CUDA.reclaim()
        end
    end
    
    return model, losses
end

# Evaluierung
function evaluate_model(model, test_inputs, test_targets)
    model_eval = cpu(model)
    
    total_loss = 0.0f0
    n_samples = length(test_inputs)
    
    for i in 1:n_samples
        prediction = model_eval(test_inputs[i])
        loss = mse(prediction, test_targets[i])
        total_loss += loss
    end
    
    avg_loss = total_loss / n_samples
    println("Test-MSE: $(round(avg_loss, digits=6))")
    
    return avg_loss
end

# Konfigurationserstellung - GPU standardmäßig deaktiviert
function create_config(;
    image_size = (256, 256),
    num_crystals = 3,
    batch_size = 4,
    learning_rate = 0.001f0,
    epochs = 30,
    dataset_size = 50,
    validation_split = 0.2f0,
    use_gpu = false  # GPU deaktiviert für Stabilität
)
    config = TrainingConfig(
        image_size = image_size,
        num_crystals = num_crystals,
        batch_size = batch_size,
        learning_rate = learning_rate,
        epochs = epochs,
        save_path = "./models",
        nx = image_size[1] - 1,
        nz = image_size[2] - 1,
        η = 1e20,
        Δρ = 200.0,
        R = 0.05,
        use_gpu = use_gpu,
        dataset_size = dataset_size,
        validation_split = validation_split
    )
    
    validate_config(config)
    return config
end

# Hauptfunktion
function run_training()
    println("Multi-Kristall Velocity Prediction Training")
    
    config = create_config(
        dataset_size = 20,
        epochs = 10,
        batch_size = 2,
        use_gpu = false  # CPU-Training für Stabilität
    )
    
    # Dataset generieren
    println("\n1. Dataset-Generierung")
    inputs, targets, metadata = create_dataset(config)
    
    if length(inputs) < 5
        error("Zu wenige Samples generiert für Training")
    end
    
    # Train/Test Split
    n_train = max(1, Int(floor((1 - config.validation_split) * length(inputs))))
    train_inputs = inputs[1:n_train]
    train_targets = targets[1:n_train]
    test_inputs = inputs[n_train+1:end]
    test_targets = targets[n_train+1:end]
    
    println("Training: $(length(train_inputs)), Test: $(length(test_inputs))")
    
    # Batches erstellen
    println("\n2. Batch-Erstellung")
    train_batches = create_batches(train_inputs, train_targets, config.batch_size)
    
    if isempty(train_batches)
        error("Keine gültigen Batches erstellt")
    end
    
    # Modell erstellen
    println("\n3. Modell-Erstellung")
    model = create_multi_crystal_unet(config)
    
    # Training
    println("\n4. Training")
    trained_model, losses = train_model(model, train_batches, config)
    
    # Modell speichern
    final_model_cpu = cpu(trained_model)
    timestamp = get_timestamp()
    final_path = joinpath(config.save_path, "multi_crystal_model_$timestamp.bson")
    mkpath(dirname(final_path))
    @save final_path final_model_cpu
    
    # Evaluierung
    test_loss = NaN
    if !isempty(test_inputs)
        println("\n5. Evaluierung")
        test_loss = evaluate_model(trained_model, test_inputs, test_targets)
    end
    
    println("\nTraining abgeschlossen:")
    println("  Samples: $(length(inputs))")
    println("  Epochen: $(config.epochs)")
    println("  Finaler Trainingsverlust: $(round(losses[end], digits=6))")
    if !isnan(test_loss)
        println("  Test-Verlust: $(round(test_loss, digits=6))")
    end
    println("  Modell gespeichert: $final_path")
    
    return trained_model, losses, test_loss, config
end

# Dataset speichern/laden
function save_dataset(inputs, targets, metadata, filename="dataset.bson")
    @save filename inputs targets metadata
    println("Dataset gespeichert: $filename")
end

function load_dataset(filename="dataset.bson")
    if !isfile(filename)
        error("Dataset-Datei nicht gefunden: $filename")
    end
    
    data = BSON.load(filename)
    inputs = data[:inputs]
    targets = data[:targets]
    metadata = data[:metadata]
    
    println("Dataset geladen: $(length(inputs)) Samples")
    return inputs, targets, metadata
end

# Direktes Training starten
println("Multi-Kristall UNet für Velocity Prediction geladen")
println("Training starten mit: run_training()")

model, losses, test_loss, config = run_training()