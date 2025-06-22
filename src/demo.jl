using CUDA, Flux, BSON, LaMEM, GeophysicalModelGenerator, GLMakie, Random, Statistics
using Flux: mse, gpu, cpu
using Optimisers
using ProgressMeter
using Dates

# Konfigurationsstruktur
Base.@kwdef mutable struct CONFIG
    image_size::Tuple{Int, Int} = (256, 256)
    num_crystals::Int = 3
    batch_size::Int = 4  # Reduziert für Speicher-Effizienz
    learning_rate::Float32 = 0.001f0
    epochs::Int = 50
    save_path::String = "./models"
    nx::Int = 255  # für 256x256 Output
    nz::Int = 255
    η::Float64 = 1e20
    Δρ::Float64 = 200.0
    R::Float64 = 0.05
    use_gpu::Bool = false  # CPU-Training für Stabilität
    dataset_size::Int = 100
    validation_split::Float32 = 0.2f0
end

# Validierung der Konfiguration
function validate_config(config::CONFIG)
    @assert config.image_size[1] == config.image_size[2] "Image size must be square."
    @assert config.num_crystals >= 1 && config.num_crystals <= 5 "Number of crystals must be between 1 and 5."
    @assert config.batch_size > 0 "Batch size must be positive."
    @assert config.learning_rate > 0 "Learning rate must be positive."
    @assert config.epochs > 0 "Number of epochs must be positive."
    @assert config.nx == config.image_size[1] - 1 "nx should be image_size - 1 for LaMEM"
    @assert config.nz == config.image_size[2] - 1 "nz should be image_size - 1 for LaMEM"
    nothing
end

# Verbesserte Kristall-Platzierung mit Mindestabstand
function place_crystals(config::CONFIG)
    crystal_positions = []
    min_distance = 2.5 * config.R  # Mindestabstand zwischen Kristallen
    max_attempts = 1000
    
    for i in 1:config.num_crystals
        placed = false
        attempts = 0
        
        while !placed && attempts < max_attempts
            # Zufällige Position mit Abstand zum Rand
            margin = config.R + 0.1
            x = rand() * (2 - 2*margin) - (1 - margin)
            z = rand() * (1.6 - 2*margin) + (margin - 0.8)  # z zwischen -0.8 und 0.8
            pos = (x, z)
            
            # Prüfe Abstand zu anderen Kristallen
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

# Robuste LaMEM-Funktion für Multi-Kristalle mit Retry-Mechanismus
function generate_multi_crystal_data_with_lamem_robust(config::CONFIG, max_retries::Int=3)
    """
    Robuste Version der LaMEM-Datengenerierung mit Retry-Mechanismus
    """
    for attempt in 1:max_retries
        try
            # Kristall-Positionen generieren
            crystal_positions = place_crystals(config)
            
            if length(crystal_positions) < config.num_crystals
                @warn "Attempt $attempt: Only placed $(length(crystal_positions)) out of $(config.num_crystals) crystals"
                if attempt == max_retries
                    throw(ErrorException("Failed to place enough crystals after $max_retries attempts"))
                end
                continue
            end
            
            # LaMEM-Modell Setup
            η_crystal = 1e4 * config.η
            ρ_magma = 2700.0
            
            model = Model(
                Grid(nel=(config.nx, config.nz), x=[-1, 1], z=[-1, 1]), 
                Time(nstep_max=1), 
                Output(out_strain_rate=1)
            )
            
            matrix = Phase(ID=0, Name="matrix", eta=config.η, rho=ρ_magma)
            crystal = Phase(ID=1, Name="crystal", eta=η_crystal, rho=ρ_magma + config.Δρ)
            add_phase!(model, crystal, matrix)
            
            # Kristalle als Sphären hinzufügen
            for (i, pos) in enumerate(crystal_positions)
                add_sphere!(model, cen=(pos[1], 0.0, pos[2]), radius=config.R, phase=ConstantPhase(1))
            end
            
            # LaMEM ausführen
            run_lamem(model, 1)
            data, _ = read_LaMEM_timestep(model, 1)
            
            # Daten extrahieren und validieren
            phase = data.fields.phase[:, 1, :]
            Vx = data.fields.velocity[1][:, 1, :]
            Vz = data.fields.velocity[3][:, 1, :]
            
            # Validierung der Daten
            if any(isnan.(phase)) || any(isnan.(Vx)) || any(isnan.(Vz))
                @warn "Attempt $attempt: NaN values detected in LaMEM output"
                if attempt == max_retries
                    throw(ErrorException("LaMEM produced NaN values after $max_retries attempts"))
                end
                continue
            end
            
            # Dimensionen-Check und Anpassung
            target_h, target_w = config.image_size
            actual_h, actual_w = size(phase)
            
            if (actual_h, actual_w) != (target_h, target_w)
                # Zuschneiden oder auffüllen auf Zielgröße
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
            
            # Stokes-Geschwindigkeit berechnen
            V_stokes = 2/9 * config.Δρ * 9.81 * (config.R * 1000)^2 / config.η
            V_stokes_cm_year = V_stokes * 100 * (3600 * 24 * 365.25)
            
            # Erfolgreiche Generierung
            return phase, Vx, Vz, V_stokes_cm_year, crystal_positions
            
        catch e
            @warn "Attempt $attempt failed: $e"
            if attempt == max_retries
                rethrow(e)
            end
            # Kurze Pause vor nächstem Versuch
            sleep(0.1)
        end
    end
end

# Verbesserte Phasenfeld-Kodierung
function encode_multi_phase_field(phase_field::Matrix{Float32})
    """
    Kodiert Phasenfeld in binäre Kanäle:
    Kanal 1: Matrix (Phase 0)
    Kanal 2: Kristalle (Phase 1)
    """
    h, w = size(phase_field)
    encoded = zeros(Float32, h, w, 2)
    
    # Matrix-Kanal
    encoded[:, :, 1] = (phase_field .≈ 0.0)
    # Kristall-Kanal  
    encoded[:, :, 2] = (phase_field .≈ 1.0)
    
    return encoded
end

# Datenvorverarbeitung
function preprocess_sample(phase, vx, vz, v_stokes, config::CONFIG)
    """
    Bereitet ein Sample für das Training vor
    """
    # Phase-Kodierung
    phase_encoded = encode_multi_phase_field(Float32.(phase))
    
    # Geschwindigkeiten normalisieren
    vx_norm = Float32.(vx ./ v_stokes)
    vz_norm = Float32.(vz ./ v_stokes)
    
    # Input: (H, W, 2, 1) - Phase-Kanäle
    input_tensor = reshape(phase_encoded, config.image_size..., 2, 1)
    
    # Target: (H, W, 2, 1) - Geschwindigkeits-Kanäle
    target_tensor = zeros(Float32, config.image_size..., 2, 1)
    target_tensor[:, :, 1, 1] = vx_norm
    target_tensor[:, :, 2, 1] = vz_norm
    
    return input_tensor, target_tensor
end

# Dataset-Generierung mit Memory-Monitoring
function create_dataset_with_monitoring(config::CONFIG)
    """
    Dataset-Erstellung mit Memory-Monitoring
    """
    println("Generiere $(config.dataset_size) Samples mit Memory-Monitoring...")
    
    inputs = []
    targets = []
    metadata = []
    failed_samples = 0
    
    # Memory-Tracking
    initial_memory = Base.gc_bytes()
    
    progress = Progress(config.dataset_size, desc="Generating samples: ")
    
    for i in 1:config.dataset_size
        try
            # Memory-Check vor Sample-Generierung
            current_memory = Base.gc_bytes()
            memory_used_mb = (current_memory - initial_memory) / (1024^2)
            
            if memory_used_mb > 8000  # Warnung bei > 8GB
                @warn "Memory usage high: $(round(memory_used_mb, digits=1)) MB"
                GC.gc()  # Aggressive garbage collection
            end
            
            # Sample generieren
            phase, vx, vz, v_stokes, positions = generate_multi_crystal_data_with_lamem_robust(config)
            input_tensor, target_tensor = preprocess_sample(phase, vx, vz, v_stokes, config)
            
            # Tensor-Validierung
            if any(isnan.(input_tensor)) || any(isnan.(target_tensor))
                @warn "NaN detected in preprocessed tensors for sample $i"
                failed_samples += 1
                continue
            end
            
            push!(inputs, input_tensor)
            push!(targets, target_tensor)
            push!(metadata, (v_stokes=v_stokes, positions=positions, sample_id=i))
            
        catch e
            @warn "Fehler bei Sample $i: $e"
            failed_samples += 1
        end
        
        # Update progress mit zusätzlichen Infos
        next!(progress, showvalues = [
            (:successful, length(inputs)),
            (:failed, failed_samples),
            (:success_rate, "$(round(100*length(inputs)/i, digits=1))%")
        ])
        
        # Memory cleanup
        if i % 20 == 0
            GC.gc()
            current_memory = Base.gc_bytes()
            memory_used_mb = (current_memory - initial_memory) / (1024^2)
            println("  Memory nach $i Samples: $(round(memory_used_mb, digits=1)) MB")
        end
    end
    
    finish!(progress)
    
    success_rate = round(100 * length(inputs) / config.dataset_size, digits=1)
    println("\nDataset-Generierung abgeschlossen:")
    println("  Erfolgreich: $(length(inputs))/$(config.dataset_size) ($success_rate%)")
    println("  Fehlgeschlagen: $failed_samples")
    
    if length(inputs) < config.dataset_size * 0.5
        @warn "Weniger als 50% der Samples erfolgreich generiert!"
    end
    
    return inputs, targets, metadata
end

# Verbesserte UNet-Architektur
struct MultiCrystalUNet
    # Encoder
    encoder1
    encoder2
    encoder3
    encoder4
    bottleneck
    
    # Decoder
    decoder4
    decoder4_conv
    decoder3
    decoder3_conv
    decoder2
    decoder2_conv
    decoder1
    final_conv
end

Flux.@functor MultiCrystalUNet

# Hilfsfunktion für Skip-Connections
function crop_and_concat(upsampled, skip_connection)
    """
    Kombiniert Upsampled-Feature mit Skip-Connection
    """
    up_h, up_w = size(upsampled)[1:2]
    skip_h, skip_w = size(skip_connection)[1:2]
    
    # Auf kleinere Größe zuschneiden
    target_h = min(up_h, skip_h)
    target_w = min(up_w, skip_w)
    
    up_cropped = upsampled[1:target_h, 1:target_w, :, :]
    skip_cropped = skip_connection[1:target_h, 1:target_w, :, :]
    
    return cat(up_cropped, skip_cropped, dims=3)
end

# UNet-Konstruktor
function create_multi_crystal_unet(config::CONFIG)
    # Encoder (Downsampling)
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
    
    # Bottleneck
    bottleneck = Chain(
        MaxPool((2, 2)),
        Conv((3, 3), 256 => 512, relu, pad=SamePad()),
        BatchNorm(512),
        Dropout(0.5),
        Conv((3, 3), 512 => 512, relu, pad=SamePad()),
        BatchNorm(512)
    )
    
    # Decoder (Upsampling)
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
        Conv((1, 1), 32 => 2)  # Output: 2 Kanäle für vx, vz
    )
    
    return MultiCrystalUNet(
        encoder1, encoder2, encoder3, encoder4, bottleneck,
        decoder4, decoder4_conv, decoder3, decoder3_conv,
        decoder2, decoder2_conv, decoder1, final_conv
    )
end

# Forward-Pass
function (model::MultiCrystalUNet)(x)
    # Encoder
    e1 = model.encoder1(x)
    e2 = model.encoder2(e1)
    e3 = model.encoder3(e2)
    e4 = model.encoder4(e3)
    
    # Bottleneck
    b = model.bottleneck(e4)
    
    # Decoder mit Skip-Connections
    d4 = model.decoder4(b)
    d4 = model.decoder4_conv(crop_and_concat(d4, e4))
    
    d3 = model.decoder3(d4)
    d3 = model.decoder3_conv(crop_and_concat(d3, e3))
    
    d2 = model.decoder2(d3)
    d2 = model.decoder2_conv(crop_and_concat(d2, e2))
    
    d1 = model.decoder1(d2)
    output = model.final_conv(crop_and_concat(d1, e1))
    
    return output
end

# Verbesserte Batch-Erstellung mit Validierung
function create_batches_validated(inputs, targets, batch_size)
    """
    Erstellt Batches mit Dimensionsvalidierung
    """
    n_samples = length(inputs)
    batches = []
    
    println("Erstelle Batches...")
    println("  Samples: $n_samples")
    println("  Batch-Größe: $batch_size")
    
    # Validiere alle Input-Dimensionen
    first_input_size = size(inputs[1])
    first_target_size = size(targets[1])
    
    for i in 1:n_samples
        if size(inputs[i]) != first_input_size
            error("Input $i hat inkonsistente Dimensionen: $(size(inputs[i])) vs $first_input_size")
        end
        if size(targets[i]) != first_target_size
            error("Target $i hat inkonsistente Dimensionen: $(size(targets[i])) vs $first_target_size")
        end
    end
    
    println("  Input-Dimensionen: $first_input_size")
    println("  Target-Dimensionen: $first_target_size")
    
    for i in 1:batch_size:n_samples
        batch_end = min(i + batch_size - 1, n_samples)
        batch_indices = i:batch_end
        actual_batch_size = length(batch_indices)
        
        try
            # Tensoren zusammenfügen
            input_batch = cat(inputs[batch_indices]..., dims=4)
            target_batch = cat(targets[batch_indices]..., dims=4)
            
            # Dimensionscheck
            expected_input_dims = (first_input_size[1], first_input_size[2], first_input_size[3], actual_batch_size)
            expected_target_dims = (first_target_size[1], first_target_size[2], first_target_size[3], actual_batch_size)
            
            if size(input_batch) != expected_input_dims
                error("Batch input dimensions mismatch: $(size(input_batch)) vs $expected_input_dims")
            end
            if size(target_batch) != expected_target_dims
                error("Batch target dimensions mismatch: $(size(target_batch)) vs $expected_target_dims")
            end
            
            push!(batches, (input_batch, target_batch))
            
        catch e
            @warn "Fehler bei Batch-Erstellung für Indices $batch_indices: $e"
        end
    end
    
    println("  Erfolgreich $(length(batches)) Batches erstellt")
    return batches
end

# Training-Loop
function train_multi_crystal_model(model, train_batches, config::CONFIG)
    println("\n=== TRAINING STARTET ===")
    println("Batches: $(length(train_batches))")
    println("Epochen: $(config.epochs)")
    println("Lernrate: $(config.learning_rate)")
    
    # Optimizer
    opt_state = Optimisers.setup(Optimisers.Adam(config.learning_rate), model)
    
    # GPU-Setup
    if config.use_gpu && CUDA.functional()
        model = gpu(model)
        println("Modell auf GPU verschoben")
    end
    
    # Training
    losses = Float32[]
    
    for epoch in 1:config.epochs
        println("\n--- Epoche $epoch/$(config.epochs) ---")
        
        epoch_loss = 0.0f0
        batch_count = 0
        
        for (batch_idx, (input_batch, target_batch)) in enumerate(train_batches)
            try
                # Auf Device verschieben
                if config.use_gpu && CUDA.functional()
                    input_batch = gpu(input_batch)
                    target_batch = gpu(target_batch)
                end
                
                # Gradienten berechnen
                ∇model = gradient(m -> mse(m(input_batch), target_batch), model)[1]
                batch_loss = mse(model(input_batch), target_batch)
                
                # Parameter updaten
                opt_state, model = Optimisers.update!(opt_state, model, ∇model)
                
                epoch_loss += batch_loss
                batch_count += 1
                
                if batch_idx % 10 == 0
                    println("  Batch $batch_idx: Loss = $(round(batch_loss, digits=6))")
                end
                
            catch e
                @warn "Fehler bei Batch $batch_idx: $e"
            end
            
            # Memory cleanup
            if config.use_gpu && CUDA.functional()
                CUDA.reclaim()
            end
        end
        
        avg_loss = batch_count > 0 ? epoch_loss / batch_count : NaN32
        push!(losses, avg_loss)
        
        println("Epoche $epoch: Durchschnittsverlust = $(round(avg_loss, digits=6))")
        
        # Checkpoint speichern
        if epoch % 10 == 0
            model_cpu = cpu(model)
            checkpoint_path = joinpath(config.save_path, "multi_crystal_checkpoint_epoch_$(epoch).bson")
            mkpath(dirname(checkpoint_path))
            @save checkpoint_path model_cpu
            println("  Checkpoint gespeichert: $checkpoint_path")
        end
        
        # Memory cleanup
        GC.gc()
        if config.use_gpu && CUDA.functional()
            CUDA.reclaim()
        end
    end
    
    return model, losses
end

# Evaluierung
function evaluate_model(model, test_inputs, test_targets, config::CONFIG)
    """
    Evaluiert das Modell auf Testdaten
    """
    model_eval = cpu(model)  # Evaluation auf CPU
    
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

# Visualisierung
function visualize_prediction(model, test_input, test_target, config::CONFIG, save_path="prediction.png")
    """
    Visualisiert Vorhersage vs Ground Truth
    """
    model_eval = cpu(model)
    prediction = model_eval(test_input)
    
    # Daten extrahieren
    phase_matrix = test_input[:, :, 1, 1]  # Matrix-Kanal
    phase_crystal = test_input[:, :, 2, 1]  # Kristall-Kanal
    
    vx_true = test_target[:, :, 1, 1]
    vz_true = test_target[:, :, 2, 1]
    vx_pred = prediction[:, :, 1, 1]
    vz_pred = prediction[:, :, 2, 1]
    
    # Plot erstellen
    fig = Figure(resolution=(1200, 800))
    
    # Phasenfeld
    ax1 = Axis(fig[1, 1], title="Phasenfeld", aspect=DataAspect())
    heatmap!(ax1, phase_crystal, colormap=:grays)
    
    # Ground Truth vz
    ax2 = Axis(fig[1, 2], title="Ground Truth vz", aspect=DataAspect())
    heatmap!(ax2, vz_true, colormap=:RdBu, colorrange=(-3, 1))
    contour!(ax2, phase_crystal, levels=[0.5], color=:black)
    
    # Vorhersage vz
    ax3 = Axis(fig[1, 3], title="Vorhersage vz", aspect=DataAspect())
    heatmap!(ax3, vz_pred, colormap=:RdBu, colorrange=(-3, 1))
    contour!(ax3, phase_crystal, levels=[0.5], color=:black)
    
    # Differenz
    ax4 = Axis(fig[2, 1], title="Differenz vz", aspect=DataAspect())
    heatmap!(ax4, abs.(vz_true - vz_pred), colormap=:hot)
    
    # MSE
    mse_vx = mean((vx_true - vx_pred).^2)
    mse_vz = mean((vz_true - vz_pred).^2)
    
    Label(fig[2, 2:3], "MSE vx: $(round(mse_vx, digits=6))\nMSE vz: $(round(mse_vz, digits=6))")
    
    save(save_path, fig)
    display(fig)
    
    return fig
end

# Verbesserte Konfiguration mit Validierung
function create_robust_config(;
    image_size = (256, 256),
    num_crystals = 3,
    batch_size = 4,
    learning_rate = 0.001f0,
    epochs = 30,  # Reduziert für Testing
    dataset_size = 50,  # Reduziert für Testing
    validation_split = 0.2f0,
    use_gpu = false
)
    """
    Erstellt robuste Konfiguration mit Validierung
    """
    config = CONFIG(
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
    
    println("Konfiguration erstellt:")
    println("  Bildgröße: $(config.image_size)")
    println("  Kristalle: $(config.num_crystals)")
    println("  Dataset-Größe: $(config.dataset_size)")
    println("  Epochen: $(config.epochs)")
    println("  GPU: $(config.use_gpu)")
    
    return config
end

# Hauptfunktion mit verbesserter Robustheit
function run_robust_multi_crystal_training()
    """
    Robuste Version des Multi-Kristall Trainings
    """
    println("=== ROBUSTES MULTI-KRISTALL TRAINING ===")
    
    try
        # Konfiguration mit reduzierten Werten für Testing
        config = create_robust_config(
            dataset_size = 20,  # Klein für Testing
            epochs = 10,        # Weniger Epochen für Testing
            batch_size = 2      # Kleinere Batches
        )
        
        # Dataset generieren mit Monitoring
        println("\n1. DATASET-GENERIERUNG MIT MONITORING")
        inputs, targets, metadata = create_dataset_with_monitoring(config)
        
        if length(inputs) < 5
            error("Zu wenige Samples generiert für Training!")
        end
        
        # Train/Test Split
        n_train = max(1, Int(floor((1 - config.validation_split) * length(inputs))))
        train_inputs = inputs[1:n_train]
        train_targets = targets[1:n_train]
        test_inputs = inputs[n_train+1:end]
        test_targets = targets[n_train+1:end]
        
        println("Training: $(length(train_inputs)), Test: $(length(test_inputs))")
        
        # Batches erstellen mit Validierung
        println("\n2. VALIDIERTE BATCH-ERSTELLUNG")
        train_batches = create_batches_validated(train_inputs, train_targets, config.batch_size)
        
        if isempty(train_batches)
            error("Keine gültigen Batches erstellt!")
        end
        
        # Modell erstellen
        println("\n3. MODELL-ERSTELLUNG")
        model = create_multi_crystal_unet(config)
        
        # Training mit besserer Fehlerbehandlung
        println("\n4. ROBUSTES TRAINING")
        trained_model, losses = train_multi_crystal_model(model, train_batches, config)
        
        # Finales Modell speichern
        final_model_cpu = cpu(trained_model)
        timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
        final_path = joinpath(config.save_path, "robust_multi_crystal_model_$timestamp.bson")
        mkpath(dirname(final_path))
        @save final_path final_model_cpu
        
        # Evaluierung falls Testdaten vorhanden
        test_loss = NaN
        if !isempty(test_inputs)
            println("\n5. EVALUIERUNG")
            test_loss = evaluate_model(trained_model, test_inputs, test_targets, config)
            
            # Visualisierung
            println("\n6. VISUALISIERUNG")
            visualize_prediction(trained_model, test_inputs[1], test_targets[1], config, 
                               "robust_prediction_$timestamp.png")
        end
        
        # Zusammenfassung
        println("\n" * "="^60)
        println("ROBUSTES TRAINING ERFOLGREICH ABGESCHLOSSEN")
        println("="^60)
        println("Konfiguration:")
        println("  Samples generiert: $(length(inputs))")
        println("  Training Samples: $(length(train_inputs))")
        println("  Test Samples: $(length(test_inputs))")
        println("  Epochen: $(config.epochs)")
        println("  Finaler Trainingsverlust: $(round(losses[end], digits=6))")
        if !isnan(test_loss)
            println("  Test-Verlust: $(round(test_loss, digits=6))")
        end
        println("  Modell gespeichert: $final_path")
        println("="^60)
        
        return trained_model, losses, test_loss, config
        
    catch e
        println("\n" * "="^60)
        println("FEHLER IM TRAINING")
        println("="^60)
        println("Fehler: $e")
        println("Stacktrace:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        println("="^60)
        rethrow(e)
    end
end

# Alternative: Originalfunktion für größere Datasets
function run_multi_crystal_training()
    """
    Ursprüngliche Hauptfunktion für größere Datasets
    """
    println("=== MULTI-KRISTALL VELOCITY PREDICTION ===")
    
    # Konfiguration
    config = CONFIG()
    validate_config(config)
    
    # Dataset generieren
    println("\n1. DATASET-GENERIERUNG")
    inputs, targets, metadata = create_dataset_with_monitoring(config)
    
    if length(inputs) < 10
        error("Zu wenige Samples generiert!")
    end
    
    # Train/Test Split
    n_train = Int(floor((1 - config.validation_split) * length(inputs)))
    train_inputs = inputs[1:n_train]
    train_targets = targets[1:n_train]
    test_inputs = inputs[n_train+1:end]
    test_targets = targets[n_train+1:end]
    
    println("Training: $(length(train_inputs)), Test: $(length(test_inputs))")
    
    # Batches erstellen
    println("\n2. BATCH-ERSTELLUNG")
    train_batches = create_batches_validated(train_inputs, train_targets, config.batch_size)
    
    # Modell erstellen
    println("\n3. MODELL-ERSTELLUNG")
    model = create_multi_crystal_unet(config)
    
    # Training
    println("\n4. TRAINING")
    trained_model, losses = train_multi_crystal_model(model, train_batches, config)
    
    # Finales Modell speichern
    final_model_cpu = cpu(trained_model)
    final_path = joinpath(config.save_path, "final_multi_crystal_model.bson")
    mkpath(dirname(final_path))
    @save final_path final_model_cpu
    
    # Evaluierung
    println("\n5. EVALUIERUNG")
    test_loss = evaluate_model(trained_model, test_inputs, test_targets, config)
    
    # Visualisierung
    if !isempty(test_inputs)
        println("\n6. VISUALISIERUNG")
        visualize_prediction(trained_model, test_inputs[1], test_targets[1], config)
    end
    
    println("\n" * "="^50)
    println("TRAINING ABGESCHLOSSEN")
    println("Samples: $(length(inputs))")
    println("Finaler Trainingsverlust: $(round(losses[end], digits=6))")
    println("Test-Verlust: $(round(test_loss, digits=6))")
    println("Modell gespeichert: $final_path")
    println("="^50)
    
    return trained_model, losses, test_loss
end

# Hilfsfunktionen für Analyse und Debugging
function analyze_dataset(inputs, targets, metadata)
    """
    Analysiert das generierte Dataset
    """
    println("\n=== DATASET-ANALYSE ===")
    
    # Grundlegende Statistiken
    n_samples = length(inputs)
    println("Anzahl Samples: $n_samples")
    
    if n_samples > 0
        input_shape = size(inputs[1])
        target_shape = size(targets[1])
        println("Input-Shape: $input_shape")
        println("Target-Shape: $target_shape")
        
        # Analysiere Phasenfelder
        crystal_counts = []
        for input in inputs
            crystal_channel = input[:, :, 2, 1]
            crystal_pixels = sum(crystal_channel .> 0.5)
            push!(crystal_counts, crystal_pixels)
        end
        
        println("Kristall-Pixel pro Sample:")
        println("  Min: $(minimum(crystal_counts))")
        println("  Max: $(maximum(crystal_counts))")
        println("  Mittel: $(round(mean(crystal_counts), digits=1))")
        
        # Analysiere Geschwindigkeiten
        vz_ranges = []
        for target in targets
            vz = target[:, :, 2, 1]
            vz_range = maximum(vz) - minimum(vz)
            push!(vz_ranges, vz_range)
        end
        
        println("V_z Bereiche pro Sample:")
        println("  Min: $(round(minimum(vz_ranges), digits=3))")
        println("  Max: $(round(maximum(vz_ranges), digits=3))")
        println("  Mittel: $(round(mean(vz_ranges), digits=3))")
        
        # Stokes-Geschwindigkeiten
        if !isempty(metadata)
            v_stokes_values = [m.v_stokes for m in metadata]
            println("Stokes-Geschwindigkeiten:")
            println("  Min: $(round(minimum(v_stokes_values), digits=6))")
            println("  Max: $(round(maximum(v_stokes_values), digits=6))")
            println("  Mittel: $(round(mean(v_stokes_values), digits=6))")
        end
    end
    
    println("=" * 25)
    return nothing
end

function test_single_sample(config::CONFIG)
    """
    Testet die Generierung eines einzelnen Samples
    """
    println("=== EINZELNER SAMPLE-TEST ===")
    
    try
        phase, vx, vz, v_stokes, positions = generate_multi_crystal_data_with_lamem_robust(config)
        input_tensor, target_tensor = preprocess_sample(phase, vx, vz, v_stokes, config)
        
        println("Erfolgreich generiert:")
        println("  Phase Shape: $(size(phase))")
        println("  Vx Shape: $(size(vx))")
        println("  Vz Shape: $(size(vz))")
        println("  Input Tensor Shape: $(size(input_tensor))")
        println("  Target Tensor Shape: $(size(target_tensor))")
        println("  Kristall-Positionen: $positions")
        println("  Stokes-Geschwindigkeit: $(round(v_stokes, digits=6))")
        
        return true
    catch e
        println("Fehler: $e")
        return false
    end
end

# Verwendung und Beispiele
println("="^70)
println("KOMPLETTER KORRIGIERTER MULTI-KRISTALL UNET CODE")
println("="^70)
println("Verfügbare Funktionen:")
println()
println("1. ROBUSTES TRAINING (empfohlen für Testing):")
println("   model, losses, test_loss, config = run_robust_multi_crystal_training()")
println()
println("2. VOLLSTÄNDIGES TRAINING (für Produktion):")
println("   model, losses, test_loss = run_multi_crystal_training()")
println()
println("3. EINZELNER SAMPLE-TEST:")
println("   config = create_robust_config()")
println("   success = test_single_sample(config)")
println()
println("4. DATASET-ANALYSE:")
println("   inputs, targets, metadata = create_dataset_with_monitoring(config)")
println("   analyze_dataset(inputs, targets, metadata)")
println()
println("5. EIGENE KONFIGURATION:")
println("   config = create_robust_config(")
println("       dataset_size = 100,")
println("       epochs = 30,")
println("       num_crystals = 3")
println("   )")
println()
println("Empfehlung: Starten Sie mit run_robust_multi_crystal_training()")
println("für initiale Tests und Debugging.")
println("="^70)