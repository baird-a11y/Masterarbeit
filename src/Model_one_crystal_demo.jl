using Flux
using Flux: mse, gpu, cpu
using CUDA
using Statistics
using FileIO
using LinearAlgebra
using Optimisers
using ProgressMeter
using BSON: @save, @load
using Random
using LaMEM, GeophysicalModelGenerator

println("=== VERBESSERTER VELOCITY UNET TRAINER ===")

# ==================== MODELL-STRUKTUREN ====================

struct VelocityUNet
    encoder1; encoder2; encoder3; encoder4; bottleneck
    decoder4; decoder4_1; decoder3; decoder3_1
    decoder2; decoder2_1; decoder1; decoder1_1
end

Flux.@functor VelocityUNet

function crop_and_concat(x, skip, dims=3)
    """
    GPU-freundliche Skip-Connection Funktion
    """
    x_size = size(x)
    skip_size = size(skip)
    
    # Einfache Lösung: Falls Größen nicht passen, nimm die kleinere
    min_h = min(x_size[1], skip_size[1])
    min_w = min(x_size[2], skip_size[2])
    
    # Schneide beide auf gleiche Größe zu
    x_cropped = x[1:min_h, 1:min_w, :, :]
    skip_cropped = skip[1:min_h, 1:min_w, :, :]
    
    # Zusammenfügen
    return cat(x_cropped, skip_cropped, dims=dims)
end

function (model::VelocityUNet)(x)
    """
    Forward-Pass durch das UNet
    """
    # Encoder
    e1 = model.encoder1(x)
    e2 = model.encoder2(e1)
    e3 = model.encoder3(e2)
    e4 = model.encoder4(e3)
    b = model.bottleneck(e4)
    
    # Decoder mit Skip-Connections
    d4 = model.decoder4(b)
    d4 = model.decoder4_1(crop_and_concat(d4, e4))
    
    d3 = model.decoder3(d4)
    d3 = model.decoder3_1(crop_and_concat(d3, e3))
    
    d2 = model.decoder2(d3)
    d2 = model.decoder2_1(crop_and_concat(d2, e2))
    
    d1 = model.decoder1(d2)
    d1 = model.decoder1_1(crop_and_concat(d1, e1))
    
    return d1
end

# ==================== ZENTRALE KONFIGURATION ====================

# Alle Parameter an einem Ort - einfach zu ändern!
const CONFIG = (
    # Modell-Parameter
    image_size = (256, 256),           # Konsistente 256×256
    input_channels = 1,                # Phasenfeld
    output_channels = 2,               # v_x, v_z
    
    # Training-Parameter
    learning_rate = 0.001,
    num_epochs = 30,                   # Erstmal weniger für Testing
    batch_size = 2,                    # Kleine Batches für 256×256
    
    # Dataset-Parameter
    dataset_size = 100,                 # Erstmal weniger für Testing
    test_split = 0.1,                  # 10% für Testing
    
    # LaMEM-Parameter (Bereiche für Variation)
    eta_range = (1e19, 1e21),         # Viskosität-Bereich
    delta_rho_range = (100, 500),      # Dichtedifferenz-Bereich
    position_range = (-0.6, 0.6),     # Kristall-Position Bereich
    radius_range = (0.03, 0.08),      # Kristall-Radius Bereich
    
    # Speicher-Parameter
    checkpoint_dir = "velocity_checkpoints_256_v2",
    save_every_n_epochs = 5,           # Häufiger speichern bei weniger Epochen
    
    # Hardware-Parameter
    use_gpu = false,                   # CPU-Training für Stabilität!
    memory_cleanup_frequency = 10,     # Häufiger cleanup
    
    # Debug-Parameter
    verbose = true,
    save_sample_images = true
)

println("Konfiguration geladen:")
println("  Bildgröße: $(CONFIG.image_size)")
println("  Trainingsdaten: $(CONFIG.dataset_size)")
println("  Epochen: $(CONFIG.num_epochs)")
println("  Lernrate: $(CONFIG.learning_rate)")

# ==================== VERBESSERTE LAMEM-FUNKTION ====================

function LaMEM_Single_crystal_fixed(; η=1e20, Δρ=200, cen_2D=[(0.0, 0.0)], R=[0.1])
    """
    LaMEM Funktion mit garantierten $(CONFIG.image_size) Dimensionen
    """
    η_crystal = 1e4*η
    ρ_magma = 2700
    
    # KRITISCH: nel=(255,255) für exakt 256×256 Output
    target_h, target_w = CONFIG.image_size
    nel_h, nel_w = target_h - 1, target_w - 1  # LaMEM erzeugt nel+1 Punkte
    
    model = Model(
        Grid(nel=(nel_h, nel_w), x=[-1,1], z=[-1,1]), 
        Time(nstep_max=1), 
        Output(out_strain_rate=1)
    )
    
    matrix = Phase(ID=0, Name="matrix", eta=η, rho=ρ_magma)
    crystal = Phase(ID=1, Name="crystal", eta=η_crystal, rho=ρ_magma+Δρ)
    add_phase!(model, crystal, matrix)

    for i = 1:length(cen_2D)
        add_sphere!(model, cen=(cen_2D[i][1], 0.0, cen_2D[i][2]), radius=R[i], phase=ConstantPhase(1))
    end
   
    run_lamem(model, 1)
    data, _ = read_LaMEM_timestep(model, 1)

    # Daten extrahieren
    x_vec_1D = data.x.val[:,1,1]
    z_vec_1D = data.z.val[1,1,:]
    phase = data.fields.phase[:,1,:]
    Vx = data.fields.velocity[1][:,1,:]
    Vz = data.fields.velocity[3][:,1,:]
    
    V_stokes = 2/9*Δρ*9.81*(R[1]*1000)^2/(η)
    V_stokes_cm_year = V_stokes * 100 * (3600*24*365.25)
    
    # Dimensionen-Sicherheitscheck
    actual_size = size(phase)
    if actual_size != CONFIG.image_size
        if CONFIG.verbose
            println("   Warnung: LaMEM lieferte $actual_size, schneide auf $(CONFIG.image_size) zu")
        end
        
        # Zuschneiden auf Zielgröße
        phase = phase[1:target_h, 1:target_w]
        Vx = Vx[1:target_h, 1:target_w]
        Vz = Vz[1:target_h, 1:target_w]
        x_vec_1D = x_vec_1D[1:target_h]
        z_vec_1D = z_vec_1D[1:target_w]
    end

    return x_vec_1D, z_vec_1D, phase, Vx, Vz, V_stokes_cm_year
end

# ==================== KONFIGURIERBARE DATENGENERIERUNG ====================

function generate_random_params()
    """
    Generiert zufällige Parameter basierend auf CONFIG
    """
    # Logarithmische Verteilung für Viskosität
    η_min, η_max = CONFIG.eta_range
    η = 10^(rand() * (log10(η_max) - log10(η_min)) + log10(η_min))
    
    # Lineare Verteilungen
    Δρ = rand(CONFIG.delta_rho_range[1]:CONFIG.delta_rho_range[2])
    
    pos_min, pos_max = CONFIG.position_range
    x_pos = rand() * (pos_max - pos_min) + pos_min
    z_pos = rand() * (0.8 - 0.2) + 0.2  # Z immer zwischen 0.2 und 0.8
    
    rad_min, rad_max = CONFIG.radius_range
    R = rand() * (rad_max - rad_min) + rad_min
    
    return (
        η = η,
        Δρ = Δρ,
        cen_2D = [(x_pos, z_pos)],
        R = [R]
    )
end

function generate_single_sample_configured(sample_id)
    """
    Generiert ein Sample mit konfigurierbaren Parametern
    """
    if CONFIG.verbose && sample_id % 50 == 0
        println("Generiere Sample $sample_id/$(CONFIG.dataset_size)...")
    end
    
    # Zufällige Parameter generieren
    params = generate_random_params()
    
    try
        # LaMEM-Simulation
        x, z, phase, Vx, Vz, V_stokes = LaMEM_Single_crystal_fixed(;params...)
        
        # Dimensionsprüfung
        if size(phase) != CONFIG.image_size
            error("Sample $sample_id: Falsche Dimensionen $(size(phase))")
        end
        
        return (
            phase = phase, 
            vx = Vx, 
            vz = Vz, 
            v_stokes = V_stokes,
            params = params
        )
    catch e
        if CONFIG.verbose
            println("   Fehler bei Sample $sample_id: $e")
        end
        return nothing
    end
end

function generate_training_dataset_configured()
    """
    Generiert konfigurierbaren Trainingsdatensatz
    """
    println("\n=== GENERIERE TRAININGSDATENSATZ ===")
    println("Parameter:")
    println("  Samples: $(CONFIG.dataset_size)")
    println("  Bildgröße: $(CONFIG.image_size)")
    println("  Eta-Bereich: $(CONFIG.eta_range)")
    println("  Delta-Rho-Bereich: $(CONFIG.delta_rho_range)")
    
    # LaMEM-Test
    println("\n1. Teste LaMEM-Konsistenz...")
    test_params = generate_random_params()
    try
        x, z, phase, vx, vz, v_stokes = LaMEM_Single_crystal_fixed(;test_params...)
        println("   ✓ LaMEM-Test erfolgreich: $(size(phase))")
    catch e
        error("LaMEM-Test fehlgeschlagen: $e")
    end
    
    # Datengenerierung
    println("\n2. Generiere Samples...")
    dataset = []
    successful = 0
    
    progress = Progress(CONFIG.dataset_size, desc="Generating samples: ")
    
    for i in 1:CONFIG.dataset_size
        sample = generate_single_sample_configured(i)
        
        if sample !== nothing
            push!(dataset, sample)
            successful += 1
        end
        
        # Memory-Management
        if i % CONFIG.memory_cleanup_frequency == 0
            GC.gc()
            if CUDA.functional() && CONFIG.use_gpu
                CUDA.reclaim()
            end
        end
        
        next!(progress)
    end
    
    finish!(progress)
    
    success_rate = round(100 * successful / CONFIG.dataset_size, digits=1)
    println("\n3. Datengenerierung abgeschlossen:")
    println("   Erfolgreich: $successful/$(CONFIG.dataset_size) ($success_rate%)")
    
    if successful == 0
        error("Keine Samples generiert!")
    end
    
    return dataset
end

# ==================== KONFIGURIERBARE DATENVORVERARBEITUNG ====================

function preprocess_configured(sample_data)
    """
    Vorverarbeitung mit konfigurierbaren Parametern
    """
    h, w = CONFIG.image_size
    
    # Dimensionsprüfung
    if size(sample_data.phase) != (h, w)
        error("Sample hat falsche Dimensionen: $(size(sample_data.phase)) statt $(string(h, ", ", w))")
    end
    
    # Phase für UNet
    phase_input = reshape(Float32.(sample_data.phase), h, w, 1, 1)
    
    # Geschwindigkeiten normalisieren
    vx_norm = Float32.(sample_data.vx ./ sample_data.v_stokes)
    vz_norm = Float32.(sample_data.vz ./ sample_data.v_stokes)
    
    # Velocity target
    velocity_target = zeros(Float32, h, w, 2, 1)
    velocity_target[:, :, 1, 1] .= vx_norm
    velocity_target[:, :, 2, 1] .= vz_norm
    
    return (phase_input, velocity_target)
end

function create_training_batches_configured(raw_dataset)
    """
    Erstellt Trainings-Batches mit Konfiguration
    """
    println("Erstelle Trainings-Batches...")
    println("  Batch-Größe: $(CONFIG.batch_size)")
    
    processed_data = []
    
    for (i, sample) in enumerate(raw_dataset)
        try
            phase_input, velocity_target = preprocess_configured(sample)
            push!(processed_data, (phase_input, velocity_target))
        catch e
            println("Fehler bei Sample $i: $e")
        end
    end
    
    # Batches erstellen (erstmal einzeln)
    batched_data = []
    for data in processed_data
        push!(batched_data, data)
    end
    
    println("✓ $(length(batched_data)) Batches erstellt")
    return batched_data
end

# ==================== KONFIGURIERBARE MODELL-ARCHITEKTUR ====================

function create_velocity_unet_configured()
    """
    Erstellt UNet mit konfigurierbaren Parametern
    """
    input_ch = CONFIG.input_channels
    output_ch = CONFIG.output_channels
    
    # Encoder
    encoder1 = Chain(
        Conv((3, 3), input_ch => 32, relu, pad=SamePad()),
        BatchNorm(32),
        Conv((3, 3), 32 => 32, relu, pad=SamePad()),
        BatchNorm(32)
    )
    
    encoder2 = Chain(
        MaxPool((2,2)),
        Conv((3, 3), 32 => 64, relu, pad=SamePad()),
        BatchNorm(64),
        Conv((3, 3), 64 => 64, relu, pad=SamePad()),
        BatchNorm(64)
    )
    
    encoder3 = Chain(
        MaxPool((2,2)),
        Conv((3, 3), 64 => 128, relu, pad=SamePad()),
        BatchNorm(128),
        Conv((3, 3), 128 => 128, relu, pad=SamePad()),
        BatchNorm(128)
    )
    
    encoder4 = Chain(
        MaxPool((2,2)),
        Conv((3, 3), 128 => 256, relu, pad=SamePad()),
        BatchNorm(256),
        Conv((3, 3), 256 => 256, relu, pad=SamePad()),
        BatchNorm(256)
    )
    
    bottleneck = Chain(
        MaxPool((2,2)),
        Conv((3,3), 256 => 512, relu, pad=SamePad()),
        BatchNorm(512),
        Dropout(0.5),
        Conv((3,3), 512 => 512, relu, pad=SamePad()),
        BatchNorm(512)
    )
    
    # Decoder
    decoder4 = ConvTranspose((2, 2), 512 => 256, stride=2)
    decoder4_1 = Chain(
        Conv((3,3), 512 => 256, relu, pad=SamePad()),
        BatchNorm(256),
        Conv((3,3), 256 => 256, relu, pad=SamePad()),
        BatchNorm(256)
    )
    
    decoder3 = ConvTranspose((2,2), 256 => 128, stride=2)
    decoder3_1 = Chain(
        Conv((3,3), 256 => 128, relu, pad=SamePad()),
        BatchNorm(128),
        Conv((3,3), 128 => 128, relu, pad=SamePad()),
        BatchNorm(128)
    )
    
    decoder2 = ConvTranspose((2,2), 128 => 64, stride=2)
    decoder2_1 = Chain(
        Conv((3,3), 128 => 64, relu, pad=SamePad()),
        BatchNorm(64),
        Conv((3,3), 64 => 64, relu, pad=SamePad()),
        BatchNorm(64)
    )
    
    decoder1 = ConvTranspose((2,2), 64 => 32, stride=2)
    decoder1_1 = Chain(
        Conv((3,3), 64 => 32, relu, pad=SamePad()),
        BatchNorm(32),
        Conv((3,3), 32 => 32, relu, pad=SamePad()),
        BatchNorm(32),
        Dropout(0.2),
        Conv((1,1), 32 => output_ch)  # Keine Aktivierung für Regression
    )
    
    return VelocityUNet(encoder1, encoder2, encoder3, encoder4, bottleneck,
                        decoder4, decoder4_1, decoder3, decoder3_1,
                        decoder2, decoder2_1, decoder1, decoder1_1)
end

# ==================== KONFIGURIERBARE TRAINING-LOOP ====================

function train_velocity_unet_configured(model, train_data)
    """
    Training mit konfigurierbaren Parametern
    """
    println("\n=== STARTE TRAINING ===")
    println("Parameter:")
    println("  Epochen: $(CONFIG.num_epochs)")
    println("  Lernrate: $(CONFIG.learning_rate)")
    println("  Batches: $(length(train_data))")
    println("  GPU: $(CONFIG.use_gpu && CUDA.functional())")
    
    # Checkpoint-Verzeichnis erstellen
    mkpath(CONFIG.checkpoint_dir)
    
    # Optimizer
    opt_state = Optimisers.setup(Optimisers.Adam(CONFIG.learning_rate), model)
    losses = Float32[]
    
    # GPU setup
    if CONFIG.use_gpu && CUDA.functional()
        model = gpu(model)
        println("  Modell auf GPU verschoben")
    end
    
    # Training loop
    for epoch in 1:CONFIG.num_epochs
        println("\n--- Epoche $epoch/$(CONFIG.num_epochs) ---")
        
        total_loss = 0f0
        batch_count = 0
        
        for (batch_idx, (phase_batch, velocity_batch)) in enumerate(train_data)
            try
                # Auf GPU verschieben falls verfügbar
                if CONFIG.use_gpu && CUDA.functional()
                    phase_batch = gpu(phase_batch)
                    velocity_batch = gpu(velocity_batch)
                end
                
                # Verlust und Gradienten berechnen
                ∇model = gradient(m -> mse(m(phase_batch), velocity_batch), model)[1]
                batch_loss = mse(model(phase_batch), velocity_batch)
                
                # Parameter aktualisieren
                opt_state, model = Optimisers.update!(opt_state, model, ∇model)
                
                total_loss += batch_loss
                batch_count += 1
                
                if CONFIG.verbose && batch_idx % 100 == 0
                    println("  Batch $batch_idx: Loss = $(round(batch_loss, digits=6))")
                end
                
            catch e
                println("  Fehler bei Batch $batch_idx: $e")
            end
            
            # Memory cleanup
            if batch_idx % CONFIG.memory_cleanup_frequency == 0 && CONFIG.use_gpu && CUDA.functional()
                CUDA.reclaim()
            end
        end
        
        # Epoche abgeschlossen
        avg_loss = batch_count > 0 ? total_loss / batch_count : NaN32
        push!(losses, avg_loss)
        
        println("Epoche $epoch: Durchschnittsverlust = $(round(avg_loss, digits=6))")
        
        # Checkpoint speichern
        if epoch % CONFIG.save_every_n_epochs == 0
            model_cpu = cpu(model)
            checkpoint_path = joinpath(CONFIG.checkpoint_dir, "checkpoint_epoch_$(epoch).bson")
            @save checkpoint_path model_cpu
            println("  Checkpoint gespeichert: $checkpoint_path")
        end
        
        # Memory cleanup
        GC.gc()
        if CONFIG.use_gpu && CUDA.functional()
            CUDA.reclaim()
        end
    end
    
    # Finales Modell speichern
    final_model_cpu = cpu(model)
    final_path = joinpath(CONFIG.checkpoint_dir, "final_model_configured.bson")
    @save final_path final_model_cpu
    println("\nFinales Modell gespeichert: $final_path")
    
    return model, losses
end

# ==================== HAUPTFUNKTION ====================

function run_complete_training_configured()
    """
    Führt komplettes Training mit Konfiguration durch
    """
    println("=== KOMPLETTES KONFIGURIERTES TRAINING ===")
    
    # 1. Daten generieren
    println("\n1. DATENGENERIERUNG")
    raw_dataset = generate_training_dataset_configured()
    
    # 2. Daten vorverarbeiten
    println("\n2. DATENVORVERARBEITUNG")
    train_batches = create_training_batches_configured(raw_dataset)
    
    # 3. Modell erstellen
    println("\n3. MODELL-ERSTELLUNG")
    model = create_velocity_unet_configured()
    println("Modell erstellt mit $(CONFIG.input_channels) → $(CONFIG.output_channels) Kanälen")
    
    # 4. Training
    println("\n4. TRAINING")
    trained_model, losses = train_velocity_unet_configured(model, train_batches)
    
    # 5. Zusammenfassung
    println("\n" * "="^50)
    println("TRAINING ABGESCHLOSSEN")
    println("="^50)
    println("Konfiguration:")
    println("  Samples: $(length(raw_dataset))")
    println("  Epochen: $(CONFIG.num_epochs)")
    println("  Finaler Loss: $(round(losses[end], digits=6))")
    println("  Modell gespeichert: $(CONFIG.checkpoint_dir)")
    println("="^50)
    
    return trained_model, losses, raw_dataset
end

# ==================== VERWENDUNG ====================

println("\n" * "="^60)
println("KONFIGURIERTER VELOCITY UNET TRAINER GELADEN")
println("="^60)
println("Verwendung:")
println("  # Alle Parameter sind in CONFIG definiert")
println("  # Training starten:")
println("  model, losses, data = run_complete_training_configured()")
println("  ")
println("  # Parameter ändern:")
println("  # CONFIG.dataset_size = 1000")
println("  # CONFIG.learning_rate = 0.0005")
println("  # etc.")
println("="^60)

# ==================== TRAINING STARTEN ====================
model, losses, data = run_complete_training_configured()