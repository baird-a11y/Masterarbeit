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
#using GLMakie
CUDA.allowscalar(true)
println("=== ZWEI-KRISTALL VELOCITY UNET TRAINER ===")

# ==================== MODELL-STRUKTUREN ====================

struct VelocityUNet
    encoder1; encoder2; encoder3; encoder4; bottleneck
    decoder4; decoder4_1; decoder3; decoder3_1
    decoder2; decoder2_1; decoder1; decoder1_1
end

Flux.@functor VelocityUNet

function crop_and_concat_flux_native(x, skip, dims=3)
    """
    Flux-native Skip-Connections ohne manuelle Array-Operationen
    """
    # Nutze Flux-interne Größenanpassung
    x_size = size(x)
    skip_size = size(skip)
    
    # Falls Größen unterschiedlich sind, nutze einfache Flux-Operationen
    if x_size[1] != skip_size[1] || x_size[2] != skip_size[2]
        # Verwende Flux-native Interpolation/Pooling für Größenanpassung
        min_h = min(x_size[1], skip_size[1])
        min_w = min(x_size[2], skip_size[2])
        
        # Adaptive Pooling für einheitliche Größen (GPU-nativ)
        x_resized = Flux.AdaptiveMeanPool((min_h, min_w))(x)
        skip_resized = Flux.AdaptiveMeanPool((min_h, min_w))(skip)
        
        return cat(x_resized, skip_resized, dims=dims)
    else
        # Gleiche Größen - direkte Concatenation
        return cat(x, skip, dims=dims)
    end
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
    image_size = (256, 256),
    input_channels = 1,
    output_channels = 2,
    
    # Training-Parameter - GPU-optimiert
    learning_rate = 0.001,
    num_epochs = 5,                   # Mehr Epochen, da GPU schnell ist
    batch_size = 1,                    # Größere Batches für bessere GPU-Auslastung
    
    # Dataset-Parameter
    dataset_size = 5,                 # Moderater Datensatz für Test
    test_split = 0.1,
    
    # Multi-Kristall Parameter
    num_crystals = 2,
    crystal_min_distance = 0.15,
    allow_different_densities = true,
    
    # LaMEM-Parameter
    eta_range = (1e19, 1e21),
    delta_rho_range = (100, 500),
    position_range = (-0.6, 0.6),
    radius_range = (0.03, 0.08),
    
    # Speicher-Parameter
    checkpoint_dir = "velocity_checkpoints_gpu",
    save_every_n_epochs = 2,           # Häufiger speichern
    
    # Hardware-Parameter - GPU AKTIVIERT
    use_gpu = true,                    # ← WICHTIG: GPU aktiviert!
    memory_cleanup_frequency = 5,      # Häufiger GPU-Memory cleanup
    
    # Debug-Parameter
    verbose = true,
    save_sample_images = true
)

println("Konfiguration geladen:")
println("  Bildgröße: $(CONFIG.image_size)")
println("  Trainingsdaten: $(CONFIG.dataset_size)")
println("  Anzahl Kristalle: $(CONFIG.num_crystals)")
println("  Epochen: $(CONFIG.num_epochs)")
println("  Lernrate: $(CONFIG.learning_rate)")

# ==================== ERWEITERTE LAMEM-FUNKTION ====================

function LaMEM_Multi_crystal_fixed(; η=1e20, Δρ=200, cen_2D=[(0.0, 0.0)], R=[0.1])
    """
    LaMEM Funktion für mehrere Kristalle mit garantierten Dimensionen
    
    Parameter:
    - η: Viskosität der Matrix
    - Δρ: Dichtedifferenz (kann Array für verschiedene Kristalle sein)
    - cen_2D: Array von (x,z) Positionen für jeden Kristall
    - R: Array von Radien für jeden Kristall
    """
    η_crystal = 1e4*η
    ρ_magma = 2700
    
    # Dimensionen wie zuvor
    target_h, target_w = CONFIG.image_size
    nel_h, nel_w = target_h - 1, target_w - 1
    
    model = Model(
        Grid(nel=(nel_h, nel_w), x=[-1,1], z=[-1,1]), 
        Time(nstep_max=1), 
        Output(out_strain_rate=1)
    )
    
    # Matrix Phase
    matrix = Phase(ID=0, Name="matrix", eta=η, rho=ρ_magma)
    
    # Kristall Phasen erstellen
    crystal_phases = []
    for i in 1:length(cen_2D)
        # Falls Δρ ein Array ist, verwende verschiedene Werte, sonst gleichen Wert
        if isa(Δρ, Array)
            rho_crystal = ρ_magma + Δρ[min(i, length(Δρ))]
        else
            rho_crystal = ρ_magma + Δρ
        end
        
        crystal = Phase(ID=i, Name="crystal_$i", eta=η_crystal, rho=rho_crystal)
        push!(crystal_phases, crystal)
    end
    
    # Alle Phasen hinzufügen
    add_phase!(model, matrix)
    for crystal in crystal_phases
        add_phase!(model, crystal)
    end

    # Kristalle als Sphären hinzufügen
    for i = 1:length(cen_2D)
        add_sphere!(model, 
                   cen=(cen_2D[i][1], 0.0, cen_2D[i][2]), 
                   radius=R[min(i, length(R))],  # Falls weniger Radien als Positionen
                   phase=ConstantPhase(i))       # Phase ID = i
    end
   
    run_lamem(model, 1)
    data, _ = read_LaMEM_timestep(model, 1)

    # Daten extrahieren
    x_vec_1D = data.x.val[:,1,1]
    z_vec_1D = data.z.val[1,1,:]
    phase = data.fields.phase[:,1,:]
    Vx = data.fields.velocity[1][:,1,:]
    Vz = data.fields.velocity[3][:,1,:]
    
    # Stokes-Geschwindigkeit basierend auf erstem Kristall
    first_delta_rho = isa(Δρ, Array) ? Δρ[1] : Δρ
    first_radius = isa(R, Array) ? R[1] : R
    V_stokes = 2/9*first_delta_rho*9.81*(first_radius*1000)^2/(η)
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

# ==================== ERWEITERTE DATENGENERIERUNG ====================

function generate_random_params_two_crystals()
    """
    Generiert zufällige Parameter für zwei Kristalle
    """
    # Basis-Parameter
    η_min, η_max = CONFIG.eta_range
    η = 10^(rand() * (log10(η_max) - log10(η_min)) + log10(η_min))
    
    # Zwei verschiedene Dichtedifferenzen (optional)
    if CONFIG.allow_different_densities
        Δρ1 = rand(CONFIG.delta_rho_range[1]:CONFIG.delta_rho_range[2])
        Δρ2 = rand(CONFIG.delta_rho_range[1]:CONFIG.delta_rho_range[2])
        Δρ = [Δρ1, Δρ2]
    else
        Δρ_value = rand(CONFIG.delta_rho_range[1]:CONFIG.delta_rho_range[2])
        Δρ = [Δρ_value, Δρ_value]
    end
    
    # Mindestabstand zwischen Kristallen
    min_distance = CONFIG.crystal_min_distance
    
    # Erste Kristallposition
    pos_min, pos_max = CONFIG.position_range
    x1 = rand() * (pos_max - pos_min) + pos_min
    z1 = rand() * (0.8 - 0.2) + 0.2
    
    # Zweite Kristallposition mit Mindestabstand
    attempts = 0
    max_attempts = 100
    
    while attempts < max_attempts
        x2 = rand() * (pos_max - pos_min) + pos_min
        z2 = rand() * (0.8 - 0.2) + 0.2
        
        # Prüfe Abstand
        distance = sqrt((x2 - x1)^2 + (z2 - z1)^2)
        
        if distance >= min_distance
            # Radien
            rad_min, rad_max = CONFIG.radius_range
            R1 = rand() * (rad_max - rad_min) + rad_min
            R2 = rand() * (rad_max - rad_min) + rad_min
            
            return (
                η = η,
                Δρ = Δρ,
                cen_2D = [(x1, z1), (x2, z2)],
                R = [R1, R2]
            )
        end
        
        attempts += 1
    end
    
    # Falls kein gültiger Abstand gefunden wird, forciere Position
    if CONFIG.verbose
        println("Warnung: Fallback auf forcierte Kristallpositionen")
    end
    x2 = x1 + min_distance * (rand() > 0.5 ? 1 : -1)
    z2 = z1 + min_distance * (rand() > 0.5 ? 1 : -1)
    
    # Clamp auf gültigen Bereich
    x2 = clamp(x2, pos_min, pos_max)
    z2 = clamp(z2, 0.2, 0.8)
    
    rad_min, rad_max = CONFIG.radius_range
    R1 = rand() * (rad_max - rad_min) + rad_min
    R2 = rand() * (rad_max - rad_min) + rad_min
    
    return (
        η = η,
        Δρ = Δρ,
        cen_2D = [(x1, z1), (x2, z2)],
        R = [R1, R2]
    )
end

function generate_single_sample_two_crystals(sample_id)
    """
    Generiert ein Sample mit zwei Kristallen
    """
    if CONFIG.verbose && sample_id % 50 == 0
        println("Generiere Zwei-Kristall Sample $sample_id/$(CONFIG.dataset_size)...")
    end
    
    # Zufällige Parameter generieren
    params = generate_random_params_two_crystals()
    
    try
        # LaMEM-Simulation mit angepasster Funktion
        x, z, phase, Vx, Vz, V_stokes = LaMEM_Multi_crystal_fixed(;params...)
        
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
            println("   Fehler bei Zwei-Kristall Sample $sample_id: $e")
        end
        return nothing
    end
end

function generate_training_dataset_two_crystals()
    """
    Generiert konfigurierbaren Trainingsdatensatz für zwei Kristalle
    """
    println("\n=== GENERIERE ZWEI-KRISTALL TRAININGSDATENSATZ ===")
    println("Parameter:")
    println("  Samples: $(CONFIG.dataset_size)")
    println("  Bildgröße: $(CONFIG.image_size)")
    println("  Anzahl Kristalle: $(CONFIG.num_crystals)")
    println("  Mindestabstand: $(CONFIG.crystal_min_distance)")
    println("  Eta-Bereich: $(CONFIG.eta_range)")
    println("  Delta-Rho-Bereich: $(CONFIG.delta_rho_range)")
    
    # LaMEM-Test
    println("\n1. Teste LaMEM-Konsistenz für zwei Kristalle...")
    test_params = generate_random_params_two_crystals()
    try
        x, z, phase, vx, vz, v_stokes = LaMEM_Multi_crystal_fixed(;test_params...)
        println("   LaMEM-Test erfolgreich: $(size(phase))")
        println("   Kristall-Positionen: $(test_params.cen_2D)")
        println("   Kristall-Radien: $(test_params.R)")
        println("   Dichtedifferenzen: $(test_params.Δρ)")
    catch e
        error("LaMEM-Test fehlgeschlagen: $e")
    end
    
    # Datengenerierung
    println("\n2. Generiere Samples...")
    dataset = []
    successful = 0
    
    progress = Progress(CONFIG.dataset_size, desc="Generating two-crystal samples: ")
    
    for i in 1:CONFIG.dataset_size
        sample = generate_single_sample_two_crystals(i)
        
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
    
    println("$(length(batched_data)) Batches erstellt")
    return batched_data
end

# ==================== KONFIGURIERBARE MODELL-ARCHITEKTUR ====================

function create_velocity_unet_flux_native()
    """
    Vereinfachtes UNet mit Flux-nativen Operationen (wie Cityscapes-Code)
    """
    
    # Encoder - identisch zu funktionierendem Code
    encoder = Chain(
        # Block 1
        Conv((3, 3), 1 => 32, relu, pad=SamePad()),
        BatchNorm(32),
        Conv((3, 3), 32 => 32, relu, pad=SamePad()),
        BatchNorm(32),
        MaxPool((2,2)),
        
        # Block 2  
        Conv((3, 3), 32 => 64, relu, pad=SamePad()),
        BatchNorm(64),
        Conv((3, 3), 64 => 64, relu, pad=SamePad()),
        BatchNorm(64),
        MaxPool((2,2)),
        
        # Block 3
        Conv((3, 3), 64 => 128, relu, pad=SamePad()),
        BatchNorm(128),
        Conv((3, 3), 128 => 128, relu, pad=SamePad()),
        BatchNorm(128),
        MaxPool((2,2)),
        
        # Block 4
        Conv((3, 3), 128 => 256, relu, pad=SamePad()),
        BatchNorm(256),
        Conv((3, 3), 256 => 256, relu, pad=SamePad()),
        BatchNorm(256),
        MaxPool((2,2)),
    )
    
    # Bottleneck
    bottleneck = Chain(
        Conv((3,3), 256 => 512, relu, pad=SamePad()),
        BatchNorm(512),
        Dropout(0.5),
        Conv((3,3), 512 => 512, relu, pad=SamePad()),
        BatchNorm(512)
    )
    
    # Decoder - ohne komplexe Skip-Connections
    decoder = Chain(
        # Upsampling Block 1
        ConvTranspose((2, 2), 512 => 256, stride=2),
        Conv((3,3), 256 => 256, relu, pad=SamePad()),
        BatchNorm(256),
        
        # Upsampling Block 2
        ConvTranspose((2, 2), 256 => 128, stride=2),
        Conv((3,3), 128 => 128, relu, pad=SamePad()),
        BatchNorm(128),
        
        # Upsampling Block 3
        ConvTranspose((2,2), 128 => 64, stride=2),
        Conv((3,3), 64 => 64, relu, pad=SamePad()),
        BatchNorm(64),
        
        # Upsampling Block 4
        ConvTranspose((2,2), 64 => 32, stride=2),
        Conv((3,3), 32 => 32, relu, pad=SamePad()),
        BatchNorm(32),
        Dropout(0.2),
        
        # Output Layer
        Conv((1,1), 32 => 2)  # 2 Kanäle für v_x, v_z
    )
    
    # Kombiniere zu einfachem Sequential-Modell
    return Chain(encoder, bottleneck, decoder)
end

# ==================== FLUX-NATIVE VERLUSTFUNKTION ====================

function velocity_loss_flux_native(model, x, y_true)
    """
    Flux-native Verlustfunktion ohne manuelle Broadcast-Operationen
    """
    y_pred = model(x)
    
    # Nutze Flux.mse statt manuelle Implementierung
    return Flux.mse(y_pred, y_true)
end

# ==================== KONFIGURIERBARE TRAINING-LOOP ====================

function train_velocity_flux_style(model, train_data)
    """
    Training im Flux-Stil (wie funktionierender Cityscapes-Code)
    """
    println("\n=== STARTE FLUX-STYLE GPU-TRAINING ===")
    
    # GPU-Check
    if !CUDA.functional()
        error("CUDA nicht verfügbar!")
    end
    
    println("Flux-Style Training:")
    println("  GPU verfügbar: $(CUDA.functional())")
    println("  Epochen: $(CONFIG.num_epochs)")
    println("  Batches: $(length(train_data))")
    
    # Optimizer wie im funktionierenden Code
    opt_state = Optimisers.setup(Optimisers.Adam(CONFIG.learning_rate), model)
    losses = Float32[]
    
    # Modell auf GPU (wie Cityscapes)
    model = gpu(model)
    println("✓ Modell auf GPU verschoben")
    
    # Training loop im Cityscapes-Stil
    for epoch in 1:CONFIG.num_epochs
        println("\n--- Epoche $epoch/$(CONFIG.num_epochs) ---")
        
        total_loss = 0f0
        batch_count = 0
        
        for (batch_idx, (input_batch, target_batch)) in enumerate(train_data)
            try
                # Daten auf GPU (wie Cityscapes)
                input_batch = gpu(input_batch)
                target_batch = gpu(target_batch)
                
                # Gradient-Berechnung (wie Cityscapes)
                ∇model = gradient(m -> velocity_loss_flux_native(m, input_batch, target_batch), model)[1]
                batch_loss = velocity_loss_flux_native(model, input_batch, target_batch)
                
                # Parameter-Update (wie Cityscapes)
                opt_state, model = Optimisers.update!(opt_state, model, ∇model)
                
                total_loss += batch_loss
                batch_count += 1
                
                println("  Batch $batch_idx: Loss = $(round(batch_loss, digits=6)) ✓")
                
            catch e
                println("  ✗ Fehler bei Batch $batch_idx: $(typeof(e))")
                if batch_idx <= 2
                    println("    Detail: $e")
                end
            end
            
            # Memory cleanup (wie Cityscapes)
            if batch_idx % 5 == 0
                CUDA.reclaim()
            end
        end
        
        # Epoche abgeschlossen
        if batch_count > 0
            avg_loss = total_loss / batch_count
            push!(losses, avg_loss)
            println("Epoche $epoch: Durchschnittsverlust = $(round(avg_loss, digits=6))")
        else
            push!(losses, NaN32)
            println("Epoche $epoch: ALLE BATCHES FEHLGESCHLAGEN!")
        end
        
        # Checkpoint (wie Cityscapes)
        if epoch % 2 == 0
            model_cpu = cpu(model)
            checkpoint_path = joinpath(CONFIG.checkpoint_dir, "flux_checkpoint_epoch_$(epoch).bson")
            @save checkpoint_path model_cpu
            println("  ✓ Checkpoint gespeichert")
        end
    end
    
    # Finales Modell
    final_model_cpu = cpu(model)
    final_path = joinpath(CONFIG.checkpoint_dir, "final_model_flux_style.bson")
    @save final_path final_model_cpu
    println("\n✓ Flux-Style Modell gespeichert: $final_path")
    
    return model, losses
end

# ==================== HAUPTFUNKTION ====================

function run_flux_style_training()
    """
    Flux-Style Training ohne allowscalar (wie Cityscapes)
    """
    println("=== FLUX-STYLE ZWEI-KRISTALL TRAINING ===")
    
    # Daten generieren (bestehende Funktionen)
    println("\n1. DATENGENERIERUNG")
    raw_dataset = generate_training_dataset_two_crystals()
    
    println("\n2. DATENVORVERARBEITUNG") 
    train_batches = create_training_batches_configured(raw_dataset)
    
    println("\n3. FLUX-STYLE MODELL")
    model = create_velocity_unet_flux_native()
    println("Flux-Style Modell erstellt (Sequential)")
    
    println("\n4. FLUX-STYLE TRAINING")
    trained_model, losses = train_velocity_flux_style(model, train_batches)
    
    println("\n" * "="^50)
    println("FLUX-STYLE TRAINING ABGESCHLOSSEN")
    println("="^50)
    println("Finaler Loss: $(round(losses[end], digits=6))")
    println("Training erfolgreich: $(isnan(losses[end]) ? "NEIN" : "JA")")
    println("="^50)
    
    return trained_model, losses, raw_dataset
end

# ==================== ERWEITERTE DEBUG-FUNKTIONEN ====================

function find_crystal_centers_multi(phase_field)
    """
    Findet die Zentren aller Kristalle im Phasenfeld
    Returns: Array von (x_center, z_center) für jeden Kristall
    """
    crystal_centers = []
    
    # Finde alle Kristall-IDs (größer als 0.5)
    unique_phases = unique(phase_field[phase_field .> 0.5])
    
    for phase_id in unique_phases
        # Finde alle Punkte für diese Phase
        crystal_indices = findall(x -> abs(x - phase_id) < 0.1, phase_field)
        
        if !isempty(crystal_indices)
            x_coords = [idx[1] for idx in crystal_indices]
            z_coords = [idx[2] for idx in crystal_indices]
            
            center = (mean(x_coords), mean(z_coords))
            push!(crystal_centers, center)
        end
    end
    
    return crystal_centers
end

function find_velocity_minima_multi(vz_field, num_minima=2)
    """
    Findet die N stärksten lokalen Minima in v_z
    """
    # Sortiere alle Punkte nach v_z Wert
    linear_indices = sortperm(vec(vz_field))
    
    minima = []
    min_distance = 20  # Mindestabstand zwischen Minima in Pixeln
    
    for idx in linear_indices
        cart_idx = CartesianIndices(vz_field)[idx]
        pos = (cart_idx[1], cart_idx[2])
        value = vz_field[idx]
        
        # Prüfe ob weit genug von bestehenden Minima entfernt
        too_close = false
        for existing_min in minima
            distance = sqrt((pos[1] - existing_min[1])^2 + (pos[2] - existing_min[2])^2)
            if distance < min_distance
                too_close = true
                break
            end
        end
        
        if !too_close
            push!(minima, (pos[1], pos[2], value))
            
            if length(minima) >= num_minima
                break
            end
        end
    end
    
    return minima
end

function load_velocity_model(model_path)
    """
    Lädt ein gespeichertes Velocity-Modell
    """
    println("Lade Modell: $model_path")
    if !isfile(model_path)
        error("Modelldatei nicht gefunden: $model_path")
    end
    
    model_dict = BSON.load(model_path)
    model_cpu = nothing
    
    for key in [:final_model_cpu, :model_cpu, :model, :trained_model, :final_velocity_model]
        if haskey(model_dict, key)
            model_cpu = model_dict[key]
            println("Modell unter Schlüssel '$key' gefunden")
            break
        end
    end
    
    if model_cpu === nothing
        model_cpu = first(values(model_dict))
        println("Verwende ersten Wert aus BSON-Datei")
    end
    
    return model_cpu
end

function debug_vz_coordinates_two_crystals(model_path; target_size=(256, 256))
    """
    Debug-Funktion für zwei Kristalle
    """
    println("\n=== ZWEI-KRISTALL V_Z KOORDINATEN-DEBUGGING ===")
    
    # 1. Modell laden
    model_cpu = load_velocity_model(model_path)
    
    # 2. LaMEM Ground Truth mit zwei Kristallen generieren
    println("\n1. Generiere Zwei-Kristall LaMEM Ground Truth...")
    params = generate_random_params_two_crystals()
    
    x, z, phase_gt, vx_gt, vz_gt, v_stokes = LaMEM_Multi_crystal_fixed(;params...)
    
    println("   Dimensionen - Phase: $(size(phase_gt)), Vz: $(size(vz_gt))")
    println("   Kristall-Parameter:")
    println("     Positionen: $(params.cen_2D)")
    println("     Radien: $(params.R)")
    println("     Dichtedifferenzen: $(params.Δρ)")
    
    # 3. Kristall-Positionen und Geschwindigkeits-Minima analysieren
    println("\n2. Analysiere Ground Truth...")
    gt_crystal_centers = find_crystal_centers_multi(phase_gt)
    gt_vz_minima = find_velocity_minima_multi(vz_gt, 2)
    
    println("   Gefundene Kristall-Zentren (GT): $gt_crystal_centers")
    println("   Gefundene v_z Minima (GT):")
    for (i, minimum) in enumerate(gt_vz_minima)
        println("     Minimum $i: Position $(minimum[1:2]), Wert: $(round(minimum[3], digits=6))")
    end
    
    # Berechne Alignment-Fehler für jeden Kristall
    gt_alignment_errors = []
    for (i, crystal_center) in enumerate(gt_crystal_centers)
        if i <= length(gt_vz_minima)
            vz_min = gt_vz_minima[i]
            error = sqrt((crystal_center[1] - vz_min[1])^2 + (crystal_center[2] - vz_min[2])^2)
            push!(gt_alignment_errors, error)
            println("   GT Alignment-Fehler Kristall $i: $(round(error, digits=1)) Pixel")
        end
    end
    
    # 4. UNet Vorhersage
    println("\n3. UNet Vorhersage...")
    actual_size = size(phase_gt)
    phase_input = reshape(Float32.(phase_gt), actual_size[1], actual_size[2], 1, 1)
    
    try
        prediction = model_cpu(phase_input)
        vz_pred = prediction[:, :, 2, 1]
        
        # UNet Minima finden
        pred_vz_minima = find_velocity_minima_multi(vz_pred, 2)
        
        println("   UNet v_z Minima:")
        for (i, minimum) in enumerate(pred_vz_minima)
            println("     Minimum $i: Position $(minimum[1:2]), Wert: $(round(minimum[3], digits=6))")
        end
        
        # Alignment-Fehler für UNet
        pred_alignment_errors = []
        for (i, crystal_center) in enumerate(gt_crystal_centers)
            if i <= length(pred_vz_minima)
                vz_min = pred_vz_minima[i]
                error = sqrt((crystal_center[1] - vz_min[1])^2 + (crystal_center[2] - vz_min[2])^2)
                push!(pred_alignment_errors, error)
                println("   UNet Alignment-Fehler Kristall $i: $(round(error, digits=1)) Pixel")
            end
        end
        
        # 5. Visualisierung
        println("\n4. Erstelle Visualisierung...")
        
        
        
        fig = Figure(resolution=(1500, 400))
        
        # Phasenfeld
        ax1 = Axis(fig[1, 1], title="Phasenfeld (2 Kristalle)", xlabel="x", ylabel="z", aspect=DataAspect())
        heatmap!(ax1, phase_gt, colormap=:grays)
        
        # Markiere Kristall-Zentren
        colors = [:red, :blue, :green, :orange]
        for (i, center) in enumerate(gt_crystal_centers)
            scatter!(ax1, [center[2]], [center[1]], color=colors[min(i, length(colors))], markersize=10)
        end
        
        # LaMEM v_z
        ax2 = Axis(fig[1, 2], title="LaMEM: v_z", xlabel="x", ylabel="z", aspect=DataAspect())
        heatmap!(ax2, vz_gt ./ v_stokes, colormap=:RdBu, colorrange=(-3, 1))
        contour!(ax2, phase_gt, levels=[0.5, 1.5, 2.5], color=:black, linewidth=2)
        
        # Markiere GT Minima
        marker_colors = [:yellow, :orange, :cyan, :magenta]
        for (i, minimum) in enumerate(gt_vz_minima)
            scatter!(ax2, [minimum[2]], [minimum[1]], color=marker_colors[min(i, length(marker_colors))], markersize=10)
        end
        
        # UNet v_z
        ax3 = Axis(fig[1, 3], title="UNet: v_z", xlabel="x", ylabel="z", aspect=DataAspect())
        heatmap!(ax3, vz_pred, colormap=:RdBu, colorrange=(-3, 1))
        contour!(ax3, phase_gt, levels=[0.5, 1.5, 2.5], color=:black, linewidth=2)
        
        # Markiere UNet Minima
        for (i, minimum) in enumerate(pred_vz_minima)
            scatter!(ax3, [minimum[2]], [minimum[1]], color=marker_colors[min(i, length(marker_colors))], markersize=10)
        end
        
        # Statistiken
        avg_gt_alignment = length(gt_alignment_errors) > 0 ? mean(gt_alignment_errors) : NaN
        avg_pred_alignment = length(pred_alignment_errors) > 0 ? mean(pred_alignment_errors) : NaN
        
        stats_text = """
        Zwei-Kristall Analyse:
        Durchschn. GT Alignment: $(round(avg_gt_alignment, digits=1)) px
        Durchschn. UNet Alignment: $(round(avg_pred_alignment, digits=1)) px
        Kristalle erkannt: $(length(gt_crystal_centers))/2
        Minima gefunden: $(length(pred_vz_minima))/2
        """
        
        Label(fig[2, 1:3], stats_text, fontsize=12)
        
        save("two_crystal_debug.png", fig)
        display(fig)
        
        # 6. Zusammenfassung
        println("\n" * "="^50)
        println("ZUSAMMENFASSUNG ZWEI-KRISTALL DEBUGGING:")
        if length(gt_crystal_centers) == 2 && length(pred_vz_minima) == 2
            println("Kristall-Erkennung: Erfolgreich")
            if avg_pred_alignment < 30
                println("UNet-Leistung: Gut")
            else
                println("UNet-Leistung: Verbesserungsbedarf")
            end
        else
            println("Kristall-Erkennung: Problematisch")
        end
        println("="^50)
        
        return (
            gt_crystal_centers = gt_crystal_centers,
            gt_vz_minima = gt_vz_minima,
            pred_vz_minima = pred_vz_minima,
            gt_alignment_errors = gt_alignment_errors,
            pred_alignment_errors = pred_alignment_errors,
            params = params,
            phase_gt = phase_gt,
            vz_gt = vz_gt,
            vz_pred = vz_pred
        )
        
    catch e
        println("   UNet Fehler: $e")
        return nothing
    end
end

# ==================== EVALUIERUNGSFUNKTIONEN ====================

function evaluate_two_crystal_model(model, test_samples)
    """
    Evaluiert das Modell auf mehreren Testbeispielen mit zwei Kristallen
    """
    println("\n=== EVALUIERUNG ZWEI-KRISTALL MODELL ===")
    
    total_mse = 0.0
    total_alignment_error = 0.0
    successful_predictions = 0
    
    for (i, sample) in enumerate(test_samples[1:min(10, length(test_samples))])
        try
            # Eingabe vorbereiten
            phase_input = reshape(Float32.(sample.phase), size(sample.phase)..., 1, 1)
            
            # Vorhersage
            prediction = model(phase_input)
            vz_pred = prediction[:, :, 2, 1]
            
            # Ground Truth
            vz_true = sample.vz ./ sample.v_stokes
            
            # MSE berechnen
            mse_val = mean((vz_pred .- vz_true).^2)
            total_mse += mse_val
            
            # Alignment-Fehler berechnen
            crystal_centers = find_crystal_centers_multi(sample.phase)
            pred_minima = find_velocity_minima_multi(vz_pred, length(crystal_centers))
            
            if length(crystal_centers) == length(pred_minima)
                alignment_errors = []
                for (crystal, minimum) in zip(crystal_centers, pred_minima)
                    error = sqrt((crystal[1] - minimum[1])^2 + (crystal[2] - minimum[2])^2)
                    push!(alignment_errors, error)
                end
                total_alignment_error += mean(alignment_errors)
                successful_predictions += 1
            end
            
            if i <= 3
                println("  Sample $i: MSE = $(round(mse_val, digits=6))")
            end
            
        catch e
            println("  Fehler bei Sample $i: $e")
        end
    end
    
    avg_mse = total_mse / length(test_samples[1:min(10, length(test_samples))])
    avg_alignment = successful_predictions > 0 ? total_alignment_error / successful_predictions : NaN
    
    println("\nEvaluierungsergebnisse:")
    println("  Durchschnittlicher MSE: $(round(avg_mse, digits=6))")
    println("  Durchschnittlicher Alignment-Fehler: $(round(avg_alignment, digits=1)) px")
    println("  Erfolgreiche Vorhersagen: $successful_predictions/$(min(10, length(test_samples)))")
    
    return (avg_mse = avg_mse, avg_alignment = avg_alignment, success_rate = successful_predictions)
end

# ==================== VERWENDUNG UND HAUPTPROGRAMM ====================

println("=== WARUM CITYSCAPES FUNKTIONIERT, ZWEI-KRISTALL NICHT ===")
println("Cityscapes-Code:")
println("  ✓ Nutzt nur Flux-native Operationen")
println("  ✓ Keine manuellen Array-Manipulationen") 
println("  ✓ Standard Conv/BatchNorm/MaxPool")
println("  ✓ Flux.mse() statt manuelle Broadcasts")
println("")
println("Zwei-Kristall-Code:")
println("  ✗ Manuelle crop_and_concat Operationen")
println("  ✗ Komplexe Array-Indexing")
println("  ✗ Custom Tensor-Manipulationen")
println("  ✗ Löst GPU Scalar Indexing aus")
println("")
println("LÖSUNG: Flux-Style Training verwenden!")
println("Ausführen: model, losses, data = run_flux_style_training()")
model, losses, data = run_flux_style_training()
