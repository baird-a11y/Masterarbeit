using Flux          # Neuronale Netzwerk-Bibliothek
using Flux: mse, gpu, cpu
using CUDA          # GPU-Unterstützung
using Statistics    # Für statistische Funktionen (mean, etc.)
using FileIO        # Datei-Ein/Ausgabe
using LinearAlgebra # Für mathematische Operationen
using Optimisers    # Optimierungsalgorithmen
using ProgressMeter # Fortschrittsanzeige
using BSON: @save, @load  # Modellspeicherung
using Random        # Zufallszahlengenerator
using LaMEM, GeophysicalModelGenerator  # Für Datengenerierung

# ==================== KONFIGURATION ====================

# Bildgrößen für UNet-Architektur - jetzt konsistent!
STANDARD_HEIGHT = 256   
STANDARD_WIDTH = 256    
NUM_EPOCHS = 100         
LEARNING_RATE = 0.001   
INPUT_CHANNELS = 1      
OUTPUT_CHANNELS = 2     
BATCH_SIZE = 1          # 
CHECKPOINT_DIR = "velocity_checkpoints_256"
DATASET_SIZE = 100      # 

# ==================== HILFSFUNKTIONEN ====================

function clear_gpu_memory()
    GC.gc()
    CUDA.reclaim()
    println("GPU-Speicher freigegeben")
end

# Standardisiert die Größe auf feste Dimensionen
function standardize_size(data::AbstractArray{T,2}) where {T}
    h, w = size(data)
    final = zeros(T, STANDARD_HEIGHT, STANDARD_WIDTH)
    
    h_range = 1:min(h, STANDARD_HEIGHT)
    w_range = 1:min(w, STANDARD_WIDTH)
    final[h_range, w_range] .= view(data, h_range, w_range)
    
    return final
end

function standardize_size_3d(data::AbstractArray{T,3}) where {T}
    h, w, c = size(data)
    final = zeros(T, STANDARD_HEIGHT, STANDARD_WIDTH, c)
    
    h_range = 1:min(h, STANDARD_HEIGHT)
    w_range = 1:min(w, STANDARD_WIDTH)
    final[h_range, w_range, :] .= view(data, h_range, w_range, 1:c)
    
    return final
end

# ==================== LAMEM FUNKTION ====================

function LaMEM_Single_crystal(; nx=64, nz=64, η=1e20, Δρ=200, cen_2D=[(0.0, 0.0)], R=[0.1])
    # Create a model with a single crystal phase
    # nx, nz: number of grid points in x and z direction
    # η: viscosity of matrix in Pa.s
    # Δρ: density contrast in kg/m^3
    # cen_2D: center of sphere in km
    # R: radius of sphere in km

    # Define the model parameters
    η_crystal = 1e4*η       # viscosity of crystal in Pa.s
    ρ_magma   = 2700

    model     = Model(Grid(nel=(nx,nz), x=[-1,1], z=[-1,1]), Time(nstep_max=1), Output(out_strain_rate=1) )
    matrix    = Phase(ID=0,Name="matrix", eta=η,        rho=ρ_magma);
    crystal   = Phase(ID=1,Name="crystal",eta=η_crystal,rho=ρ_magma+Δρ);
    add_phase!(model, crystal, matrix)

    for i =1:length(cen_2D)
        # Add a sphere with the crystal phase
        add_sphere!(model, cen=(cen_2D[i][1], 0.0, cen_2D[i][2]), radius=R[i],  phase=ConstantPhase(1))
    end
   
    # Run LaMEM
    run_lamem(model,1)

    # Read results back into julia
    data, _ = read_LaMEM_timestep(model, 1)

    # Extract the data we need 
    x_vec_1D = data.x.val[:,1,1]
    z_vec_1D = data.z.val[1,1,:]

    phase = data.fields.phase[:,1,:]
    Vx    = data.fields.velocity[1][:,1,:]          # velocity in x [cm/year]
    Vz    = data.fields.velocity[3][:,1,:]          # velocity in z [cm/year]
    Exx   = data.fields.strain_rate[1][:,1,:]       # strainrate in x (=dVx/dx) in [1/s]
    Ezz   = data.fields.strain_rate[9][:,1,:]       # strainrate in z (=dVz/dz) in [1/s]
    rho   = data.fields.density[:,1,:]              # density in kg/m^3    
    log10eta  = data.fields.visc_creep[:,1,:]       # log10(viscosity)

    V_stokes =  2/9*Δρ*9.81*(R[1]*1000)^2/(η)            # Stokes velocity in m/s
    V_stokes_cm_year = V_stokes * 100 * (3600*24*365.25) # convert to cm/year

    return x_vec_1D, z_vec_1D, phase, Vx, Vz, Exx, Ezz, V_stokes_cm_year
end

# ==================== DATENGENERIERUNG MIT LAMEM ====================

function generate_single_sample(sample_id; nx=256, nz=256)  # ERHÖHT auf 256!
    """
    Generiert ein einzelnes Trainingsbeispiel mit zufälligen Parametern
    JETZT MIT KONSISTENTER 256x256 AUFLÖSUNG
    """
    println("Generiere Sample $sample_id (256x256)...")
    
    # Zufällige Parameter für Variation
    η = 10^(rand() * 2 + 19)  # 1e19 bis 1e21 Pa.s
    Δρ = rand(100:500)        # 100-500 kg/m³ Dichtedifferenz
    
    # Zufällige Kristallposition (nicht zu nah am Rand)
    x_pos = rand(-0.6:0.1:0.6)
    z_pos = rand(0.2:0.1:0.8)
    cen_2D = [(x_pos, z_pos)]
    
    # Zufälliger Radius
    R = [rand(0.03:0.01:0.08)]
    
    try
        # LaMEM-Simulation mit 256x256 Auflösung
        x, z, phase, Vx, Vz, Exx, Ezz, V_stokes = LaMEM_Single_crystal(
            nx=nx, nz=nz, η=η, Δρ=Δρ, cen_2D=cen_2D, R=R
        )
        
        println("  Sample $sample_id (256x256) erfolgreich generiert")
        
        return (
            phase=phase, 
            vx=Vx, 
            vz=Vz, 
            v_stokes=V_stokes,
            params=(η=η, Δρ=Δρ, center=cen_2D[1], radius=R[1])
        )
    catch e
        println("Fehler bei Sample $sample_id: $e")
        return nothing
    end
end

function generate_training_dataset(n_samples=500)  # Weniger Samples wegen höherer Auflösung
    """
    Generiert einen kompletten Trainingsdatensatz mit 256x256 Auflösung
    """
    println("Generiere $n_samples Trainingsbeispiele (256x256)...")
    
    # Test mit einem einfachen Beispiel
    println("Teste LaMEM_Single_crystal mit 256x256...")
    try
        test_result = LaMEM_Single_crystal(nx=256, nz=256, η=1e20, Δρ=200, cen_2D=[(0.0, 0.5)], R=[0.05])
        println("  ✓ LaMEM 256x256-Test erfolgreich!")
    catch e
        error("LaMEM 256x256-Test fehlgeschlagen: $e")
    end
    
    dataset = []
    successful_samples = 0
    
    for i in 1:n_samples
        sample = generate_single_sample(i, nx=256, nz=256)  # Explizit 256x256
        
        if sample !== nothing
            push!(dataset, sample)
            successful_samples += 1
            println("  ✓ Sample $i erfolgreich (256x256)")
        else
            println("  ✗ Sample $i fehlgeschlagen")
        end
        
        # Speicher-Management wichtiger bei höherer Auflösung
        if CUDA.functional()
            CUDA.reclaim()
        end
        GC.gc()
    end
    
    println("Datengenerierung abgeschlossen: $successful_samples/$n_samples Samples (256x256)")
    
    if successful_samples == 0
        error("Keine Trainingssamples generiert! Prüfe LaMEM-Setup.")
    end
    
    return dataset
end

# ==================== DATENVORVERARBEITUNG ====================

function preprocess_phase_field(phase_data)
    """
    Bereitet Phasenfeld für UNet vor - KEINE SKALIERUNG mehr nötig!
    """
    # Daten sollten bereits 256x256 sein
    if size(phase_data) != (256, 256)
        println("WARNUNG: Unerwartete Phasenfeldgröße: $(size(phase_data))")
        # Fallback zur Standardisierung
        phase_std = standardize_size(Float32.(phase_data))
    else
        phase_std = Float32.(phase_data)
    end
    
    # Reshape zu (H, W, C, B) für Flux
    return reshape(phase_std, 256, 256, 1, 1)
end

function preprocess_velocity_fields(vx, vz, v_stokes)
    """
    Bereitet Geschwindigkeitsfelder für Training vor - KEINE SKALIERUNG mehr nötig!
    """
    # Mit Stokes-Geschwindigkeit normalisieren
    vx_norm = Float32.(vx ./ v_stokes)
    vz_norm = Float32.(vz ./ v_stokes)
    
    # Daten sollten bereits 256x256 sein
    if size(vx_norm) != (256, 256)
        println("WARNUNG: Unerwartete Geschwindigkeitsfeldgröße: $(size(vx_norm))")
        # Fallback zur Standardisierung
        vx_std = standardize_size(vx_norm)
        vz_std = standardize_size(vz_norm)
    else
        vx_std = vx_norm
        vz_std = vz_norm
    end
    
    # Zu (H, W, 2, 1) kombinieren
    velocity_field = zeros(Float32, 256, 256, 2, 1)
    velocity_field[:, :, 1, 1] .= vx_std
    velocity_field[:, :, 2, 1] .= vz_std
    
    return velocity_field
end

function create_training_batches(raw_dataset)
    """
    Konvertiert Rohdaten zu Trainings-Batches - optimiert für 256x256
    """
    println("Erstelle Trainings-Batches für 256x256...")
    
    processed_data = []
    
    for (i, sample) in enumerate(raw_dataset)
        try
            # Phase und Geschwindigkeiten vorverarbeiten
            phase_input = preprocess_phase_field(sample.phase)
            velocity_target = preprocess_velocity_fields(
                sample.vx, sample.vz, sample.v_stokes
            )
            
            push!(processed_data, (phase_input, velocity_target))
            
            if i % 25 == 0  # Häufigere Updates wegen weniger Samples
                println("  $i/$(length(raw_dataset)) Samples verarbeitet")
                # Zwischenspeicher freigeben
                GC.gc()
            end
        catch e
            println("Fehler bei Verarbeitung von Sample $i: $e")
        end
    end
    
    # Kleinere Batches wegen höherem Speicherverbrauch
    n_batches = length(processed_data)  # Batch-Size = 1
    batched_data = []
    
    for i in 1:n_batches
        phase_batch = processed_data[i][1]
        velocity_batch = processed_data[i][2]
        
        push!(batched_data, (phase_batch, velocity_batch))
    end
    
    println("$(length(batched_data)) Batches mit je 1 Sample (256x256) erstellt")
    return batched_data
end

# ==================== MODELL-DEFINITION ====================

struct VelocityUNet
    # Encoder
    encoder1
    encoder2
    encoder3
    encoder4
    bottleneck
    # Decoder
    decoder4
    decoder4_1
    decoder3
    decoder3_1
    decoder2
    decoder2_1
    decoder1
    decoder1_1
end

Flux.@functor VelocityUNet

function crop_and_concat(x, skip, dims=3)
    x_size = size(x)
    skip_size = size(skip)
    
    height_diff = skip_size[1] - x_size[1]
    width_diff = skip_size[2] - x_size[2]
    
    if height_diff < 0 || width_diff < 0
        padded_skip = zeros(eltype(skip),
                            max(x_size[1], skip_size[1]),
                            max(x_size[2], skip_size[2]),
                            skip_size[3], skip_size[4])
        
        h_start = abs(min(0, height_diff)) ÷ 2 + 1
        w_start = abs(min(0, width_diff)) ÷ 2 + 1
        
        padded_skip[h_start:h_start+skip_size[1]-1,
                    w_start:w_start+skip_size[2]-1, :, :] .= skip
        
        return cat(x, padded_skip, dims=dims)
    else
        h_start = height_diff ÷ 2 + 1
        w_start = width_diff ÷ 2 + 1
        
        cropped_skip = skip[h_start:h_start+x_size[1]-1,
                            w_start:w_start+x_size[2]-1, :, :]
        
        return cat(x, cropped_skip, dims=dims)
    end
end

function create_velocity_unet(input_channels=1, output_channels=2)
    # Encoder
    encoder1 = Chain(
        Conv((3, 3), input_channels => 32, relu, pad=SamePad()),
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
        Conv((1,1), 32 => output_channels)  # KEINE AKTIVIERUNG für Regression!
    )
    
    return VelocityUNet(encoder1, encoder2, encoder3, encoder4, bottleneck,
                        decoder4, decoder4_1, decoder3, decoder3_1,
                        decoder2, decoder2_1, decoder1, decoder1_1)
end

function (model::VelocityUNet)(x)
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

# ==================== TRAINING ====================

function velocity_loss_fn(model, x, y_true)
    y_pred = model(x)
    return mse(y_pred, y_true)
end

function train_velocity_unet(model, train_data, num_epochs, learning_rate)
    mkpath(CHECKPOINT_DIR)
    
    opt_state = Optimisers.setup(Optimisers.Adam(learning_rate), model)
    losses = Float32[]
    
    for epoch in 1:num_epochs
        println("====== Epoche $epoch/$num_epochs ======")
        
        total_loss = 0f0
        batch_count = 0
        
        for (batch_idx, (phase_batch, velocity_batch)) in enumerate(train_data)
            try
                # Auf GPU verschieben
                phase_batch = gpu(phase_batch)
                velocity_batch = gpu(velocity_batch)
                
                # Gradienten berechnen
                ∇model = gradient(m -> velocity_loss_fn(m, phase_batch, velocity_batch), model)[1]
                batch_loss = velocity_loss_fn(model, phase_batch, velocity_batch)
                
                # Parameter aktualisieren
                opt_state, model = Optimisers.update!(opt_state, model, ∇model)
                
                total_loss += batch_loss
                batch_count += 1
                
                println("  Batch $batch_idx: MSE Verlust = $(round(batch_loss, digits=6))")
                
            catch e
                println("  FEHLER bei Batch $batch_idx: $e")
            end
            
            CUDA.reclaim()
        end
        
        avg_loss = batch_count > 0 ? total_loss / batch_count : NaN32
        push!(losses, avg_loss)
        
        println("Epoche $epoch: Durchschnittsverlust = $(round(avg_loss, digits=6))")
        
        # Checkpoint speichern
        if epoch % 10 == 0
            model_cpu = cpu(model)
            @save joinpath(CHECKPOINT_DIR, "velocity_checkpoint_epoch$(epoch).bson") model_cpu
            println("  Checkpoint für Epoche $epoch gespeichert")
        end
        
        clear_gpu_memory()
    end
    
    return model, losses
end

# ==================== EVALUIERUNG ====================

function predict_velocity(model, phase_input)
    """
    Vorhersage von Geschwindigkeitsfeldern
    """
    phase_gpu = gpu(phase_input)
    velocity_pred = cpu(model(phase_gpu))
    
    # Zurück zu separaten v_x und v_z Arrays
    vx_pred = velocity_pred[:, :, 1, 1]
    vz_pred = velocity_pred[:, :, 2, 1]
    
    return vx_pred, vz_pred
end

function evaluate_model(model, test_sample)
    """
    Evaluiert das Modell auf einem Testbeispiel
    """
    # Eingabe vorbereiten
    phase_input = preprocess_phase_field(test_sample.phase)
    
    # Vorhersage
    vx_pred, vz_pred = predict_velocity(model, phase_input)
    
    # Ground Truth normalisieren
    vx_true = test_sample.vx ./ test_sample.v_stokes
    vz_true = test_sample.vz ./ test_sample.v_stokes
    
    # Ground Truth auf gleiche Größe bringen
    vx_true_std = standardize_size(Float32.(vx_true))
    vz_true_std = standardize_size(Float32.(vz_true))
    
    # MSE berechnen
    mse_vx = mean((vx_pred .- vx_true_std).^2)
    mse_vz = mean((vz_pred .- vz_true_std).^2)
    mse_total = (mse_vx + mse_vz) / 2
    
    println("Evaluierung:")
    println("  MSE v_x: $(round(mse_vx, digits=6))")
    println("  MSE v_z: $(round(mse_vz, digits=6))")
    println("  MSE gesamt: $(round(mse_total, digits=6))")
    
    return (
        vx_pred=vx_pred, vz_pred=vz_pred,
        vx_true=vx_true_std, vz_true=vz_true_std,
        mse_vx=mse_vx, mse_vz=mse_vz, mse_total=mse_total
    )
end

# ==================== HAUPTPROGRAMM ====================

println("===== GESCHWINDIGKEITSFELD-VORHERSAGE MIT UNET =====")

# 1. Trainingsdaten generieren
println("\n1. Generiere Trainingsdaten...")
raw_dataset = generate_training_dataset(DATASET_SIZE)

if length(raw_dataset) < 10
    println("FEHLER: Zu wenige Trainingssamples generiert!")
    exit()
end

# 2. Daten vorverarbeiten
println("\n2. Verarbeite Daten für Training...")
train_batches = create_training_batches(raw_dataset)

# 3. Modell erstellen
println("\n3. Erstelle UNet-Modell...")
model = create_velocity_unet(INPUT_CHANNELS, OUTPUT_CHANNELS)
model = gpu(model)

# 4. Training
println("\n4. Starte Training...")
trained_model, training_losses = train_velocity_unet(
    model, train_batches, NUM_EPOCHS, LEARNING_RATE
)

# 5. Finales Modell speichern
final_model_cpu = cpu(trained_model)
@save joinpath(CHECKPOINT_DIR, "final_velocity_model.bson") final_model_cpu
println("\nFinales Modell gespeichert!")

# 6. Evaluierung auf Testbeispiel
println("\n5. Evaluiere Modell...")
if length(raw_dataset) > 0
    test_idx = rand(1:length(raw_dataset))
    test_result = evaluate_model(trained_model, raw_dataset[test_idx])
    
    println("\nTraining abgeschlossen!")
    println("Finaler Trainingsverlust: $(round(training_losses[end], digits=6))")
    println("Test-MSE: $(round(test_result.mse_total, digits=6))")
end