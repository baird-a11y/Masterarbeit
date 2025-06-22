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

println("=== GPU-NATIVE MULTI-KRISTALL UNET ===")

# ==================== KONFIGURATION ====================

const CONFIG = (
    # Modell-Parameter
    image_size = (128, 128),
    input_channels = 4,
    output_channels = 2,
    
    # Multi-Kristall Parameter
    max_crystals = 4,
    min_crystals = 1,
    collision_buffer = 0.03,
    
    # Training-Parameter
    learning_rate = 0.001,
    num_epochs = 60,
    batch_size = 2,
    dataset_size = 200,
    
    # LaMEM-Parameter
    eta_range = (1e19, 1e21),
    delta_rho_range = (150, 400),
    position_range_x = (-0.6, 0.6),
    position_range_z = (0.3, 0.7),
    radius_range = (0.04, 0.07),
    
    # System-Parameter
    checkpoint_dir = "multi_crystal_gpu_native",
    save_every_n_epochs = 5,
    use_gpu = true,
    verbose = true
)

# ==================== GPU-NATIVE UNET ====================

struct GPUNativeUNet
    encoder_conv1; encoder_pool1
    encoder_conv2; encoder_pool2
    encoder_conv3; encoder_pool3
    bottleneck
    decoder_up3; decoder_conv3
    decoder_up2; decoder_conv2
    decoder_up1; decoder_conv1
    final_conv
end

Flux.@functor GPUNativeUNet

function (model::GPUNativeUNet)(x)
    # Encoder - komplett ohne Scalar-Indexing
    e1 = model.encoder_conv1(x)
    e1_pool = model.encoder_pool1(e1)
    
    e2 = model.encoder_conv2(e1_pool)
    e2_pool = model.encoder_pool2(e2)
    
    e3 = model.encoder_conv3(e2_pool)
    e3_pool = model.encoder_pool3(e3)
    
    # Bottleneck
    b = model.bottleneck(e3_pool)
    
    # Decoder - ohne Skip-Connections für GPU-Kompatibilität
    d3 = model.decoder_up3(b)
    d3 = model.decoder_conv3(d3)
    
    d2 = model.decoder_up2(d3)
    d2 = model.decoder_conv2(d2)
    
    d1 = model.decoder_up1(d2)
    d1 = model.decoder_conv1(d1)
    
    # Final output
    output = model.final_conv(d1)
    
    return output
end

function create_gpu_native_unet()
    input_ch = CONFIG.input_channels
    output_ch = CONFIG.output_channels
    
    # Encoder
    encoder_conv1 = Chain(
        Conv((3, 3), input_ch => 32, relu, pad=SamePad()),
        BatchNorm(32),
        Conv((3, 3), 32 => 32, relu, pad=SamePad()),
        BatchNorm(32)
    )
    encoder_pool1 = MaxPool((2, 2))
    
    encoder_conv2 = Chain(
        Conv((3, 3), 32 => 64, relu, pad=SamePad()),
        BatchNorm(64),
        Conv((3, 3), 64 => 64, relu, pad=SamePad()),
        BatchNorm(64)
    )
    encoder_pool2 = MaxPool((2, 2))
    
    encoder_conv3 = Chain(
        Conv((3, 3), 64 => 128, relu, pad=SamePad()),
        BatchNorm(128),
        Conv((3, 3), 128 => 128, relu, pad=SamePad()),
        BatchNorm(128)
    )
    encoder_pool3 = MaxPool((2, 2))
    
    # Bottleneck
    bottleneck = Chain(
        Conv((3, 3), 128 => 256, relu, pad=SamePad()),
        BatchNorm(256),
        Dropout(0.5),
        Conv((3, 3), 256 => 256, relu, pad=SamePad()),
        BatchNorm(256)
    )
    
    # Decoder
    decoder_up3 = ConvTranspose((2, 2), 256 => 128, stride=2)
    decoder_conv3 = Chain(
        Conv((3, 3), 128 => 128, relu, pad=SamePad()),
        BatchNorm(128),
        Conv((3, 3), 128 => 128, relu, pad=SamePad()),
        BatchNorm(128)
    )
    
    decoder_up2 = ConvTranspose((2, 2), 128 => 64, stride=2)
    decoder_conv2 = Chain(
        Conv((3, 3), 64 => 64, relu, pad=SamePad()),
        BatchNorm(64),
        Conv((3, 3), 64 => 64, relu, pad=SamePad()),
        BatchNorm(64)
    )
    
    decoder_up1 = ConvTranspose((2, 2), 64 => 32, stride=2)
    decoder_conv1 = Chain(
        Conv((3, 3), 32 => 32, relu, pad=SamePad()),
        BatchNorm(32),
        Conv((3, 3), 32 => 32, relu, pad=SamePad()),
        BatchNorm(32)
    )
    
    final_conv = Conv((1, 1), 32 => output_ch)
    
    return GPUNativeUNet(
        encoder_conv1, encoder_pool1,
        encoder_conv2, encoder_pool2,
        encoder_conv3, encoder_pool3,
        bottleneck,
        decoder_up3, decoder_conv3,
        decoder_up2, decoder_conv2,
        decoder_up1, decoder_conv1,
        final_conv
    )
end

# ==================== LAMEM FUNKTIONEN ====================

function LaMEM_Multi_crystal(params)
    η_crystal = 1e4 * params.η
    ρ_magma = 2700
    
    target_h, target_w = CONFIG.image_size
    nel_h, nel_w = target_h - 1, target_w - 1
    
    model = Model(
        Grid(nel=(nel_h, nel_w), x=[-1,1], z=[-1,1]), 
        Time(nstep_max=1), 
        Output(out_strain_rate=1)
    )
    
    matrix = Phase(ID=0, Name="matrix", eta=params.η, rho=ρ_magma)
    add_phase!(model, matrix)
    
    for i in 1:params.num_crystals
        crystal = Phase(ID=i, Name="crystal_$i", eta=η_crystal, rho=ρ_magma+params.Δρ)
        add_phase!(model, crystal)
    end
    
    for i in 1:params.num_crystals
        center = params.centers[i]
        radius = params.radii[i]
        add_sphere!(model, cen=(center[1], 0.0, center[2]), 
                   radius=radius, phase=ConstantPhase(i))
    end
   
    run_lamem(model, 1)
    data, _ = read_LaMEM_timestep(model, 1)

    x_vec_1D = data.x.val[:,1,1]
    z_vec_1D = data.z.val[1,1,:]
    phase = data.fields.phase[:,1,:]
    Vx = data.fields.velocity[1][:,1,:]
    Vz = data.fields.velocity[3][:,1,:]
    
    max_radius = maximum(params.radii)
    V_stokes = 2/9*params.Δρ*9.81*(max_radius*1000)^2/(params.η)
    V_stokes_cm_year = V_stokes * 100 * (3600*24*365.25)
    
    actual_size = size(phase)
    if actual_size != CONFIG.image_size
        phase = phase[1:target_h, 1:target_w]
        Vx = Vx[1:target_h, 1:target_w]
        Vz = Vz[1:target_h, 1:target_w]
    end

    return x_vec_1D, z_vec_1D, phase, Vx, Vz, V_stokes_cm_year, params
end

# ==================== DATENGENERIERUNG ====================

function check_collision(new_center, new_radius, existing_crystals)
    for (center, radius) in existing_crystals
        distance = sqrt((new_center[1] - center[1])^2 + (new_center[2] - center[2])^2)
        min_distance = new_radius + radius + CONFIG.collision_buffer
        if distance < min_distance
            return true
        end
    end
    return false
end

function generate_multi_crystal_params()
    num_crystals = rand(CONFIG.min_crystals:CONFIG.max_crystals)
    
    η_min, η_max = CONFIG.eta_range
    η = 10^(rand() * (log10(η_max) - log10(η_min)) + log10(η_min))
    Δρ = rand(CONFIG.delta_rho_range[1]:CONFIG.delta_rho_range[2])
    
    centers = Tuple{Float64, Float64}[]
    radii = Float64[]
    existing_crystals = Tuple{Tuple{Float64, Float64}, Float64}[]
    
    for i in 1:num_crystals
        placed = false
        attempts = 0
        
        while !placed && attempts < 30
            x_min, x_max = CONFIG.position_range_x
            z_min, z_max = CONFIG.position_range_z
            r_min, r_max = CONFIG.radius_range
            
            x_pos = rand() * (x_max - x_min) + x_min
            z_pos = rand() * (z_max - z_min) + z_min
            radius = rand() * (r_max - r_min) + r_min
            
            new_center = (x_pos, z_pos)
            
            if !check_collision(new_center, radius, existing_crystals)
                push!(centers, new_center)
                push!(radii, radius)
                push!(existing_crystals, (new_center, radius))
                placed = true
            end
            attempts += 1
        end
        
        if !placed
            break
        end
    end
    
    if length(centers) == 0
        push!(centers, (0.0, 0.5))
        push!(radii, 0.05)
    end
    
    return (
        η = η,
        Δρ = Δρ,
        centers = centers,
        radii = radii,
        num_crystals = length(centers)
    )
end

function encode_multi_channel_phase(phase_field, max_crystals=CONFIG.max_crystals)
    h, w = size(phase_field)
    encoded = zeros(Float32, h, w, max_crystals + 1)
    
    encoded[:, :, 1] = Float32.(phase_field .≈ 0)
    
    for crystal_id in 1:max_crystals
        crystal_mask = Float32.(abs.(phase_field .- crystal_id) .< 0.5)
        encoded[:, :, crystal_id + 1] = crystal_mask
    end
    
    for i in 1:h, j in 1:w
        channel_sum = sum(encoded[i, j, :])
        if channel_sum > 0
            encoded[i, j, :] ./= channel_sum
        else
            encoded[i, j, 1] = 1.0f0
        end
    end
    
    return encoded
end

function generate_multi_crystal_sample(sample_id)
    if CONFIG.verbose && sample_id % 20 == 0
        println("Sample $sample_id/$(CONFIG.dataset_size)")
    end
    
    params = generate_multi_crystal_params()
    
    try
        x, z, phase, Vx, Vz, V_stokes, final_params = LaMEM_Multi_crystal(params)
        
        if size(phase) != CONFIG.image_size
            error("Falsche Dimensionen: $(size(phase))")
        end
        
        return (
            phase = phase, 
            vx = Vx, 
            vz = Vz, 
            v_stokes = V_stokes,
            params = final_params
        )
    catch e
        return nothing
    end
end

function generate_dataset()
    println("Generiere Dataset...")
    
    # Test
    test_params = generate_multi_crystal_params()
    try
        x, z, phase, vx, vz, v_stokes, final_params = LaMEM_Multi_crystal(test_params)
        println("LaMEM-Test erfolgreich: $(size(phase))")
    catch e
        error("LaMEM-Test fehlgeschlagen: $e")
    end
    
    dataset = []
    progress = Progress(CONFIG.dataset_size)
    
    for i in 1:CONFIG.dataset_size
        sample = generate_multi_crystal_sample(i)
        if sample !== nothing
            push!(dataset, sample)
        end
        next!(progress)
    end
    
    println("Dataset erstellt: $(length(dataset)) Samples")
    return dataset
end

# ==================== GPU-BATCH ERSTELLUNG ====================

function preprocess_sample(sample_data)
    h, w = CONFIG.image_size
    
    phase_encoded = encode_multi_channel_phase(sample_data.phase)
    phase_input = reshape(phase_encoded, h, w, CONFIG.input_channels, 1)
    
    vx_norm = Float32.(sample_data.vx ./ sample_data.v_stokes)
    vz_norm = Float32.(sample_data.vz ./ sample_data.v_stokes)
    
    velocity_target = zeros(Float32, h, w, 2, 1)
    velocity_target[:, :, 1, 1] .= vx_norm
    velocity_target[:, :, 2, 1] .= vz_norm
    
    return (Float32.(phase_input), Float32.(velocity_target))
end

function create_gpu_batches(raw_dataset)
    println("Erstelle GPU-Batches...")
    
    processed_data = []
    
    for (i, sample) in enumerate(raw_dataset)
        try
            phase_input, velocity_target = preprocess_sample(sample)
            
            if any(isnan.(phase_input)) || any(isnan.(velocity_target))
                continue
            end
            
            push!(processed_data, (phase_input, velocity_target))
        catch e
            continue
        end
    end
    
    # Echte Batches erstellen
    batched_data = []
    batch_size = CONFIG.batch_size
    
    for i in 1:batch_size:length(processed_data)
        batch_end = min(i + batch_size - 1, length(processed_data))
        batch_indices = i:batch_end
        
        if length(batch_indices) >= 1
            try
                # Batch zusammenfügen
                phase_batch = cat([processed_data[j][1] for j in batch_indices]...; dims=4)
                velocity_batch = cat([processed_data[j][2] for j in batch_indices]...; dims=4)
                
                push!(batched_data, (phase_batch, velocity_batch))
            catch e
                # Bei Fehler einzeln hinzufügen
                for j in batch_indices
                    push!(batched_data, processed_data[j])
                end
            end
        end
    end
    
    println("$(length(batched_data)) GPU-Batches erstellt")
    return batched_data
end

# ==================== GPU-NATIVES TRAINING ====================

function train_gpu_native(model, train_data)
    println("Starte GPU-natives Training...")
    println("  GPU verfügbar: $(CUDA.functional())")
    println("  Epochen: $(CONFIG.num_epochs)")
    println("  Batches: $(length(train_data))")
    
    if !CUDA.functional()
        error("GPU nicht verfügbar!")
    end
    
    mkpath(CONFIG.checkpoint_dir)
    
    # Modell auf GPU
    model = gpu(model)
    
    # Optimizer
    opt = Optimisers.Adam(CONFIG.learning_rate)
    opt_state = Optimisers.setup(opt, model)
    losses = Float32[]
    
    for epoch in 1:CONFIG.num_epochs
        println("Epoche $epoch/$(CONFIG.num_epochs)")
        epoch_losses = Float32[]
        
        for (i, (x, y)) in enumerate(train_data)
            try
                # Daten auf GPU
                x_gpu = gpu(x)
                y_gpu = gpu(y)
                
                # Loss-Funktion ohne Scalar-Indexing
                loss_fn = function(m)
                    pred = m(x_gpu)
                    return mse(pred, y_gpu)
                end
                
                # Gradienten mit Flux.withgradient (GPU-sicher)
                result = Flux.withgradient(loss_fn, model)
                loss_val = result.val
                grads = result.grad[1]
                
                # Parameter update
                opt_state, model = Optimisers.update!(opt_state, model, grads)
                
                # Loss auf CPU übertragen für Tracking
                push!(epoch_losses, Float32(loss_val))
                
                if i % 5 == 0 && CONFIG.verbose
                    println("  Batch $i: $(round(Float32(loss_val), digits=4))")
                end
                
            catch e
                println("Fehler bei Batch $i: $e")
                continue
            end
            
            # GPU Memory Management
            if i % 10 == 0
                CUDA.reclaim()
            end
        end
        
        if length(epoch_losses) > 0
            avg_loss = mean(epoch_losses)
            push!(losses, avg_loss)
            println("Durchschnittsloss: $(round(avg_loss, digits=4))")
        else
            push!(losses, NaN32)
        end
        
        # Checkpoint
        if epoch % CONFIG.save_every_n_epochs == 0
            model_cpu = cpu(model)
            checkpoint_path = joinpath(CONFIG.checkpoint_dir, "gpu_checkpoint_epoch_$(epoch).bson")
            @save checkpoint_path model_cpu
            println("GPU-Checkpoint gespeichert")
            
            # Modell zurück auf GPU
            model = gpu(model_cpu)
        end
        
        # GPU cleanup
        CUDA.reclaim()
        GC.gc()
    end
    
    # Finales Modell
    final_model_cpu = cpu(model)
    final_path = joinpath(CONFIG.checkpoint_dir, "final_gpu_model.bson")
    @save final_path final_model_cpu
    println("Finales GPU-Modell gespeichert: $final_path")
    
    return model, losses
end

# ==================== EVALUIERUNG ====================

function evaluate_gpu_model(model, test_sample)
    println("Evaluiere GPU-Modell...")
    
    phase_input, velocity_target = preprocess_sample(test_sample)
    
    # GPU-Vorhersage
    model_gpu = gpu(model)
    phase_gpu = gpu(phase_input)
    
    velocity_pred_gpu = model_gpu(phase_gpu)
    velocity_pred = cpu(velocity_pred_gpu)
    
    vx_pred = velocity_pred[:, :, 1, 1]
    vz_pred = velocity_pred[:, :, 2, 1]
    
    vx_true = velocity_target[:, :, 1, 1]
    vz_true = velocity_target[:, :, 2, 1]
    
    mse_vx = mean((vx_pred .- vx_true).^2)
    mse_vz = mean((vz_pred .- vz_true).^2)
    mse_total = (mse_vx + mse_vz) / 2
    
    num_crystals = test_sample.params.num_crystals
    
    println("GPU-Evaluierung:")
    println("  Kristalle: $num_crystals")
    println("  MSE v_x: $(round(mse_vx, digits=6))")
    println("  MSE v_z: $(round(mse_vz, digits=6))")
    println("  MSE total: $(round(mse_total, digits=6))")
    
    return mse_total
end

# ==================== HAUPTFUNKTION ====================

function run_gpu_training()
    println("="^60)
    println("GPU-NATIVES MULTI-KRISTALL TRAINING")
    println("="^60)
    
    if !CUDA.functional()
        error("GPU nicht verfügbar! CUDA muss funktionieren.")
    end
    
    
    
    # 1. Dataset
    dataset = generate_dataset()
    
    # 2. GPU-Batches
    batches = create_gpu_batches(dataset)
    
    # 3. GPU-Modell
    model = create_gpu_native_unet()
    
    # 4. GPU-Training
    trained_model, losses = train_gpu_native(model, batches)
    
    # 5. GPU-Test
    if length(dataset) > 0
        test_sample = dataset[1]
        test_mse = evaluate_gpu_model(cpu(trained_model), test_sample)
        
        println("\nGPU-Training abgeschlossen!")
        println("Test MSE: $(round(test_mse, digits=6))")
        
    end
    
    return trained_model, losses, dataset
end

# ==================== AUSFÜHRUNG ====================

println("GPU-native Multi-Kristall UNet geladen.")
println("GPU verfügbar: $(CUDA.functional())")

if CUDA.functional()
    
    println("Starte Training mit: model, losses, data = run_gpu_training()")
    
    # Auto-Start
    model, losses, data = run_gpu_training()
else
    println("FEHLER: GPU nicht verfügbar!")
end