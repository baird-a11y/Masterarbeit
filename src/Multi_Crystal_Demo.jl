using LaMEM, GeophysicalModelGenerator, GLMakie
using CUDA
using Flux  # Für gpu() Funktion
using Statistics
using Random
using StatsBase  # Für sample() Funktion
using Printf  # Für @sprintf
# =============================================================================
# SINKING SPHERE MODEL MIT MULTIPLE CRYSTALS
# =============================================================================

"""Erstellt ein LaMEM-Modell mit mehreren Kristallphasen
    - resolution: Zielauflösung in (nx, nz)
    - n_crystals: Anzahl Kristalle
    - radius_crystal: Array mit Radien der Kristalle in km
    - η_magma: Viskosität des Magmas in Pa.s
    - ρ_magma: Dichte des Magmas in kg/m³
    - Δρ: Dichteunterschied zwischen Kristall und Magma in kg/m³
    - domain_size: Physikalische Dimensionen (x, z)
    - cen_2D: Zentren der Kristalle in km
    - max_attempts: Max. Versuche für Kristallplatzierung
    - collision_threshold: Minimaler Abstand zwischen Kristallen
    """


function LaMEM_Multi_crystal(;
    resolution = (64, 64),              # Zielauflösung in (nx, nz) - konkrete Standardwerte
    n_crystals = 1,                     # Anzahl Kristalle - Integer, nicht Variable
    radius_crystal = [0.05],            # Radius der Kristalle in km - Array für mehrere Kristalle
    η_magma = 1e20,                     # Viskosität des Magmas in Pa.s
    ρ_magma = 2700,                     # Dichte des Magmas in kg/m³
    Δρ = 200,                           # Dichteunterschied zwischen Kristall und Magma in kg/m³
    domain_size = (-1.0, 1.0),          # Physikalische Dimensionen (x, z)
    cen_2D = [(0.0, 0.5)],              # Zentren der Kristalle in km - Standardposition
    max_attempts = 100,                 # Max. Versuche für Kristallplatzierung
    collision_threshold = 0.3          # Minimaler Abstand zwischen Kristallen
)
    # Berechne abgeleitete Parameter
    η_crystal = 1e4 * η_magma           # Viskosität des Kristalls
    ρ_crystal = ρ_magma + Δρ            # Dichte des Kristalls
    
    # Create a model with multiple crystal phases
    model = Model(
        Grid(nel=(resolution[1]-1, resolution[2]-1), x=[-1,1], z=[-1,1]), 
        Time(nstep_max=1), 
        Output(out_strain_rate=1)
    )
    
    # Define the Matrix
    matrix = Phase(ID=0, Name="matrix", eta=η_magma, rho=ρ_magma)
    # Define the Crystal Phase
    crystal = Phase(ID=1, Name="crystal", eta=η_crystal, rho=ρ_crystal)
    add_phase!(model, crystal, matrix)

    # Fügt die Kristalle zum Model hinzu
    for i = 1:n_crystals
        # Bestimme Radius für diesen Kristall
        current_radius = length(radius_crystal) >= i ? radius_crystal[i] : radius_crystal[1]
        
        # Bestimme Position für diesen Kristall
        if length(cen_2D) >= i
            current_center = cen_2D[i]
        else
            # Generiere zufällige Position wenn nicht genug Zentren angegeben
            x_pos = rand(-0.6:0.1:0.6)
            z_pos = rand(0.2:0.1:0.8)
            current_center = (x_pos, z_pos)
        end
        
        # Add a sphere with the crystal phase
        add_sphere!(model, 
            cen=(current_center[1], 0.0, current_center[2]), 
            radius=current_radius, 
            phase=ConstantPhase(1)
        )
    end

    # LaMEM ausführen
    run_lamem(model, 1)

    # Read results back into julia
    data, _ = read_LaMEM_timestep(model, 1)

    # Extract the data we need 
    x_vec_1D = data.x.val[:,1,1]
    z_vec_1D = data.z.val[1,1,:]
    phase = data.fields.phase[:,1,:]
    Vx = data.fields.velocity[1][:,1,:]          # velocity in x [cm/year]
    Vz = data.fields.velocity[3][:,1,:]          # velocity in z [cm/year]
    Exx = data.fields.strain_rate[1][:,1,:]      # strainrate in x (=dVx/dx) in [1/s]
    Ezz = data.fields.strain_rate[9][:,1,:]      # strainrate in z (=dVz/dz) in [1/s]
    rho = data.fields.density[:,1,:]             # density in kg/m³    
    log10eta = data.fields.visc_creep[:,1,:]     # log10(viscosity)

    # Stokes velocity calculation - verwende ersten Kristall als Referenz
    ref_radius = radius_crystal[1]
    V_stokes = 2/9 * Δρ * 9.81 * (ref_radius * 1000)^2 / η_magma  # Stokes velocity in m/s
    V_stokes_cm_year = V_stokes * 100 * (3600*24*365.25)      # convert to cm/year
        
    return x_vec_1D, z_vec_1D, phase, Vx, Vz, Exx, Ezz, V_stokes_cm_year
end

function downsample_avg(data, factor)
    """
    Downsampling durch Mittelwertbildung über factor×factor Blöcke
    Beispiel: 256×256 mit factor=2 → 128×128
    """
    h, w = size(data)
    
    # Validierung
    if h % factor != 0 || w % factor != 0
        error("Dimensionen ($h, $w) nicht durch factor=$factor teilbar!")
    end
    
    new_h, new_w = h ÷ factor, w ÷ factor
    
    # Reshape für Blockweise-Verarbeitung und Mittelwert
    reshaped = reshape(data, factor, new_h, factor, new_w)
    averaged = mean(reshaped, dims=(1, 3))
    return dropdims(averaged, dims=(1, 3))
end

function upsample_nearest(data, factor)
    """
    Upsampling durch Pixel-Wiederholung
    Beispiel: 64×64 mit factor=2 → 128×128
    """
    return repeat(data, inner=(factor, factor))
end

function detect_power_of_2(size_val)
    """
    Prüft ob eine Zahl eine Potenz von 2 ist und im gültigen Bereich liegt
    """
    if size_val & (size_val - 1) != 0
        error("Auflösung $size_val ist keine Potenz von 2!")
    end
    
    if size_val < 64 || size_val > 512
        error("Auflösung $size_val außerhalb erlaubtem Bereich [64, 512]!")
    end
    
    return true
end

function resize_power_of_2(data, target_size)
    """
    Automatische Größenanpassung für Potenzen von 2
    target_size kann (128, 128) oder 128 sein
    """
    current_size = size(data, 1)
    
    # Handle verschiedene Input-Formate
    if isa(target_size, Tuple)
        target_size_val = target_size[1]
        # Validiere dass target_size quadratisch ist
        if target_size[1] != target_size[2]
            error("Nur quadratische Zielgrößen unterstützt: $target_size")
        end
    else
        target_size_val = target_size
    end
    
    # Validiere dass beide Größen Potenzen von 2 sind
    detect_power_of_2(current_size)
    detect_power_of_2(target_size_val)
    
    # Keine Änderung nötig
    if current_size == target_size_val
        return copy(data)  # Explizite Kopie für Sicherheit
    
    # Downsampling
    elseif current_size > target_size_val
        factor = current_size ÷ target_size_val
        if current_size % target_size_val != 0
            error("Downsampling-Factor $(current_size)÷$(target_size_val) ist nicht ganzzahlig!")
        end
        return downsample_avg(data, factor)
    
    # Upsampling  
    else
        factor = target_size_val ÷ current_size
        if target_size_val % current_size != 0
            error("Upsampling-Factor $(target_size_val)÷$(current_size) ist nicht ganzzahlig!")
        end
        return upsample_nearest(data, factor)
    end
end

# =============================================================================

function detect_resolution(data)
    size_val = size(data, 1)
    
    # Validierung: Ist es eine Potenz von 2?
    if size_val & (size_val - 1) != 0
        error("Auflösung $size_val ist keine Potenz von 2!")
    end
    
    # Validierung: Ist es im erlaubten Bereich?
    if size_val < 64 || size_val > 512
        error("Auflösung $size_val außerhalb erlaubtem Bereich [64, 512]!")
    end
    
    return size_val
end



# =============================================================================
# KONFIGURATION UND HILFSFUNKTIONEN
# =============================================================================

# Batch-Größen können auflösungsabhängig sein:
const BATCH_SIZES = Dict(
    64 => 16,    # Kleine Bilder → große Batches
    128 => 8,    # Mittlere Bilder → mittlere Batches  
    256 => 4,    # Große Bilder → kleine Batches
    512 => 1     # Sehr große Bilder → einzeln
)

function get_gpu_memory_info()
    """
    Gibt aktuelle GPU-Speicher-Information zurück
    """
    if !CUDA.functional()
        return (total=0, available=0, used=0, percent_used=0.0)
    end
    
    total = CUDA.total_memory()
    available = CUDA.available_memory()
    used = total - available
    percent_used = (used / total) * 100
    
    return (
        total = total,
        available = available,
        used = used,
        percent_used = percent_used
    )
end

function estimate_tensor_memory(resolution, batch_size, channels=3)
    """
    Schätzt Speicherbedarf für Tensoren in Bytes
    """
    # Input: phase (H, W, 1, B)
    # Output: velocity (H, W, 2, B)
    # Zusätzlich: Gradienten, Activations (Faktor ~4)
    
    elements_input = resolution * resolution * 1 * batch_size
    elements_output = resolution * resolution * 2 * batch_size
    total_elements = (elements_input + elements_output) * 4  # Faktor für Training
    
    bytes_needed = total_elements * sizeof(Float32)
    return bytes_needed
end

function safe_batch_size(resolution, max_memory_percent=80)
    """
    Berechnet sichere Batch-Größe basierend auf verfügbarem GPU-Speicher
    """
    if !CUDA.functional()
        return get(BATCH_SIZES, resolution, 1)
    end
    
    mem_info = get_gpu_memory_info()
    available_bytes = mem_info.available * (max_memory_percent / 100)
    
    # Binäre Suche für maximale Batch-Größe
    max_theoretical = get(BATCH_SIZES, resolution, 1) * 2
    safe_size = 1
    
    for test_size in 1:max_theoretical
        needed_bytes = estimate_tensor_memory(resolution, test_size)
        if needed_bytes <= available_bytes
            safe_size = test_size
        else
            break
        end
    end
    
    # Sicherheitspuffer: 75% der berechneten Größe
    return max(1, Int(floor(safe_size * 0.75)))
end

# =============================================================================
# ADAPTIVE BATCH-ERSTELLUNG
# =============================================================================

function create_adaptive_batch(samples, target_resolution; max_gpu_memory_percent=80, verbose=false)
    """
    Bestimme optimale Batch-Größe für aktuellen GPU-Zustand
    Erstelle Batch ohne GPU-OOM
    """
    if isempty(samples)
        error("Keine Samples für Batch-Erstellung!")
    end
    
    # 1. Bestimme optimale Batch-Größe
    theoretical_batch_size = get(BATCH_SIZES, target_resolution, 1)
    safe_batch_size_val = safe_batch_size(target_resolution, max_gpu_memory_percent)
    actual_batch_size = min(theoretical_batch_size, safe_batch_size_val, length(samples))
    
    if verbose
        mem_info = get_gpu_memory_info()
        println("=== ADAPTIVE BATCH CREATION ===")
        println("Target resolution: $(target_resolution)x$(target_resolution)")
        println("Available samples: $(length(samples))")
        println("Theoretical batch size: $theoretical_batch_size")
        println("Safe batch size: $safe_batch_size_val")
        println("Actual batch size: $actual_batch_size")
        println("GPU memory: $(round(mem_info.percent_used, digits=1))% used")
    end
    
    # 2. Wähle zufällige Samples
    selected_indices = StatsBase.sample(1:length(samples), actual_batch_size, replace=false)
    selected_samples = samples[selected_indices]
    
    # 3. Preprocessing
    batch_phases = []
    batch_velocities = []
    successful_samples = 0
    
    for (i, sample) in enumerate(selected_samples)
        try
            # Sample entpacken (verschiedene Formate handhaben)
            if length(sample) == 8
                x, z, phase, vx, vz, exx, ezz, v_stokes = sample
            elseif length(sample) == 7
                x, z, phase, vx, vz, exx, v_stokes = sample
            else
                error("Unbekanntes Sample-Format: $(length(sample)) Elemente")
            end
            
            # Preprocessing
            phase_tensor, velocity_tensor = preprocess_lamem_sample(
                x, z, phase, vx, vz, v_stokes,
                target_resolution=target_resolution
            )
            
            push!(batch_phases, phase_tensor)
            push!(batch_velocities, velocity_tensor)
            successful_samples += 1
            
        catch e
            if verbose
                println("  Warnung: Sample $i fehlgeschlagen: $e")
            end
            continue
        end
    end
    
    if successful_samples == 0
        error("Keine Samples konnten verarbeitet werden!")
    end
    
    # 4. Tensoren zusammenfügen
    try
        phase_batch = cat(batch_phases..., dims=4)
        velocity_batch = cat(batch_velocities..., dims=4)
        
        if verbose
            println("Batch erstellt: $(size(phase_batch)), $(size(velocity_batch))")
            println("Erfolgreiche Samples: $successful_samples/$actual_batch_size")
        end
        
        return phase_batch, velocity_batch, successful_samples
        
    catch e
        error("Fehler beim Zusammenfügen der Tensoren: $e")
    end
end

# =============================================================================
# SMART BATCH MANAGER
# =============================================================================

mutable struct BatchManager
    dataset::Vector
    target_resolution::Int
    max_memory_percent::Float64
    current_batch_size::Int
    memory_history::Vector{Float64}
    oom_count::Int
    adaptive_mode::Bool
    
    function BatchManager(dataset, target_resolution; max_memory_percent=80.0, adaptive_mode=true)
        new(dataset, target_resolution, max_memory_percent, 
            get(BATCH_SIZES, target_resolution, 1), Float64[], 0, adaptive_mode)
    end
end

function update_memory_stats!(manager::BatchManager)
    """
    Aktualisiert Memory-Statistiken
    """
    mem_info = get_gpu_memory_info()
    push!(manager.memory_history, mem_info.percent_used)
    
    # Behalte nur die letzten 10 Messungen
    if length(manager.memory_history) > 10
        manager.memory_history = manager.memory_history[end-9:end]
    end
end

function adapt_batch_size!(manager::BatchManager)
    """
    Passt Batch-Größe basierend auf Memory-History an
    """
    if !manager.adaptive_mode || length(manager.memory_history) < 3
        return
    end
    
    avg_memory = mean(manager.memory_history[end-2:end])
    
    if avg_memory > 90.0  # Sehr hohe Speichernutzung
        manager.current_batch_size = max(1, manager.current_batch_size ÷ 2)
        manager.oom_count += 1
        println("  Memory kritisch ($(round(avg_memory, digits=1))%) - Batch-Größe reduziert auf $(manager.current_batch_size)")
    elseif avg_memory < 60.0 && manager.oom_count == 0  # Niedrige Speichernutzung
        max_allowed = get(BATCH_SIZES, manager.target_resolution, 1)
        if manager.current_batch_size < max_allowed
            manager.current_batch_size = min(max_allowed, Int(ceil(manager.current_batch_size * 1.2)))
            println("  Memory niedrig ($(round(avg_memory, digits=1))%) - Batch-Größe erhöht auf $(manager.current_batch_size)")
        end
    end
end

function smart_batch_manager(dataset, target_resolution; max_memory_percent=80, verbose=true)
    """
    Überwacht GPU-Speicher
    Passt Batch-Größen automatisch an
    Cleanup zwischen Batches
    """
    manager = BatchManager(dataset, target_resolution, max_memory_percent=max_memory_percent)
    
    if verbose
        println("=== SMART BATCH MANAGER ===")
        println("Dataset: $(length(dataset)) samples")
        println("Target resolution: $(target_resolution)x$(target_resolution)")
        println("Initial batch size: $(manager.current_batch_size)")
        if CUDA.functional()
            mem_info = get_gpu_memory_info()
            println("GPU memory: $(round(mem_info.total / 1e9, digits=2)) GB total")
        else
            println("CPU mode (no GPU)")
        end
    end
    
    # Generator für Batches - Channel-basiert für echte Iteration
    batches = Channel{Tuple}(32) do ch
        batch_count = 0
        dataset_shuffled = shuffle(dataset)
        
        while batch_count * manager.current_batch_size < length(dataset_shuffled)
            batch_count += 1
            
            # Memory cleanup vor jedem Batch
            if CUDA.functional()
                GC.gc()
                CUDA.reclaim()
            else
                GC.gc()
            end
            
            # Memory-Status aktualisieren
            update_memory_stats!(manager)
            
            # Batch-Größe anpassen falls nötig
            if batch_count > 1
                adapt_batch_size!(manager)
            end
            
            # Sample-Bereich für aktuellen Batch
            start_idx = (batch_count - 1) * manager.current_batch_size + 1
            end_idx = min(start_idx + manager.current_batch_size - 1, length(dataset_shuffled))
            batch_samples = dataset_shuffled[start_idx:end_idx]
            
            if verbose && batch_count % 5 == 1
                mem_info = get_gpu_memory_info()
                println("Batch $batch_count: $(length(batch_samples)) samples, Memory: $(round(mem_info.percent_used, digits=1))%")
            end
            
            try
                # Erstelle Batch
                phase_batch, velocity_batch, successful = create_adaptive_batch(
                    batch_samples, target_resolution,
                    max_gpu_memory_percent=max_memory_percent,
                    verbose=false
                )
                
                # GPU-Transfer falls verfügbar
                if CUDA.functional()
                    phase_batch = gpu(phase_batch)
                    velocity_batch = gpu(velocity_batch)
                end
                
                put!(ch, (phase_batch, velocity_batch, successful, batch_count))
                
            catch e
                if occursin("out of memory", string(e)) || occursin("OOM", string(e))
                    manager.oom_count += 1
                    manager.current_batch_size = max(1, manager.current_batch_size ÷ 2)
                    println("  GPU OOM! Batch-Größe reduziert auf $(manager.current_batch_size)")
                    continue
                else
                    println("  Batch $batch_count Fehler: $e")
                    continue
                end
            end
        end
        
        if verbose
            println("Batch-Manager abgeschlossen: $batch_count Batches, $(manager.oom_count) OOM-Events")
        end
    end
    
    return batches
end

# =============================================================================
# MIXED-RESOLUTION DATASET GENERATOR
# =============================================================================

function generate_mixed_resolution_dataset(n_samples; resolutions=[64, 128, 256], verbose=true)
    """
    Generiert Samples mit verschiedenen Auflösungen
    Alle werden zu einheitlicher Zielauflösung verarbeitet
    """
    if verbose
        println("=== MIXED-RESOLUTION DATASET GENERATOR ===")
        println("Generiere $n_samples Samples")
        println("Auflösungen: $resolutions")
    end
    
    dataset = []
    stats = Dict(res => 0 for res in resolutions)
    
    for i in 1:n_samples
        # Zufällige Auflösung wählen
        resolution = rand(resolutions)
        stats[resolution] += 1
        
        # Zufällige Parameter
        n_crystals = rand(1:4)
        radius_crystal = [rand(0.03:0.005:0.08) for _ in 1:n_crystals]
        
        # Zufällige Positionen (Kollisionsvermeidung)
        centers = []
        for j in 1:n_crystals
            attempts = 0
            while attempts < 20
                x_pos = rand(-0.7:0.05:0.7)
                z_pos = rand(0.1:0.05:0.9)
                new_center = (x_pos, z_pos)
                
                # Prüfe Kollision mit existierenden Zentren
                collision = false
                for existing_center in centers
                    distance = sqrt((new_center[1] - existing_center[1])^2 + 
                                   (new_center[2] - existing_center[2])^2)
                    if distance < 0.15  # Minimaler Abstand
                        collision = true
                        break
                    end
                end
                
                if !collision
                    push!(centers, new_center)
                    break
                end
                attempts += 1
            end
            
            # Fallback wenn keine Position gefunden
            if length(centers) < j
                push!(centers, (rand(-0.3:0.1:0.3), rand(0.3:0.1:0.7)))
            end
        end
        
        if verbose && i % 50 == 1
            println("Sample $i/$n_samples: $(resolution)x$(resolution), $n_crystals Kristalle")
        end
        
        try
            # LaMEM-Simulation
            sample = LaMEM_Multi_crystal(
                resolution=(resolution, resolution),
                n_crystals=n_crystals,
                radius_crystal=radius_crystal,
                cen_2D=centers
            )
            
            push!(dataset, sample)
            
        catch e
            if verbose && i % 50 == 1
                println("  Fehler bei Sample $i: $e")
            end
            # Bei Fehler: Einfacheres Sample generieren
            try
                simple_sample = LaMEM_Multi_crystal(
                    resolution=(resolution, resolution),
                    n_crystals=1,
                    radius_crystal=[0.05],
                    cen_2D=[(0.0, 0.5)]
                )
                push!(dataset, simple_sample)
            catch e2
                println("  Auch einfaches Sample fehlgeschlagen: $e2")
                continue
            end
        end
        
        # Memory cleanup alle 10 Samples
        if i % 10 == 0
            GC.gc()
            if CUDA.functional()
                CUDA.reclaim()
            end
        end
    end
    
    if verbose
        println("\nDataset-Statistiken:")
        for (res, count) in stats
            percentage = round(100 * count / n_samples, digits=1)
            println("  $(res)x$(res): $count Samples ($percentage%)")
        end
        println("Gesamt: $(length(dataset)) erfolgreich generiert")
    end
    
    return dataset
end


# =============================================================================
# UNET KONFIGURATIONSSTRUKTUREN
# =============================================================================

"""
Konfiguration für ein adaptives UNet
"""
struct UNetConfig
    input_resolution::Int           # Eingabeauflösung (64, 128, 256, 512)
    input_channels::Int            # Anzahl Input-Kanäle (1 für Phasenfeld)
    output_channels::Int           # Anzahl Output-Kanäle (2 für vx, vz)
    
    # Architektur-Parameter
    depth::Int                     # Anzahl Pooling-Schritte (3-5)
    base_filters::Int              # Anzahl Filter im ersten Layer (32 oder 64)
    filter_progression::Vector{Int} # Filter pro Layer [32, 64, 128, 256, 512]
    
    # Pooling und Upsampling
    pooling_factor::Int            # Pooling-Faktor (immer 2)
    bottleneck_size::Int           # Minimale Feature-Map-Größe im Bottleneck
    
    # Regularisierung
    dropout_rate::Float32          # Dropout-Rate (0.0-0.5)
    use_batchnorm::Bool           # BatchNorm verwenden
    
    # Aktivierungen
    activation::Function           # Aktivierungsfunktion (relu, leakyrelu, etc.)
    final_activation::Union{Function, Nothing}  # Finale Aktivierung (nothing für Regression)
    
    # GPU-Optimierung
    use_checkpointing::Bool        # Gradient Checkpointing für Memory-Sparen
    mixed_precision::Bool          # Float16 für Forward Pass
end

"""
Zeigt UNet-Konfiguration übersichtlich an
"""
function Base.show(io::IO, config::UNetConfig)
    println(io, "UNetConfig für $(config.input_resolution)×$(config.input_resolution):")
    println(io, "  Tiefe: $(config.depth) Pooling-Schritte")
    println(io, "  Filter: $(config.filter_progression)")
    println(io, "  Bottleneck: $(config.bottleneck_size)×$(config.bottleneck_size)")
    println(io, "  Dropout: $(config.dropout_rate)")
    println(io, "  BatchNorm: $(config.use_batchnorm)")
    println(io, "  Checkpointing: $(config.use_checkpointing)")
end

# =============================================================================
# AUFLÖSUNGS-ANALYSE UND VALIDIERUNG
# =============================================================================

"""
Berechnet resultierende Feature-Map-Größen nach Pooling-Schritten
"""
function calculate_feature_map_sizes(input_resolution::Int, depth::Int, pooling_factor::Int=2)
    sizes = [input_resolution]
    current_size = input_resolution
    
    for i in 1:depth
        current_size = current_size ÷ pooling_factor
        push!(sizes, current_size)
    end
    
    return sizes
end

"""
Validiert ob eine UNet-Konfiguration sinnvoll ist
"""
function validate_unet_config(input_resolution::Int, depth::Int, min_bottleneck_size::Int=4)
    # Prüfe ob Eingabeauflösung eine Potenz von 2 ist
    if input_resolution & (input_resolution - 1) != 0
        return false, "Eingabeauflösung $input_resolution ist keine Potenz von 2"
    end
    
    # Prüfe erlaubten Bereich
    if input_resolution < 64 || input_resolution > 512
        return false, "Eingabeauflösung $input_resolution außerhalb erlaubtem Bereich [64, 512]"
    end
    
    # Berechne Feature-Map-Größen
    sizes = calculate_feature_map_sizes(input_resolution, depth)
    bottleneck_size = sizes[end]
    
    # Prüfe Bottleneck-Größe
    if bottleneck_size < min_bottleneck_size
        return false, "Bottleneck zu klein: $(bottleneck_size)×$(bottleneck_size) < $(min_bottleneck_size)×$(min_bottleneck_size)"
    end
    
    # Prüfe maximale Tiefe
    max_possible_depth = Int(floor(log2(input_resolution / min_bottleneck_size)))
    if depth > max_possible_depth
        return false, "Tiefe $depth zu groß für Auflösung $input_resolution (max: $max_possible_depth)"
    end
    
    return true, "Konfiguration gültig"
end

"""
Bestimmt optimale Tiefe für eine gegebene Auflösung
"""
function determine_optimal_depth(input_resolution::Int; min_bottleneck_size::Int=4, prefer_deeper::Bool=true)
    max_depth = Int(floor(log2(input_resolution / min_bottleneck_size)))
    
    # Auflösungsabhängige Heuristiken
    if input_resolution <= 64
        return min(3, max_depth)  # Flache Netze für kleine Bilder
    elseif input_resolution <= 128
        return min(4, max_depth)  # Standard-Tiefe
    elseif input_resolution <= 256
        return min(prefer_deeper ? 5 : 4, max_depth)  # Mittlere Tiefe
    else  # 512x512
        return min(prefer_deeper ? 6 : 5, max_depth)  # Tiefe Netze für große Bilder
    end
end

# =============================================================================
# FILTER-PROGRESSIONS-STRATEGIEN
# =============================================================================

"""
Verschiedene Strategien für Filter-Anzahl pro Layer
"""
function create_filter_progression(base_filters::Int, depth::Int, strategy::Symbol=:exponential)
    if strategy == :exponential
        # Exponentiell: 32 → 64 → 128 → 256 → 512
        return [base_filters * (2^i) for i in 0:depth]
        
    elseif strategy == :linear
        # Linear: 32 → 64 → 96 → 128 → 160
        return [base_filters + i * base_filters÷2 for i in 0:depth]
        
    elseif strategy == :conservative
        # Konservativ: weniger Parameter für Overfitting-Vermeidung
        progression = [base_filters]
        for i in 1:depth
            next_filters = min(progression[end] * 2, 256)  # Max 256 Filter
            push!(progression, next_filters)
        end
        return progression
        
    elseif strategy == :custom_small
        # Für kleine Auflösungen optimiert
        if depth <= 3
            return [base_filters, base_filters*2, base_filters*4, base_filters*4]
        else
            return [base_filters * (2^min(i, 2)) for i in 0:depth]
        end
        
    else
        error("Unbekannte Filter-Strategie: $strategy")
    end
end

"""
Bestimmt optimale Base-Filter-Anzahl basierend auf Auflösung
"""
function determine_base_filters(input_resolution::Int; memory_efficient::Bool=false)
    if memory_efficient
        return 32  # Immer 32 für Memory-Effizienz
    else
        if input_resolution <= 128
            return 32  # Kleinere Netze für kleine Bilder
        else
            return 64  # Größere Netze für große Bilder
        end
    end
end

# =============================================================================
# HAUPTKONFIGURATIONSFUNKTION
# =============================================================================

"""
Erstellt optimale UNet-Konfiguration für gegebene Auflösung
"""
function design_adaptive_unet(input_resolution::Int; 
                              input_channels::Int=1,
                              output_channels::Int=2,
                              prefer_deeper::Bool=true,
                              memory_efficient::Bool=false,
                              filter_strategy::Symbol=:exponential,
                              min_bottleneck_size::Int=4)
    
    println("=== ADAPTIVE UNET DESIGN ===")
    println("Zielauflösung: $(input_resolution)×$(input_resolution)")
    
    # 1. Validiere Eingabeauflösung
    is_valid, message = validate_unet_config(input_resolution, 1, min_bottleneck_size)
    if !is_valid
        error("Ungültige Auflösung: $message")
    end
    
    # 2. Bestimme optimale Tiefe
    optimal_depth = determine_optimal_depth(input_resolution, 
                                           min_bottleneck_size=min_bottleneck_size,
                                           prefer_deeper=prefer_deeper)
    
    # 3. Validiere gewählte Tiefe
    is_valid, message = validate_unet_config(input_resolution, optimal_depth, min_bottleneck_size)
    if !is_valid
        error("Optimale Tiefe ungültig: $message")
    end
    
    # 4. Bestimme Base-Filter
    base_filters = determine_base_filters(input_resolution, memory_efficient=memory_efficient)
    
    # 5. Erstelle Filter-Progression
    filter_progression = create_filter_progression(base_filters, optimal_depth, filter_strategy)
    
    # 6. Berechne finale Feature-Map-Größen
    feature_sizes = calculate_feature_map_sizes(input_resolution, optimal_depth)
    bottleneck_size = feature_sizes[end]
    
    # 7. Bestimme Regularisierungs-Parameter
    dropout_rate = if input_resolution >= 256
        0.2f0  # Mehr Dropout für große Bilder (Overfitting-Gefahr)
    elseif input_resolution >= 128
        0.1f0  # Mittleres Dropout
    else
        0.05f0  # Wenig Dropout für kleine Bilder
    end
    
    # 8. GPU-Optimierungen
    use_checkpointing = input_resolution >= 256 || memory_efficient
    mixed_precision = input_resolution >= 256
    
    # 9. Erstelle finale Konfiguration
    config = UNetConfig(
        input_resolution, input_channels, output_channels,
        optimal_depth, base_filters, filter_progression,
        2, bottleneck_size,  # pooling_factor=2
        dropout_rate, true,  # use_batchnorm=true
        relu, nothing,       # activation, final_activation
        use_checkpointing, mixed_precision
    )
    
    # 10. Debug-Ausgabe
    println("  Gewählte Tiefe: $optimal_depth")
    println("  Base-Filter: $base_filters")
    println("  Filter-Progression: $filter_progression")
    println("  Feature-Map-Größen: $feature_sizes")
    println("  Bottleneck: $(bottleneck_size)×$(bottleneck_size)")
    println("  Dropout: $(dropout_rate)")
    println("  Checkpointing: $use_checkpointing")
    println("  Mixed-Precision: $mixed_precision")
    
    return config
end

# =============================================================================
# UNET-BLÖCKE - HINZUFÜGUNG ZU IHREM BESTEHENDEN CODE
# =============================================================================
# Fügen Sie diesen Code zu Ihrem bestehenden Code hinzu

# =============================================================================
# SKIP-CONNECTION-MANAGER
# =============================================================================

"""
Verwaltet Skip-Connections zwischen Encoder und Decoder
"""
mutable struct SkipConnectionManager
    stored_features::Dict{Int, Any}
    target_resolution::Int
    
    function SkipConnectionManager(target_resolution::Int)
        new(Dict{Int, Any}(), target_resolution)
    end
end

"""
Passt Skip-Feature-Dimensionen an Decoder-Eingabe an - ROBUSTE VERSION
"""
function adapt_skip_dimensions(skip_features, decoder_features)
    skip_size = size(skip_features)
    decoder_size = size(decoder_features)
    
    # Prüfe ob Anpassung nötig ist
    if skip_size[1:2] == decoder_size[1:2]
        return skip_features  # Keine Anpassung nötig
    end
    
    # Größenanpassung (Center-Crop oder Padding)
    target_h, target_w = decoder_size[1:2]
    current_h, current_w = skip_size[1:2]
    
    if current_h >= target_h && current_w >= target_w
        # Center-Crop
        h_start = max(1, (current_h - target_h) ÷ 2 + 1)
        w_start = max(1, (current_w - target_w) ÷ 2 + 1)
        h_end = min(current_h, h_start + target_h - 1)
        w_end = min(current_w, w_start + target_w - 1)
        
        return skip_features[h_start:h_end, w_start:w_end, :, :]
    else
        # Padding - einfachste Lösung: Repeat edge pixels
        if target_h > current_h || target_w > current_w
            # Für Simplizität: Interpolation mit repeat
            repeated_h = max(current_h, target_h)
            repeated_w = max(current_w, target_w)
            
            # Einfache Wiederholung der letzten Pixel (bessere Lösung wäre Interpolation)
            padded = zeros(eltype(skip_features), repeated_h, repeated_w, skip_size[3], skip_size[4])
            
            # Kopiere vorhandene Daten
            padded[1:current_h, 1:current_w, :, :] .= skip_features
            
            # Wiederhole Randpixel
            if repeated_h > current_h
                padded[current_h+1:end, 1:current_w, :, :] .= skip_features[current_h:current_h, 1:current_w, :, :]
            end
            if repeated_w > current_w
                padded[:, current_w+1:end, :, :] .= padded[:, current_w:current_w, :, :]
            end
            
            # Crop auf exakte Zielgröße
            return padded[1:target_h, 1:target_w, :, :]
        else
            return skip_features[1:target_h, 1:target_w, :, :]
        end
    end
end

# =============================================================================
# ENCODER-BLÖCKE
# =============================================================================

"""
Erstellt einen Standard-Encoder-Block
"""
function create_encoder_block(config::UNetConfig, in_channels::Int, out_channels::Int, level::Int)
    layers = []
    
    # Erste Convolution
    push!(layers, Conv((3, 3), in_channels => out_channels, pad=SamePad()))
    
    # BatchNorm falls aktiviert
    if config.use_batchnorm
        push!(layers, BatchNorm(out_channels))
    end
    
    # Aktivierungsfunktion
    push!(layers, config.activation)
    
    # Zweite Convolution (Standard UNet)
    push!(layers, Conv((3, 3), out_channels => out_channels, pad=SamePad()))
    
    if config.use_batchnorm
        push!(layers, BatchNorm(out_channels))
    end
    
    push!(layers, config.activation)
    
    # Dropout für tiefere Layer
    if config.dropout_rate > 0 && level >= 2
        push!(layers, Dropout(config.dropout_rate))
    end
    
    return Chain(layers...)
end

"""
Erstellt Encoder-Block mit Pooling - KORRIGIERTE VERSION
"""
function create_encoder_with_pooling(config::UNetConfig, in_channels::Int, out_channels::Int, level::Int)
    encoder_block = create_encoder_block(config, in_channels, out_channels, level)
    
    # KRITISCHE ÄNDERUNG: Nur Pooling wenn es NICHT der letzte Encoder ist
    # Der letzte Encoder ist level == config.depth
    if level < config.depth  # Kein Pooling im letzten Encoder
        return Chain(encoder_block, MaxPool((2, 2)))
    else
        return encoder_block  # Letzter Encoder ohne Pooling
    end
end

# =============================================================================
# DECODER-BLÖCKE
# =============================================================================

"""
Spezielle Decoder-Block-Funktion die Skip-Connection-Handling einbaut - KORRIGIERT
"""
struct DecoderBlockWithSkip
    upsample::ConvTranspose
    conv_block::Chain
    level::Int
end

Flux.@functor DecoderBlockWithSkip

function (block::DecoderBlockWithSkip)(x, skip_features)
    # Upsampling
    upsampled = block.upsample(x)
    
    # Skip-Connection: Dimensionen anpassen und concatenaten
    adapted_skip = adapt_skip_dimensions(skip_features, upsampled)
    concatenated = cat(upsampled, adapted_skip, dims=3)  # Channel-Dimension
    
    # Convolution-Block
    return block.conv_block(concatenated)
end

function create_decoder_with_skip(config::UNetConfig, in_channels::Int, skip_channels::Int, out_channels::Int, level::Int)
    # Upsampling-Layer - KORRIGIERT: Exakte 2x Vergrößerung
    upsample = ConvTranspose((2, 2), in_channels => out_channels, stride=2, pad=0)
    
    # Convolution-Block nach Concatenation
    concat_channels = out_channels + skip_channels
    conv_layers = []
    
    push!(conv_layers, Conv((3, 3), concat_channels => out_channels, pad=SamePad()))
    if config.use_batchnorm
        push!(conv_layers, BatchNorm(out_channels))
    end
    push!(conv_layers, config.activation)
    
    push!(conv_layers, Conv((3, 3), out_channels => out_channels, pad=SamePad()))
    if config.use_batchnorm
        push!(conv_layers, BatchNorm(out_channels))
    end
    push!(conv_layers, config.activation)
    
    if config.dropout_rate > 0 && level <= 2
        push!(conv_layers, Dropout(config.dropout_rate))
    end
    
    conv_block = Chain(conv_layers...)
    
    return DecoderBlockWithSkip(upsample, conv_block, level)
end

# =============================================================================
# BOTTLENECK-BLOCK
# =============================================================================

"""
Erstellt den Bottleneck-Block (tiefster Punkt des UNet)
"""
function create_bottleneck_block(config::UNetConfig, in_channels::Int, out_channels::Int)
    layers = []
    
    # Erste Convolution
    push!(layers, Conv((3, 3), in_channels => out_channels, pad=SamePad()))
    
    if config.use_batchnorm
        push!(layers, BatchNorm(out_channels))
    end
    
    push!(layers, config.activation)
    
    # Dropout im Bottleneck (höchste Rate)
    if config.dropout_rate > 0
        push!(layers, Dropout(min(config.dropout_rate * 2, 0.5)))  # Höhere Dropout-Rate
    end
    
    # Zweite Convolution
    push!(layers, Conv((3, 3), out_channels => out_channels, pad=SamePad()))
    
    if config.use_batchnorm
        push!(layers, BatchNorm(out_channels))
    end
    
    push!(layers, config.activation)
    
    return Chain(layers...)
end

# =============================================================================
# OUTPUT-LAYER
# =============================================================================

"""
Erstellt den finalen Output-Layer
"""
function create_output_layer(config::UNetConfig, in_channels::Int)
    layers = []
    
    # 1x1 Convolution für finale Klassifikation/Regression
    push!(layers, Conv((1, 1), in_channels => config.output_channels))
    
    # Finale Aktivierung falls definiert
    if config.final_activation !== nothing
        push!(layers, config.final_activation)
    end
    
    return Chain(layers...)
end

# =============================================================================
# ADAPTIVE UNET-STRUKTUR
# =============================================================================

"""
Hauptstruktur des adaptiven UNet
"""
struct AdaptiveUNet
    config::UNetConfig
    
    # Encoder-Komponenten
    encoder_blocks::Vector{Chain}
    bottleneck::Chain
    
    # Decoder-Komponenten
    decoder_blocks::Vector{DecoderBlockWithSkip}
    output_layer::Chain
    
    # Skip-Connection-Manager
    skip_manager::SkipConnectionManager
end

Flux.@functor AdaptiveUNet

"""
Erstellt komplettes adaptives UNet basierend auf Konfiguration - DIMENSIONEN-KORRIGIERT
"""
function create_adaptive_unet(config::UNetConfig)
    println("=== ERSTELLE ADAPTIVE UNET ===")
    println("Konfiguration: $(config.input_resolution)×$(config.input_resolution)")
    println("Tiefe: $(config.depth), Filter: $(config.filter_progression)")
    
    # KRITISCH: Encoder sollten nur bis config.depth gehen, nicht depth+1
    # Encoder-Blöcke erstellen
    encoder_blocks = Chain[]
    for level in 1:config.depth
        in_ch = level == 1 ? config.input_channels : config.filter_progression[level-1]
        out_ch = config.filter_progression[level]
        
        encoder_block = create_encoder_with_pooling(config, in_ch, out_ch, level)
        push!(encoder_blocks, encoder_block)
        
        println("  Encoder $level: $in_ch → $out_ch")
    end
    
    # Bottleneck erstellt - WICHTIG: Verwendet config.depth als Input, config.depth+1 als Output
    bottleneck_in = config.filter_progression[config.depth]
    bottleneck_out = config.filter_progression[config.depth + 1]
    bottleneck = create_bottleneck_block(config, bottleneck_in, bottleneck_out)
    println("  Bottleneck: $bottleneck_in → $bottleneck_out")
    
    # KRITISCH: Decoder sollten exakt config.depth Schritte haben
    # Decoder-Blöcke erstellen (in umgekehrter Reihenfolge)
    decoder_blocks = DecoderBlockWithSkip[]
    for level in config.depth:-1:1
        # Decoder Input kommt vom Bottleneck (level+1) oder vorherigen Decoder
        decoder_in = config.filter_progression[level + 1]
        # Skip Connection kommt vom entsprechenden Encoder-Level
        skip_ch = config.filter_progression[level]
        # Decoder Output sollte der Skip-Channel-Anzahl entsprechen
        decoder_out = config.filter_progression[level]
        
        decoder_block = create_decoder_with_skip(config, decoder_in, skip_ch, decoder_out, level)
        push!(decoder_blocks, decoder_block)
        
        println("  Decoder $level: $decoder_in + $skip_ch → $decoder_out")
    end
    
    # Output-Layer erstellen - verwendet ersten Filter-Wert
    final_in = config.filter_progression[1]
    output_layer = create_output_layer(config, final_in)
    println("  Output: $final_in → $(config.output_channels)")
    
    println("  Encoder-Anzahl: $(length(encoder_blocks))")
    println("  Decoder-Anzahl: $(length(decoder_blocks))")
    
    # Validierung
    if length(encoder_blocks) != config.depth
        error("Encoder-Anzahl $(length(encoder_blocks)) != config.depth $(config.depth)")
    end
    if length(decoder_blocks) != config.depth
        error("Decoder-Anzahl $(length(decoder_blocks)) != config.depth $(config.depth)")
    end
    
    # Vereinfachte Struktur ohne Skip-Manager
    return AdaptiveUNet(config, encoder_blocks, bottleneck, decoder_blocks, output_layer, SkipConnectionManager(config.input_resolution))
end

# =============================================================================
# FORWARD-PASS IMPLEMENTATION
# =============================================================================

"""
Forward-Pass durch das adaptive UNet - FINAL KORRIGIERT
"""
function (model::AdaptiveUNet)(x)
    # Input-Validierung
    input_size = size(x)
    expected_size = (model.config.input_resolution, model.config.input_resolution, model.config.input_channels)
    
    if input_size[1:3] != expected_size
        error("Input-Größe $(input_size[1:3]) passt nicht zu erwarteter Größe $expected_size")
    end
    
    # Encoder-Phase: Sammle Skip-Features
    skip_features = []
    current = x
    
    # WICHTIG: Durchlaufe NUR die tatsächlich vorhandenen Encoder
    for (level, encoder) in enumerate(model.encoder_blocks)
        if level < length(model.encoder_blocks)
            # Alle Encoder außer dem letzten haben Pooling
            if length(encoder.layers) > 1 && isa(encoder.layers[end], typeof(MaxPool((2,2))))
                # Trenne Conv-Teil von MaxPool
                conv_layers = encoder.layers[1:end-1]
                conv_part = Chain(conv_layers...)
                pool_layer = encoder.layers[end]
                
                # Führe Convolution aus (für Skip-Connection)
                features = conv_part(current)
                push!(skip_features, features)
                
                # Dann Pooling für nächsten Level
                current = pool_layer(features)
            else
                # Fallback - Encoder ohne explizites Pooling
                features = encoder(current)
                push!(skip_features, features)
                current = MaxPool((2,2))(features)  # Manuelles Pooling
            end
        else
            # LETZTER Encoder (vor Bottleneck) - DEFINITIV KEIN POOLING!
            features = encoder(current)
            push!(skip_features, features)
            current = features  # KEIN POOLING!
        end
    end
    
    # Bottleneck
    current = model.bottleneck(current)
    
    # WICHTIG: Decoder sollten Skip-Features in umgekehrter Reihenfolge verwenden
    # Aber nur so viele Decoder wie es Encoder gab
    num_decoders = min(length(model.decoder_blocks), length(skip_features))
    
    for i in 1:num_decoders
        decoder = model.decoder_blocks[i]
        skip_idx = length(skip_features) - i + 1
        skip = skip_features[skip_idx]
        current = decoder(current, skip)
    end
    
    # Output-Layer
    output = model.output_layer(current)
    
    return output
end

# =============================================================================
# TESTING UND VALIDIERUNG
# =============================================================================

"""
Testet UNet mit verschiedenen Eingabegrößen
"""
function test_unet_forward_pass(model::AdaptiveUNet; batch_size::Int=2, verbose::Bool=true)
    config = model.config
    
    if verbose
        println("=== TESTE UNET FORWARD-PASS ===")
        println("Auflösung: $(config.input_resolution)×$(config.input_resolution)")
        println("Batch-Größe: $batch_size")
    end
    
    # Erstelle Test-Input
    input_shape = (config.input_resolution, config.input_resolution, config.input_channels, batch_size)
    test_input = randn(Float32, input_shape...)
    
    if verbose
        println("Input-Shape: $(size(test_input))")
    end
    
    # Forward-Pass
    try
        output = model(test_input)
        
        expected_output_shape = (config.input_resolution, config.input_resolution, config.output_channels, batch_size)
        
        if verbose
            println("Output-Shape: $(size(output))")
            println("Erwartet: $expected_output_shape")
        end
        
        if size(output) == expected_output_shape
            if verbose
                println("✓ Forward-Pass erfolgreich!")
            end
            return true, size(output)
        else
            if verbose
                println("✗ Output-Shape inkorrekt!")
            end
            return false, size(output)
        end
        
    catch e
        if verbose
            println("✗ Forward-Pass fehlgeschlagen: $e")
        end
        return false, nothing
    end
end

"""
Testet UNet-Erstellung für verschiedene Konfigurationen
"""
function test_unet_configurations(resolutions::Vector{Int}; verbose::Bool=true)
    println("=== TESTE UNET-KONFIGURATIONEN ===")
    
    results = Dict{Int, Bool}()
    
    for res in resolutions
        println("\n" * "-"^50)
        println("Teste $(res)×$(res):")
        
        try
            # Erstelle Konfiguration
            config = design_adaptive_unet(res)
            
            # Erstelle UNet
            model = create_adaptive_unet(config)
            
            # Teste Forward-Pass
            success, output_shape = test_unet_forward_pass(model, verbose=verbose)
            results[res] = success
            
            if success
                println("✓ $(res)×$(res) UNet erfolgreich erstellt und getestet")
            else
                println("✗ $(res)×$(res) UNet-Test fehlgeschlagen")
            end
            
        catch e
            println("✗ $(res)×$(res) UNet-Erstellung fehlgeschlagen: $e")
            results[res] = false
        end
    end
    
    # Zusammenfassung
    println("\n" * "="^50)
    println("ZUSAMMENFASSUNG:")
    successful = sum(values(results))
    total = length(results)
    println("Erfolgreich: $successful/$total")
    
    for (res, success) in sort(collect(results))
        status = success ? "✓" : "✗"
        println("  $(res)×$(res): $status")
    end
    
    return results
end

# =============================================================================
# DEMO
# =============================================================================

function demo_unet_blocks()
    """
    Demonstriert die modularen UNet-Blöcke
    """
    println("="^80)
    println("DEMO: MODULARE UNET-BLÖCKE")
    println("="^80)
    
    # Teste verschiedene Auflösungen
    test_resolutions = [64, 128, 256]
    results = test_unet_configurations(test_resolutions)
    
    # Erstelle ein Beispiel-UNet für detaillierte Analyse
    println("\n" * "="^50)
    println("DETAILLIERTE ANALYSE (128×128):")
    println("="^50)
    
    config_128 = design_adaptive_unet(128)
    model_128 = create_adaptive_unet(config_128)
    
    # Teste verschiedene Batch-Größen
    for batch_size in [1, 2, 4]
        success, output_shape = test_unet_forward_pass(model_128, batch_size=batch_size, verbose=false)
        status = success ? "✓" : "✗"
        println("Batch $batch_size: $status (Output: $output_shape)")
    end
    
    println("\n" * "="^80)
    println("MODULARE UNET-BLÖCKE DEMO ABGESCHLOSSEN")
    println("="^80)
    
    return results, model_128
end

# =============================================================================
# PREPROCESSING-FUNKTION (für Vollständigkeit)
# =============================================================================

function preprocess_lamem_sample(x, z, phase, vx, vz, v_stokes; target_resolution=128)
    """
    Vollständige Vorverarbeitung eines LaMEM-Samples
    """
    # 1. Größenanpassung
    phase_resized = resize_power_of_2(phase, target_resolution)
    vx_resized = resize_power_of_2(vx, target_resolution)
    vz_resized = resize_power_of_2(vz, target_resolution)
    
    # 2. Normalisierung
    vx_norm = Float32.(vx_resized ./ v_stokes)
    vz_norm = Float32.(vz_resized ./ v_stokes)
    phase_float = Float32.(phase_resized)
    
    # 3. Tensor-Format für UNet (H, W, C, B)
    phase_tensor = reshape(phase_float, target_resolution, target_resolution, 1, 1)
    velocity_tensor = cat(vx_norm, vz_norm, dims=3)
    velocity_tensor = reshape(velocity_tensor, target_resolution, target_resolution, 2, 1)
    
    return phase_tensor, velocity_tensor
end

println("UNet-Blöcke erfolgreich hinzugefügt!")
println("Verfügbare Funktionen:")
println("  - create_adaptive_unet(config)")
println("  - test_unet_forward_pass(model)")
println("  - demo_unet_blocks()")
println("Testen Sie jetzt: demo_unet_blocks()")

# =============================================================================
# UNET KONFIGURATIONSSTRUKTUREN
# =============================================================================

"""
Konfiguration für ein adaptives UNet
"""
struct UNetConfig
    input_resolution::Int           # Eingabeauflösung (64, 128, 256, 512)
    input_channels::Int            # Anzahl Input-Kanäle (1 für Phasenfeld)
    output_channels::Int           # Anzahl Output-Kanäle (2 für vx, vz)
    
    # Architektur-Parameter
    depth::Int                     # Anzahl Pooling-Schritte (3-5)
    base_filters::Int              # Anzahl Filter im ersten Layer (32 oder 64)
    filter_progression::Vector{Int} # Filter pro Layer [32, 64, 128, 256, 512]
    
    # Pooling und Upsampling
    pooling_factor::Int            # Pooling-Faktor (immer 2)
    bottleneck_size::Int           # Minimale Feature-Map-Größe im Bottleneck
    
    # Regularisierung
    dropout_rate::Float32          # Dropout-Rate (0.0-0.5)
    use_batchnorm::Bool           # BatchNorm verwenden
    
    # Aktivierungen
    activation::Function           # Aktivierungsfunktion (relu, leakyrelu, etc.)
    final_activation::Union{Function, Nothing}  # Finale Aktivierung (nothing für Regression)
    
    # GPU-Optimierung
    use_checkpointing::Bool        # Gradient Checkpointing für Memory-Sparen
    mixed_precision::Bool          # Float16 für Forward Pass
end

"""
Zeigt UNet-Konfiguration übersichtlich an
"""
function Base.show(io::IO, config::UNetConfig)
    println(io, "UNetConfig für $(config.input_resolution)×$(config.input_resolution):")
    println(io, "  Tiefe: $(config.depth) Pooling-Schritte")
    println(io, "  Filter: $(config.filter_progression)")
    println(io, "  Bottleneck: $(config.bottleneck_size)×$(config.bottleneck_size)")
    println(io, "  Dropout: $(config.dropout_rate)")
    println(io, "  BatchNorm: $(config.use_batchnorm)")
    println(io, "  Checkpointing: $(config.use_checkpointing)")
end

# =============================================================================
# AUFLÖSUNGS-ANALYSE UND VALIDIERUNG
# =============================================================================

"""
Berechnet resultierende Feature-Map-Größen nach Pooling-Schritten
"""
function calculate_feature_map_sizes(input_resolution::Int, depth::Int, pooling_factor::Int=2)
    sizes = [input_resolution]
    current_size = input_resolution
    
    for i in 1:depth
        current_size = current_size ÷ pooling_factor
        push!(sizes, current_size)
    end
    
    return sizes
end

"""
Validiert ob eine UNet-Konfiguration sinnvoll ist
"""
function validate_unet_config(input_resolution::Int, depth::Int, min_bottleneck_size::Int=4)
    # Prüfe ob Eingabeauflösung eine Potenz von 2 ist
    if input_resolution & (input_resolution - 1) != 0
        return false, "Eingabeauflösung $input_resolution ist keine Potenz von 2"
    end
    
    # Prüfe erlaubten Bereich
    if input_resolution < 64 || input_resolution > 512
        return false, "Eingabeauflösung $input_resolution außerhalb erlaubtem Bereich [64, 512]"
    end
    
    # Berechne Feature-Map-Größen
    sizes = calculate_feature_map_sizes(input_resolution, depth)
    bottleneck_size = sizes[end]
    
    # Prüfe Bottleneck-Größe
    if bottleneck_size < min_bottleneck_size
        return false, "Bottleneck zu klein: $(bottleneck_size)×$(bottleneck_size) < $(min_bottleneck_size)×$(min_bottleneck_size)"
    end
    
    # Prüfe maximale Tiefe
    max_possible_depth = Int(floor(log2(input_resolution / min_bottleneck_size)))
    if depth > max_possible_depth
        return false, "Tiefe $depth zu groß für Auflösung $input_resolution (max: $max_possible_depth)"
    end
    
    return true, "Konfiguration gültig"
end

"""
Bestimmt optimale Tiefe für eine gegebene Auflösung - KORRIGIERT
"""
function determine_optimal_depth(input_resolution::Int; min_bottleneck_size::Int=8, prefer_deeper::Bool=true)
    # KRITISCHE KORREKTUR: Die Tiefe ist die Anzahl der Pooling-Schritte
    # Für 256×256: 256→128→64→32→16 = 4 Pooling-Schritte, Bottleneck 16×16
    # Für 128×128: 128→64→32→16 = 3 Pooling-Schritte, Bottleneck 16×16
    # Für 64×64: 64→32→16 = 2 Pooling-Schritte, Bottleneck 16×16
    
    max_depth = Int(floor(log2(input_resolution / min_bottleneck_size)))
    
    # Auflösungsabhängige Heuristiken - KORRIGIERT
    if input_resolution <= 64
        return min(2, max_depth)  # 64→32→16, Bottleneck 16×16
    elseif input_resolution <= 128
        return min(3, max_depth)  # 128→64→32→16, Bottleneck 16×16
    elseif input_resolution <= 256
        return min(4, max_depth)  # 256→128→64→32→16, Bottleneck 16×16
    else  # 512×512
        return min(5, max_depth)  # 512→256→128→64→32→16, Bottleneck 16×16
    end
end

# =============================================================================
# FILTER-PROGRESSIONS-STRATEGIEN
# =============================================================================

"""
Verschiedene Strategien für Filter-Anzahl pro Layer
"""
function create_filter_progression(base_filters::Int, depth::Int, strategy::Symbol=:exponential)
    if strategy == :exponential
        # Exponentiell: 32 → 64 → 128 → 256 → 512
        return [base_filters * (2^i) for i in 0:depth]
        
    elseif strategy == :linear
        # Linear: 32 → 64 → 96 → 128 → 160
        return [base_filters + i * base_filters÷2 for i in 0:depth]
        
    elseif strategy == :conservative
        # Konservativ: weniger Parameter für Overfitting-Vermeidung
        progression = [base_filters]
        for i in 1:depth
            next_filters = min(progression[end] * 2, 256)  # Max 256 Filter
            push!(progression, next_filters)
        end
        return progression
        
    elseif strategy == :custom_small
        # Für kleine Auflösungen optimiert
        if depth <= 3
            return [base_filters, base_filters*2, base_filters*4, base_filters*4]
        else
            return [base_filters * (2^min(i, 2)) for i in 0:depth]
        end
        
    else
        error("Unbekannte Filter-Strategie: $strategy")
    end
end

"""
Bestimmt optimale Base-Filter-Anzahl basierend auf Auflösung
"""
function determine_base_filters(input_resolution::Int; memory_efficient::Bool=false)
    if memory_efficient
        return 32  # Immer 32 für Memory-Effizienz
    else
        if input_resolution <= 128
            return 32  # Kleinere Netze für kleine Bilder
        else
            return 64  # Größere Netze für große Bilder
        end
    end
end

# =============================================================================
# HAUPTKONFIGURATIONSFUNKTION
# =============================================================================

"""
Erstellt optimale UNet-Konfiguration für gegebene Auflösung
"""
function design_adaptive_unet(input_resolution::Int; 
                              input_channels::Int=1,
                              output_channels::Int=2,
                              prefer_deeper::Bool=true,
                              memory_efficient::Bool=false,
                              filter_strategy::Symbol=:exponential,
                              min_bottleneck_size::Int=4)
    
    println("=== ADAPTIVE UNET DESIGN ===")
    println("Zielauflösung: $(input_resolution)×$(input_resolution)")
    
    # 1. Validiere Eingabeauflösung
    is_valid, message = validate_unet_config(input_resolution, 1, min_bottleneck_size)
    if !is_valid
        error("Ungültige Auflösung: $message")
    end
    
    # 2. Bestimme optimale Tiefe
    optimal_depth = determine_optimal_depth(input_resolution, 
                                           min_bottleneck_size=min_bottleneck_size,
                                           prefer_deeper=prefer_deeper)
    
    # 3. Validiere gewählte Tiefe
    is_valid, message = validate_unet_config(input_resolution, optimal_depth, min_bottleneck_size)
    if !is_valid
        error("Optimale Tiefe ungültig: $message")
    end
    
    # 4. Bestimme Base-Filter
    base_filters = determine_base_filters(input_resolution, memory_efficient=memory_efficient)
    
    # 5. Erstelle Filter-Progression
    filter_progression = create_filter_progression(base_filters, optimal_depth, filter_strategy)
    
    # 6. Berechne finale Feature-Map-Größen
    feature_sizes = calculate_feature_map_sizes(input_resolution, optimal_depth)
    bottleneck_size = feature_sizes[end]
    
    # 7. Bestimme Regularisierungs-Parameter
    dropout_rate = if input_resolution >= 256
        0.2f0  # Mehr Dropout für große Bilder (Overfitting-Gefahr)
    elseif input_resolution >= 128
        0.1f0  # Mittleres Dropout
    else
        0.05f0  # Wenig Dropout für kleine Bilder
    end
    
    # 8. GPU-Optimierungen
    use_checkpointing = input_resolution >= 256 || memory_efficient
    mixed_precision = input_resolution >= 256
    
    # 9. Erstelle finale Konfiguration
    config = UNetConfig(
        input_resolution, input_channels, output_channels,
        optimal_depth, base_filters, filter_progression,
        2, bottleneck_size,  # pooling_factor=2
        dropout_rate, true,  # use_batchnorm=true
        relu, nothing,       # activation, final_activation
        use_checkpointing, mixed_precision
    )
    
    # 10. Debug-Ausgabe
    println("  Gewählte Tiefe: $optimal_depth")
    println("  Base-Filter: $base_filters")
    println("  Filter-Progression: $filter_progression")
    println("  Feature-Map-Größen: $feature_sizes")
    println("  Bottleneck: $(bottleneck_size)×$(bottleneck_size)")
    println("  Dropout: $(dropout_rate)")
    println("  Checkpointing: $use_checkpointing")
    println("  Mixed-Precision: $mixed_precision")
    
    return config
end

# =============================================================================
# KONFIGURATIONSVERGLEICH UND ANALYSE
# =============================================================================

"""
Vergleicht Konfigurationen für verschiedene Auflösungen
"""
function compare_unet_configs(resolutions::Vector{Int}; kwargs...)
    println("=== UNET KONFIGURATIONSVERGLEICH ===")
    
    configs = Dict{Int, UNetConfig}()
    
    for res in resolutions
        println("\n" * "="^50)
        try
            config = design_adaptive_unet(res; kwargs...)
            configs[res] = config
        catch e
            println("Fehler für Auflösung $res: $e")
        end
    end
    
    # Zusammenfassungstabelle
    println("\n" * "="^80)
    println("ZUSAMMENFASSUNG:")
    println("="^80)
    println(@sprintf("%-12s %-6s %-6s %-20s %-12s %-8s", 
                     "Auflösung", "Tiefe", "Base", "Filter", "Bottleneck", "Dropout"))
    println("-"^80)
    
    for res in sort(collect(keys(configs)))
        config = configs[res]
        filter_str = join(string.(config.filter_progression), "→")
        if length(filter_str) > 18
            filter_str = filter_str[1:15] * "..."
        end
        
        println(@sprintf("%-12s %-6d %-6d %-20s %-12s %-8.1f%%", 
                        "$(res)×$(res)", 
                        config.depth, 
                        config.base_filters,
                        filter_str,
                        "$(config.bottleneck_size)×$(config.bottleneck_size)",
                        config.dropout_rate*100))
    end
    
    return configs
end

"""
Schätzt Speicherbedarf einer UNet-Konfiguration
"""
function estimate_memory_usage(config::UNetConfig; batch_size::Int=1, include_gradients::Bool=true)
    total_memory = 0
    precision_factor = config.mixed_precision ? 2 : 4  # Float16 vs Float32
    
    # Input/Output
    input_memory = config.input_resolution^2 * config.input_channels * batch_size * precision_factor
    output_memory = config.input_resolution^2 * config.output_channels * batch_size * precision_factor
    total_memory += input_memory + output_memory
    
    # Feature Maps (grobe Schätzung)
    current_size = config.input_resolution
    for i in 1:length(config.filter_progression)
        filters = config.filter_progression[i]
        feature_memory = current_size^2 * filters * batch_size * precision_factor
        total_memory += feature_memory
        
        # Skip-Connection-Speicher
        if i < length(config.filter_progression)
            total_memory += feature_memory  # Für Skip-Connections
        end
        
        if i < config.depth
            current_size ÷= 2
        end
    end
    
    # Gradienten (falls Training)
    if include_gradients
        total_memory *= 2  # Parameter + Gradienten
    end
    
    return total_memory / (1024^3)  # In GB
end

# =============================================================================
# DEMO UND TESTING
# =============================================================================

function demo_unet_configuration()
    """
    Demonstriert die UNet-Konfigurationslogik
    """
    println("="^80)
    println("DEMO: ADAPTIVE UNET KONFIGURATIONSLOGIK")
    println("="^80)
    
    # 1. Teste verschiedene Auflösungen
    test_resolutions = [64, 128, 256, 512]
    configs = compare_unet_configs(test_resolutions)
    
    # 2. Teste verschiedene Strategien
    println("\n" * "="^50)
    println("FILTER-STRATEGIEN-VERGLEICH (256×256):")
    println("="^50)
    
    strategies = [:exponential, :linear, :conservative, :custom_small]
    for strategy in strategies
        println("\nStrategie: $strategy")
        try
            config = design_adaptive_unet(256, filter_strategy=strategy)
            println("  Filter: $(config.filter_progression)")
        catch e
            println("  Fehler: $e")
        end
    end
    
    # 3. Memory-Analyse
    println("\n" * "="^50)
    println("SPEICHER-ANALYSE:")
    println("="^50)
    
    for res in test_resolutions
        if haskey(configs, res)
            config = configs[res]
            memory_gb = estimate_memory_usage(config, batch_size=4)
            println("$(res)×$(res): $(round(memory_gb, digits=2)) GB (Batch 4)")
        end
    end
    
    # 4. Validierungs-Tests
    println("\n" * "="^50)
    println("VALIDIERUNGS-TESTS:")
    println("="^50)
    
    # Teste ungültige Eingaben
    invalid_tests = [
        (63, "Keine Potenz von 2"),
        (32, "Zu kleine Auflösung"),
        (1024, "Zu große Auflösung")
    ]
    
    for (res, reason) in invalid_tests
        try
            design_adaptive_unet(res)
            println("$res: ✗ Sollte fehlschlagen ($reason)")
        catch e
            println("$res: ✓ Korrekt abgelehnt ($reason)")
        end
    end
    
    println("\n" * "="^80)
    println("KONFIGURATIONSLOGIK-DEMO ABGESCHLOSSEN")
    println("="^80)
    
    return configs
end

# Führe Demo aus
println("Starte UNet-Konfigurationslogik...")
configs = demo_unet_configuration()

# 1. Erstelle 256×256 Konfiguration:
config_256 = design_adaptive_unet(256)

# 2. Erstelle das UNet:
model_256 = create_adaptive_unet(config_256)

# 3. Teste es:
test_input = randn(Float32, 256, 256, 1, 2)
output = model_256(test_input)

println("Input: $(size(test_input))")
println("Output: $(size(output))")
println("Erwartet: (256, 256, 2, 2)")
println("Erfolgreich: $(size(output) == (256, 256, 2, 2))")