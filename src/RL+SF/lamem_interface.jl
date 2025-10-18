# =============================================================================
# LAMEM INTERFACE MODULE
# =============================================================================
# Speichern als: lamem_interface.jl

using LaMEM, GeophysicalModelGenerator
using Statistics
using Random

"""
Erstellt ein LaMEM-Modell mit mehreren Kristallphasen
"""
function LaMEM_Multi_crystal(;
    resolution = (64, 64),              
    n_crystals = 10,                     
    radius_crystal = [0.05],            
    η_magma = 1e20,                     
    ρ_magma = 2700,                     
    Δρ = 200,                           
    domain_size = (-1.0, 1.0),          
    cen_2D = [(0.0, 0.5)],              
    max_attempts = 100,                 
    collision_threshold = 0.3          
)
    # Berechne abgeleitete Parameter
    η_crystal = 1e4 * η_magma           
    ρ_crystal = ρ_magma + Δρ            
    
    # Create model
    model = Model(
        Grid(nel=(resolution[1]-1, resolution[2]-1), x=[-1,1], z=[-1,1]), 
        LaMEM.Time(nstep_max=1), # Von Time(nstep_max=1) zu LaMEM.Time(nstep_max=1) geändert
        Output(out_strain_rate=1)
    )
    
    # Define phases
    matrix = Phase(ID=0, Name="matrix", eta=η_magma, rho=ρ_magma)
    crystal = Phase(ID=1, Name="crystal", eta=η_crystal, rho=ρ_crystal)
    add_phase!(model, crystal, matrix)

    # Add crystals
    for i = 1:n_crystals
        current_radius = length(radius_crystal) >= i ? radius_crystal[i] : radius_crystal[1]
        
        if length(cen_2D) >= i
            current_center = cen_2D[i]
        else
            x_pos = rand(-0.6:0.1:0.6)
            z_pos = rand(0.2:0.1:0.8)
            current_center = (x_pos, z_pos)
        end
        
        add_sphere!(model, 
            cen=(current_center[1], 0.0, current_center[2]), 
            radius=current_radius, 
            phase=ConstantPhase(1)
        )
    end

    # Run LaMEM
    run_lamem(model, 1)
    data, _ = read_LaMEM_timestep(model, 1)

    # Extract data
    x_vec_1D = data.x.val[:,1,1]
    z_vec_1D = data.z.val[1,1,:]
    phase = data.fields.phase[:,1,:]
    Vx = data.fields.velocity[1][:,1,:]          
    Vz = data.fields.velocity[3][:,1,:]          
    Exx = data.fields.strain_rate[1][:,1,:]      
    Ezz = data.fields.strain_rate[9][:,1,:]      
    rho = data.fields.density[:,1,:]             
    log10eta = data.fields.visc_creep[:,1,:]     

    # Stokes velocity calculation
    ref_radius = radius_crystal[1]
    V_stokes = 2/9 * Δρ * 9.81 * (ref_radius * 1000)^2 / η_magma  
    V_stokes_cm_year = V_stokes * 100 * (3600*24*365.25)      
        
    return x_vec_1D, z_vec_1D, phase, Vx, Vz, Exx, Ezz, V_stokes_cm_year
end

"""
Generiert Samples mit verschiedenen Auflösungen
"""
function generate_mixed_resolution_dataset(n_samples; resolutions=[64, 128, 256], verbose=true)
    if verbose
        println("=== MIXED-RESOLUTION DATASET GENERATOR ===")
        println("Generiere $n_samples Samples")
        println("Auflösungen: $resolutions")
    end
    
    dataset = []
    stats = Dict(res => 0 for res in resolutions)
    
    for i in 1:n_samples
        resolution = rand(resolutions)
        stats[resolution] += 1
        
        n_crystals = rand(1:4)
        radius_crystal = [rand(0.03:0.005:0.08) for _ in 1:n_crystals]
        
        # Generate non-colliding centers
        centers = []
        for j in 1:n_crystals
            attempts = 0
            while attempts < 20
                x_pos = rand(-0.7:0.05:0.7)
                z_pos = rand(0.1:0.05:0.9)
                new_center = (x_pos, z_pos)
                
                collision = false
                for existing_center in centers
                    distance = sqrt((new_center[1] - existing_center[1])^2 + 
                                   (new_center[2] - existing_center[2])^2)
                    if distance < 0.15
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
            
            if length(centers) < j
                push!(centers, (rand(-0.3:0.1:0.3), rand(0.3:0.1:0.7)))
            end
        end
        
        if verbose && i % 50 == 1
            println("Sample $i/$n_samples: $(resolution)x$(resolution), $n_crystals Kristalle")
        end
        
        try
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
        
        if i % 10 == 0
            GC.gc()
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

println("LaMEM Interface Module geladen!")
println("Verfügbare Funktionen:")
println("  - LaMEM_Multi_crystal(...)")
println("  - generate_mixed_resolution_dataset(n_samples)")


# =============================================================================
# RESIDUAL LEARNING DATA PIPELINE
# =============================================================================
# Erweiterte Funktionen für lamem_interface.jl
# Füge diese zu deiner existierenden lamem_interface.jl hinzu

using Statistics
using ImageMorphology
using ImageFiltering

# =============================================================================
# KRISTALL-PARAMETER EXTRAKTION
# =============================================================================

"""
    CrystalParams

Struktur für Kristall-Parameter (für Stokes-Berechnung).

# Felder
- `x::Float32`: x-Position des Zentrums (normalisiert)
- `z::Float32`: z-Position des Zentrums (normalisiert)
- `radius::Float32`: Radius des Kristalls (normalisiert)
- `density::Float32`: Dichte-Kontrast Δρ
- `phase_id::Int`: Phase-ID im Phasenfeld (typisch: 1)
"""
struct CrystalParams
    x::Float32
    z::Float32
    radius::Float32
    density::Float32
    phase_id::Int
end

"""
    extract_crystal_params(phase_field; kwargs...)

Extrahiert Kristall-Parameter aus Phasenfeld.

# Argumente
- `phase_field`: 2D-Array mit Phase-IDs (0=Matrix, 1=Kristall)

# Keyword-Argumente
- `domain_size`: Tuple (x_min, x_max, z_min, z_max) (default: (-1.0, 1.0, -1.0, 1.0))
- `default_density_contrast`: Δρ wenn nicht bekannt (default: 200.0)
- `min_crystal_size`: Minimale Pixelanzahl für Kristall (default: 10)
- `verbose`: Debug-Ausgaben (default: false)

# Rückgabe
- `Vector{CrystalParams}`: Liste der extrahierten Kristalle

# Methodik
1. Finde zusammenhängende Regionen (Connected Components)
2. Berechne Zentrum jeder Region (Schwerpunkt)
3. Schätze Radius (äquivalenter Kreis-Radius)
4. Konvertiere zu physikalischen Koordinaten
"""
function extract_crystal_params(
    phase_field::AbstractMatrix{T};
    domain_size = (-1.0, 1.0, -1.0, 1.0),
    default_density_contrast = 200.0f0,
    min_crystal_size = 10,
    verbose = false
) where T
    
    H, W = size(phase_field)
    x_min, x_max, z_min, z_max = domain_size
    
    # Binäre Maske: wo ist Kristall?
    crystal_mask = phase_field .> 0.5
    
    # Connected Components Labeling
    labeled_crystals = label_components(crystal_mask)
    n_crystals = maximum(labeled_crystals)
    
    if verbose
        println("Gefundene Kristalle: $n_crystals")
    end
    
    crystals = CrystalParams[]
    
    for crystal_id in 1:n_crystals
        # Maske für diesen Kristall
        mask = labeled_crystals .== crystal_id
        n_pixels = sum(mask)
        
        # Filter zu kleine Regionen
        if n_pixels < min_crystal_size
            if verbose
                println("  Kristall $crystal_id zu klein ($n_pixels Pixel), überspringe")
            end
            continue
        end
        
        # Finde Indizes aller Pixel in diesem Kristall
        indices = findall(mask)
        
        # Berechne Schwerpunkt (Zentrum)
        center_i = mean(idx[1] for idx in indices)  # Zeilen-Index (z)
        center_j = mean(idx[2] for idx in indices)  # Spalten-Index (x)
        
        # Konvertiere Pixel-Koordinaten zu physikalischen Koordinaten
        # Pixel (1,1) entspricht (z_min, x_min)
        # Pixel (H,W) entspricht (z_max, x_max)
        x_phys = x_min + (center_j - 1) / (W - 1) * (x_max - x_min)
        z_phys = z_min + (center_i - 1) / (H - 1) * (z_max - z_min)
        
        # Schätze Radius: Äquivalenter Kreis-Radius
        # Area = π * r²  →  r = sqrt(Area / π)
        area_pixels = n_pixels
        radius_pixels = sqrt(area_pixels / π)
        
        # Konvertiere Radius zu physikalischen Einheiten
        # Durchschnittliche Pixel-Größe
        pixel_size_x = (x_max - x_min) / W
        pixel_size_z = (z_max - z_min) / H
        pixel_size = (pixel_size_x + pixel_size_z) / 2  # Durchschnitt
        
        radius_phys = radius_pixels * pixel_size
        
        # Erstelle CrystalParams
        crystal = CrystalParams(
            Float32(x_phys),
            Float32(z_phys),
            Float32(radius_phys),
            Float32(default_density_contrast),
            Int(crystal_id)
        )
        
        push!(crystals, crystal)
        
        if verbose
            println("  Kristall $crystal_id:")
            println("    Position: (x=$(round(x_phys, digits=3)), z=$(round(z_phys, digits=3)))")
            println("    Radius: $(round(radius_phys, digits=3))")
            println("    Pixels: $n_pixels")
        end
    end
    
    return crystals
end

"""
    extract_crystal_params_from_lamem(x_vec, z_vec, phase)

Alternative Extraktion wenn LaMEM-Daten direkt verfügbar sind.

# Argumente
- `x_vec`: x-Koordinaten Vektor
- `z_vec`: z-Koordinaten Vektor
- `phase`: 2D Phase-Array

# Rückgabe
- `Vector{CrystalParams}`: Liste der extrahierten Kristalle
"""
function extract_crystal_params_from_lamem(
    x_vec::AbstractVector,
    z_vec::AbstractVector,
    phase::AbstractMatrix;
    default_density_contrast = 200.0f0,
    min_crystal_size = 10,
    verbose = false
)
    # Domain-Größe aus Vektoren
    x_min, x_max = extrema(x_vec)
    z_min, z_max = extrema(z_vec)
    domain_size = (x_min, x_max, z_min, z_max)
    
    # Verwende Standard-Extraktion
    return extract_crystal_params(
        phase;
        domain_size = domain_size,
        default_density_contrast = default_density_contrast,
        min_crystal_size = min_crystal_size,
        verbose = verbose
    )
end

# =============================================================================
# RESIDUAL DATA PREPARATION
# =============================================================================

"""
    prepare_residual_training_data(lamem_data; kwargs...)

Bereitet Daten für Residual Learning vor.

# Argumente
- `lamem_data`: Tuple von LaMEM_Multi_crystal Output
  Format: (x_vec, z_vec, phase, Vx, Vz, Exx, Ezz, V_stokes)

# Keyword-Argumente
- `compute_stokes_field`: Funktion zum Berechnen von v_stokes (default: nothing)
- `verbose`: Debug-Ausgaben (default: false)

# Rückgabe
- Tuple: (phase_field, crystal_params, velocity_lamem, residual_preview)
  - `phase_field`: 2D Phase-Array [H, W]
  - `crystal_params`: Vector{CrystalParams}
  - `velocity_lamem`: 3D Velocity-Array [H, W, 2] (vx, vz)
  - `residual_preview`: Optional 3D Residual [H, W, 2] (wenn Stokes berechenbar)
"""
function prepare_residual_training_data(
    lamem_data;
    compute_stokes_field = nothing,
    verbose = false
)
    # Entpacke LaMEM-Daten
    x_vec, z_vec, phase, Vx, Vz, Exx, Ezz, V_stokes_scalar = lamem_data
    
    if verbose
        println("=== RESIDUAL DATA PREPARATION ===")
        println("Phase-Feld: $(size(phase))")
        println("Velocity: $(size(Vx)) / $(size(Vz))")
    end
    
    # 1. Extrahiere Kristall-Parameter
    crystal_params = extract_crystal_params_from_lamem(
        x_vec, z_vec, phase;
        verbose = verbose
    )
    
    if verbose
        println("Extrahierte Kristalle: $(length(crystal_params))")
    end
    
    # 2. Kombiniere Geschwindigkeiten zu einem Array
    H, W = size(phase)
    velocity_lamem = zeros(Float32, H, W, 2)
    velocity_lamem[:, :, 1] .= Float32.(Vx)
    velocity_lamem[:, :, 2] .= Float32.(Vz)
    
    # 3. Optional: Berechne Stokes-Feld und Residuum
    residual_preview = nothing
    
    if !isnothing(compute_stokes_field)
        try
            v_stokes = compute_stokes_field(phase, crystal_params)
            residual_preview = velocity_lamem .- v_stokes
            
            if verbose
                stokes_mag = mean(sqrt.(v_stokes[:,:,1].^2 .+ v_stokes[:,:,2].^2))
                residual_mag = mean(sqrt.(residual_preview[:,:,1].^2 .+ residual_preview[:,:,2].^2))
                ratio = residual_mag / stokes_mag * 100
                
                println("\nStokes vs. Residuum:")
                println("  Stokes Magnitude: $(round(stokes_mag, digits=6))")
                println("  Residual Magnitude: $(round(residual_mag, digits=6))")
                println("  Residual/Stokes: $(round(ratio, digits=2))%")
            end
        catch e
            if verbose
                println("Warnung: Stokes-Berechnung fehlgeschlagen: $e")
            end
        end
    end
    
    return phase, crystal_params, velocity_lamem, residual_preview
end

# =============================================================================
# BATCH CREATION FÜR RESIDUAL LEARNING
# =============================================================================

"""
    create_residual_batch(samples, target_resolution; kwargs...)

Erstellt Batch für Residual Learning Training.

# Argumente
- `samples`: Liste von Samples, jedes Sample ist Tuple:
  (phase, crystal_params, velocity_target, ...)
- `target_resolution`: Ziel-Auflösung (H, W)

# Keyword-Argumente
- `interpolation_method`: :nearest, :bilinear (default: :bilinear)
- `normalize_velocity`: Geschwindigkeiten normalisieren? (default: true)
- `return_crystal_params`: Gebe crystal_params zurück? (default: true)

# Rückgabe
Wenn `return_crystal_params=true`:
  - (phase_batch, crystal_params_batch, velocity_batch)
    - `phase_batch`: [H, W, 1, B]
    - `crystal_params_batch`: Vector{Vector{CrystalParams}} (länge B)
    - `velocity_batch`: [H, W, 2, B]
    
Wenn `return_crystal_params=false`:
  - (phase_batch, velocity_batch)
"""
function create_residual_batch(
    samples,
    target_resolution;
    interpolation_method = :bilinear,
    normalize_velocity = true,
    return_crystal_params = true
)
    batch_size = length(samples)
    H, W = target_resolution
    
    # Initialisiere Batch-Arrays
    phase_batch = zeros(Float32, H, W, 1, batch_size)
    velocity_batch = zeros(Float32, H, W, 2, batch_size)
    crystal_params_batch = Vector{Vector{CrystalParams}}(undef, batch_size)
    
    # Normalisierungs-Statistiken
    velocity_magnitudes = Float32[]
    
    for (b, sample) in enumerate(samples)
        # Entpacke Sample
        # Annahme: sample = (phase, crystal_params, velocity, ...)
        # Passe an deine Datenstruktur an!
        
        if length(sample) >= 3
            phase = sample[1]
            crystal_params = sample[2]
            velocity = sample[3]
        else
            # Fallback für alte Datenstruktur
            phase = sample[1]
            velocity = sample[end-1]  # Velocity oft vorletztes Element
            crystal_params = CrystalParams[]  # Leer wenn nicht verfügbar
        end
        
        # Resize Phase
        phase_resized = resize_phase_field(phase, (H, W), interpolation_method)
        phase_batch[:, :, 1, b] .= phase_resized
        
        # Resize Velocity
        velocity_resized = resize_velocity_field(velocity, (H, W), interpolation_method)
        velocity_batch[:, :, :, b] .= velocity_resized
        
        # Speichere Crystal Params
        crystal_params_batch[b] = crystal_params
        
        # Sammle Magnitude für Normalisierung
        if normalize_velocity
            mag = sqrt(mean(velocity_resized[:,:,1].^2 .+ velocity_resized[:,:,2].^2))
            push!(velocity_magnitudes, mag)
        end
    end
    
    # Optional: Normalisierung
    if normalize_velocity && length(velocity_magnitudes) > 0
        mean_mag = mean(velocity_magnitudes)
        if mean_mag > 1e-10  # Vermeide Division durch Null
            velocity_batch ./= mean_mag
        end
    end
    
    if return_crystal_params
        return phase_batch, crystal_params_batch, velocity_batch
    else
        return phase_batch, velocity_batch
    end
end

# =============================================================================
# HILFSFUNKTIONEN
# =============================================================================

"""
    resize_phase_field(phase, target_size, method)

Resize Phase-Feld mit geeigneter Interpolation.
"""
function resize_phase_field(
    phase::AbstractMatrix,
    target_size::Tuple{Int,Int},
    method::Symbol = :nearest
)
    H_src, W_src = size(phase)
    H_tgt, W_tgt = target_size
    
    if (H_src, W_src) == (H_tgt, W_tgt)
        return Float32.(phase)
    end
    
    # Einfaches Nearest-Neighbor Resampling
    # Für Production: verwende bessere Interpolation (z.B. ImageTransformations.jl)
    phase_resized = zeros(Float32, H_tgt, W_tgt)
    
    for i in 1:H_tgt, j in 1:W_tgt
        # Map zu Source-Koordinaten
        i_src = round(Int, (i - 1) / (H_tgt - 1) * (H_src - 1) + 1)
        j_src = round(Int, (j - 1) / (W_tgt - 1) * (W_src - 1) + 1)
        
        i_src = clamp(i_src, 1, H_src)
        j_src = clamp(j_src, 1, W_src)
        
        phase_resized[i, j] = phase[i_src, j_src]
    end
    
    return phase_resized
end

"""
    resize_velocity_field(velocity, target_size, method)

Resize Velocity-Feld mit Interpolation.
"""
function resize_velocity_field(
    velocity::AbstractArray{T,3},
    target_size::Tuple{Int,Int},
    method::Symbol = :bilinear
) where T
    H_src, W_src, C = size(velocity)
    H_tgt, W_tgt = target_size
    
    if (H_src, W_src) == (H_tgt, W_tgt)
        return Float32.(velocity)
    end
    
    velocity_resized = zeros(Float32, H_tgt, W_tgt, C)
    
    # Bilineare Interpolation
    for i in 1:H_tgt, j in 1:W_tgt
        # Map zu Source-Koordinaten (kontinuierlich)
        i_src = (i - 1) / (H_tgt - 1) * (H_src - 1) + 1
        j_src = (j - 1) / (W_tgt - 1) * (W_src - 1) + 1
        
        # Bilineare Interpolation
        i1 = floor(Int, i_src)
        i2 = min(i1 + 1, H_src)
        j1 = floor(Int, j_src)
        j2 = min(j1 + 1, W_src)
        
        α = i_src - i1
        β = j_src - j1
        
        for c in 1:C
            v11 = velocity[i1, j1, c]
            v12 = velocity[i1, j2, c]
            v21 = velocity[i2, j1, c]
            v22 = velocity[i2, j2, c]
            
            # Bilineare Interpolation
            v = (1-α) * (1-β) * v11 +
                (1-α) * β * v12 +
                α * (1-β) * v21 +
                α * β * v22
            
            velocity_resized[i, j, c] = v
        end
    end
    
    return velocity_resized
end

# =============================================================================
# DATASET GENERATION MIT RESIDUAL DATA
# =============================================================================

"""
    generate_residual_dataset(n_samples; kwargs...)

Generiert Dataset speziell für Residual Learning.

Wie generate_mixed_resolution_dataset aber gibt Daten im Format zurück:
(phase, crystal_params, velocity, x_vec, z_vec)
"""
function generate_residual_dataset(
    n_samples;
    resolutions = [64, 128, 256],
    crystal_range = 1:5,
    verbose = true
)
    if verbose
        println("=== RESIDUAL LEARNING DATASET GENERATION ===")
        println("Samples: $n_samples")
        println("Resolutions: $resolutions")
        println("Crystal Range: $crystal_range")
    end
    
    dataset = []
    
    for i in 1:n_samples
        resolution = rand(resolutions)
        n_crystals = rand(crystal_range)
        
        try
            # Generiere LaMEM-Sample
            lamem_data = LaMEM_Multi_crystal(
                resolution = (resolution, resolution),
                n_crystals = n_crystals
            )
            
            # Prepare für Residual Learning
            phase, crystal_params, velocity, _ = prepare_residual_training_data(
                lamem_data;
                verbose = false
            )
            
            # Erweiterte Sample-Struktur
            sample = (
                phase,
                crystal_params,
                velocity,
                lamem_data[1],  # x_vec
                lamem_data[2]   # z_vec
            )
            
            push!(dataset, sample)
            
            if verbose && i % 50 == 0
                println("Sample $i/$n_samples: $(resolution)x$(resolution), $n_crystals Kristalle, $(length(crystal_params)) extrahiert")
            end
            
        catch e
            if verbose && i % 50 == 0
                println("  Fehler bei Sample $i: $e")
            end
            continue
        end
        
        if i % 10 == 0
            GC.gc()
        end
    end
    
    if verbose
        println("\nDataset erfolgreich generiert: $(length(dataset)) Samples")
    end
    
    return dataset
end

# =============================================================================
# VISUALISIERUNG & DEBUGGING
# =============================================================================

"""
    visualize_crystal_extraction(phase, crystal_params)

Visualisiert extrahierte Kristall-Parameter.
Nützlich zum Debugging der Extraktion.
"""
function visualize_crystal_extraction(phase, crystal_params)
    println("=== CRYSTAL EXTRACTION VISUALIZATION ===")
    println("Phase Field Size: $(size(phase))")
    println("Extracted Crystals: $(length(crystal_params))")
    
    for (i, crystal) in enumerate(crystal_params)
        println("\nCrystal $i:")
        println("  Position: ($(crystal.x), $(crystal.z))")
        println("  Radius: $(crystal.radius)")
        println("  Density Contrast: $(crystal.density)")
        println("  Phase ID: $(crystal.phase_id)")
    end
end

println("Residual Learning Data Pipeline geladen!")
println("Neue Funktionen:")
println("  - extract_crystal_params(phase) → Vector{CrystalParams}")
println("  - prepare_residual_training_data(lamem_data)")
println("  - create_residual_batch(samples, resolution)")
println("  - generate_residual_dataset(n_samples)")