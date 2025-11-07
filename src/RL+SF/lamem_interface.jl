# =============================================================================
# LAMEM INTERFACE - CLEAN VERSION (256x256 only)
# =============================================================================
# Basiert auf funktionierender alter LaMEM API

using LaMEM
using GeophysicalModelGenerator
using Statistics

println("LaMEM Interface (Clean) wird geladen...")

# =============================================================================
# DATENSTRUKTUREN
# =============================================================================

"""
    CrystalParams

Kristall-Parameter für Stokes-Berechnung.
"""
struct CrystalParams
    center_x::Float64
    center_z::Float64
    radius::Float64
    density::Float64
end

# =============================================================================
# LAMEM MULTI-CRYSTAL SIMULATION
# =============================================================================

"""
    LaMEM_Multi_crystal(; kwargs...)

Generiert LaMEM Simulation mit mehreren Kristallen.
IMMER 256x256 Auflösung.

# Arguments
- `resolution::Tuple{Int,Int}`: Wird ignoriert - immer (256, 256)
- `n_crystals::Int`: Anzahl Kristalle (1-15 empfohlen)
- `radius_crystal::Vector{Float64}`: Radius pro Kristall
- `cen_2D::Vector{Tuple{Float64,Float64}}`: Zentren [(x,z), ...]
- `verbose::Bool`: Ausgabe aktivieren

# Returns
- `Tuple`: (x_vec, z_vec, Phase, Velocity.x, Velocity.z, v_stokes)
"""
function LaMEM_Multi_crystal(;
    resolution::Tuple{Int,Int} = (256, 256),
    n_crystals::Int = 1,
    radius_crystal::Vector{Float64} = [0.05],
    cen_2D::Vector{Tuple{Float64,Float64}} = [(0.0, 0.5)],
    verbose::Bool = false
)
    # ERZWINGE 256x256
    if resolution != (256, 256)
        verbose && println("WARNUNG: Auflösung $resolution wird ignoriert - verwende immer 256x256")
    end
    
    # Parameter-Validierung
    if n_crystals != length(radius_crystal)
        error("n_crystals ($n_crystals) != length(radius_crystal) ($(length(radius_crystal)))")
    end
    
    if n_crystals != length(cen_2D)
        error("n_crystals ($n_crystals) != length(cen_2D) ($(length(cen_2D)))")
    end
    
    # LaMEM Grid: nel = 255 → 256 Knoten
    nel = (255, 255)
    
    # Material-Eigenschaften
    η_magma = 1e21
    ρ_magma = 2800.0
    ρ_crystal = 3300.0
    η_crystal = 1e21
    
    matrix = Phase(
        ID=0,
        Name="Matrix",
        rho=ρ_magma,
        eta=η_magma,
        G=1e10
    )
    
    crystal = Phase(
        ID=1,
        Name="Crystal",
        rho=ρ_crystal,
        eta=η_crystal,
        G=1e10
    )
    
    # LaMEM Model erstellen (ALTE API)
    model = Model(
        Grid(nel=nel, x=[-1.0, 1.0], z=[-1.0, 1.0]),
        LaMEM.Time(nstep_max=1),
        Solver(SolverType="direct", DirectPenalty=1e5),
        BoundaryConditions(
            open_top_bound=1,
            permeable_phase_inflow=1
        ),
        FreeSurface(surf_use=0),
        Output(out_density=1, out_velocity=1)
    )
    
    # Füge Phases hinzu
    add_phase!(model, crystal, matrix)
    
    # Füge Kristalle hinzu
    for i in 1:n_crystals
        cen_x, cen_z = cen_2D[i]
        r = radius_crystal[i]
        
        add_sphere!(model, 
            cen=(cen_x, 0.0, cen_z), 
            radius=r, 
            phase=ConstantPhase(1))
    end
    
    # Run LaMEM
    verbose && println("LaMEM Simulation laeuft ($n_crystals Kristalle)...")
    
    run_lamem(model, 1)
    
    verbose && println("LaMEM Simulation abgeschlossen")
    
    # Load results (ALTE API)
    data, _ = read_LaMEM_timestep(model, 1)
    
    # Extract 2D data
    x_vec = data.x.val[:,1,1]
    z_vec = data.z.val[1,1,:]
    
    Phase_2D = data.fields.phase[:,1,:]'  # Transpose für korrekte Orientierung
    vx_2D = data.fields.velocity[1][:,1,:]'
    vz_2D = data.fields.velocity[3][:,1,:]'
    
    # Stokes velocity (placeholder)
    v_stokes = 0.01
    
    verbose && println("📊 Output: $(size(Phase_2D)) (sollte 256x256 sein)")
    
    return x_vec, z_vec, Phase_2D, vx_2D, vz_2D, v_stokes
end

# =============================================================================
# KRISTALL-PARAMETER EXTRAKTION
# =============================================================================

"""
    extract_crystal_params(phase_field, x_vec, z_vec)

Extrahiert Kristall-Parameter aus Phasenfeld.

WICHTIG: Phase ist kontinuierlich (0.0 - 1.0 = Phase-Fraktion)!
Wir nutzen einen Threshold um Kristalle zu identifizieren.
"""
function extract_crystal_params(
    phase_field::AbstractArray{T,2},
    x_vec::AbstractVector,
    z_vec::AbstractVector;
    crystal_threshold::Float64 = 0.5,  # Phase > 0.5 ist Kristall
    min_size::Int = 50
) where T
    
    # Binäre Maske mit Threshold (kontinuierliche Phase!)
    crystal_mask = phase_field .> crystal_threshold
    
    params = CrystalParams[]
    
    # Simple Blob Detection
    H, W = size(phase_field)
    visited = falses(H, W)
    
    for i in 1:H, j in 1:W
        if crystal_mask[i, j] && !visited[i, j]
            # Neue Komponente
            blob_pixels = find_connected_blob(crystal_mask, visited, i, j)
            
            if length(blob_pixels) >= min_size
                # Zentrum und Radius
                z_indices = [p[1] for p in blob_pixels]
                x_indices = [p[2] for p in blob_pixels]
                
                center_i = mean(z_indices)
                center_j = mean(x_indices)
                
                # Zu physikalischen Koordinaten
                center_x = x_vec[round(Int, center_j)]
                center_z = z_vec[round(Int, center_i)]
                
                # Radius
                radius = sqrt(length(blob_pixels) / π) * (x_vec[2] - x_vec[1])
                
                push!(params, CrystalParams(
                    center_x, center_z, radius, 3300.0
                ))
            end
        end
    end
    
    return params
end

"""
Flood-Fill Helper
"""
function find_connected_blob(mask, visited, start_i, start_j)
    H, W = size(mask)
    pixels = Tuple{Int,Int}[]
    stack = [(start_i, start_j)]
    
    while !isempty(stack)
        i, j = pop!(stack)
        
        if i < 1 || i > H || j < 1 || j > W
            continue
        end
        
        if visited[i, j] || !mask[i, j]
            continue
        end
        
        visited[i, j] = true
        push!(pixels, (i, j))
        
        # 4-connected
        push!(stack, (i+1, j))
        push!(stack, (i-1, j))
        push!(stack, (i, j+1))
        push!(stack, (i, j-1))
    end
    
    return pixels
end

# =============================================================================
# DATASET GENERATION
# =============================================================================

"""
    generate_dataset(n_samples; kwargs...)

Generiert Dataset für Residual Learning.
"""
function generate_dataset(
    n_samples::Int;
    crystal_range::UnitRange{Int} = 1:10,
    verbose::Bool = true
)
    verbose && println("\n=== DATASET GENERATION (256x256) ===")
    verbose && println("Samples: $n_samples")
    verbose && println("Kristall-Range: $crystal_range")
    
    dataset = []
    
    for i in 1:n_samples
        n_crystals = rand(crystal_range)
        
        # Random Konfiguration
        radius_crystal = [rand(0.03:0.01:0.07) for _ in 1:n_crystals]
        
        centers = Tuple{Float64,Float64}[]
        for j in 1:n_crystals
            attempts = 0
            while length(centers) < j && attempts < 100
                new_center = (rand(-0.3:0.1:0.3), rand(0.3:0.1:0.7))
                
                # Check Mindestabstand
                too_close = any(centers) do c
                    dist = sqrt((c[1] - new_center[1])^2 + (c[2] - new_center[2])^2)
                    dist < 0.15
                end
                
                if !too_close
                    push!(centers, new_center)
                    break
                end
                attempts += 1
            end
            
            # Fallback
            if length(centers) < j
                push!(centers, (rand(-0.3:0.1:0.3), rand(0.3:0.1:0.7)))
            end
        end
        
        try
            # LaMEM Simulation
            x_vec, z_vec, phase, vx, vz, v_stokes = LaMEM_Multi_crystal(
                n_crystals=n_crystals,
                radius_crystal=radius_crystal,
                cen_2D=centers,
                verbose=false
            )
            
            # Extrahiere Kristall-Parameter
            crystal_params = extract_crystal_params(phase, x_vec, z_vec)
            
            # Velocity Stack
            velocity = cat(vx, vz, dims=3)
            
            sample = (phase, crystal_params, velocity, x_vec, z_vec)
            push!(dataset, sample)
            
            if verbose && i % 10 == 0
                println("  Sample $i/$n_samples: $n_crystals Kristalle ✓")
            end
            
        catch e
            verbose && println("  Sample $i fehlgeschlagen: $e")
        end
        
        # Memory Management
        if i % 20 == 0
            GC.gc()
        end
    end
    
    verbose && println("\nDataset: $(length(dataset))/$n_samples erfolgreich generiert")
    
    return dataset
end

# =============================================================================
# MODUL-INFO
# =============================================================================

println("LaMEM Interface (Clean) geladen!")
println("   - Feste 256x256 Aufloesung")
println("   - Multi-Kristall Support (1-15)")
println("   - Kristall-Parameter Extraktion")
println("")
println("Wichtige Funktionen:")
println("   - LaMEM_Multi_crystal(...)")
println("   - extract_crystal_params(phase, x, z)")
println("   - generate_dataset(n_samples)")