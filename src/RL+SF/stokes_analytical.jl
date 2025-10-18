# =============================================================================
# ANALYTICAL STOKES SOLUTION MODULE - COMPLETE VERSION
# =============================================================================
# Analytische Stokes-Lösung für sinkende Kristalle in viskoser Matrix
# Basiert auf der klassischen Stokes-Lösung für sphärische Partikel

using Statistics
using LinearAlgebra

# =============================================================================
# PHYSIKALISCHE KONSTANTEN (kompatibel mit LaMEM)
# =============================================================================

const DEFAULT_ETA_MATRIX = 1e20      # Viskosität Matrix [Pa·s]
const DEFAULT_DELTA_RHO = 200.0      # Dichtedifferenz [kg/m³]
const DEFAULT_GRAVITY = 9.81         # Gravitationskonstante [m/s²]
const DOMAIN_SIZE_KM = 2.0          # Domain: -1 bis 1 (= 2 km)

# =============================================================================
# STOKES-GESCHWINDIGKEIT (Einzelner Kristall)
# =============================================================================

"""
    compute_stokes_settling_velocity(radius, Δρ, g, η)

Berechnet die Stokes-Sinkgeschwindigkeit eines sphärischen Kristalls.

# Physik
Stokes-Gesetz für eine Kugel in viskoser Flüssigkeit:
V_stokes = (2/9) × (Δρ × g × R²) / η

# Argumente
- `radius`: Kristall-Radius [m]
- `Δρ`: Dichtedifferenz [kg/m³]
- `g`: Gravitationskonstante [m/s²]
- `η`: Dynamische Viskosität der Matrix [Pa·s]

# Rückgabe
- Sinkgeschwindigkeit [m/s] (positiv = nach unten)
"""
function compute_stokes_settling_velocity(
    radius::Real,
    Δρ::Real = DEFAULT_DELTA_RHO,
    g::Real = DEFAULT_GRAVITY,
    η::Real = DEFAULT_ETA_MATRIX
)
    # Stokes-Formel
    V_stokes = (2.0 / 9.0) * (Δρ * g * radius^2) / η
    
    return Float64(V_stokes)
end

# =============================================================================
# GESCHWINDIGKEITSFELD UM EINZELNEN KRISTALL
# =============================================================================

"""
    analytical_stokes_field_single(x_grid, z_grid, crystal; kwargs...)

Berechnet das analytische Stokes-Strömungsfeld um einen einzelnen Kristall.

# Physikalischer Hintergrund
Stokes-Lösung für eine sinkende Kugel in unendlich ausgedehnter viskoser Flüssigkeit:

Geschwindigkeitskomponenten in sphärischen Koordinaten:
- u_r = V_stokes × cos(θ) × [1 - (3R)/(2r) + (R³)/(2r³)]
- u_θ = -V_stokes × sin(θ) × [1 - (3R)/(4r) - (R³)/(4r³)]

Wobei:
- r: Abstand zum Kristall-Zentrum
- θ: Polarwinkel (relativ zur Vertikalen)
- R: Kristall-Radius
- V_stokes: Stokes-Sinkgeschwindigkeit

# Argumente
- `x_grid`: 2D-Array mit x-Koordinaten [normalisiert]
- `z_grid`: 2D-Array mit z-Koordinaten [normalisiert]
- `crystal`: CrystalParams Struktur

# Keyword-Argumente
- `η_matrix`: Viskosität Matrix [Pa·s] (default: 1e20)
- `Δρ`: Dichtedifferenz [kg/m³] (default: 200.0)
- `g`: Gravitation [m/s²] (default: 9.81)
- `enforce_zero_inside`: Geschwindigkeit innerhalb Kristall = 0 (default: true)

# Rückgabe
- `(vx, vz)`: Tuple mit Geschwindigkeitskomponenten [m/s]
"""
function analytical_stokes_field_single(
    x_grid::AbstractMatrix,
    z_grid::AbstractMatrix,
    crystal::CrystalParams;
    η_matrix::Real = DEFAULT_ETA_MATRIX,
    Δρ::Real = DEFAULT_DELTA_RHO,
    g::Real = DEFAULT_GRAVITY,
    enforce_zero_inside::Bool = true
)
    H, W = size(x_grid)
    @assert size(z_grid) == (H, W) "x_grid und z_grid müssen gleiche Größe haben"
    
    # Initialisiere Geschwindigkeitsfelder
    vx = zeros(Float64, H, W)
    vz = zeros(Float64, H, W)
    
    # Konvertiere Kristall-Radius zu Metern
    # Annahme: Normalisierte Koordinaten entsprechen 2 km Domain
    R_m = crystal.radius * DOMAIN_SIZE_KM * 1000  # [m]
    
    # Berechne Stokes-Sinkgeschwindigkeit
    V_stokes = compute_stokes_settling_velocity(R_m, Δρ, g, η_matrix)
    
    # Für jeden Punkt im Gitter
    for i in 1:H
        for j in 1:W
            # Abstand zum Kristall-Zentrum (normalisiert)
            dx = x_grid[i, j] - crystal.x
            dz = z_grid[i, j] - crystal.z
            
            # Absolute Distanz (normalisiert)
            r_norm = sqrt(dx^2 + dz^2)
            
            # Vermeide Division durch Null
            if r_norm < 1e-10
                # Am Zentrum: keine Bewegung
                vx[i, j] = 0.0
                vz[i, j] = 0.0
                continue
            end
            
            # Prüfe ob innerhalb des Kristalls
            if enforce_zero_inside && r_norm <= crystal.radius
                # Innerhalb: starrer Körper, keine relative Bewegung
                vx[i, j] = 0.0
                vz[i, j] = 0.0
                continue
            end
            
            # Sphärische Koordinaten
            # θ: Winkel zur Vertikalen (z-Achse)
            # cos(θ) = dz / r
            # sin(θ) = dx / r
            cos_theta = dz / r_norm
            sin_theta = dx / r_norm
            
            # Dimensionslose Verhältnisse
            R_over_r = crystal.radius / r_norm
            R3_over_r3 = (crystal.radius / r_norm)^3
            
            # Stokes-Lösung in sphärischen Koordinaten
            # Radiale Komponente
            u_r = V_stokes * cos_theta * (1.0 - 1.5*R_over_r + 0.5*R3_over_r3)
            
            # Tangentiale Komponente (θ-Richtung)
            u_theta = -V_stokes * sin_theta * (1.0 - 0.75*R_over_r - 0.25*R3_over_r3)
            
            # Transformation zu kartesischen Koordinaten
            # Koordinatensystem: x horizontal, z vertikal (nach oben positiv)
            # 
            # Transformation:
            # vx = u_r × sin(θ) + u_θ × cos(θ)
            # vz = u_r × cos(θ) - u_θ × sin(θ)
            #
            # Wichtig: In unserem System ist z nach OBEN positiv,
            # daher ist das Vorzeichen von vz korrekt negativ für sinkende Kristalle
            
            vx[i, j] = u_r * sin_theta + u_theta * cos_theta
            vz[i, j] = u_r * cos_theta - u_theta * sin_theta
        end
    end
    
    return vx, vz
end

# =============================================================================
# SUPERPOSITION FÜR MEHRERE KRISTALLE
# =============================================================================

"""
    compute_stokes_velocity(phase_field, crystal_params; kwargs...)

Berechnet das vollständige Stokes-Geschwindigkeitsfeld durch Superposition
der Einzelfelder aller Kristalle.

# Prinzip
Das Gesamtfeld ist die lineare Superposition (Summe) der Einzelfelder,
da die Stokes-Gleichung linear ist:

v_total = Σ v_i(x, z)

wobei v_i das Feld des i-ten Kristalls ist.

# Argumente
- `phase_field`: 2D oder 3D Array mit Phasen-IDs
- `crystal_params`: Vector{CrystalParams} mit Kristall-Informationen

# Keyword-Argumente
- `η_matrix`: Viskosität Matrix [Pa·s] (default: 1e20)
- `Δρ`: Dichtedifferenz [kg/m³] (default: 200.0)
- `g`: Gravitation [m/s²] (default: 9.81)
- `verbose`: Debug-Ausgaben (default: false)

# Rückgabe
- 4D-Array [H, W, 2, 1] mit Geschwindigkeiten
  - [:, :, 1, 1]: vx (horizontal)
  - [:, :, 2, 1]: vz (vertikal)

# Hinweise
- Koordinaten werden automatisch aus phase_field-Größe generiert
- Domain: -1 bis 1 in x und z
- Output ist kompatibel mit UNet-Format
"""
function compute_stokes_velocity(
    phase_field::AbstractArray,
    crystal_params::Vector{CrystalParams};
    η_matrix::Real = DEFAULT_ETA_MATRIX,
    Δρ::Real = DEFAULT_DELTA_RHO,
    g::Real = DEFAULT_GRAVITY,
    verbose::Bool = false
)
    # Extrahiere Dimensionen
    H, W = size(phase_field)[1:2]
    
    if verbose
        println("=== STOKES VELOCITY COMPUTATION ===")
        println("Grid: $(H)×$(W)")
        println("Kristalle: $(length(crystal_params))")
        println("η_matrix: $(η_matrix) Pa·s")
        println("Δρ: $(Δρ) kg/m³")
    end
    
    # Erstelle Koordinaten-Gitter (normalisiert: -1 bis 1)
    x_range = range(-1.0, 1.0, length=W)
    z_range = range(-1.0, 1.0, length=H)
    
    x_grid = repeat(reshape(collect(x_range), 1, W), H, 1)
    z_grid = repeat(reshape(collect(z_range), H, 1), 1, W)
    
    # Initialisiere Gesamt-Geschwindigkeitsfelder
    vx_total = zeros(Float64, H, W)
    vz_total = zeros(Float64, H, W)
    
    # Superposition: Addiere Beiträge aller Kristalle
    for (i, crystal) in enumerate(crystal_params)
        if verbose
            println("  Kristall $i: pos=($(round(crystal.x, digits=3)), $(round(crystal.z, digits=3))), R=$(round(crystal.radius, digits=4))")
        end
        
        vx_single, vz_single = analytical_stokes_field_single(
            x_grid, z_grid, crystal;
            η_matrix = η_matrix,
            Δρ = Δρ,
            g = g
        )
        
        vx_total .+= vx_single
        vz_total .+= vz_single
    end
    
    # Konvertiere zu Float32 für Konsistenz mit UNet
    vx_f32 = Float32.(vx_total)
    vz_f32 = Float32.(vz_total)
    
    # Kombiniere zu 4D-Tensor: (H, W, 2, 1)
    velocity_field = zeros(Float32, H, W, 2, 1)
    velocity_field[:, :, 1, 1] .= vx_f32
    velocity_field[:, :, 2, 1] .= vz_f32
    
    if verbose
        v_mag = sqrt.(vx_f32.^2 .+ vz_f32.^2)
        println("  |v| mean: $(mean(v_mag)) m/s")
        println("  |v| max: $(maximum(v_mag)) m/s")
        println("  vx range: [$(minimum(vx_f32)), $(maximum(vx_f32))] m/s")
        println("  vz range: [$(minimum(vz_f32)), $(maximum(vz_f32))] m/s")
    end
    
    return velocity_field
end

# =============================================================================
# ALL-IN-ONE FUNKTION (für einfache Verwendung)
# =============================================================================

"""
    compute_stokes_from_phase(phase_field; kwargs...)

Vereinfachte All-in-One Funktion: Extrahiert Kristalle und berechnet Stokes-Feld.

Dies ist die einfachste Methode zur Nutzung - gibt Phasenfeld rein, bekommt
Stokes-Geschwindigkeitsfeld raus.

# Verwendung
```julia
v_stokes = compute_stokes_from_phase(phase_field, verbose=true)
```

# Argumente
- `phase_field`: 2D oder 3D Array mit Phase-IDs (0=Matrix, 1=Kristall)

# Keyword-Argumente
- `η_matrix`: Viskosität Matrix (default: 1e20)
- `Δρ`: Dichtedifferenz (default: 200.0)
- `g`: Gravitation (default: 9.81)
- `min_crystal_size`: Minimale Pixelanzahl für Kristall (default: 10)
- `verbose`: Debug-Ausgaben (default: false)

# Rückgabe
- 4D-Array [H, W, 2, 1] mit Stokes-Geschwindigkeiten
"""
function compute_stokes_from_phase(
    phase_field::AbstractArray;
    η_matrix::Real = DEFAULT_ETA_MATRIX,
    Δρ::Real = DEFAULT_DELTA_RHO,
    g::Real = DEFAULT_GRAVITY,
    min_crystal_size::Int = 10,
    verbose::Bool = false
)
    # Extrahiere Kristall-Parameter aus Phasenfeld
    # Nutze die Funktion aus lamem_interface.jl
    crystal_params = extract_crystal_params(
        phase_field;
        min_crystal_size = min_crystal_size,
        default_density_contrast = Float32(Δρ),
        verbose = verbose
    )
    
    if isempty(crystal_params)
        if verbose
            println("Warnung: Keine Kristalle gefunden!")
        end
        # Gebe Null-Feld zurück
        H, W = size(phase_field)[1:2]
        return zeros(Float32, H, W, 2, 1)
    end
    
    # Berechne Stokes-Feld
    return compute_stokes_velocity(
        phase_field,
        crystal_params;
        η_matrix = η_matrix,
        Δρ = Δρ,
        g = g,
        verbose = verbose
    )
end

# =============================================================================
# NORMALISIERUNG (für Training)
# =============================================================================

"""
    normalize_stokes_velocity(v_stokes, v_reference)

Normalisiert Stokes-Geschwindigkeitsfeld durch Referenzgeschwindigkeit.

Nützlich für Training, um Größenordnungen anzupassen.

# Argumente
- `v_stokes`: Stokes-Geschwindigkeitsfeld [H, W, 2, B]
- `v_reference`: Referenz-Geschwindigkeit (Skalar oder Array)

# Rückgabe
- Normalisiertes Geschwindigkeitsfeld
"""
function normalize_stokes_velocity(
    v_stokes::AbstractArray,
    v_reference::Real
)
    if v_reference < 1e-10
        @warn "Referenz-Geschwindigkeit zu klein, keine Normalisierung"
        return v_stokes
    end
    
    return v_stokes ./ Float32(v_reference)
end

# =============================================================================
# BATCH-VERARBEITUNG
# =============================================================================

"""
    compute_stokes_batch(phase_batch, crystal_params_batch; kwargs...)

Berechnet Stokes-Felder für einen ganzen Batch.

# Argumente
- `phase_batch`: 4D Array [H, W, 1, B] mit Phasenfeldern
- `crystal_params_batch`: Vector{Vector{CrystalParams}} (Länge B)

# Rückgabe
- 4D Array [H, W, 2, B] mit Stokes-Geschwindigkeiten für alle Batch-Elemente
"""
function compute_stokes_batch(
    phase_batch::AbstractArray{T, 4},
    crystal_params_batch::Vector{Vector{CrystalParams}};
    η_matrix::Real = DEFAULT_ETA_MATRIX,
    Δρ::Real = DEFAULT_DELTA_RHO,
    g::Real = DEFAULT_GRAVITY,
    verbose::Bool = false
) where T
    H, W, C, B = size(phase_batch)
    @assert C == 1 "Phase batch sollte 1 Kanal haben"
    @assert length(crystal_params_batch) == B "Crystal params batch Länge muss zu Batch-Size passen"
    
    # Initialisiere Output
    v_stokes_batch = zeros(Float32, H, W, 2, B)
    
    # Berechne für jedes Batch-Element
    for b in 1:B
        phase_2d = phase_batch[:, :, 1, b]
        crystal_params = crystal_params_batch[b]
        
        v_stokes = compute_stokes_velocity(
            phase_2d,
            crystal_params;
            η_matrix = η_matrix,
            Δρ = Δρ,
            g = g,
            verbose = false
        )
        
        v_stokes_batch[:, :, :, b] .= v_stokes[:, :, :, 1]
    end
    
    if verbose
        println("Batch Stokes computation: $(B) samples processed")
    end
    
    return v_stokes_batch
end

# =============================================================================
# TEST & VALIDATION
# =============================================================================

"""
    test_stokes_analytical(resolution=128)

Testet die Stokes-Implementation mit einem synthetischen Fall.
"""
function test_stokes_analytical(resolution::Int = 128)
    println("="^60)
    println("TEST: ANALYTISCHE STOKES-LÖSUNG")
    println("="^60)
    
    # Erstelle Test-Phasenfeld mit 2 Kristallen
    phase_field = zeros(Float64, resolution, resolution)
    
    # Kristall 1: Oben (x=0.0, z=0.3)
    # Kristall 2: Unten (x=0.0, z=-0.3)
    for i in 1:resolution
        for j in 1:resolution
            x = -1.0 + (j - 1) / (resolution - 1) * 2.0
            z = -1.0 + (i - 1) / (resolution - 1) * 2.0
            
            # Kristall 1
            if sqrt((x - 0.0)^2 + (z - 0.3)^2) <= 0.1
                phase_field[i, j] = 1.0
            end
            
            # Kristall 2
            if sqrt((x - 0.0)^2 + (z + 0.3)^2) <= 0.1
                phase_field[i, j] = 1.0
            end
        end
    end
    
    println("\n1. KRISTALL-EXTRAKTION")
    crystal_params = extract_crystal_params(phase_field, verbose=true)
    
    println("\n2. STOKES-FELD BERECHNUNG")
    v_stokes = compute_stokes_from_phase(phase_field, verbose=true)
    
    println("\n3. VALIDIERUNG")
    println("Shape: $(size(v_stokes))")
    
    # Prüfe ob Feld physikalisch sinnvoll ist
    v_mag = sqrt.(v_stokes[:,:,1,1].^2 .+ v_stokes[:,:,2,1].^2)
    println("Geschwindigkeits-Magnitude:")
    println("  Mean: $(mean(v_mag)) m/s")
    println("  Max: $(maximum(v_mag)) m/s")
    println("  Min: $(minimum(v_mag)) m/s")
    
    # Prüfe Sinkrichtung (vz sollte überwiegend negativ sein)
    vz_mean = mean(v_stokes[:,:,2,1])
    println("\nSinkrichtung:")
    println("  Mean vz: $(vz_mean) m/s")
    println("  $(vz_mean < 0 ? "✓" : "✗") Kristalle sinken nach unten")
    
    # Prüfe Symmetrie (vx sollte nahe 0 sein für zentrierte Kristalle)
    vx_mean = mean(abs.(v_stokes[:,:,1,1]))
    println("  Mean |vx|: $(vx_mean) m/s")
    println("  $(vx_mean < 1e-6 ? "✓" : "○") Feld ist symmetrisch")
    
    println("\n" * "="^60)
    println(vz_mean < 0 ? "✓ TEST BESTANDEN" : "✗ TEST FEHLGESCHLAGEN")
    println("="^60)
    
    return v_stokes, crystal_params
end

"""
    validate_stokes_against_lamem(lamem_data; tolerance=0.5)

Validiert Stokes-Berechnung gegen LaMEM-Daten.

# Rückgabe
- `true` wenn relative Differenz < tolerance
"""
function validate_stokes_against_lamem(
    lamem_data;
    tolerance::Real = 0.5,
    verbose::Bool = true
)
    # Entpacke LaMEM-Daten
    x_vec, z_vec, phase, Vx_lamem, Vz_lamem, Exx, Ezz, V_stokes_scalar = lamem_data
    
    # Berechne Stokes-Feld
    crystal_params = extract_crystal_params_from_lamem(x_vec, z_vec, phase)
    v_stokes = compute_stokes_velocity(phase, crystal_params)
    
    # Vergleiche mit LaMEM
    vx_stokes = v_stokes[:, :, 1, 1]
    vz_stokes = v_stokes[:, :, 2, 1]
    
    # Berechne Fehler
    vx_diff = abs.(vx_stokes .- Vx_lamem)
    vz_diff = abs.(vz_stokes .- Vz_lamem)
    
    mean_vx_error = mean(vx_diff)
    mean_vz_error = mean(vz_diff)
    
    # Relative Fehler
    vx_mag = mean(abs.(Vx_lamem))
    vz_mag = mean(abs.(Vz_lamem))
    
    rel_vx_error = vx_mag > 0 ? mean_vx_error / vx_mag : 0.0
    rel_vz_error = vz_mag > 0 ? mean_vz_error / vz_mag : 0.0
    
    if verbose
        println("=== STOKES vs. LaMEM VALIDATION ===")
        println("Absolute Fehler:")
        println("  vx: $(round(mean_vx_error, digits=8)) m/s")
        println("  vz: $(round(mean_vz_error, digits=8)) m/s")
        println("Relative Fehler:")
        println("  vx: $(round(rel_vx_error*100, digits=2))%")
        println("  vz: $(round(rel_vz_error*100, digits=2))%")
        println("\nValidierung: $(max(rel_vx_error, rel_vz_error) < tolerance ? "✓ PASS" : "✗ FAIL")")
    end
    
    return max(rel_vx_error, rel_vz_error) < tolerance
end

# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

println("Analytical Stokes Solution Module geladen!")
println("Verfügbare Funktionen:")
println("  - compute_stokes_from_phase(phase) → v_stokes [einfachste Nutzung]")
println("  - compute_stokes_velocity(phase, crystal_params) → v_stokes")
println("  - compute_stokes_batch(phase_batch, crystal_params_batch)")
println("  - test_stokes_analytical() → Test-Funktion")
println("  - validate_stokes_against_lamem(lamem_data)")
println("")
println("Quick Start:")
println("  v_stokes = compute_stokes_from_phase(phase_field, verbose=true)")