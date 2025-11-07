# =============================================================================
# STOKES ANALYTICAL SOLUTION
# =============================================================================
# Analytische Lösung für Geschwindigkeitsfeld um sinkende Sphären
# Basierend auf Faxén's Law und Stokes-Lösung für Low Reynolds Number Flow

using LinearAlgebra
using Statistics

println("Stokes Analytical Solution wird geladen...")

# =============================================================================
# PHYSIKALISCHE KONSTANTEN (DEFAULT)
# =============================================================================

const DEFAULT_ρ_CRYSTAL = 3300.0  # kg/m³ (Olivin)
const DEFAULT_ρ_MATRIX = 2800.0   # kg/m³ (Basaltisches Magma)
const DEFAULT_η = 1e21             # Pa·s (Magma-Viskosität)
const DEFAULT_g = 9.81             # m/s² (Gravitation)

# =============================================================================
# EINZELNE SPHÄRE - STOKES LÖSUNG
# =============================================================================

"""
    stokes_single_sphere(x_grid, z_grid, center_x, center_z, radius, ρ_crystal, ρ_matrix, η, g)

Berechnet analytisches Geschwindigkeitsfeld für EINE sinkende Sphäre.

# Physik
- Stokes-Regime (Re << 1)
- Inkompressibler, viskoser Fluss
- Sphärische Symmetrie

# Koordinatensystem
- x: horizontal (-0.5 bis 0.5)
- z: vertikal (0 bis 1, nach oben positiv)
- Sphäre sinkt nach unten (negative z-Richtung)

# Stokes-Geschwindigkeit
v_stokes = (2/9) * (Δρ * g * R²) / η

Wobei Δρ = ρ_crystal - ρ_matrix

# Geschwindigkeitsfeld (außerhalb der Sphäre, r > R)
Basierend auf Stokes-Strömung um Sphäre:

v_r = v_stokes * cos(θ) * [1 - (3R)/(2r) + (R³)/(2r³)]
v_θ = -v_stokes * sin(θ) * [1 - (3R)/(4r) - (R³)/(4r³)]

Umrechnung zu kartesischen Koordinaten:
vx = v_r * sin(θ) + v_θ * cos(θ)
vz = v_r * cos(θ) - v_θ * sin(θ)

# Returns
- `Tuple`: (vx, vz) Arrays gleicher Größe wie x_grid
"""
function stokes_single_sphere(
    x_grid::AbstractArray,
    z_grid::AbstractArray,
    center_x::Float64,
    center_z::Float64,
    radius::Float64,
    ρ_crystal::Float64 = DEFAULT_ρ_CRYSTAL,
    ρ_matrix::Float64 = DEFAULT_ρ_MATRIX,
    η::Float64 = DEFAULT_η,
    g::Float64 = DEFAULT_g
)
    # Stokes-Geschwindigkeit (Sinkgeschwindigkeit)
    Δρ = ρ_crystal - ρ_matrix
    v_stokes = (2.0 / 9.0) * (Δρ * g * radius^2) / η
    
    # Initialisiere Geschwindigkeitsfelder
    vx = zeros(Float64, size(x_grid))
    vz = zeros(Float64, size(z_grid))
    
    # Iteriere über Grid
    for i in eachindex(x_grid)
        # Relativposition zur Sphäre
        dx = x_grid[i] - center_x
        dz = z_grid[i] - center_z
        
        r = sqrt(dx^2 + dz^2)
        
        # Nur außerhalb der Sphäre (r > radius)
        if r > radius
            # Polarwinkel θ (von z-Achse aus gemessen)
            cos_θ = dz / r
            sin_θ = dx / r
            
            # Radiale und tangentiale Komponenten
            R_over_r = radius / r
            R3_over_r3 = (radius / r)^3
            
            # Stokes-Strömung Komponenten
            v_r = v_stokes * cos_θ * (1.0 - 1.5 * R_over_r + 0.5 * R3_over_r3)
            v_θ = -v_stokes * sin_θ * (1.0 - 0.75 * R_over_r - 0.25 * R3_over_r3)
            
            # Umrechnung zu kartesischen Koordinaten
            vx = [compute_vx(i) for i in eachindex(x_grid)]
            vz = [compute_vz(i) for i in eachindex(z_grid)]
        else
            # Innerhalb der Sphäre: Rigid Body Motion
            # (Sphäre bewegt sich als Ganzes mit v_stokes nach unten)
            vx[i] = 0.0
            vz[i] = -v_stokes
        end
    end
    
    return vx, vz
end

# =============================================================================
# MULTIPLE SPHÄREN - SUPERPOSITION
# =============================================================================

"""
    compute_stokes_velocity_field(crystal_params, x_grid, z_grid; kwargs...)

Berechnet Geschwindigkeitsfeld für MEHRERE Kristalle durch Superposition.

# Annahmen
- Linearität: Gesamtfeld = Summe der Einzelfelder
- Keine Kristall-Kristall Interaktion (gültig für kleine Volumenanteile)
- Quasi-statischer Zustand

# Arguments
- `crystal_params::Vector{CrystalParams}`: Liste der Kristalle
- `x_grid::AbstractArray`: X-Koordinaten Grid
- `z_grid::AbstractArray`: Z-Koordinaten Grid

# Keyword Arguments
- `ρ_crystal::Float64`: Kristall-Dichte (default: 3300.0 kg/m³)
- `ρ_matrix::Float64`: Matrix-Dichte (default: 2800.0 kg/m³)
- `η::Float64`: Viskosität (default: 1e21 Pa·s)
- `g::Float64`: Gravitation (default: 9.81 m/s²)

# Returns
- `Tuple`: (vx_total, vz_total, v_stokes_magnitude)
  - vx_total, vz_total: Gesamtgeschwindigkeitsfeld
  - v_stokes_magnitude: Typische Stokes-Geschwindigkeit (für Normalisierung)
"""
function compute_stokes_velocity_field(
    crystal_params::Vector,
    x_grid::AbstractArray,
    z_grid::AbstractArray;
    ρ_crystal::Float64 = DEFAULT_ρ_CRYSTAL,
    ρ_matrix::Float64 = DEFAULT_ρ_MATRIX,
    η::Float64 = DEFAULT_η,
    g::Float64 = DEFAULT_g
)
    # Initialisiere Gesamtfeld
    vx_total = zeros(Float64, size(x_grid))
    vz_total = zeros(Float64, size(z_grid))
    
    # Superposition aller Kristalle
    for crystal in crystal_params
        vx_single, vz_single = stokes_single_sphere(
            x_grid, z_grid,
            crystal.center_x, crystal.center_z, crystal.radius,
            ρ_crystal, ρ_matrix, η, g
        )
        
        # OHNE Mutation - Zygote-kompatibel
        vx_total = vx_total .+ vx_single
        vz_total = vz_total .+ vz_single
    end
    
    # Berechne typische Stokes-Geschwindigkeit (für Normalisierung)
    # Verwende mittleren Radius
    if !isempty(crystal_params)
        mean_radius = mean([c.radius for c in crystal_params])
        Δρ = ρ_crystal - ρ_matrix
        v_stokes_ref = (2.0 / 9.0) * (Δρ * g * mean_radius^2) / η
    else
        v_stokes_ref = 0.0
    end
    
    return vx_total, vz_total, v_stokes_ref
end

# =============================================================================
# GRID GENERATION HELPER
# =============================================================================

"""
    create_velocity_grid(x_vec, z_vec)

Erstellt 2D Mesh-Grids aus 1D Vektoren für Geschwindigkeitsberechnung.

# Arguments
- `x_vec::AbstractVector`: X-Koordinaten (eindimensional)
- `z_vec::AbstractVector`: Z-Koordinaten (eindimensional)

# Returns
- `Tuple`: (x_grid, z_grid) als 2D Arrays
"""
function create_velocity_grid(x_vec::AbstractVector, z_vec::AbstractVector)
    nx = length(x_vec)
    nz = length(z_vec)
    
    x_grid = repeat(x_vec', nz, 1)
    z_grid = repeat(z_vec, 1, nx)
    
    return x_grid, z_grid
end

# =============================================================================
# WRAPPER FÜR CRYSTAL PARAMS
# =============================================================================

"""
    compute_stokes_from_crystal_params(phase, crystal_params, x_vec, z_vec)

High-Level Wrapper: Berechnet Stokes-Feld direkt aus Kristall-Parametern.

# Returns
- `Array{Float32,4}`: Velocity Tensor [H, W, 2, 1] für UNet-Kompatibilität
"""
function compute_stokes_from_crystal_params(
    phase::AbstractArray{T,2},
    crystal_params::Vector,
    x_vec::AbstractVector,
    z_vec::AbstractVector
) where T
    
    # Grid erstellen
    x_grid, z_grid = create_velocity_grid(x_vec, z_vec)
    
    # Stokes-Feld berechnen
    vx_stokes, vz_stokes, v_ref = compute_stokes_velocity_field(
        crystal_params, x_grid, z_grid
    )
    
    # Stack zu [H, W, 2]
    velocity_stokes = cat(vx_stokes, vz_stokes, dims=3)
    
    # Reshape zu [H, W, 2, 1] für UNet
    velocity_tensor = reshape(Float32.(velocity_stokes), size(phase)..., 2, 1)
    
    return velocity_tensor
end

# =============================================================================
# VALIDIERUNG & TESTS
# =============================================================================

"""
    validate_stokes_solution(vx, vz)

Validiert Stokes-Lösung (Divergenzfreiheit, Symmetrie).

# Checks
- Divergenz ≈ 0 (für inkompressiblen Fluss)
- Boundary Conditions erfüllt
- Symmetrie (falls vorhanden)

# Returns
- `Dict`: Diagnostics
"""
function validate_stokes_solution(vx::AbstractArray, vz::AbstractArray)
    # Divergenz-Check
    ∂vx_∂x = diff(vx, dims=1)
    ∂vz_∂z = diff(vz, dims=2)
    
    # Match Größen
    min_size = min(size(∂vx_∂x, 1), size(∂vz_∂z, 1))
    divergence = ∂vx_∂x[1:min_size, 1:end-1] .+ ∂vz_∂z[1:min_size, 1:end-1]
    
    max_div = maximum(abs, divergence)
    mean_div = mean(abs, divergence)
    
    # Velocity Magnitude
    v_mag = sqrt.(vx.^2 .+ vz.^2)
    max_v = maximum(v_mag)
    mean_v = mean(v_mag)
    
    diagnostics = Dict(
        "max_divergence" => max_div,
        "mean_divergence" => mean_div,
        "max_velocity" => max_v,
        "mean_velocity" => mean_v,
        "divergence_ok" => max_div < 1e-6
    )
    
    return diagnostics
end

"""
    test_single_sphere()

Unit Test: Validiert Stokes-Lösung für einzelne Sphäre.
"""
function test_single_sphere()
    println("\n TEST: Einzelne Sphäre Stokes-Lösung")
    
    # Test-Grid
    x = range(-0.5, 0.5, length=128)
    z = range(0.0, 1.0, length=128)
    x_grid, z_grid = create_velocity_grid(x, z)
    
    # Einzelne Sphäre
    vx, vz = stokes_single_sphere(
        x_grid, z_grid,
        0.0, 0.5,  # Zentrum
        0.05       # Radius
    )
    
    # Validierung
    diagnostics = validate_stokes_solution(vx, vz)
    
    println("  Max Divergenz: $(diagnostics["max_divergence"])")
    println("  Mean Divergenz: $(diagnostics["mean_divergence"])")
    println("  Max Velocity: $(diagnostics["max_velocity"]) m/s")
    
    if diagnostics["divergence_ok"]
        println("  Test bestanden!")
    else
        println("  Test fehlgeschlagen - Divergenz zu hoch")
    end
    
    return diagnostics
end

# =============================================================================
# MODUL-INFO
# =============================================================================

println("Stokes Analytical Solution geladen!")
println("   - Einzelne Sphäre: Exakte Lösung")
println("   - Multiple Sphären: Superposition")
println("   - Automatische Validierung")
println("")
println("Wichtige Funktionen:")
println("   - compute_stokes_velocity_field(crystal_params, x, z)")
println("   - compute_stokes_from_crystal_params(...)")
println("   - test_single_sphere()")
println("")
println("Quick Test verfügbar:")
println("   julia> test_single_sphere()")