# =============================================================================
# ANALYTICAL STOKES SOLUTION MODULE
# =============================================================================

using Statistics
using LinearAlgebra

"""
Parameter für einen einzelnen Kristall
"""
struct CrystalParams
    x::Float64          # x-Position (normalisiert, -1 bis 1)
    z::Float64          # z-Position (normalisiert, -1 bis 1)
    radius::Float64     # Radius (normalisiert)
    density::Float64    # Dichte oder Dichtedifferenz
end

"""
Extrahiert Kristallparameter aus einem Phasenfeld
"""
function extract_crystal_params(phase_field::AbstractArray; 
                               min_crystal_size=50,
                               domain_size=2.0)  # Domain von -1 bis 1
    
    # Finde Kristall-Bereiche
    crystal_mask = phase_field .> 0.5
    
    if sum(crystal_mask) == 0
        return CrystalParams[]  # Keine Kristalle gefunden
    end
    
    # Connected Components Analysis
    labeled_regions, num_regions = find_connected_components(crystal_mask)
    
    crystal_params = CrystalParams[]
    resolution = size(phase_field, 1)
    
    for region_id in 1:num_regions
        # Finde alle Pixel dieser Region
        region_mask = labeled_regions .== region_id
        region_size = sum(region_mask)
        
        # Filtere zu kleine Regionen
        if region_size < min_crystal_size
            continue
        end
        
        # Berechne Schwerpunkt
        indices = findall(region_mask)
        
        if isempty(indices)
            continue
        end
        
        # Schwerpunkt in Pixel-Koordinaten
        center_y_px = mean([idx[1] for idx in indices])
        center_x_px = mean([idx[2] for idx in indices])
        
        # Konvertiere zu normalisierten Koordinaten (-1 bis 1)
        x_norm = -1.0 + (center_x_px / resolution) * domain_size
        z_norm = -1.0 + (center_y_px / resolution) * domain_size
        
        # Schätze Radius aus Fläche (Annahme: kreisförmig)
        area_px = region_size
        radius_px = sqrt(area_px / π)
        radius_norm = (radius_px / resolution) * domain_size
        
        # Dichte aus Phasenfeld (vereinfacht: Mittelwert der Phase-Werte)
        phase_values = phase_field[region_mask]
        avg_density = mean(phase_values)
        
        push!(crystal_params, CrystalParams(x_norm, z_norm, radius_norm, avg_density))
    end
    
    return crystal_params
end

"""
Vereinfachte Connected Components Analysis (aus evaluate_model.jl übernommen)
"""
function find_connected_components(binary_mask)
    H, W = size(binary_mask)
    labeled = zeros(Int, H, W)
    current_label = 0
    
    # 4-connected neighbors
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for i in 1:H
        for j in 1:W
            if binary_mask[i, j] && labeled[i, j] == 0
                current_label += 1
                
                # Flood fill
                stack = [(i, j)]
                labeled[i, j] = current_label
                
                while !isempty(stack)
                    y, x = pop!(stack)
                    
                    for (dy, dx) in neighbors
                        ny, nx = y + dy, x + dx
                        
                        if 1 <= ny <= H && 1 <= nx <= W
                            if binary_mask[ny, nx] && labeled[ny, nx] == 0
                                labeled[ny, nx] = current_label
                                push!(stack, (ny, nx))
                            end
                        end
                    end
                end
            end
        end
    end
    
    return labeled, current_label
end

"""
Analytische Stokes-Lösung für einen einzelnen sphärischen Kristall
Basierend auf der klassischen Stokes-Lösung für eine sinkende Kugel in viskoser Flüssigkeit
"""
function analytical_stokes_field_single(x_grid, z_grid, crystal::CrystalParams;
                                       η_matrix=1e20, Δρ=200.0, g=9.81)
    
    H, W = size(x_grid)
    vx = zeros(Float64, H, W)
    vz = zeros(Float64, H, W)
    
    # Stokes-Geschwindigkeit des Kristalls (Sinkgeschwindigkeit)
    # V_stokes = (2/9) * (Δρ * g * R²) / η
    R = crystal.radius * 1000  # Konvertiere zu Metern (angenommen Domain ist in km)
    V_stokes = (2.0/9.0) * (Δρ * g * R^2) / η_matrix
    
    # Für jeden Punkt im Gitter
    for i in 1:H
        for j in 1:W
            # Abstand zum Kristall-Zentrum
            dx = x_grid[i, j] - crystal.x
            dz = z_grid[i, j] - crystal.z
            r = sqrt(dx^2 + dz^2)
            
            # Vermeide Division durch Null am Kristall-Zentrum
            if r < 1e-10
                continue
            end
            
            # Normalisierte Radialrichtung
            r_norm = r / crystal.radius
            
            # Stokes-Lösung für Geschwindigkeitsfeld um sinkende Kugel
            # Innerhalb der Kugel: starre Körper-Rotation (vereinfacht: 0)
            if r <= crystal.radius
                # Im Kristall: keine Geschwindigkeit (starrer Körper)
                vx[i, j] = 0.0
                vz[i, j] = 0.0
            else
                # Außerhalb: Stokes-Lösung
                # Vereinfachte axisymmetrische Lösung in 2D
                
                # Radiale und vertikale Komponenten
                cos_theta = dz / r  # Winkel zur Vertikalen
                sin_theta = dx / r
                
                # Stokes-Strömungsfeld (vereinfachte 2D-Approximation)
                # Basierend auf: u_r = V * cos(θ) * [1 - (3R)/(2r) + (R³)/(2r³)]
                #                u_θ = -V * sin(θ) * [1 - (3R)/(4r) - (R³)/(4r³)]
                
                R_over_r = crystal.radius / r
                R3_over_r3 = (crystal.radius / r)^3
                
                # Radiale Komponente
                u_r = V_stokes * cos_theta * (1.0 - 1.5*R_over_r + 0.5*R3_over_r3)
                
                # Tangentiale Komponente
                u_theta = -V_stokes * sin_theta * (1.0 - 0.75*R_over_r - 0.25*R3_over_r3)
                
                # Konvertiere zu kartesischen Koordinaten
                # vx = u_r * sin(θ) + u_θ * cos(θ)
                # vz = u_r * cos(θ) - u_θ * sin(θ)
                vx[i, j] = u_r * sin_theta + u_theta * cos_theta
                vz[i, j] = u_r * cos_theta - u_theta * sin_theta
            end
        end
    end
    
    return vx, vz
end

"""
Berechnet das vollständige Stokes-Geschwindigkeitsfeld für mehrere Kristalle
durch Superposition der Einzelfelder
"""
function compute_stokes_velocity(phase_field::AbstractArray, 
                                crystal_params::Vector{CrystalParams};
                                η_matrix=1e20, Δρ=200.0, g=9.81)
    
    H, W = size(phase_field)[1:2]
    
    # Erstelle Koordinaten-Gitter (normalisiert von -1 bis 1)
    x_range = range(-1.0, 1.0, length=W)
    z_range = range(-1.0, 1.0, length=H)
    
    x_grid = repeat(reshape(x_range, 1, W), H, 1)
    z_grid = repeat(reshape(z_range, H, 1), 1, W)
    
    # Initialisiere Gesamt-Geschwindigkeitsfelder
    vx_total = zeros(Float64, H, W)
    vz_total = zeros(Float64, H, W)
    
    # Superposition: Addiere Beiträge aller Kristalle
    for crystal in crystal_params
        vx_single, vz_single = analytical_stokes_field_single(
            x_grid, z_grid, crystal,
            η_matrix=η_matrix, Δρ=Δρ, g=g
        )
        
        vx_total .+= vx_single
        vz_total .+= vz_single
    end
    
    # Konvertiere zu Float32 für Konsistenz mit UNet
    vx_total_f32 = Float32.(vx_total)
    vz_total_f32 = Float32.(vz_total)
    
    # Kombiniere zu 4D-Tensor: (H, W, 2, 1)
    velocity_field = cat(vx_total_f32, vz_total_f32, dims=3)
    velocity_field = reshape(velocity_field, H, W, 2, 1)
    
    return velocity_field
end

"""
Vereinfachte Version: Extrahiert Kristallparameter und berechnet Stokes-Feld direkt
"""
function compute_stokes_from_phase(phase_field::AbstractArray;
                                  η_matrix=1e20, Δρ=200.0, g=9.81,
                                  verbose=false)
    
    # Extrahiere Kristallparameter
    crystal_params = extract_crystal_params(phase_field)
    
    if verbose
        println("Extrahierte Kristalle: $(length(crystal_params))")
        for (i, crystal) in enumerate(crystal_params)
            println("  Kristall $i: pos=($(round(crystal.x, digits=2)), $(round(crystal.z, digits=2))), R=$(round(crystal.radius, digits=3))")
        end
    end
    
    # Berechne Stokes-Feld
    if isempty(crystal_params)
        H, W = size(phase_field)[1:2]
        return zeros(Float32, H, W, 2, 1)
    end
    
    return compute_stokes_velocity(phase_field, crystal_params,
                                  η_matrix=η_matrix, Δρ=Δρ, g=g)
end

"""
Test-Funktion für Stokes-Berechnung
"""
function test_stokes_analytical()
    println("=== TEST: ANALYTISCHE STOKES-LÖSUNG ===")
    
    # Erstelle Test-Phasenfeld mit 2 Kristallen
    resolution = 128
    phase_field = zeros(Float64, resolution, resolution)
    
    # Kristall 1: Zentrum bei (0.0, 0.3), Radius ~0.1
    for i in 1:resolution
        for j in 1:resolution
            x = -1.0 + (j / resolution) * 2.0
            z = -1.0 + (i / resolution) * 2.0
            
            # Kristall 1
            if sqrt((x - 0.0)^2 + (z - 0.3)^2) <= 0.1
                phase_field[i, j] = 1.0
            end
            
            # Kristall 2
            if sqrt((x - 0.0)^2 + (z - 0.7)^2) <= 0.1
                phase_field[i, j] = 1.0
            end
        end
    end
    
    # Teste Extraktion
    crystal_params = extract_crystal_params(phase_field)
    println("Gefundene Kristalle: $(length(crystal_params))")
    
    for (i, crystal) in enumerate(crystal_params)
        println("  Kristall $i:")
        println("    Position: ($(round(crystal.x, digits=2)), $(round(crystal.z, digits=2)))")
        println("    Radius: $(round(crystal.radius, digits=3))")
    end
    
    # Teste Stokes-Berechnung
    v_stokes = compute_stokes_from_phase(phase_field, verbose=true)
    
    println("\nStokes-Feld berechnet:")
    println("  Shape: $(size(v_stokes))")
    println("  vx range: [$(minimum(v_stokes[:,:,1,1])), $(maximum(v_stokes[:,:,1,1]))]")
    println("  vz range: [$(minimum(v_stokes[:,:,2,1])), $(maximum(v_stokes[:,:,2,1]))]")
    println("  Mean |v|: $(mean(sqrt.(v_stokes[:,:,1,1].^2 + v_stokes[:,:,2,1].^2)))")
    
    return size(v_stokes) == (resolution, resolution, 2, 1)
end

"""
Normalisiert Stokes-Feld für Training (kompatibel mit existierendem Code)
"""
function normalize_stokes_velocity(v_stokes, v_stokes_reference)
    # Normalisiere durch Referenz-Stokes-Geschwindigkeit
    return v_stokes ./ Float32(v_stokes_reference)
end

println("Analytical Stokes Solution Module geladen!")
println("Verfügbare Funktionen:")
println("  - extract_crystal_params(phase_field)")
println("  - compute_stokes_velocity(phase_field, crystal_params)")
println("  - compute_stokes_from_phase(phase_field) - Vereinfachte All-in-One Funktion")
println("  - test_stokes_analytical() - Test der Implementierung")
println("")
println("Zum Testen: test_stokes_analytical()")