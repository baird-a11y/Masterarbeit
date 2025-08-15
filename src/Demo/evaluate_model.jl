# =============================================================================
# FEHLENDE FUNKTIONEN FÜR KRISTALL-ERKENNUNG UND ALIGNMENT
# =============================================================================
# Diese Funktionen zu evaluate_model.jl hinzufügen oder als separates Modul

using Statistics

"""
Findet Kristall-Zentren im Phasenfeld durch Clustering
"""
function find_crystal_centers(phase_field; min_crystal_size=50)
    # Finde alle Pixel mit Phase > 0.5 (Kristall-Bereiche)
    crystal_mask = phase_field .> 0.5
    
    if sum(crystal_mask) == 0
        return []  # Keine Kristalle gefunden
    end
    
    # Connected Components Analysis (vereinfacht)
    labeled_regions, num_regions = find_connected_components(crystal_mask)
    
    crystal_centers = []
    
    for region_id in 1:num_regions
        # Finde alle Pixel dieser Region
        region_mask = labeled_regions .== region_id
        region_size = sum(region_mask)
        
        # Filtere zu kleine Regionen (Rauschen)
        if region_size < min_crystal_size
            continue
        end
        
        # Berechne Schwerpunkt der Region
        indices = findall(region_mask)
        
        if length(indices) > 0
            center_y = mean([idx[1] for idx in indices])
            center_x = mean([idx[2] for idx in indices])
            
            push!(crystal_centers, (center_x, center_y))
        end
    end
    
    return crystal_centers
end

"""
Vereinfachte Connected Components Analysis
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
                
                # Flood fill für diese Komponente
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
Findet Geschwindigkeits-Minima (z.B. stärkste negative v_z Werte)
"""
function find_velocity_minima(velocity_field, expected_count; search_radius=20)
    minima_positions = []
    velocity_copy = copy(velocity_field)
    
    for i in 1:expected_count
        # Finde globales Minimum
        min_idx = argmin(velocity_copy)
        min_position = (min_idx[2], min_idx[1])  # (x, y) Format
        
        push!(minima_positions, min_position)
        
        # Lösche Umgebung um dieses Minimum für nächste Suche
        y_center, x_center = min_idx[1], min_idx[2]
        H, W = size(velocity_copy)
        
        y_range = max(1, y_center - search_radius):min(H, y_center + search_radius)
        x_range = max(1, x_center - search_radius):min(W, x_center + search_radius)
        
        velocity_copy[y_range, x_range] .= Inf  # Ausschließen für weitere Suche
    end
    
    return minima_positions
end

"""
Berechnet durchschnittlichen Alignment-Fehler zwischen zwei Punktlisten
"""
function calculate_alignment_error(points1, points2)
    if length(points1) == 0 || length(points2) == 0
        return Inf
    end
    
    # Wenn unterschiedliche Anzahl Punkte: Verwende Minimum
    n_points = min(length(points1), length(points2))
    
    if n_points == 0
        return Inf
    end
    
    total_distance = 0.0
    
    # Finde beste Zuordnung (einfachster Ansatz: nearest neighbor)
    used_indices = Set{Int}()
    
    for i in 1:n_points
        point1 = points1[i]
        
        best_distance = Inf
        best_j = 0
        
        for j in 1:length(points2)
            if j in used_indices
                continue
            end
            
            point2 = points2[j]
            distance = sqrt((point1[1] - point2[1])^2 + (point1[2] - point2[2])^2)
            
            if distance < best_distance
                best_distance = distance
                best_j = j
            end
        end
        
        if best_j > 0
            total_distance += best_distance
            push!(used_indices, best_j)
        end
    end
    
    return total_distance / n_points
end

"""
Test-Funktion für Kristall-Erkennung
"""
function test_crystal_detection()
    println("=== TEST: KRISTALL-ERKENNUNG ===")
    
    # Erstelle Test-Phasenfeld mit 2 Kristallen
    test_phase = zeros(256, 256)
    
    # Kristall 1: Zentriert bei (64, 128)
    for i in 55:75, j in 119:137
        if (i-65)^2 + (j-128)^2 <= 100  # Radius ~10
            test_phase[i, j] = 1.0
        end
    end
    
    # Kristall 2: Zentriert bei (192, 128)  
    for i in 183:201, j in 119:137
        if (i-192)^2 + (j-128)^2 <= 100  # Radius ~10
            test_phase[i, j] = 1.0
        end
    end
    
    # Teste Kristall-Erkennung
    centers = find_crystal_centers(test_phase)
    
    println("Gefundene Kristall-Zentren: $centers")
    println("Erwartet: ungefähr [(64, 128), (192, 128)]")
    
    # Teste Geschwindigkeits-Minima
    test_velocity = randn(256, 256)
    test_velocity[65, 128] = -5.0  # Starkes Minimum bei Kristall 1
    test_velocity[192, 128] = -4.0  # Starkes Minimum bei Kristall 2
    
    minima = find_velocity_minima(test_velocity, 2)
    println("Gefundene Geschwindigkeits-Minima: $minima")
    
    # Teste Alignment-Berechnung
    alignment_error = calculate_alignment_error(centers, minima)
    println("Alignment-Fehler: $(round(alignment_error, digits=2)) Pixel")
    
    return length(centers) == 2 && alignment_error < 20
end

println("Kristall-Erkennungs-Funktionen geladen!")
println("Verfügbare Funktionen:")
println("  - find_crystal_centers(phase_field)")
println("  - find_velocity_minima(velocity_field, expected_count)")
println("  - calculate_alignment_error(points1, points2)")
println("")
println("Zum Testen: test_crystal_detection()")