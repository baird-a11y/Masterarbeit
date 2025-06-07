# Physikalische Plausibilitätsprüfung
function physics_validation(model_path)
    println("=== PHYSIKALISCHE VALIDIERUNG ===")
    
    # Lade Modell
    model_dict = BSON.load(model_path)
    model = first(values(model_dict))
    
    # Test 1: Stokes-Gesetz
    println("\n1. Stokes-Gesetz Validierung:")
    
    # Verschiedene Kristallgrößen testen
    radii = [0.03, 0.05, 0.07]
    velocities = []
    
    for R in radii
        phase_field = zeros(Float32, 128, 128, 1, 1)
        center = (64, 64)
        
        # Kristall hinzufügen
        for i in 1:128, j in 1:128
            if sqrt((i-center[1])^2 + (j-center[2])^2) < R*1000/16  # Skalierung
                phase_field[i, j, 1, 1] = 1.0f0
            end
        end
        
        prediction = model(phase_field)
        vz = prediction[:, :, 2, 1]
        
        # Durchschnittsgeschwindigkeit am Kristall
        kristall_mask = phase_field[:, :, 1, 1] .== 1.0f0
        avg_vz = mean(abs.(vz[kristall_mask]))
        push!(velocities, avg_vz)
        
        println("  Radius $R: |v_z| = $(round(avg_vz, digits=4))")
    end
    
    # Prüfe, ob Geschwindigkeit mit R² skaliert (Stokes-Gesetz)
    println("  Verhältnis v(R2)/v(R1): $(round(velocities[2]/velocities[1], digits=2))")
    println("  Erwartet (R2/R1)²: $(round((radii[2]/radii[1])^2, digits=2))")
    
    # Test 2: Symmetrie
    println("\n2. Symmetrie-Test:")
    
    phase_field = zeros(Float32, 128, 128, 1, 1)
    center = (64, 64)  # Exakt zentral
    
    for i in 1:128, j in 1:128
        if sqrt((i-center[1])^2 + (j-center[2])^2) < 15
            phase_field[i, j, 1, 1] = 1.0f0
        end
    end
    
    prediction = model(phase_field)
    vx = prediction[:, :, 1, 1]
    
    # Links-Rechts Symmetrie
    vx_left = mean(vx[:, 1:64])
    vx_right = mean(vx[:, 65:128])
    symmetry_error = abs(vx_left + vx_right)
    
    println("  v_x links: $(round(vx_left, digits=4))")
    println("  v_x rechts: $(round(vx_right, digits=4))")
    println("  Symmetrie-Fehler: $(round(symmetry_error, digits=4))")
    
    if symmetry_error < 0.1
        println("  ✓ Gute Symmetrie")
    else
        println("  ⚠ Asymmetrische Strömung")
    end
    
    # Test 3: Kontinuität (∇·v = 0)
    println("\n3. Kontinuitäts-Test:")
    
    # Berechne Divergenz numerisch
    vx = prediction[:, :, 1, 1]
    vz = prediction[:, :, 2, 1]
    
    # Finite Differenzen
    dvx_dx = diff(vx, dims=2)
    dvz_dz = diff(vz, dims=1)
    
    # Divergenz (reduzierte Größe wegen diff)
    h_min, w_min = min(size(dvx_dx, 1), size(dvz_dz, 1)), min(size(dvx_dx, 2), size(dvz_dz, 2))
    divergence = dvx_dx[1:h_min, :] .+ dvz_dz[:, 1:w_min]
    
    avg_div = mean(abs.(divergence))
    max_div = maximum(abs.(divergence))
    
    println("  Durchschnittliche |∇·v|: $(round(avg_div, digits=5))")
    println("  Maximale |∇·v|: $(round(max_div, digits=5))")
    
    if avg_div < 0.1
        println("  ✓ Kontinuität gut erfüllt")
    else
        println("  ⚠ Kontinuität verletzt - prüfe Training")
    end
    
    return (
        stokes_scaling = velocities[2]/velocities[1],
        symmetry_error = symmetry_error,
        avg_divergence = avg_div
    )
end

# Beispielaufruf:
# physics_results = physics_validation("final_velocity_model.bson")