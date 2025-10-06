# =============================================================================
# 3D LAMEM INTERFACE
# =============================================================================


using LaMEM, GeophysicalModelGenerator
using Statistics
using Random

# =============================================================================
# 3D LAMEM MULTI-CRYSTAL FUNCTION
# =============================================================================

"""
3D Version deiner LaMEM_Multi_crystal Funktion
Erstellt ein 3D LaMEM-Modell mit mehreren Kristallphasen
"""
function LaMEM_Multi_crystal(;
    resolution = (64, 64, 64),              # 3D Auflösung (W, H, D)
    n_crystals = 2,                         # Reduziert für 3D Tests
    radius_crystal = [0.1],                 # Größere Radien für 3D
    η_magma = 1e20,                         # Gleiche Physik wie 2D
    ρ_magma = 2700,                         
    Δρ = 200,                               
    domain_size = (-1.0, 1.0),              
    cen_3D = [(0.0, 0.0, 0.5)],            # 3D Kristall-Zentren (x,y,z)
    max_attempts = 100,                 
    collision_threshold = 0.4               # Größerer Threshold für 3D
)
    
    # Berechne abgeleitete Parameter (gleich wie dein 2D Code)
    η_crystal = 1e4 * η_magma           
    ρ_crystal = ρ_magma + Δρ            
    
    # 3D Grid Setup - HAUPTÄNDERUNG VON DEINEM 2D CODE
    model = Model(
        Grid(
            nel=(resolution[1]-1, resolution[2]-1, resolution[3]-1),  # 3D Grid!
            x=[domain_size[1], domain_size[2]], 
            y=[domain_size[1], domain_size[2]],                       # Neue Y-Dimension
            z=[domain_size[1], domain_size[2]]
        ), 
        LaMEM.Time(nstep_max=1),  # Dein Fix für Time-Konflikt
        Output(out_strain_rate=1)
    )
    
    # Define phases (gleich wie dein 2D Code)
    matrix = Phase(ID=0, Name="matrix", eta=η_magma, rho=ρ_magma)
    crystal = Phase(ID=1, Name="crystal", eta=η_crystal, rho=ρ_crystal)
    add_phase!(model, crystal, matrix)

    # 3D Kristalle hinzufügen - ERWEITERT VON DEINEM CODE
    for i = 1:n_crystals
        current_radius = length(radius_crystal) >= i ? radius_crystal[i] : radius_crystal[1]
        
        # 3D Zentrum bestimmen
        if length(cen_3D) >= i
            current_center = cen_3D[i]
        else
            # Zufällige 3D Position generieren
            x_pos = rand(-0.6:0.1:0.6)
            y_pos = rand(-0.6:0.1:0.6)  # Neue Y-Koordinate
            z_pos = rand(0.2:0.1:0.8)
            current_center = (x_pos, y_pos, z_pos)
        end
        
        # 3D Sphäre hinzufügen (erweitert von deinem 2D code)
        add_sphere!(model, 
            cen=current_center,                    # 3D Zentrum (x,y,z)
            radius=current_radius, 
            phase=ConstantPhase(1)
        )
    end

    # LaMEM ausführen (gleich wie dein Code)
    run_lamem(model, 1)
    data, _ = read_LaMEM_timestep(model, 1)

    # 3D Daten extrahieren - ERWEITERT VON DEINEM 2D CODE
    x_vec_1D = data.x.val[:, 1, 1]      # X-Koordinaten
    y_vec_1D = data.y.val[1, :, 1]      # Y-Koordinaten (NEU für 3D)
    z_vec_1D = data.z.val[1, 1, :]      # Z-Koordinaten
    
    # 3D Felder extrahieren
    phase = data.fields.phase                           # 3D Phasenfeld
    Vx = data.fields.velocity[1]                        # 3D X-Geschwindigkeit
    Vy = data.fields.velocity[2]                        # 3D Y-Geschwindigkeit (NEU)
    Vz = data.fields.velocity[3]                        # 3D Z-Geschwindigkeit
    
    # Optionale Felder (falls benötigt)
    Exx = data.fields.strain_rate[1]                    # 3D Strain rates
    Eyy = data.fields.strain_rate[5]                    # NEU für 3D
    Ezz = data.fields.strain_rate[9]                    
    rho = data.fields.density                           # 3D Dichte
    log10eta = data.fields.visc_creep                   # 3D Viskosität

    # Stokes-Geschwindigkeit (gleiche Berechnung wie 2D)
    ref_radius = radius_crystal[1]
    V_stokes = 2/9 * Δρ * 9.81 * (ref_radius * 1000)^2 / η_magma  
    V_stokes_cm_year = V_stokes * 100 * (3600*24*365.25)      
        
    # 3D Return - ERWEITERT VON DEINEM FORMAT
    return x_vec_1D, y_vec_1D, z_vec_1D, phase, Vx, Vy, Vz, Exx, Eyy, Ezz, V_stokes_cm_year
end

# =============================================================================
# 3D DATASET GENERATION
# =============================================================================

"""
3D Version deiner generate_mixed_resolution_dataset Funktion
"""
function generate_mixed_resolution_dataset(n_samples; 
                                         resolutions=[(32,32,32), (48,48,48), (64,64,64)],  # Kleinere 3D Auflösungen
                                         verbose=true)
    if verbose
        println("=== 3D MIXED-RESOLUTION DATASET GENERATOR ===")
        println("Generiere $n_samples 3D Samples")
        println("3D Auflösungen: $resolutions")
    end
    
    dataset = []
    stats = Dict(res => 0 for res in resolutions)
    
    for i in 1:n_samples
        resolution = rand(resolutions)
        stats[resolution] += 1
        
        # Weniger Kristalle für 3D (Speicher + Komplexität)
        n_crystals = rand(1:3)  # Statt 1:4 in 2D
        radius_crystal = [rand(0.05:0.01:0.12) for _ in 1:n_crystals]  # Größere Radien für 3D
        
        # 3D nicht-kollidierende Zentren generieren
        centers = []
        for j in 1:n_crystals
            attempts = 0
            while attempts < 30  # Mehr Versuche für 3D
                x_pos = rand(-0.6:0.05:0.6)
                y_pos = rand(-0.6:0.05:0.6)  # NEU: Y-Position
                z_pos = rand(0.1:0.05:0.9)
                new_center = (x_pos, y_pos, z_pos)
                
                # 3D Kollisionserkennung
                collision = false
                for existing_center in centers
                    distance = sqrt((new_center[1] - existing_center[1])^2 + 
                                   (new_center[2] - existing_center[2])^2 +   # NEU: Y-Differenz
                                   (new_center[3] - existing_center[3])^2)
                    if distance < 0.2  # Größerer Abstand für 3D
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
            
            # Fallback falls keine Position gefunden
            if length(centers) < j
                fallback_x = rand(-0.3:0.1:0.3)
                fallback_y = rand(-0.3:0.1:0.3)  # NEU
                fallback_z = rand(0.3:0.1:0.7)
                push!(centers, (fallback_x, fallback_y, fallback_z))
            end
        end
        
        if verbose && i % 10 == 1  # Häufigere Updates für langsamere 3D
            res_str = "$(resolution[1])×$(resolution[2])×$(resolution[3])"
            println("Sample $i/$n_samples: $res_str, $n_crystals Kristalle")
        end
        
        try
            sample = LaMEM_Multi_crystal(
                resolution=resolution,
                n_crystals=n_crystals,
                radius_crystal=radius_crystal,
                cen_3D=centers  # 3D Zentren statt cen_2D
            )
            
            push!(dataset, sample)
            
        catch e
            if verbose && i % 10 == 1
                println("  Fehler bei 3D Sample $i: $e")
            end
            
            # 3D Fallback: Einfaches einzelnes Kristall
            try
                simple_sample = LaMEM_Multi_crystal(
                    resolution=resolution,
                    n_crystals=1,
                    radius_crystal=[0.08],
                    cen_3D=[(0.0, 0.0, 0.5)]  # 3D Zentrum
                )
                push!(dataset, simple_sample)
            catch e2
                println("  Auch einfaches 3D Sample fehlgeschlagen: $e2")
                continue
            end
        end
        
        # Häufigere Garbage Collection für 3D (mehr Speicherverbrauch)
        if i % 5 == 0
            GC.gc()
        end
    end
    
    if verbose
        println("\n3D Dataset-Statistiken:")
        for (res, count) in stats
            percentage = round(100 * count / n_samples, digits=1)
            res_str = "$(res[1])×$(res[2])×$(res[3])"
            println("  $res_str: $count Samples ($percentage%)")
        end
        println("Gesamt: $(length(dataset)) 3D Samples erfolgreich generiert")
    end
    
    return dataset
end

# =============================================================================
# 3D KRISTALL-PLATZIERUNG UTILITIES
# =============================================================================

"""
Generiert optimale 3D Kristall-Positionen (erweitert von deiner 2D Logik)
"""
function generate_3d_crystal_positions(n_crystals; domain_bounds=(-0.8, 0.8))
    if n_crystals == 1
        return [(0.0, 0.0, 0.5)]
    elseif n_crystals == 2
        return [(-0.4, 0.0, 0.5), (0.4, 0.0, 0.5)]
    elseif n_crystals <= 4
        # Tetraeder-ähnliche Anordnung
        return [
            (-0.3, -0.3, 0.3),
            (0.3, -0.3, 0.3), 
            (0.0, 0.3, 0.3),
            (0.0, 0.0, 0.7)
        ][1:n_crystals]
    else
        # Kubisches Grid für mehr Kristalle
        centers = []
        grid_size = ceil(Int, n_crystals^(1/3))  # Kubische Wurzel für 3D Grid
        spacing = 1.6 / (grid_size + 1)
        
        for i in 1:n_crystals
            x_idx = (i-1) % grid_size
            y_idx = div(i-1, grid_size) % grid_size
            z_idx = div(i-1, grid_size^2)
            
            x_pos = domain_bounds[1] + x_idx * spacing + spacing
            y_pos = domain_bounds[1] + y_idx * spacing + spacing
            z_pos = 0.2 + z_idx * spacing + spacing  # Z startet höher
            
            push!(centers, (x_pos, y_pos, z_pos))
        end
        
        return centers
    end
end

# =============================================================================
# 3D SPEZIELLE DATASET FUNKTIONEN
# =============================================================================

"""
Generiert 3D Dataset für spezifische Kristall-Anzahl (für systematische Tests)
"""
function generate_3d_crystal_dataset(n_crystals, n_samples; resolution=(64,64,64), verbose=true)
    if verbose
        println("=== 3D $n_crystals-KRISTALL DATASET ===")
        println("Generiere $n_samples Samples mit $n_crystals Kristallen")
        println("3D Auflösung: $(resolution[1])×$(resolution[2])×$(resolution[3])")
    end
    
    dataset = []
    
    for i in 1:n_samples
        # Verschiedene Radien für Diversität
        radius_crystal = [rand(0.06:0.01:0.1) for _ in 1:n_crystals]
        
        # Optimale 3D Positionen generieren
        centers = generate_3d_crystal_positions(n_crystals)
        
        # Kleine zufällige Verschiebungen hinzufügen
        centers_perturbed = []
        for center in centers
            x_pert = center[1] + rand(-0.1:0.02:0.1)
            y_pert = center[2] + rand(-0.1:0.02:0.1)
            z_pert = center[3] + rand(-0.1:0.02:0.1)
            push!(centers_perturbed, (x_pert, y_pert, z_pert))
        end
        
        try
            sample = LaMEM_Multi_crystal(
                resolution=resolution,
                n_crystals=n_crystals,
                radius_crystal=radius_crystal,
                cen_3D=centers_perturbed
            )
            
            push!(dataset, sample)
            
            if verbose && i % 5 == 1
                println("  3D Sample $i/$n_samples generiert")
            end
            
        catch e
            if verbose
                println("  Fehler bei 3D Sample $i: $e")
            end
            continue
        end
        
        # Memory management für 3D
        if i % 3 == 0
            GC.gc()
        end
    end
    
    if verbose
        println("✓ $n_crystals-Kristall 3D Dataset: $(length(dataset)) Samples")
    end
    
    return dataset
end

# =============================================================================
# KOMPATIBILITÄTS-WRAPPER
# =============================================================================

"""
Wrapper um 3D LaMEM für Kompatibilität mit deinem bestehenden Code
"""
function LaMEM_Multi_crystal_wrapper(args...; kwargs...)
    # Konvertiere 2D Argumente zu 3D falls nötig
    
    # Falls cen_2D übergeben wird, konvertiere zu cen_3D
    if haskey(kwargs, :cen_2D)
        cen_2D = kwargs[:cen_2D]
        cen_3D = [(pos[1], 0.0, pos[2]) for pos in cen_2D]  # Y=0 für 2D→3D
        kwargs_3d = merge(kwargs, (cen_3D=cen_3D,))
        delete!(kwargs_3d, :cen_2D)
    else
        kwargs_3d = kwargs
    end
    
    # Falls 2D Resolution übergeben wird, erweitere zu 3D
    if haskey(kwargs_3d, :resolution) && length(kwargs_3d[:resolution]) == 2
        res_2d = kwargs_3d[:resolution]
        res_3d = (res_2d[1], res_2d[2], res_2d[1])  # Verwende X-Resolution für Y
        kwargs_3d = merge(kwargs_3d, (resolution=res_3d,))
    end
    
    return LaMEM_Multi_crystal(args...; kwargs_3d...)
end


# =============================================================================
# FIX FÜR 3D LAMEM TEST
# =============================================================================

"""
KORRIGIERTE quick_test_3d_safe Funktion
"""
function quick_test_3d_safe()
    println("=== 3D LAMEM TEST ===")
    
    try
        # 1. Erst die funktionierenden Tests
        println("1. Teste 3D UNet und Datenverarbeitung...")
        if !quick_test_3d_safe_minimal()
            return false
        end
        
        # 2. Jetzt vorsichtiger LaMEM Test
        println("2. Teste 3D LaMEM (vorsichtig)...")
        
        # KLEINSTE mögliche 3D LaMEM Simulation
        test_sample = LaMEM_Multi_crystal(
            resolution=(16, 16, 16),            # SEHR klein für Test
            n_crystals=1,                       # NUR ein Kristall
            radius_crystal=[0.1],               # Größerer Radius für 3D
            cen_3D=[(0.0, 0.0, 0.5)]           # Einfache zentrale Position
        )
        
        println("✓ 3D LaMEM Sample erfolgreich erstellt")
        
        # 3. Vorsichtige Datenextraktion
        println("3. Extrahiere 3D LaMEM Daten...")
        
        # Prüfe Sample-Format
        println("  Sample hat $(length(test_sample)) Elemente")
        
        if length(test_sample) >= 11  # x, y, z, phase, vx, vy, vz, ..., v_stokes
            x, y, z, phase, vx, vy, vz = test_sample[1:7]
            v_stokes = test_sample[end]
            
            println("  ✓ 3D Daten extrahiert:")
            println("    X: $(length(x)) Punkte")
            println("    Y: $(length(y)) Punkte") 
            println("    Z: $(length(z)) Punkte")
            println("    Phase: $(size(phase))")
            println("    Vx: $(size(vx))")
            println("    Vy: $(size(vy))")
            println("    Vz: $(size(vz))")
            println("    V_stokes: $v_stokes")
            
        else
            println("  ✗ Unerwartetes Sample-Format: $(length(test_sample)) Elemente")
            return false
        end
        
        # 4. 3D Preprocessing mit echten LaMEM Daten
        println("4. Teste 3D Preprocessing mit echten LaMEM Daten...")
        
        phase_tensor, velocity_tensor = preprocess_lamem_sample_3d(
            x, y, z, phase, vx, vy, vz, v_stokes, target_resolution=16  # Klein für Test
        )
        
        println("✓ 3D LaMEM Preprocessing erfolgreich")
        
        # 5. 3D UNet Test mit echten Daten
        println("5. Teste 3D UNet mit echten LaMEM Daten...")
        
        model = create_simplified_unet_3d(1, 3, 16)  # Kleines Modell
        output = model(phase_tensor)
        
        expected_size = (16, 16, 16, 3, 1)
        if size(output) == expected_size
            println("✓ 3D UNet mit echten LaMEM Daten erfolgreich: $(size(output))")
            return true
        else
            println("✗ 3D UNet Output falsch: $(size(output)) ≠ $expected_size")
            return false
        end
        
    catch e
        println("✗ 3D LaMEM Test fehlgeschlagen: $e")
        println("Fehler-Details:")
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        return false
    end
end

# =============================================================================
# VERBESSERTE 3D MAIN FUNKTION MIT BESSERER FEHLERBEHANDLUNG
# =============================================================================

"""
Sichere 3D Training-Funktion mit schrittweisem Debugging
"""
function run_3d_crystal_training_safe()
    println("="^80)
    println("SICHERES 3D UNET TRAINING")
    println("="^80)
    
    start_time = time()
    
    # Erstelle Output-Verzeichnisse
    mkpath(SERVER_CONFIG_3D.checkpoint_dir)
    mkpath(SERVER_CONFIG_3D.results_dir)
    
    try
        # 1. VORSICHTIGER 3D SYSTEM-CHECK
        println("\n1. VORSICHTIGER 3D SYSTEM-CHECK")
        println("-"^50)
        
        if !quick_test_3d_safe()
            error("3D System-Test fehlgeschlagen!")
        end
        println("✓ Vollständiger 3D System-Test erfolgreich!")
        
        # 2. SCHRITT-FÜR-SCHRITT DATENGENERIERUNG
        println("\n2. VORSICHTIGE 3D DATENGENERIERUNG")
        println("-"^50)
        
        # Starte mit NUR 2 Samples für ersten Test
        test_samples = 2
        println("Starte mit nur $test_samples Test-Samples...")
        
        dataset = []
        for i in 1:test_samples
            println("Generiere 3D Test-Sample $i/$test_samples...")
            
            try
                sample = LaMEM_Multi_crystal(
                    resolution=(32, 32, 32),       # Kleine, sichere Auflösung
                    n_crystals=1,                  # NUR ein Kristall
                    radius_crystal=[0.08],         
                    cen_3D=[(0.0, 0.0, 0.5)]      
                )
                
                push!(dataset, sample)
                println("  ✓ Sample $i erfolgreich")
                
                # Memory cleanup
                GC.gc()
                
            catch e
                println("  ✗ Sample $i fehlgeschlagen: $e")
                continue
            end
        end
        
        if length(dataset) == 0
            error("Keine 3D Test-Samples erstellt!")
        end
        
        println("✓ $(length(dataset)) 3D Test-Samples erstellt")
        
        # 3. MINIMALES 3D MODELL
        println("\n3. MINIMALES 3D MODELL")
        println("-"^50)
        
        model = create_simplified_unet_3d(1, 3, 16)  # Kleine Architektur
        println("✓ Minimales 3D UNet erstellt")
        
        # 4. EIN TRAINING-SCHRITT ZUM TESTEN
        println("\n4. TEST-TRAINING (1 EPOCHE)")
        println("-"^50)
        
        # Minimale Training-Konfiguration
        minimal_config = create_training_config_3d(
            learning_rate = 0.001f0,
            num_epochs = 1,              # NUR eine Epoche zum Testen!
            batch_size = 1,
            use_gpu = false,
            validation_split = 0.5f0     # Hälftig teilen bei 2 Samples
        )
        
        println("Teste 1 Training-Epoche...")
        
        trained_model, train_losses, val_losses, physics_losses = train_velocity_unet_3d_safe(
            model, dataset, (32, 32, 32),
            config=minimal_config
        )
        
        if length(train_losses) > 0
            println("✓ Test-Training erfolgreich!")
            println("  Training Loss: $(round(train_losses[1], digits=6))")
            println("  Validation Loss: $(round(val_losses[1], digits=6))")
            
            if length(physics_losses) > 0
                println("  Physics Loss: $(round(physics_losses[1], digits=6))")
            end
        end
        
        # Erfolg!
        end_time = time()
        total_time = end_time - start_time
        
        println("\n" * "="^80)
        println("3D TEST-TRAINING ERFOLGREICH!")
        println("="^80)
        println("Zeit: $(round(total_time/60, digits=2)) Minuten")
        println("Jetzt kannst du die volle 3D Pipeline verwenden!")
        
        return true
        
    catch e
        end_time = time()
        total_time = end_time - start_time
        
        println("\n" * "="^80)
        println("3D TEST-TRAINING FEHLGESCHLAGEN")
        println("="^80)
        println("Fehler: $e")
        println("Zeit bis Fehler: $(round(total_time/60, digits=1)) Minuten")
        
        for (exc, bt) in Base.catch_stack()
            showerror(stdout, exc, bt)
            println()
        end
        
        return false
    end
end