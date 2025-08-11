# =============================================================================
# ERWEITERTE LAMEM INTERFACE - 1 BIS 10 KRISTALLE (BASIEREND AUF FUNKTIONIERENDEM CODE)
# =============================================================================
# Speichern als: lamem_interface.jl

using LaMEM, GeophysicalModelGenerator
using Statistics
using Random

# Fix für Time-Namenskonflikt
import LaMEM: Time as LaMEMTime

"""
Erweiterte LaMEM-Funktion für 1-10 Kristalle (basierend auf LaMEM_Multi_crystal)
"""
function LaMEM_Multi_crystal(;
    resolution = (64, 64),              
    n_crystals = 1,                     # Jetzt 1-10 Kristalle möglich
    radius_crystal = [0.05],            
    η_magma = 1e20,                     
    ρ_magma = 2700,                     
    Δρ = 200,                           
    domain_size = (-1.0, 1.0),          
    cen_2D = [(0.0, 0.5)],              
    max_attempts = 200,                 # Mehr Versuche für Kollisionsvermeidung
    collision_threshold = 0.15          # Mindestabstand zwischen Kristallen
)
    @assert 1 <= n_crystals <= 10 "Kristallanzahl muss zwischen 1 und 10 liegen"
    
    # Berechne abgeleitete Parameter
    η_crystal = 1e4 * η_magma           
    ρ_crystal = ρ_magma + Δρ            
    
    # Create model - verwende LaMEMTime statt Time
    model = Model(
        Grid(nel=(resolution[1]-1, resolution[2]-1), x=[-1,1], z=[-1,1]), 
        LaMEMTime(nstep_max=1), 
        Output(out_strain_rate=1)
    )
    
    # Define phases - Matrix + verschiedene Kristall-Phasen
    matrix = Phase(ID=0, Name="matrix", eta=η_magma, rho=ρ_magma)
    phases_to_add = [matrix]
    
    # Erstelle separate Phase für jeden Kristall (für bessere Unterscheidung)
    for i in 1:n_crystals
        crystal_phase = Phase(ID=i, Name="crystal_$i", eta=η_crystal, rho=ρ_crystal)
        push!(phases_to_add, crystal_phase)
    end
    
    # Füge alle Phasen hinzu
    for phase in phases_to_add
        add_phase!(model, phase)
    end

    # Intelligente Kristall-Platzierung mit Kollisionsvermeidung
    placed_centers = []
    
    for i = 1:n_crystals
        current_radius = length(radius_crystal) >= i ? radius_crystal[i] : radius_crystal[1]
        placed = false
        
        # Versuche gültige Position zu finden
        for attempt in 1:max_attempts
            if length(cen_2D) >= i
                # Verwende vorgegebene Position
                current_center = cen_2D[i]
            else
                # Generiere zufällige Position
                x_pos = rand(-0.6:0.05:0.6)
                z_pos = rand(0.2:0.05:0.8)
                current_center = (x_pos, z_pos)
            end
            
            # Prüfe Kollision mit bereits platzierten Kristallen
            collision = false
            for existing_center in placed_centers
                distance = sqrt((current_center[1] - existing_center[1])^2 + 
                               (current_center[2] - existing_center[2])^2)
                min_distance = collision_threshold
                
                if distance < min_distance
                    collision = true
                    break
                end
            end
            
            if !collision
                # Platziere Kristall
                add_sphere!(model, 
                    cen=(current_center[1], 0.0, current_center[2]), 
                    radius=current_radius, 
                    phase=ConstantPhase(i)  # Verwende Kristall-spezifische Phase-ID
                )
                
                push!(placed_centers, current_center)
                placed = true
                break
            end
        end
        
        # Fallback falls keine Kollisionsvermeidung funktioniert
        if !placed
            println("Warnung: Kristall $i - Fallback zu zufälliger Position")
            fallback_x = rand(-0.3:0.1:0.3)
            fallback_z = rand(0.3:0.1:0.7)
            
            add_sphere!(model, 
                cen=(fallback_x, 0.0, fallback_z), 
                radius=current_radius, 
                phase=ConstantPhase(i)
            )
            
            push!(placed_centers, (fallback_x, fallback_z))
        end
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
Generiert Samples für Generalisierungsstudium (70% Ziel-Kristallanzahl, 30% andere)
"""
function generate_generalization_dataset(n_samples; 
    target_crystal_count=10, 
    resolutions=[128], 
    verbose=true)
    
    if verbose
        println("=== GENERALISIERUNGS-DATASET GENERATOR ===")
        println("Generiere $n_samples Samples")
        println("Ziel-Kristallanzahl: $target_crystal_count")
        println("Auflösungen: $resolutions")
    end
    
    dataset = []
    crystal_stats = Dict(i => 0 for i in 1:10)
    
    for i in 1:n_samples
        # 70% Ziel-Kristallanzahl, 30% andere
        if rand() < 0.7
            n_crystals = target_crystal_count
        else
            # Wähle andere Kristallanzahl
            other_counts = [j for j in 1:10 if j != target_crystal_count]
            n_crystals = rand(other_counts)
        end
        
        crystal_stats[n_crystals] += 1
        
        # Zufällige Parameter
        resolution = rand(resolutions)
        radius_crystal = [rand(0.03:0.005:0.08) for _ in 1:n_crystals]
        
        if verbose && i % 50 == 1
            println("Sample $i/$n_samples: $(resolution)x$(resolution), $n_crystals Kristalle")
        end
        
        try
            sample = LaMEM_Multi_crystal(
                resolution=(resolution, resolution),
                n_crystals=n_crystals,
                radius_crystal=radius_crystal
            )
            
            push!(dataset, sample)
            
        catch e
            if verbose && i % 25 == 1
                println("  Fehler bei Sample $i: $e")
            end
            
            # Fallback: Einzelkristall
            try
                simple_sample = LaMEM_Multi_crystal(
                    resolution=(resolution, resolution),
                    n_crystals=1,
                    radius_crystal=[0.05],
                    cen_2D=[(0.0, 0.5)]
                )
                push!(dataset, simple_sample)
                crystal_stats[1] += 1
                crystal_stats[n_crystals] -= 1
            catch e2
                println("  Auch Fallback fehlgeschlagen: $e2")
                continue
            end
        end
        
        if i % 10 == 0
            GC.gc()
        end
    end
    
    if verbose
        println("\nDataset-Statistiken (Kristallanzahl):")
        for crystal_count in 1:10
            count = crystal_stats[crystal_count]
            if count > 0
                percentage = round(100 * count / n_samples, digits=1)
                println("  $crystal_count Kristalle: $count Samples ($percentage%)")
            end
        end
        println("Gesamt: $(length(dataset)) erfolgreich generiert")
    end
    
    return dataset
end

"""
Erstellt Evaluierungsdatensatz mit spezifischer Kristallanzahl
"""
function generate_evaluation_dataset(n_crystals::Int, n_samples::Int; 
    resolution=128, verbose=true)
    
    if verbose
        println("Generiere Evaluierungsdatensatz: $n_crystals Kristalle, $n_samples Samples")
    end
    
    dataset = []
    
    for i in 1:n_samples
        try
            sample = LaMEM_Multi_crystal(
                resolution=(resolution, resolution),
                n_crystals=n_crystals
            )
            push!(dataset, sample)
            
            if verbose && i % max(1, n_samples ÷ 5) == 0
                println("  Evaluierungssample $i/$n_samples generiert")
            end
            
        catch e
            if verbose
                println("  Warnung: Sample $i fehlgeschlagen: $e")
            end
            continue
        end
        
        if i % 5 == 0
            GC.gc()
        end
    end
    
    if verbose
        println("Evaluierungsdatensatz: $(length(dataset))/$n_samples Samples erfolgreich")
    end
    
    return dataset
end

# Behalte auch die alte Funktion für Kompatibilität
function generate_mixed_resolution_dataset(n_samples; resolutions=[64, 128, 256], verbose=true)
    return generate_generalization_dataset(n_samples, resolutions=resolutions, verbose=verbose)
end

println("Erweiterte LaMEM Interface geladen!")
println("Verfügbare Funktionen:")
println("  - LaMEM_Multi_crystal(n_crystals=1-10)")
println("  - generate_generalization_dataset(n_samples, target_crystal_count=10)")
println("  - generate_evaluation_dataset(n_crystals, n_samples)")