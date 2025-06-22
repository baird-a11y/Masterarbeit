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
    n_crystals = 1,                     
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
        Time(nstep_max=1), 
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