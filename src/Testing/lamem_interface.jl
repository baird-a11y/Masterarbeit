# =============================================================================
# LAMEM INTERFACE - 1 BIS 10 KRISTALLE
# =============================================================================
# 

using LaMEM, GeophysicalModelGenerator
using Statistics
using Random

"""
LaMEM-Funktion für 1-10 Kristalle mit intelligenter Kollisionsvermeidung
"""
function LaMEM_Variable_crystals(;
    resolution = (256, 256),
    n_crystals = 1,                      # Jetzt 1-10 Kristalle möglich
    radius_range = (0.03, 0.08),         # Variablere Radien
    η_magma = 1e20,
    ρ_magma = 2700,
    Δρ_range = (150, 300),               # Variable Dichtedifferenzen
    domain_size = (-1.0, 1.0),
    max_placement_attempts = 200,        # Mehr Versuche für viele Kristalle
    min_crystal_distance = 0.12,         # Mindestabstand zwischen Kristallen
    safety_margin = 0.08                 # Sicherheitsabstand zu Rändern
)
    @assert 1 <= n_crystals <= 10 "Kristallanzahl muss zwischen 1 und 10 liegen"
    
    # Grid-Konsistenz: LaMEM erzeugt nel+1 Punkte
    target_h, target_w = resolution
    nel_h, nel_w = target_h - 1, target_w - 1
    
    # Create model
    model = Model(
        Grid(nel=(nel_h, nel_w), x=[domain_size[1], domain_size[2]], z=[domain_size[1], domain_size[2]]), 
        Time(nstep_max=1), 
        Output(out_strain_rate=1)
    )
    
    # Erstelle Phasen (Matrix + eine Phase pro Kristall)
    matrix = Phase(ID=0, Name="matrix", eta=η_magma, rho=ρ_magma)
    phases = [matrix]
    
    # Generiere Parameter für jeden Kristall
    crystal_params = []
    
    for i in 1:n_crystals
        # Zufällige Parameter für jeden Kristall
        radius = rand() * (radius_range[2] - radius_range[1]) + radius_range[1]
        Δρ = rand() * (Δρ_range[2] - Δρ_range[1]) + Δρ_range[1]
        
        η_crystal = 1e4 * η_magma
        ρ_crystal = ρ_magma + Δρ
        
        # Erstelle Phase für diesen Kristall
        crystal_phase = Phase(ID=i, Name="crystal_$i", eta=η_crystal, rho=ρ_crystal)
        push!(phases, crystal_phase)
        
        push!(crystal_params, (radius=radius, Δρ=Δρ, η=η_crystal, ρ=ρ_crystal))
    end
    
    # Füge alle Phasen zum Modell hinzu
    for phase in phases
        add_phase!(model, phase)
    end
    
    # Intelligente Kristall-Platzierung mit Kollisionsvermeidung
    placed_crystals = []
    
    for i in 1:n_crystals
        params = crystal_params[i]
        placed = false
        
        for attempt in 1:max_placement_attempts
            # Zufällige Position mit Sicherheitsabstand zu Rändern
            x_pos = rand() * (domain_size[2] - domain_size[1] - 2*safety_margin) + domain_size[1] + safety_margin
            z_pos = rand() * (domain_size[2] - domain_size[1] - 2*safety_margin) + domain_size[1] + safety_margin
            
            center = (x_pos, 0.0, z_pos)
            
            # Prüfe Kollision mit bereits platzierten Kristallen
            collision = false
            for existing in placed_crystals
                distance = sqrt((center[1] - existing.center[1])^2 + (center[3] - existing.center[3])^2)
                min_required_distance = params.radius + existing.radius + min_crystal_distance
                
                if distance < min_required_distance
                    collision = true
                    break
                end
            end
            
            if !collision
                # Kristall platzieren
                add_sphere!(model, 
                    cen=center, 
                    radius=params.radius, 
                    phase=ConstantPhase(i)
                )
                
                push!(placed_crystals, (center=center, radius=params.radius, params=params))
                placed = true
                break
            end
        end
        
        if !placed
            println("Warnung: Kristall $i konnte nach $max_placement_attempts Versuchen nicht platziert werden")
            # Fallback: Platziere an zufälliger Position
            x_pos = rand() * 0.6 - 0.3  # Kleinerer Bereich
            z_pos = rand() * 0.6 + 0.2
            center = (x_pos, 0.0, z_pos)
            
            add_sphere!(model, 
                cen=center, 
                radius=params.radius, 
                phase=ConstantPhase(i)
            )
            
            push!(placed_crystals, (center=center, radius=params.radius, params=params))
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

    # Stokes velocity calculation (Referenz: erster Kristall)
    ref_radius = crystal_params[1].radius
    ref_Δρ = crystal_params[1].Δρ
    V_stokes = 2/9 * ref_Δρ * 9.81 * (ref_radius * 1000)^2 / η_magma  
    V_stokes_cm_year = V_stokes * 100 * (3600*24*365.25)
    
    # Zusätzliche Metadaten
    metadata = (
        n_crystals = n_crystals,
        crystal_params = crystal_params,
        placed_crystals = placed_crystals,
        v_stokes = V_stokes_cm_year
    )
        
    return x_vec_1D, z_vec_1D, phase, Vx, Vz, Exx, Ezz, V_stokes_cm_year, metadata
end

"""
Generiert gemischten Trainingsdatensatz mit 1-10 Kristallen (Fokus auf 10)
"""
function generate_generalization_dataset(n_samples; 
    target_crystal_count=10,           # Hauptsächlich für 10 Kristalle trainieren
    crystal_distribution="weighted",    # "weighted", "uniform", "target_only"
    resolution=256, 
    verbose=true)
    
    if verbose
        println("=== GENERALISIERUNGS-DATASET GENERATOR ===")
        println("Generiere $n_samples Samples")
        println("Ziel-Kristallanzahl: $target_crystal_count")
        println("Verteilung: $crystal_distribution")
    end
    
    dataset = []
    crystal_count_stats = Dict(i => 0 for i in 1:10)
    
    for i in 1:n_samples
        # Bestimme Kristallanzahl basierend auf Verteilungsstrategie
        if crystal_distribution == "target_only"
            n_crystals = target_crystal_count
        elseif crystal_distribution == "uniform"
            n_crystals = rand(1:10)
        elseif crystal_distribution == "weighted"
            # 70% Ziel-Kristallanzahl, 30% andere
            if rand() < 0.7
                n_crystals = target_crystal_count
            else
                n_crystals = rand(1:9)  # Andere Anzahl
                if n_crystals >= target_crystal_count
                    n_crystals += 1  # Überspringe Ziel-Anzahl
                end
            end
        else
            error("Unbekannte Verteilungsstrategie: $crystal_distribution")
        end
        
        crystal_count_stats[n_crystals] += 1
        
        if verbose && i % 50 == 1
            println("Sample $i/$n_samples: $n_crystals Kristalle")
        end
        
        try
            sample = LaMEM_Variable_crystals(
                resolution=(resolution, resolution),
                n_crystals=n_crystals
            )
            
            push!(dataset, sample)
            
        catch e
            if verbose && i % 25 == 1
                println("  Fehler bei Sample $i: $e")
            end
            
            # Fallback: Einzelkristall
            try
                simple_sample = LaMEM_Variable_crystals(
                    resolution=(resolution, resolution),
                    n_crystals=1
                )
                push!(dataset, simple_sample)
                crystal_count_stats[1] += 1
                crystal_count_stats[n_crystals] -= 1
            catch e2
                println("  Auch Fallback fehlgeschlagen: $e2")
                continue
            end
        end
        
        # Memory cleanup
        if i % 20 == 0
            GC.gc()
        end
    end
    
    if verbose
        println("\nDataset-Statistiken (Kristallanzahl):")
        for crystal_count in 1:10
            count = crystal_count_stats[crystal_count]
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
function generate_evaluation_dataset(n_crystals::Int, n_samples::Int; resolution=256, verbose=true)
    if verbose
        println("Generiere Evaluierungsdatensatz: $n_crystals Kristalle, $n_samples Samples")
    end
    
    dataset = []
    
    for i in 1:n_samples
        try
            sample = LaMEM_Variable_crystals(
                resolution=(resolution, resolution),
                n_crystals=n_crystals
            )
            push!(dataset, sample)
            
            if verbose && i % max(1, n_samples ÷ 10) == 0
                println("  Evaluierungssample $i/$n_samples generiert")
            end
            
        catch e
            if verbose
                println("  Warnung: Sample $i fehlgeschlagen: $e")
            end
            continue
        end
        
        if i % 10 == 0
            GC.gc()
        end
    end
    
    if verbose
        println("Evaluierungsdatensatz: $(length(dataset))/$n_samples Samples erfolgreich")
    end
    
    return dataset
end

println("Erweiterte LaMEM Interface geladen!")
println("Verfügbare Funktionen:")
println("  - LaMEM_Variable_crystals(n_crystals=1-10)")
println("  - generate_generalization_dataset(n_samples)")
println("  - generate_evaluation_dataset(n_crystals, n_samples)")