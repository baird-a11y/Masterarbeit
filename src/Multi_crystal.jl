using Flux
using Flux: mse, gpu, cpu
using CUDA
using Statistics
using FileIO
using LinearAlgebra
using Optimisers
using ProgressMeter
using BSON: @save, @load
using Random
using LaMEM, GeophysicalModelGenerator

CUDA.allowscalar(true)
println("=== MULTI-KRISTALL VELOCITY UNET TRAINER ===")

# ==================== MODELL-STRUKTUREN ====================

struct VelocityUNet
    encoder1; encoder2; encoder3; encoder4; bottleneck
    decoder4; decoder4_1; decoder3; decoder3_1
    decoder2; decoder2_1; decoder1; decoder1_1
end

Flux.@functor VelocityUNet

function crop_and_concat_flux_native(x, skip, dims=3)
    """
    Flux-native Skip-Connections ohne manuelle Array-Operationen
    """
    x_size = size(x)
    skip_size = size(skip)
    
    # Falls Größen unterschiedlich sind, nutze einfache Flux-Operationen
    if x_size[1] != skip_size[1] || x_size[2] != skip_size[2]
        # Verwende Flux-native Interpolation/Pooling für Größenanpassung
        min_h = min(x_size[1], skip_size[1])
        min_w = min(x_size[2], skip_size[2])
        
        # Adaptive Pooling für einheitliche Größen (GPU-nativ)
        x_resized = Flux.AdaptiveMeanPool((min_h, min_w))(x)
        skip_resized = Flux.AdaptiveMeanPool((min_h, min_w))(skip)
        
        return cat(x_resized, skip_resized, dims=dims)
    else
        # Gleiche Größen - direkte Concatenation
        return cat(x, skip, dims=dims)
    end
end

function (model::VelocityUNet)(x)
    """
    Forward-Pass durch das UNet - KORRIGIERT
    """
    # Encoder
    e1 = model.encoder1(x)
    e2 = model.encoder2(e1)
    e3 = model.encoder3(e2)
    e4 = model.encoder4(e3)
    b = model.bottleneck(e4)
    
    # Decoder mit Skip-Connections - ALLE KORRIGIERT
    d4 = model.decoder4(b)
    d4 = model.decoder4_1(crop_and_concat_flux_native(d4, e4))
    
    d3 = model.decoder3(d4)
    d3 = model.decoder3_1(crop_and_concat_flux_native(d3, e3))
    
    d2 = model.decoder2(d3)
    d2 = model.decoder2_1(crop_and_concat_flux_native(d2, e2))
    
    d1 = model.decoder1(d2)
    d1 = model.decoder1_1(crop_and_concat_flux_native(d1, e1))
    
    return d1
end

# ==================== ZENTRALE MULTI-KRISTALL KONFIGURATION ====================

const CONFIG = (
    # Modell-Parameter
    image_size = (256, 256),           # Konsistente Bildgröße
    input_channels = 1,                # Wird automatisch angepasst bei multi_channel
    output_channels = 2,               # v_x, v_z
    
    # Training-Parameter
    learning_rate = 0.001,
    num_epochs = 5,                    # Wenige Epochen für Test
    batch_size = 1,                    # Kleine Batches bei 256×256 + Multi-Channel
    
    # Dataset-Parameter
    dataset_size = 5,                  # Wenige Samples für schnellen Test
    test_split = 0.1,
    
    # Multi-Kristall Parameter
    max_crystals = 3,                  # Maximal 3 Kristalle pro Sample
    min_crystals = 1,                  # Mindestens 1 Kristall
    crystal_spacing_min = 0.15,        # Mindestabstand zwischen Kristallen
    
    # Kristall-Variations-Parameter
    radius_variation = 0.02,           # Variation der Radien
    density_variation = 100,           # Variation der Dichtedifferenzen
    allow_different_densities = true,  # Verschiedene Dichten pro Kristall
    
    # Phasenfeld-Kodierung
    phase_encoding = "multi_channel",  # "single_channel" oder "multi_channel"
    max_phase_channels = 4,            # Matrix + 3 Kristalle
    
    # LaMEM-Parameter (Bereiche für Variation)
    eta_range = (1e19, 1e21),         # Viskosität-Bereich
    delta_rho_range = (100, 500),      # Dichtedifferenz-Bereich
    position_range = (-0.6, 0.6),     # Kristall-Position Bereich
    radius_range = (0.03, 0.08),      # Kristall-Radius Bereich
    
    # Speicher-Parameter
    checkpoint_dir = "velocity_checkpoints_gpu_multi",
    save_every_n_epochs = 2,           # Häufiger speichern bei wenigen Epochen
    
    # Hardware-Parameter
    use_gpu = true,                    # GPU für schnelleres Training
    memory_cleanup_frequency = 5,      # Häufiger GPU-Memory cleanup
    
    # Debug-Parameter
    verbose = true,
    save_sample_images = true
)

# Kompatibilität: MULTI_CONFIG als Alias für CONFIG
const MULTI_CONFIG = CONFIG

# ==================== KONFIGURATION VALIDIERUNG UND INFO ====================

function validate_and_print_config()
    """
    Validiert Konfiguration und gibt Informationen aus
    """
    println("="^60)
    println("MULTI-KRISTALL KONFIGURATION")
    println("="^60)
    
    # Basis-Parameter
    println("Modell:")
    println("  Bildgröße: $(CONFIG.image_size)")
    println("  Phasenfeld: $(CONFIG.phase_encoding)")
    if CONFIG.phase_encoding == "multi_channel"
        println("  Input-Kanäle: $(CONFIG.max_phase_channels) (auto-angepasst)")
    else
        println("  Input-Kanäle: $(CONFIG.input_channels)")
    end
    println("  Output-Kanäle: $(CONFIG.output_channels)")
    
    # Multi-Kristall Parameter
    println("\nMulti-Kristall:")
    println("  Kristalle pro Sample: $(CONFIG.min_crystals)-$(CONFIG.max_crystals)")
    println("  Mindestabstand: $(CONFIG.crystal_spacing_min)")
    println("  Verschiedene Dichten: $(CONFIG.allow_different_densities)")
    println("  Radius-Variation: ±$(CONFIG.radius_variation)")
    println("  Dichte-Variation: ±$(CONFIG.density_variation)")
    
    # Training-Parameter
    println("\nTraining:")
    println("  Samples: $(CONFIG.dataset_size)")
    println("  Epochen: $(CONFIG.num_epochs)")
    println("  Batch-Größe: $(CONFIG.batch_size)")
    println("  Lernrate: $(CONFIG.learning_rate)")
    println("  GPU: $(CONFIG.use_gpu)")
    
    # LaMEM-Bereiche
    println("\nLaMEM-Parameter-Bereiche:")
    println("  Viskosität: $(CONFIG.eta_range[1]) - $(CONFIG.eta_range[2]) Pa·s")
    println("  Dichtedifferenz: $(CONFIG.delta_rho_range[1]) - $(CONFIG.delta_rho_range[2]) kg/m³")
    println("  Positionen: $(CONFIG.position_range[1]) - $(CONFIG.position_range[2])")
    println("  Radien: $(CONFIG.radius_range[1]) - $(CONFIG.radius_range[2])")
    
    println("\nSpeicher:")
    println("  Checkpoints: $(CONFIG.checkpoint_dir)")
    println("  Speichern alle: $(CONFIG.save_every_n_epochs) Epochen")
    
    # Validierungen
    println("\n" * "="^30 * " VALIDIERUNG " * "="^30)
    
    warnings = []
    errors = []
    
    # Kritische Überprüfungen
    if CONFIG.max_crystals > CONFIG.max_phase_channels - 1
        push!(errors, "max_crystals ($(CONFIG.max_crystals)) > max_phase_channels-1 ($(CONFIG.max_phase_channels-1))")
    end
    
    if CONFIG.min_crystals > CONFIG.max_crystals
        push!(errors, "min_crystals ($(CONFIG.min_crystals)) > max_crystals ($(CONFIG.max_crystals))")
    end
    
    if CONFIG.crystal_spacing_min > 1.0
        push!(errors, "crystal_spacing_min ($(CONFIG.crystal_spacing_min)) zu groß für [-1,1] Domain")
    end
    
    # Warnungen
    if CONFIG.batch_size > 2 && CONFIG.phase_encoding == "multi_channel"
        push!(warnings, "Große Batch-Größe ($(CONFIG.batch_size)) mit multi_channel könnte Speicherprobleme verursachen")
    end
    
    if CONFIG.dataset_size < 10
        push!(warnings, "Sehr kleiner Datensatz ($(CONFIG.dataset_size)) - nur für Tests geeignet")
    end
    
    if CONFIG.num_epochs < 5
        push!(warnings, "Sehr wenige Epochen ($(CONFIG.num_epochs)) - möglicherweise unzureichend für Konvergenz")
    end
    
    if CONFIG.use_gpu && !CUDA.functional()
        push!(warnings, "GPU aktiviert, aber CUDA nicht verfügbar - wird auf CPU umschalten")
    end
    
    # Ausgabe der Validierung
    if !isempty(errors)
        println("FEHLER:")
        for error in errors
            println("  ❌ $error")
        end
        println("\n⚠️ KONFIGURATION MUSS KORRIGIERT WERDEN!")
        return false
    end
    
    if !isempty(warnings)
        println("WARNUNGEN:")
        for warning in warnings
            println("  ⚠️ $warning")
        end
    end
    
    if isempty(errors) && isempty(warnings)
        println("✅ KONFIGURATION VALID - KEINE PROBLEME GEFUNDEN")
    elseif isempty(errors)
        println("✅ KONFIGURATION VALID - NUR WARNUNGEN")
    end
    
    println("="^72)
    return true
end

# ==================== KONFIGURATIONS-HILFSFUNKTIONEN ====================

function get_effective_input_channels()
    """
    Gibt die tatsächliche Anzahl Input-Kanäle zurück
    """
    if CONFIG.phase_encoding == "multi_channel"
        return CONFIG.max_phase_channels
    else
        return CONFIG.input_channels
    end
end

function get_gpu_info()
    """
    Gibt GPU-Informationen aus
    """
    if CONFIG.use_gpu
        if CUDA.functional()
            println("GPU-Info:")
            println("  CUDA verfügbar: ✅")
            println("  GPU-Gerät: $(CUDA.device())")
            println("  GPU-Speicher: $(round(CUDA.available_memory() / 1024^3, digits=2)) GB verfügbar")
            return true
        else
            println("GPU-Info:")
            println("  CUDA verfügbar: ❌")
            println("  Fallback auf CPU")
            return false
        end
    else
        println("GPU-Info: CPU-Modus aktiviert")
        return false
    end
end

function estimate_memory_usage()
    """
    Schätzt Speicherverbrauch ab
    """
    h, w = CONFIG.image_size
    input_ch = get_effective_input_channels()
    batch_size = CONFIG.batch_size
    
    # Eingabe-Tensor Größe
    input_size = h * w * input_ch * batch_size * 4  # Float32 = 4 Bytes
    
    # Output-Tensor Größe
    output_size = h * w * CONFIG.output_channels * batch_size * 4
    
    # Grobe Modell-Parameter Schätzung (UNet)
    model_params = 15_000_000 * 4  # ~15M Parameter × 4 Bytes
    
    total_mb = (input_size + output_size + model_params) / 1024^2
    
    println("Speicher-Schätzung:")
    println("  Input: $(round(input_size / 1024^2, digits=1)) MB")
    println("  Output: $(round(output_size / 1024^2, digits=1)) MB")
    println("  Modell: $(round(model_params / 1024^2, digits=1)) MB")
    println("  Total: $(round(total_mb, digits=1)) MB")
    
    if CONFIG.use_gpu && total_mb > 4000
        println("  ⚠️ Hoher Speicherverbrauch - reduziere batch_size oder image_size")
    end
end

# ==================== MULTI-KRISTALL LAMEM FUNKTION ====================

function LaMEM_Multi_crystal_fixed(; η=1e20, Δρ=200, cen_2D=[(0.0, 0.0)], R=[0.1])
    """
    Erweiterte LaMEM Funktion für mehrere Kristalle
    
    Parameter:
    - η: Viskosität der Matrix [Pa.s]
    - Δρ: Dichtedifferenz [kg/m³] - kann Skalar oder Array sein
    - cen_2D: Array von (x,z) Positionen der Kristalle
    - R: Radien der Kristalle - kann Skalar oder Array sein
    """
    η_crystal = 1e4*η
    ρ_magma = 2700
    
    # Dimensionen wie zuvor
    target_h, target_w = CONFIG.image_size
    nel_h, nel_w = target_h - 1, target_w - 1
    
    model = Model(
        Grid(nel=(nel_h, nel_w), x=[-1,1], z=[-1,1]), 
        Time(nstep_max=1), 
        Output(out_strain_rate=1)
    )
    
    # Matrix Phase (ID=0)
    matrix = Phase(ID=0, Name="matrix", eta=η, rho=ρ_magma)
    add_phase!(model, matrix)
    
    # Kristall Phasen erstellen (ID=1,2,3,...)
    for i in 1:length(cen_2D)
        # Dichtedifferenz bestimmen
        if isa(Δρ, Array) && length(Δρ) >= i
            delta_rho_i = Δρ[i]
        elseif isa(Δρ, Array)
            delta_rho_i = Δρ[end]  # Letzten Wert wiederverwenden
        else
            delta_rho_i = Δρ  # Skalar für alle verwenden
        end
        
        rho_crystal = ρ_magma + delta_rho_i
        crystal = Phase(ID=i, Name="crystal_$i", eta=η_crystal, rho=rho_crystal)
        add_phase!(model, crystal)
    end

    # Kristalle als Sphären hinzufügen
    for i = 1:length(cen_2D)
        # Radius bestimmen
        if isa(R, Array) && length(R) >= i
            radius_i = R[i]
        elseif isa(R, Array)
            radius_i = R[end]  # Letzten Wert wiederverwenden
        else
            radius_i = R  # Skalar für alle verwenden
        end
        
        add_sphere!(model, 
                   cen=(cen_2D[i][1], 0.0, cen_2D[i][2]), 
                   radius=radius_i,
                   phase=ConstantPhase(i))  # Phase ID = i (1,2,3,...)
    end
   
    # LaMEM ausführen
    run_lamem(model, 1)
    data, _ = read_LaMEM_timestep(model, 1)

    # Daten extrahieren
    x_vec_1D = data.x.val[:,1,1]
    z_vec_1D = data.z.val[1,1,:]
    phase = data.fields.phase[:,1,:]
    Vx = data.fields.velocity[1][:,1,:]
    Vz = data.fields.velocity[3][:,1,:]
    
    # Stokes-Geschwindigkeit basierend auf erstem/stärkstem Kristall
    first_delta_rho = isa(Δρ, Array) ? Δρ[1] : Δρ
    first_radius = isa(R, Array) ? R[1] : R
    V_stokes = 2/9*first_delta_rho*9.81*(first_radius*1000)^2/(η)
    V_stokes_cm_year = V_stokes * 100 * (3600*24*365.25)
    
    # Dimensionen-Sicherheitscheck
    actual_size = size(phase)
    if actual_size != CONFIG.image_size
        if CONFIG.verbose
            println("   Warnung: LaMEM lieferte $actual_size, schneide auf $(CONFIG.image_size) zu")
        end
        
        # Zuschneiden auf Zielgröße
        phase = phase[1:target_h, 1:target_w]
        Vx = Vx[1:target_h, 1:target_w]
        Vz = Vz[1:target_h, 1:target_w]
        x_vec_1D = x_vec_1D[1:target_h]
        z_vec_1D = z_vec_1D[1:target_w]
    end

    return x_vec_1D, z_vec_1D, phase, Vx, Vz, V_stokes_cm_year
end

# ==================== MULTI-KRISTALL PARAMETERGENERIERUNG ====================

function check_crystal_overlap(positions, radii, min_spacing)
    """
    Prüft ob Kristalle zu nah beieinander sind
    """
    for i in 1:length(positions)
        for j in (i+1):length(positions)
            pos1 = positions[i]
            pos2 = positions[j]
            dist = sqrt((pos1[1] - pos2[1])^2 + (pos1[2] - pos2[2])^2)
            min_required = radii[i] + radii[j] + min_spacing
            if dist < min_required
                return false
            end
        end
    end
    return true
end

function generate_multi_crystal_params()
    """
    Generiert Parameter für mehrere Kristalle mit Kollisionsvermeidung
    """
    # Anzahl Kristalle
    num_crystals = rand(CONFIG.min_crystals:CONFIG.max_crystals)
    
    # Basis-Parameter
    η_min, η_max = CONFIG.eta_range
    η = 10^(rand() * (log10(η_max) - log10(η_min)) + log10(η_min))
    
    # Kristall-spezifische Parameter generieren
    positions = []
    radii = []
    densities = []
    
    max_attempts = 100
    attempt = 0
    
    while length(positions) < num_crystals && attempt < max_attempts
        attempt += 1
        
        # Neue Position generieren
        pos_min, pos_max = CONFIG.position_range
        x_pos = rand() * (pos_max - pos_min) + pos_min
        z_pos = rand() * (0.8 - 0.2) + 0.2
        new_position = (x_pos, z_pos)
        
        # Neuen Radius generieren
        rad_min, rad_max = CONFIG.radius_range
        base_radius = rand() * (rad_max - rad_min) + rad_min
        radius_variation = rand(-CONFIG.radius_variation:0.001:CONFIG.radius_variation)
        new_radius = max(rad_min, base_radius + radius_variation)
        
        # Neue Dichtedifferenz generieren
        base_density = rand(CONFIG.delta_rho_range[1]:CONFIG.delta_rho_range[2])
        density_variation = rand(-CONFIG.density_variation:CONFIG.density_variation)
        new_density = max(50, base_density + density_variation)  # Mindestens 50 kg/m³
        
        # Temporäre Listen für Kollisionsprüfung
        temp_positions = copy(positions)
        temp_radii = copy(radii)
        push!(temp_positions, new_position)
        push!(temp_radii, new_radius)
        
        # Kollisionsprüfung
        if length(positions) == 0 || check_crystal_overlap(temp_positions, temp_radii, CONFIG.crystal_spacing_min)
            push!(positions, new_position)
            push!(radii, new_radius)
            push!(densities, new_density)
        end
    end
    
    if length(positions) == 0
        # Fallback: einzelner Kristall
        pos_min, pos_max = CONFIG.position_range
        x_pos = rand() * (pos_max - pos_min) + pos_min
        z_pos = rand() * (0.8 - 0.2) + 0.2
        
        rad_min, rad_max = CONFIG.radius_range
        radius = rand() * (rad_max - rad_min) + rad_min
        
        density = rand(CONFIG.delta_rho_range[1]:CONFIG.delta_rho_range[2])
        
        positions = [(x_pos, z_pos)]
        radii = [radius]
        densities = [density]
    end
    
    return (
        η = η,
        Δρ = densities,
        cen_2D = positions,
        R = radii,
        num_crystals = length(positions)
    )
end

# ==================== PHASEN-KORREKTUR UND KODIERUNG ====================

function correct_phase_values(phase_field)
    """
    Korrigiert kontinuierliche Phasen-Werte zu diskreten Werten
    """
    corrected = similar(phase_field, Int)
    
    # Schwellwerte für Phasen-Zuordnung
    for i in eachindex(phase_field)
        val = phase_field[i]
        if val < 0.5
            corrected[i] = 0  # Matrix
        elseif val < 1.5
            corrected[i] = 1  # Kristall 1
        elseif val < 2.5
            corrected[i] = 2  # Kristall 2
        else
            corrected[i] = 3  # Kristall 3
        end
    end
    
    return Float32.(corrected)
end

function encode_phase_field_multi_channel(phase_field, max_channels=CONFIG.max_phase_channels)
    """
    Kodiert Phasenfeld als Multi-Channel Array
    Kanal 1: Matrix (Phase 0)
    Kanal 2-4: Kristalle (Phase 1,2,3)
    """
    h, w = size(phase_field)
    encoded = zeros(Float32, h, w, max_channels, 1)
    
    for channel in 1:max_channels
        if channel == 1
            # Matrix-Kanal (Phase 0)
            encoded[:, :, channel, 1] = Float32.(phase_field .== 0.0f0)
        else
            # Kristall-Kanäle (Phase 1,2,3,...)
            crystal_phase = Float32(channel - 1)
            encoded[:, :, channel, 1] = Float32.(phase_field .== crystal_phase)
        end
    end
    
    return encoded
end

function encode_phase_field_single_channel(phase_field)
    """
    Kodiert Phasenfeld als Single-Channel (wie bisher)
    """
    h, w = size(phase_field)
    # Normalisiere Phase-IDs auf 0-1 Bereich
    max_phase = maximum(phase_field)
    if max_phase > 0
        normalized = Float32.(phase_field ./ max_phase)
    else
        normalized = Float32.(phase_field)
    end
    
    return reshape(normalized, h, w, 1, 1)
end

# ==================== ERWEITERTE DATENGENERIERUNG ====================

function generate_multi_crystal_sample_fixed(sample_id)
    """
    Generiert ein Multi-Kristall Trainingsbeispiel - VOLLSTÄNDIG KORRIGIERT
    """
    if CONFIG.verbose && sample_id % 50 == 0
        println("Generiere Multi-Kristall Sample $sample_id/$(CONFIG.dataset_size)...")
    end
    
    # Multi-Kristall Parameter generieren
    params = generate_multi_crystal_params()
    
    try
        # Parameter für LaMEM vorbereiten (OHNE num_crystals)
        params_for_lamem = (
            η = params.η,
            Δρ = params.Δρ,
            cen_2D = params.cen_2D,
            R = params.R
        )
        
        # LaMEM-Simulation mit mehreren Kristallen
        x, z, phase, Vx, Vz, V_stokes = LaMEM_Multi_crystal_fixed(;params_for_lamem...)
        
        # Dimensionsprüfung
        if size(phase) != CONFIG.image_size
            error("Sample $sample_id: Falsche Dimensionen $(size(phase))")
        end
        
        # Debug-Info mit Phasen-Analyse
        if CONFIG.verbose && sample_id <= 5
            unique_phases = unique(phase)
            n_unique = length(unique_phases)
            phase_range = (minimum(phase), maximum(phase))
            println("   Sample $sample_id: $(params.num_crystals) Kristalle")
            println("   Phasen-Info: $n_unique eindeutige Werte, Bereich: $phase_range")
            
            # Erwartete vs. tatsächliche Phasen
            expected_phases = 0:params.num_crystals
            println("   Erwartet: $expected_phases")
            if n_unique <= 10
                println("   Tatsächlich: $unique_phases")
            else
                println("   Tatsächlich: erste 10 Werte: $(unique_phases[1:10])")
            end
        end
        
        return (
            phase = phase, 
            vx = Vx, 
            vz = Vz, 
            v_stokes = V_stokes,
            params = params
        )
    catch e
        if CONFIG.verbose
            println("   Fehler bei Multi-Kristall Sample $sample_id: $e")
        end
        return nothing
    end
end

# ==================== ERWEITERTE DATENVORVERARBEITUNG ====================

function preprocess_multi_crystal(sample_data)
    """
    Vorverarbeitung für Multi-Kristall Samples - KORRIGIERT
    """
    h, w = CONFIG.image_size
    
    # Dimensionsprüfung
    if size(sample_data.phase) != (h, w)
        error("Sample hat falsche Dimensionen: " * string(size(sample_data.phase)) * " statt " * string(h, ", ", w))
    end
    
    # DEBUG: Phasen-Werte prüfen
    unique_phases = unique(sample_data.phase)
    if CONFIG.verbose && length(unique_phases) > 10
        println("    Debug: Zu viele eindeutige Phasen-Werte ($(length(unique_phases))) - korrigiere...")
    end
    
    # KRITISCH: Phasen-Werte korrigieren
    phase_corrected = correct_phase_values(sample_data.phase)
    unique_corrected = unique(phase_corrected)
    
    if CONFIG.verbose
        println("    Debug: Nach Korrektur: $unique_corrected")
    end
    
    # Phasenfeld-Kodierung basierend auf Konfiguration
    if CONFIG.phase_encoding == "multi_channel"
        phase_input = encode_phase_field_multi_channel(phase_corrected)
        input_channels = CONFIG.max_phase_channels
    else
        phase_input = encode_phase_field_single_channel(phase_corrected)
        input_channels = 1
    end
    
    # Geschwindigkeiten normalisieren (wie bisher)
    vx_norm = Float32.(sample_data.vx ./ sample_data.v_stokes)
    vz_norm = Float32.(sample_data.vz ./ sample_data.v_stokes)
    
    # Velocity target
    velocity_target = zeros(Float32, h, w, 2, 1)
    velocity_target[:, :, 1, 1] .= vx_norm
    velocity_target[:, :, 2, 1] .= vz_norm
    
    return (phase_input, velocity_target, input_channels)
end

# ==================== ERWEITERTE MODELL-ERSTELLUNG ====================

function create_multi_crystal_unet()
    """
    Erstellt UNet für Multi-Kristall mit flexiblen Input-Kanälen
    """
    if CONFIG.phase_encoding == "multi_channel"
        input_ch = CONFIG.max_phase_channels
        println("Erstelle Multi-Channel UNet mit $input_ch Eingabe-Kanälen")
    else
        input_ch = 1
        println("Erstelle Single-Channel UNet")
    end
    
    output_ch = CONFIG.output_channels
    
    # Encoder (angepasst für variable Input-Kanäle)
    encoder1 = Chain(
        Conv((3, 3), input_ch => 32, relu, pad=SamePad()),
        BatchNorm(32),
        Conv((3, 3), 32 => 32, relu, pad=SamePad()),
        BatchNorm(32)
    )
    
    # ==================== FEHLENDE TEILE AB ZEILE ~500 ====================

# Fortsetzung von create_multi_crystal_unet():
encoder2 = Chain(
    MaxPool((2,2)),
    Conv((3, 3), 32 => 64, relu, pad=SamePad()),
    BatchNorm(64),
    Conv((3, 3), 64 => 64, relu, pad=SamePad()),
    BatchNorm(64)
)

encoder3 = Chain(
    MaxPool((2,2)),
    Conv((3, 3), 64 => 128, relu, pad=SamePad()),
    BatchNorm(128),
    Conv((3, 3), 128 => 128, relu, pad=SamePad()),
    BatchNorm(128)
)

encoder4 = Chain(
    MaxPool((2,2)),
    Conv((3, 3), 128 => 256, relu, pad=SamePad()),
    BatchNorm(256),
    Conv((3, 3), 256 => 256, relu, pad=SamePad()),
    BatchNorm(256)
)

bottleneck = Chain(
    MaxPool((2,2)),
    Conv((3,3), 256 => 512, relu, pad=SamePad()),
    BatchNorm(512),
    Dropout(0.5),
    Conv((3,3), 512 => 512, relu, pad=SamePad()),
    BatchNorm(512)
)

# Decoder
decoder4 = ConvTranspose((2, 2), 512 => 256, stride=2)
decoder4_1 = Chain(
    Conv((3,3), 512 => 256, relu, pad=SamePad()),
    BatchNorm(256),
    Conv((3,3), 256 => 256, relu, pad=SamePad()),
    BatchNorm(256)
)

decoder3 = ConvTranspose((2,2), 256 => 128, stride=2)
decoder3_1 = Chain(
    Conv((3,3), 256 => 128, relu, pad=SamePad()),
    BatchNorm(128),
    Conv((3,3), 128 => 128, relu, pad=SamePad()),
    BatchNorm(128)
)

decoder2 = ConvTranspose((2,2), 128 => 64, stride=2)
decoder2_1 = Chain(
    Conv((3,3), 128 => 64, relu, pad=SamePad()),
    BatchNorm(64),
    Conv((3,3), 64 => 64, relu, pad=SamePad()),
    BatchNorm(64)
)

decoder1 = ConvTranspose((2,2), 64 => 32, stride=2)
decoder1_1 = Chain(
    Conv((3,3), 64 => 32, relu, pad=SamePad()),
    BatchNorm(32),
    Conv((3,3), 32 => 32, relu, pad=SamePad()),
    BatchNorm(32),
    Dropout(0.2),
    Conv((1,1), 32 => output_ch)
)

return VelocityUNet(encoder1, encoder2, encoder3, encoder4, bottleneck,
                    decoder4, decoder4_1, decoder3, decoder3_1,
                    decoder2, decoder2_1, decoder1, decoder1_1)
end

# ==================== MULTI-KRISTALL TRAINING PIPELINE ====================

function run_multi_crystal_training_fixed()
"""
Komplette Multi-Kristall Training Pipeline - VOLLSTÄNDIG KORRIGIERT
"""
println("=== MULTI-KRISTALL VELOCITY UNET TRAINING (KORRIGIERT) ===")
println("Konfiguration:")
println("  Kristalle: $(CONFIG.min_crystals)-$(CONFIG.max_crystals) pro Sample")
println("  Phasenfeld: $(CONFIG.phase_encoding)")
println("  Samples: $(CONFIG.dataset_size)")

# 1. Multi-Kristall Test
println("\n1. TESTE MULTI-KRISTALL LAMEM")
test_params = generate_multi_crystal_params()
println("  Test-Parameter: $(test_params.num_crystals) Kristalle")

try
    params_for_lamem = (
        η = test_params.η,
        Δρ = test_params.Δρ,
        cen_2D = test_params.cen_2D,
        R = test_params.R
    )
    
    x, z, phase, vx, vz, v_stokes = LaMEM_Multi_crystal_fixed(;params_for_lamem...)
    unique_phases = unique(phase)
    phase_stats = (minimum(phase), maximum(phase), length(unique_phases))
    println("  Test erfolgreich: Größe $(size(phase))")
    println("  Phasen-Statistik: Min=$(phase_stats[1]), Max=$(phase_stats[2]), Anzahl=$(phase_stats[3])")
    
    # Phasen-Korrektur testen
    phase_corrected = correct_phase_values(phase)
    unique_corrected = unique(phase_corrected)
    println("  Nach Korrektur: $unique_corrected")
    
catch e
    error("Multi-Kristall LaMEM Test fehlgeschlagen: $e")
end

# 2. Datengenerierung
println("\n2. GENERIERE MULTI-KRISTALL DATEN")
dataset = []
successful = 0

for i in 1:CONFIG.dataset_size
    sample = generate_multi_crystal_sample_fixed(i)
    if sample !== nothing
        push!(dataset, sample)
        successful += 1
    end
    
    if i % CONFIG.memory_cleanup_frequency == 0
        GC.gc()
    end
end

println("  Erfolgreich: $successful/$(CONFIG.dataset_size)")

if successful == 0
    error("Keine Samples generiert!")
end

# 3. Datenvorverarbeitung
println("\n3. DATENVORVERARBEITUNG")
processed_data = []
input_channels = nothing

for (i, sample) in enumerate(dataset)
    try
        phase_input, velocity_target, channels = preprocess_multi_crystal(sample)
        push!(processed_data, (phase_input, velocity_target))
        if input_channels === nothing
            input_channels = channels
        end
        
        if i <= 3  # Debug für erste 3 Samples
            println("    Sample $i verarbeitet - Input: $(size(phase_input)), Target: $(size(velocity_target))")
        end
        
    catch e
        println("  Fehler bei Sample $i: $e")
    end
end

println("  Verarbeitet: $(length(processed_data)) Samples")
println("  Input-Kanäle: $input_channels")

if length(processed_data) == 0
    error("Keine Samples vorverarbeitet!")
end

# 4. Modell erstellen
println("\n4. ERSTELLE MULTI-KRISTALL MODELL")
model = create_multi_crystal_unet()

# GPU-Transfer falls verfügbar
if CONFIG.use_gpu && CUDA.functional()
    model = gpu(model)
    println("  Modell auf GPU verschoben")
end

# 5. Training
println("\n5. TRAINING")
mkpath(CONFIG.checkpoint_dir)

opt_state = Optimisers.setup(Optimisers.Adam(CONFIG.learning_rate), model)
losses = Float32[]

for epoch in 1:CONFIG.num_epochs
    println("\n--- Epoche $epoch/$(CONFIG.num_epochs) ---")
    
    total_loss = 0f0
    batch_count = 0
    
    for (batch_idx, (phase_batch, velocity_batch)) in enumerate(processed_data)
        try
            # GPU-Transfer falls verfügbar
            if CONFIG.use_gpu && CUDA.functional()
                phase_batch = gpu(phase_batch)
                velocity_batch = gpu(velocity_batch)
            end
            
            # Debug für erste Epoche
            if epoch == 1 && batch_idx <= 2
                println("    Batch $batch_idx - Input: $(size(phase_batch)), Target: $(size(velocity_batch))")
                println("    Input-Bereich: $(extrema(phase_batch))")
                println("    Target-Bereich: $(extrema(velocity_batch))")
            end
            
            ∇model = gradient(m -> mse(m(phase_batch), velocity_batch), model)[1]
            batch_loss = mse(model(phase_batch), velocity_batch)
            
            opt_state, model = Optimisers.update!(opt_state, model, ∇model)
            
            total_loss += batch_loss
            batch_count += 1
            
            if epoch == 1 && batch_idx <= 2
                println("    Batch $batch_idx Loss: $(round(batch_loss, digits=6))")
            end
            
        catch e
            println("  Fehler bei Batch $batch_idx: $e")
            println("  Stacktrace: ")
            for (exc, bt) in Base.catch_stack()
                showerror(stdout, exc, bt)
                println()
            end
        end
    end
    
    avg_loss = batch_count > 0 ? total_loss / batch_count : NaN32
    push!(losses, avg_loss)
    
    println("Epoche $epoch: Loss = $(round(avg_loss, digits=6))")
    
    # Checkpoint
    if epoch % CONFIG.save_every_n_epochs == 0
        model_cpu = cpu(model)
        checkpoint_path = joinpath(CONFIG.checkpoint_dir, "multi_crystal_checkpoint_epoch_$(epoch).bson")
        @save checkpoint_path model_cpu
        println("  Checkpoint gespeichert")
    end
    
    # Memory cleanup
    GC.gc()
    if CONFIG.use_gpu && CUDA.functional()
        CUDA.reclaim()
    end
end

# Finales Modell speichern
final_model_cpu = cpu(model)
final_path = joinpath(CONFIG.checkpoint_dir, "final_multi_crystal_model.bson")
@save final_path final_model_cpu

println("\n" * "="^60)
println("MULTI-KRISTALL TRAINING ABGESCHLOSSEN")
println("="^60)
println("  Samples: $(length(dataset))")
println("  Input-Kanäle: $input_channels")
println("  Finaler Loss: $(round(losses[end], digits=6))")
println("  Modell gespeichert: $final_path")
println("="^60)

return model, losses, dataset
end

# ==================== AUTOMATISCHE INITIALISIERUNG ====================

# Validierung beim Laden ausführen
println("Lade Multi-Kristall Konfiguration...")
config_valid = validate_and_print_config()

if config_valid
gpu_available = get_gpu_info()
estimate_memory_usage()

println("\n🚀 KONFIGURATION GELADEN UND BEREIT!")
println("Verwende: run_multi_crystal_training_fixed() zum Starten")
else
println("❌ KONFIGURATIONSFEHLER - Bitte korrigieren vor dem Fortfahren!")
end

# ==================== VERWENDUNGSBEISPIELE ====================

println("\n" * "="^60)
println("MULTI-KRISTALL SYSTEM GELADEN")
println("="^60)
println("Verwendung:")
println("  # Multi-Kristall Training starten:")
println("  model, losses, data = run_multi_crystal_training_fixed()")
println("  ")
println("  # Einzelne Multi-Kristall Simulation testen:")
println("  params = generate_multi_crystal_params()")
println("  x, z, phase, vx, vz, v_stokes = LaMEM_Multi_crystal_fixed(")
println("      η = params.η, Δρ = params.Δρ, cen_2D = params.cen_2D, R = params.R)")
println("  ")
println("  # Konfiguration anpassen:")
println("  # Für größere Datasets: CONFIG = merge(CONFIG, (dataset_size = 100,))")
println("  # Für mehr Epochen: CONFIG = merge(CONFIG, (num_epochs = 20,))")
println("="^60)

# ==================== CONVENIENCE-FUNKTIONEN ====================

function quick_test()
"""
Schneller Test der Multi-Kristall Funktionalität
"""
println("=== SCHNELLER MULTI-KRISTALL TEST ===")

# Parameter generieren
params = generate_multi_crystal_params()
println("Generierte Parameter: $(params.num_crystals) Kristalle")

# LaMEM-Test
try
    params_for_lamem = (η = params.η, Δρ = params.Δρ, cen_2D = params.cen_2D, R = params.R)
    x, z, phase, vx, vz, v_stokes = LaMEM_Multi_crystal_fixed(;params_for_lamem...)
    
    println("✅ LaMEM erfolgreich: $(size(phase))")
    println("  Phasen: $(length(unique(phase))) eindeutige Werte")
    println("  v_stokes: $(round(v_stokes, digits=3)) cm/Jahr")
    
    # Phasen-Korrektur testen
    phase_corrected = correct_phase_values(phase)
    println("  Nach Korrektur: $(unique(phase_corrected))")
    
    return true
catch e
    println("❌ LaMEM-Fehler: $e")
    return false
end
end

function show_config_summary()
"""
Zeigt eine kompakte Konfigurationsübersicht
"""
println("=== AKTUELLE KONFIGURATION ===")
println("Modell: $(CONFIG.image_size), $(CONFIG.phase_encoding), $(get_effective_input_channels()) Kanäle")
println("Training: $(CONFIG.dataset_size) Samples, $(CONFIG.num_epochs) Epochen, GPU=$(CONFIG.use_gpu)")
println("Kristalle: $(CONFIG.min_crystals)-$(CONFIG.max_crystals), Abstand=$(CONFIG.crystal_spacing_min)")
println("LaMEM: η=$(CONFIG.eta_range), Δρ=$(CONFIG.delta_rho_range), R=$(CONFIG.radius_range)")
end

# Zeige Zusammenfassung beim Laden
show_config_summary()