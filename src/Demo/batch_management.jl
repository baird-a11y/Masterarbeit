# =============================================================================
# BATCH MANAGEMENT MODULE
# =============================================================================
# Speichern als: batch_management.jl

using CUDA
using Statistics
using StatsBase
using Random  # HINZUGEFÜGT für shuffle

include("gpu_utils.jl")
include("data_processing.jl")  # HINZUGEFÜGT für preprocess_lamem_sample_normalized

# Batch-Größen nach Auflösung
const BATCH_SIZES = Dict(
    64 => 16,
    128 => 8,
    256 => 4,
    512 => 1
)

"""
Gibt aktuelle GPU-Speicher-Information zurück
"""
function get_gpu_memory_info()
    if !CUDA.functional()
        return (total=0, available=0, used=0, percent_used=0.0)
    end
    
    total = CUDA.total_memory()
    available = CUDA.available_memory()
    used = total - available
    percent_used = (used / total) * 100
    
    return (
        total = total,
        available = available,
        used = used,
        percent_used = percent_used
    )
end

"""
Schätzt Speicherbedarf für Tensoren in Bytes
"""
function estimate_tensor_memory(resolution, batch_size, channels=3)
    elements_input = resolution * resolution * 1 * batch_size
    elements_output = resolution * resolution * 2 * batch_size
    total_elements = (elements_input + elements_output) * 4  # Faktor für Training
    
    bytes_needed = total_elements * sizeof(Float32)
    return bytes_needed
end

"""
Berechnet sichere Batch-Größe basierend auf verfügbarem GPU-Speicher
"""
function safe_batch_size(resolution, max_memory_percent=80)
    if !CUDA.functional()
        return get(BATCH_SIZES, resolution, 1)
    end
    
    mem_info = get_gpu_memory_info()
    available_bytes = mem_info.available * (max_memory_percent / 100)
    
    max_theoretical = get(BATCH_SIZES, resolution, 1) * 2
    safe_size = 1
    
    for test_size in 1:max_theoretical
        needed_bytes = estimate_tensor_memory(resolution, test_size)
        if needed_bytes <= available_bytes
            safe_size = test_size
        else
            break
        end
    end
    
    return max(1, Int(floor(safe_size * 0.75)))
end

"""
Bestimme optimale Batch-Größe und erstelle Batch ohne GPU-OOM
"""
function create_adaptive_batch(samples, target_resolution; 
                              max_gpu_memory_percent=80, 
                              verbose=false,
                              use_gpu=false)
    if isempty(samples)
        error("Keine Samples für Batch-Erstellung!")
    end
    
    # Bestimme optimale Batch-Größe
    theoretical_batch_size = get(BATCH_SIZES, target_resolution, 1)
    safe_batch_size_val = safe_batch_size(target_resolution, max_gpu_memory_percent)
    actual_batch_size = min(theoretical_batch_size, safe_batch_size_val, length(samples))
    
    if verbose
        mem_info = get_gpu_memory_info()
        println("=== ADAPTIVE BATCH CREATION ===")
        println("Target resolution: $(target_resolution)x$(target_resolution)")
        println("Available samples: $(length(samples))")
        println("Theoretical batch size: $theoretical_batch_size")
        println("Safe batch size: $safe_batch_size_val")
        println("Actual batch size: $actual_batch_size")
        println("GPU memory: $(round(mem_info.percent_used, digits=1))% used")
    end
    
    # Wähle zufällige Samples
    selected_indices = StatsBase.sample(1:length(samples), actual_batch_size, replace=false)
    selected_samples = samples[selected_indices]
    
    # Preprocessing
    batch_phases = []
    batch_velocities = []
    successful_samples = 0
    
     for (i, sample) in enumerate(selected_samples)
        try
            # Sample entpacken
            if length(sample) == 8
                x, z, phase, vx, vz, exx, ezz, v_stokes = sample
            elseif length(sample) == 7
                x, z, phase, vx, vz, exx, v_stokes = sample
            else
                println("WARNUNG: Unbekanntes Sample-Format mit $(length(sample)) Elementen")
                continue
            end
            
            # Versuche beide Preprocessing-Funktionen
            phase_tensor, velocity_tensor = try
                # Zuerst die neue Funktion
                if isdefined(Main, :preprocess_lamem_sample_normalized)
                    preprocess_lamem_sample_normalized(
                        x, z, phase, vx, vz, v_stokes,
                        target_resolution=target_resolution
                    )
                else
                    # Fallback auf alte Funktion
                    preprocess_lamem_sample(
                        x, z, phase, vx, vz, v_stokes,
                        target_resolution=target_resolution
                    )
                end
            catch e
                if verbose
                    println("  Preprocessing-Fehler bei Sample $i: $e")
                end
                # Versuche direkte Verarbeitung
                phase_resized = Float32.(phase[1:target_resolution, 1:target_resolution])
                vx_resized = Float32.(vx[1:target_resolution, 1:target_resolution])
                vz_resized = Float32.(vz[1:target_resolution, 1:target_resolution])
                
                phase_tensor = reshape(phase_resized, target_resolution, target_resolution, 1, 1)
                vx_norm = vx_resized ./ Float32(v_stokes)
                vz_norm = vz_resized ./ Float32(v_stokes)
                velocity_tensor = cat(vx_norm, vz_norm, dims=3)
                velocity_tensor = reshape(velocity_tensor, target_resolution, target_resolution, 2, 1)
                
                (phase_tensor, velocity_tensor)
            end
            
            push!(batch_phases, phase_tensor)
            push!(batch_velocities, velocity_tensor)
            successful_samples += 1
            
        catch e
            if verbose || i == 1  # Zeige Fehler beim ersten Sample immer
                println("  FEHLER: Sample $i konnte nicht verarbeitet werden: $e")
                println("  Sample-Typ: $(typeof(sample))")
                if isa(sample, Tuple) && length(sample) >= 5
                    println("  Phase-Größe: $(size(sample[3]))")
                    println("  Vx-Größe: $(size(sample[4]))")
                    println("  Vz-Größe: $(size(sample[5]))")
                end
            end
            continue
        end
    end
    
    if successful_samples == 0
        println("KRITISCHER FEHLER: Keine Samples konnten verarbeitet werden!")
        println("  Anzahl Samples: $(length(selected_samples))")
        println("  Target Resolution: $target_resolution")
        if !isempty(selected_samples)
            println("  Erstes Sample-Format: $(typeof(selected_samples[1]))")
            if isa(selected_samples[1], Tuple)
                println("  Sample-Länge: $(length(selected_samples[1]))")
            end
        end
        error("Keine Samples konnten verarbeitet werden!")
    end
    
    # Tensoren zusammenfügen mit GPU-Support
    try
        phase_batch = cat(batch_phases..., dims=4)
        velocity_batch = cat(batch_velocities..., dims=4)
        
        # Optionaler GPU-Transfer
        if use_gpu && CUDA.functional()
            # Prüfe ob genug GPU-Speicher
            batch_bytes = sizeof(phase_batch) + sizeof(velocity_batch)
            if batch_bytes < CUDA.available_memory() * 0.8
                phase_batch = safe_gpu(phase_batch)
                velocity_batch = safe_gpu(velocity_batch)
            else
                if verbose
                    println("Batch zu groß für GPU, bleibe bei CPU")
                end
            end
        end
        
        if verbose
            println("Batch erstellt: $(size(phase_batch)), $(size(velocity_batch))")
            println("Erfolgreiche Samples: $successful_samples/$actual_batch_size")
        end
        
        return phase_batch, velocity_batch, successful_samples
        
    catch e
        if occursin("out of memory", lowercase(string(e)))
            println("Memory-Fehler beim Batch-Erstellen, versuche kleineren Batch")
            # Rekursiver Aufruf mit halbierter Sample-Anzahl
            if length(samples) > 1
                return create_adaptive_batch(samples[1:end÷2], target_resolution, 
                                           max_gpu_memory_percent=max_gpu_memory_percent,
                                           verbose=verbose, use_gpu=use_gpu)
            else
                error("Kann keinen Batch erstellen, selbst mit einem Sample!")
            end
        else
            error("Fehler beim Zusammenfügen der Tensoren: $e")
        end
    end
end

"""
Smart Batch Manager Struktur
"""
mutable struct BatchManager
    dataset::Vector
    target_resolution::Int
    max_memory_percent::Float64
    current_batch_size::Int
    memory_history::Vector{Float64}
    oom_count::Int
    adaptive_mode::Bool
    
    function BatchManager(dataset, target_resolution; max_memory_percent=80.0, adaptive_mode=true)
        new(dataset, target_resolution, max_memory_percent, 
            get(BATCH_SIZES, target_resolution, 1), Float64[], 0, adaptive_mode)
    end
end

"""
Aktualisiert Memory-Statistiken
"""
function update_memory_stats!(manager::BatchManager)
    mem_info = get_gpu_memory_info()
    push!(manager.memory_history, mem_info.percent_used)
    
    if length(manager.memory_history) > 10
        manager.memory_history = manager.memory_history[end-9:end]
    end
end

"""
Passt Batch-Größe basierend auf Memory-History an
"""
function adapt_batch_size!(manager::BatchManager)
    if !manager.adaptive_mode || length(manager.memory_history) < 3
        return
    end
    
    avg_memory = mean(manager.memory_history[end-2:end])
    
    if avg_memory > 90.0
        manager.current_batch_size = max(1, manager.current_batch_size ÷ 2)
        manager.oom_count += 1
        println("  Memory kritisch ($(round(avg_memory, digits=1))%) - Batch-Größe reduziert auf $(manager.current_batch_size)")
    elseif avg_memory < 60.0 && manager.oom_count == 0
        max_allowed = get(BATCH_SIZES, manager.target_resolution, 1)
        if manager.current_batch_size < max_allowed
            manager.current_batch_size = min(max_allowed, Int(ceil(manager.current_batch_size * 1.2)))
            println("  Memory niedrig ($(round(avg_memory, digits=1))%) - Batch-Größe erhöht auf $(manager.current_batch_size)")
        end
    end
end

"""
Überwacht GPU-Speicher und passt Batch-Größen automatisch an
"""
function smart_batch_manager(dataset, target_resolution; max_memory_percent=80, verbose=true)
    manager = BatchManager(dataset, target_resolution, max_memory_percent=max_memory_percent)
    
    if verbose
        println("=== SMART BATCH MANAGER ===")
        println("Dataset: $(length(dataset)) samples")
        println("Target resolution: $(target_resolution)x$(target_resolution)")
        println("Initial batch size: $(manager.current_batch_size)")
        if CUDA.functional()
            mem_info = get_gpu_memory_info()
            println("GPU memory: $(round(mem_info.total / 1e9, digits=2)) GB total")
        else
            println("CPU mode (no GPU)")
        end
    end
    
    # Generator für Batches
    batches = Channel{Tuple}(32) do ch
        batch_count = 0
        dataset_shuffled = Random.shuffle(dataset)  # Explizit Random.shuffle
        
        while batch_count * manager.current_batch_size < length(dataset_shuffled)
            batch_count += 1
            
            # Memory cleanup vor jedem Batch
            if CUDA.functional()
                GC.gc()
                CUDA.reclaim()
            else
                GC.gc()
            end
            
            # Memory-Status aktualisieren
            update_memory_stats!(manager)
            
            # Batch-Größe anpassen falls nötig
            if batch_count > 1
                adapt_batch_size!(manager)
            end
            
            # Sample-Bereich für aktuellen Batch
            start_idx = (batch_count - 1) * manager.current_batch_size + 1
            end_idx = min(start_idx + manager.current_batch_size - 1, length(dataset_shuffled))
            batch_samples = dataset_shuffled[start_idx:end_idx]
            
            if verbose && batch_count % 5 == 1
                mem_info = get_gpu_memory_info()
                println("Batch $batch_count: $(length(batch_samples)) samples, Memory: $(round(mem_info.percent_used, digits=1))%")
            end
            
            try
                # Erstelle Batch
                phase_batch, velocity_batch, successful = create_adaptive_batch(
                    batch_samples, target_resolution,
                    max_gpu_memory_percent=max_memory_percent,
                    verbose=false,
                    use_gpu=CUDA.functional()  # GPU nutzen wenn verfügbar
                )
                
                put!(ch, (phase_batch, velocity_batch, successful, batch_count))
                
            catch e
                if occursin("out of memory", string(e)) || occursin("OOM", string(e))
                    manager.oom_count += 1
                    manager.current_batch_size = max(1, manager.current_batch_size ÷ 2)
                    println("  GPU OOM! Batch-Größe reduziert auf $(manager.current_batch_size)")
                    continue
                else
                    println("  Batch $batch_count Fehler: $e")
                    continue
                end
            end
        end
        
        if verbose
            println("Batch-Manager abgeschlossen: $batch_count Batches, $(manager.oom_count) OOM-Events")
        end
    end
    
    return batches
end

println("Batch Management Module geladen!")
println("Verfügbare Funktionen:")
println("  - create_adaptive_batch(samples, target_resolution)")
println("  - smart_batch_manager(dataset, target_resolution)")
println("  - get_gpu_memory_info()")