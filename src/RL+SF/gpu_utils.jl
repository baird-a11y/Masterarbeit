# =============================================================================
# GPU UTILITIES
# =============================================================================
# GPU Management und Fehlerbehandlung

using CUDA
using Flux

println("GPU Utilities wird geladen...")

# =============================================================================
# GPU AVAILABILITY CHECK
# =============================================================================

"""
    check_gpu_availability()

Prüft GPU-Verfügbarkeit und gibt Status aus.

# Returns
- `Bool`: true wenn GPU verfügbar und funktional
"""
function check_gpu_availability()
    println("\n GPU STATUS CHECK")
    println("="^60)
    
    if CUDA.functional()
        println("CUDA ist funktional")
        println("   Geräte: $(length(CUDA.devices()))")
        
        for (i, dev) in enumerate(CUDA.devices())
            println("   GPU $i: $(CUDA.name(dev))")
            mem_info = CUDA.memory_status(dev)
            println("   Memory: $(round(mem_info.free / 1e9, digits=2)) GB frei")
        end
        
        return true
    else
        println("CUDA nicht verfügbar")
        println("   Training wird auf CPU durchgeführt")
        return false
    end
end

# =============================================================================
# DEVICE SELECTION
# =============================================================================

"""
    setup_device(use_gpu::Bool=true)

Setup Training Device (GPU oder CPU).

# Arguments
- `use_gpu::Bool`: GPU nutzen wenn verfügbar (default: true)

# Returns
- `Function`: Device-Transfer-Funktion (gpu oder cpu)
"""
function setup_device(use_gpu::Bool=true)
    if use_gpu && CUDA.functional()
        println("GPU-Modus aktiviert")
        device = gpu
    else
        println("CPU-Modus aktiviert")
        device = cpu
    end
    
    return device
end

# =============================================================================
# MEMORY MANAGEMENT
# =============================================================================

"""
    clear_gpu_memory()

Räumt GPU-Speicher auf (Force Garbage Collection).
"""
function clear_gpu_memory()
    if CUDA.functional()
        CUDA.reclaim()
        GC.gc(true)
        println("🧹 GPU Memory aufgeräumt")
    end
end

"""
    get_gpu_memory_info()

Gibt aktuelle GPU-Speicher-Informationen aus.
"""
function get_gpu_memory_info()
    if CUDA.functional()
        mem = CUDA.memory_status()
        used_gb = (mem.total - mem.free) / 1e9
        total_gb = mem.total / 1e9
        percent = 100 * used_gb / total_gb
        
        println("GPU Memory:")
        println("   Verwendet: $(round(used_gb, digits=2)) GB / $(round(total_gb, digits=2)) GB")
        println("   Auslastung: $(round(percent, digits=1))%")
    else
        println("Keine GPU verfügbar")
    end
end

# =============================================================================
# ERROR HANDLING
# =============================================================================

"""
    safe_gpu_operation(f::Function)

Wrapper für GPU-Operationen mit Error-Handling.

Bei GPU-Fehler: Fallback auf CPU.
"""
function safe_gpu_operation(f::Function; fallback_to_cpu::Bool=true)
    try
        return f()
    catch e
        if isa(e, CUDA.CUDAError) || isa(e, CUDA.CUDAException)
            @warn "GPU Operation fehlgeschlagen: $e"
            
            if fallback_to_cpu
                println("⚠️  Fallback auf CPU")
                # Clear GPU Memory
                clear_gpu_memory()
                # Retry auf CPU
                return f()
            else
                rethrow(e)
            end
        else
            rethrow(e)
        end
    end
end

# =============================================================================
# BATCH SIZE OPTIMIZATION
# =============================================================================

"""
    estimate_optimal_batch_size(model, sample_size::Tuple{Int,Int})

Schätzt optimale Batch-Größe basierend auf verfügbarem GPU-Speicher.

# Arguments
- `model`: Flux Modell
- `sample_size::Tuple{Int,Int}`: Größe eines Samples (H, W)

# Returns
- `Int`: Empfohlene Batch-Größe
"""
function estimate_optimal_batch_size(model, sample_size::Tuple{Int,Int}=(256,256))
    if !CUDA.functional()
        return 4  # CPU: Kleine Batches
    end
    
    mem = CUDA.memory_status()
    available_gb = mem.free / 1e9
    
    # Konservative Schätzung
    # Pro Sample (256x256x2): ~0.5 MB
    # Pro Batch mit Gradienten: ~10x
    estimated_mb_per_sample = 5.0
    
    max_batch = floor(Int, (available_gb * 1000) / estimated_mb_per_sample)
    
    # Sicherheits-Cap
    recommended = min(max_batch, 16)
    recommended = max(recommended, 1)
    
    println("Batch Size Empfehlung: $recommended")
    println("   (Basierend auf $(round(available_gb, digits=2)) GB freiem GPU-Speicher)")
    
    return recommended
end

# =============================================================================
# INITIALIZATION
# =============================================================================

# Automatischer Check beim Laden
if CUDA.functional()
    println("GPU verfügbar: $(CUDA.name(CUDA.device()))")
else
    println("Kein GPU gefunden - CPU-Modus")
end

println("GPU Utilities geladen!")