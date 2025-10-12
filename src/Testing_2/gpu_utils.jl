# =============================================================================
# GPU UTILITIES MODULE
# =============================================================================


using CUDA
using Flux

"""
Prüft GPU-Verfügbarkeit und gibt Informationen aus
"""
function check_gpu_availability()
    if CUDA.functional()
        device = CUDA.device()
        total_mem = CUDA.total_memory() / 1e9
        free_mem = CUDA.available_memory() / 1e9
        
        println("=== GPU INFORMATION ===")
        println("GPU verfügbar: $(CUDA.name(device))")
        println("Speicher: $(round(free_mem, digits=2)) GB frei von $(round(total_mem, digits=2)) GB")
        
        return true
    else
        println("=== GPU INFORMATION ===")
        println("Keine GPU verfügbar oder CUDA nicht funktional")
        println("Training wird auf CPU ausgeführt")
        
        return false
    end
end

"""
Sicherer GPU-Transfer mit Fehlerbehandlung
"""
function safe_gpu(x)
    if CUDA.functional()
        try
            return gpu(x)
        catch e
            println("GPU-Transfer fehlgeschlagen: $e")
            println("Verwende CPU als Fallback")
            return x
        end
    else
        return x
    end
end

"""
Sicherer CPU-Transfer
"""
function safe_cpu(x)
    try
        return cpu(x)
    catch e
        return x  # Bereits auf CPU
    end
end

"""
Automatisches Memory-Management für GPU
"""
function gpu_memory_cleanup()
    if CUDA.functional()
        GC.gc()
        CUDA.reclaim()
        return true
    end
    return false
end

"""
Adaptive Device-Auswahl basierend auf Modellgröße
"""
function select_device(model_params::Int, batch_size::Int)
    if !CUDA.functional()
        return :cpu
    end
    
    # Schätze Memory-Bedarf (grobe Approximation)
    bytes_per_param = 4  # Float32
    gradient_factor = 3  # Model + Gradients + Optimizer State
    estimated_bytes = model_params * bytes_per_param * gradient_factor * batch_size
    estimated_gb = estimated_bytes / 1e9
    
    available_gb = CUDA.available_memory() / 1e9
    
    if estimated_gb < available_gb * 0.8  # 80% Safety Margin
        println("GPU-Training möglich (geschätzt: $(round(estimated_gb, digits=2)) GB)")
        return :gpu
    else
        println("Nicht genug GPU-Speicher (benötigt: $(round(estimated_gb, digits=2)) GB)")
        return :cpu
    end
end

"""
Wrapper für GPU-sichere Batch-Verarbeitung
"""
function process_batch_gpu_safe(model, phase_batch, velocity_batch, loss_fn; 
                               max_retries=2)
    device = isa(model, CuArray) || any(p -> isa(p, CuArray), Flux.params(model)) ? :gpu : :cpu
    
    for attempt in 1:max_retries
        try
            if device == :gpu
                # Stelle sicher, dass alles auf GPU ist
                model_gpu = safe_gpu(model)
                phase_gpu = safe_gpu(phase_batch)
                velocity_gpu = safe_gpu(velocity_batch)
                
                loss_val, grads = Flux.withgradient(() -> loss_fn(model_gpu, phase_gpu, velocity_gpu), model_gpu)
                
                return loss_val, grads, :gpu
            else
                # CPU-Fallback
                model_cpu = safe_cpu(model)
                phase_cpu = safe_cpu(phase_batch)
                velocity_cpu = safe_cpu(velocity_batch)
                
                loss_val, grads = Flux.withgradient(() -> loss_fn(model_cpu, phase_cpu, velocity_cpu), model_cpu)
                
                return loss_val, grads, :cpu
            end
            
        catch e
            if occursin("out of memory", lowercase(string(e))) && device == :gpu
                println("GPU OOM bei Versuch $attempt - Räume auf...")
                gpu_memory_cleanup()
                
                if attempt == max_retries
                    println("Wechsle zu CPU nach $max_retries GPU-Versuchen")
                    device = :cpu
                end
            else
                rethrow(e)
            end
        end
    end
end

println("GPU Utilities Module geladen!")
println("Verfügbare Funktionen:")
println("  - check_gpu_availability()")
println("  - safe_gpu(x) / safe_cpu(x)")
println("  - gpu_memory_cleanup()")