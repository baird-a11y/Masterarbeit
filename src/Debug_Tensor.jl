using CUDA
using Flux


function tensor_debug(debugging)
    
if debugging==true
    function check_cuda_availability()
        println("=== CUDA-DIAGNOSE ===")
        
        if CUDA.functional()
            println("✓ CUDA ist verfügbar")
            println("GPU-Geräte: $(length(CUDA.devices()))")
            
            # Aktuelle GPU-Info
            device = CUDA.device()
            println("Aktuelle GPU: $(CUDA.name(device))")
            println("Compute Capability: $(CUDA.capability(device))")
        else
            println("✗ CUDA nicht verfügbar")
            return false
        end
        
        return true
    end

    function analyze_gpu_memory()
        if !CUDA.functional()
            println("Keine GPU verfügbar")
            return
        end
        
        println("\n=== GPU-SPEICHER ANALYSE ===")
        
        # Speicher vor cleanup
        total_memory = CUDA.total_memory()
        free_memory = CUDA.available_memory()
        used_memory = total_memory - free_memory
        
        println("Gesamtspeicher: $(round(total_memory / 1e9, digits=2)) GB")
        println("Verfügbar: $(round(free_memory / 1e9, digits=2)) GB")
        println("Belegt: $(round(used_memory / 1e9, digits=2)) GB")
        
        # Memory cleanup
        GC.gc()
        CUDA.reclaim()
        
        # Speicher nach cleanup
        free_after = CUDA.available_memory()
        println("Verfügbar nach Cleanup: $(round(free_after / 1e9, digits=2)) GB")
        
        return total_memory, free_after
    end

    function test_tensor_sizes()
        if !CUDA.functional()
            return
        end
        
        println("\n=== TENSOR-GRÖSSEN TEST ===")
        
        resolutions = [(64, 64), (128, 128), (256, 256), (512, 512), (1024, 1024)]
        
        for (h, w) in resolutions
            try
                # Teste verschiedene Datentypen
                for dtype in [Float32, Float16]
                    # Simuliere UNet-Input: (H, W, C, B)
                    batch_size = 1
                    channels = 1
                    
                    # Speicherbedarf berechnen
                    elements = h * w * channels * batch_size
                    bytes_per_element = sizeof(dtype)
                    total_bytes = elements * bytes_per_element
                    
                    # Teste Tensor-Erstellung
                    test_tensor = CUDA.zeros(dtype, h, w, channels, batch_size)
                    
                    println("$(h)x$(w) $(dtype): $(round(total_bytes / 1e6, digits=1)) MB - ✓")
                    
                    # Cleanup
                    test_tensor = nothing
                    CUDA.reclaim()
                    
                end
            catch e
                println("$(h)x$(w): FEHLER - $e")
            end
        end
    end
    function find_optimal_batch_sizes()
        if !CUDA.functional()
            return
        end
        
        println("\n=== BATCH-GRÖSSEN OPTIMIERUNG ===")
        
        resolutions = [(64, 64), (128, 128), (256, 256), (512, 512)]
        recommendations = Dict()
        
        for (h, w) in resolutions
            println("\nTeste $(h)x$(w):")
            
            max_batch = 1
            channels = 2  # Input + Output Kanäle
            
            # Binäre Suche für maximale Batch-Größe
            for batch_size in [1, 2, 4, 8, 16, 32, 64]
                try
                    # Simuliere Forward + Backward Pass Speicherbedarf
                    # Grober Schätzwert: 3x Input für Gradienten + Activations
                    safety_factor = 3
                    
                    elements = h * w * channels * batch_size * safety_factor
                    bytes_needed = elements * sizeof(Float32)
                    
                    available = CUDA.available_memory()
                    
                    if bytes_needed < available * 0.8  # 80% Sicherheitspuffer
                        # Teste reale Tensor-Erstellung
                        test_input = CUDA.randn(Float32, h, w, 1, batch_size)
                        test_output = CUDA.randn(Float32, h, w, 2, batch_size)
                        
                        max_batch = batch_size
                        println("  Batch $batch_size: ✓ ($(round(bytes_needed / 1e6, digits=1)) MB)")
                        
                        # Cleanup
                        test_input = nothing
                        test_output = nothing
                        CUDA.reclaim()
                    else
                        println("  Batch $batch_size: ✗ (zu wenig Speicher)")
                        break
                    end
                catch e
                    println("  Batch $batch_size: ✗ ($e)")
                    break
                end
            end
            
            recommendations[(h, w)] = max_batch
            println("  → Empfohlen für $(h)x$(w): Batch-Größe $max_batch")
        end
        
        return recommendations
    end

    function test_unet_memory_requirements()
        if !CUDA.functional()
            return
        end
        
        println("\n=== UNET-SPEICHER SIMULATION ===")
        
        # Vereinfachte UNet-Layer für Speichertest
        function create_test_unet_layer(in_ch, out_ch)
            return Chain(
                Conv((3, 3), in_ch => out_ch, relu, pad=SamePad()),
                BatchNorm(out_ch)
            ) |> gpu
        end
        
        resolutions = [(128, 128), (256, 256)]
        
        for (h, w) in resolutions
            println("\nTeste UNet-Layer für $(h)x$(w):")
            
            try
                # Teste verschiedene Layer-Tiefen
                batch_size = 2
                input_data = CUDA.randn(Float32, h, w, 1, batch_size)
                
                # Encoder-Layer (32, 64, 128, 256 Kanäle)
                current_data = input_data
                
                for (layer_idx, channels) in enumerate([32, 64, 128, 256])
                    in_ch = layer_idx == 1 ? 1 : [32, 64, 128][layer_idx-1]
                    
                    layer = create_test_unet_layer(in_ch, channels)
                    current_data = layer(current_data)
                    
                    # Memory-Status nach Layer
                    used_mb = (CUDA.total_memory() - CUDA.available_memory()) / 1e6
                    println("  Layer $layer_idx ($channels ch): $(round(used_mb, digits=1)) MB verwendet")
                    
                    # Pooling simulieren
                    if layer_idx < 4
                        current_data = current_data[:, :, :, 1:end÷2]  # Simulate pooling
                    end
                end
                
                println("  ✓ UNet-Forward für $(h)x$(w) erfolgreich")
                
            catch e
                println("  ✗ UNet-Test für $(h)x$(w) fehlgeschlagen: $e")
            finally
                # Cleanup
                GC.gc()
                CUDA.reclaim()
            end
        end
    end

    function complete_gpu_diagnosis()
        println("="^60)
        println("GPU-DIAGNOSE FÜR UNET-TRAINING")
        println("="^60)
        
        # 1. Basis-Checks
        if !check_cuda_availability()
            return
        end
        
        # 2. Speicher-Analyse
        total_mem, available_mem = analyze_gpu_memory()
        
        # 3. Tensor-Tests
        test_tensor_sizes()
        
        # 4. Batch-Optimierung
        batch_recommendations = find_optimal_batch_sizes()
        
        # 5. UNet-spezifische Tests
        test_unet_memory_requirements()
        
        # 6. Zusammenfassung
        println("\n" * "="^60)
        println("EMPFEHLUNGEN FÜR IHR SYSTEM:")
        println("="^60)
        println("GPU-Speicher: $(round(available_mem / 1e9, digits=2)) GB verfügbar")
        println("\nOptimale Batch-Größen:")
        for ((h, w), batch) in batch_recommendations
            println("  $(h)x$(w): Batch-Größe $batch")
        end
        
        println("\nEmpfohlene Trainings-Strategie:")
        if available_mem > 8e9  # > 8GB
            println("  - Beginnen Sie mit 256x256 Auflösung")
            println("  - Batch-Größe 4-8 sollte funktionieren")
            println("  - Mixed-Precision (Float16) für größere Batches")
        elseif available_mem > 4e9  # > 4GB
            println("  - Beginnen Sie mit 128x128 Auflösung")
            println("  - Batch-Größe 2-4 empfohlen")
            println("  - Schrittweise auf 256x256 erhöhen")
        else
            println("  - Beginnen Sie mit 64x64 Auflösung")
            println("  - Batch-Größe 1-2")
            println("  - Eventuell CPU-Training in Betracht ziehen")
        end
        
        println("="^60)
    end

    function debug_tensor_dimensions()
        println("=== TENSOR-DIMENSIONEN DEBUG ===")
        
        # Teste schrittweise Dimensionsänderungen
        for (h, w) in [(128, 128), (256, 256)]
            println("\nTeste $(h)x$(w) Dimensionen:")
            
            batch_size = 1  # Erstmal klein halten
            current_h, current_w = h, w
            
            println("  Start: $(current_h)x$(current_w)")
            
            # Simuliere 4 Pooling-Schritte (typisch für UNet)
            for pool_step in 1:4
                current_h = current_h ÷ 2
                current_w = current_w ÷ 2
                
                println("  Nach Pool $pool_step: $(current_h)x$(current_w)")
                
                # Prüfe ob Dimensionen gültig sind
                if current_h < 4 || current_w < 4
                    println("    ⚠ Dimensionen zu klein für Conv-Layer!")
                end
            end
        end
    end

    debug_tensor_dimensions()

    function test_unet_memory_safe()
        if !CUDA.functional()
            return
        end
        
        println("\n=== VERBESSERTE UNET-SPEICHER SIMULATION ===")
        
        function create_safe_test_layer(in_ch, out_ch, h, w)
            # Prüfe Dimensionen vor Layer-Erstellung
            if h < 8 || w < 8
                error("Dimensionen zu klein: $(h)x$(w)")
            end
            
            return Chain(
                Conv((3, 3), in_ch => out_ch, relu, pad=SamePad()),
                BatchNorm(out_ch)
            )
        end
        
        # Teste verschiedene Auflösungen
        for initial_size in [64, 128, 256, 512]
            println("\nTeste $(initial_size)x$(initial_size):")
            
            try
                batch_size = 1
                h, w = initial_size, initial_size
                
                # GPU Memory vor Test
                GC.gc(); CUDA.reclaim()
                mem_before = CUDA.available_memory()
                
                # Erstelle Test-Input
                input_data = CUDA.randn(Float32, h, w, 1, batch_size)
                current_data = input_data
                
                # UNet Encoder simulation
                channels = [1, 32, 64, 128, 256]
                current_h, current_w = h, w
                
                for layer_idx in 1:4
                    in_ch = channels[layer_idx]
                    out_ch = channels[layer_idx + 1]
                    
                    println("  Layer $layer_idx: $(current_h)x$(current_w), $in_ch→$out_ch")
                    
                    # Prüfe ob Dimensionen ausreichend sind
                    if current_h < 8 || current_w < 8
                        println("    ⚠ Stoppe bei zu kleinen Dimensionen")
                        break
                    end
                    
                    # Erstelle Layer
                    layer = create_safe_test_layer(in_ch, out_ch, current_h, current_w) |> gpu
                    
                    # Forward Pass
                    current_data = layer(current_data)
                    
                    # Memory Status
                    mem_used = (mem_before - CUDA.available_memory()) / 1e6
                    println("    Memory: $(round(mem_used, digits=1)) MB")
                    
                    # Simuliere MaxPool (2x2)
                    if layer_idx < 4
                        pool = MaxPool((2,2)) |> gpu
                        current_data = pool(current_data)
                        current_h = current_h ÷ 2
                        current_w = current_w ÷ 2
                        println("    Nach Pool: $(current_h)x$(current_w)")
                    end
                end
                
                println("  ✓ $(initial_size)x$(initial_size) erfolgreich getestet")
                
            catch e
                println("  ✗ Fehler bei $(initial_size)x$(initial_size): $e")
            finally
                # Aggressive Cleanup
                GC.gc()
                CUDA.reclaim()
            end
        end
    end

    test_unet_memory_safe()
    function find_realistic_batch_sizes()
        if !CUDA.functional()
            return Dict()
        end
        
        println("\n=== REALISTISCHE BATCH-GRÖSSEN ===")
        
        recommendations = Dict()
        
        # Teste nur sinnvolle Auflösungen
        test_resolutions = [64, 128, 256]
        
        for size in test_resolutions
            println("\nTeste $(size)x$(size):")
            
            max_working_batch = 0
            
            for batch_size in [1, 2, 4, 8]
                try
                    # Cleanup vor Test
                    GC.gc(); CUDA.reclaim()
                    
                    # Erstelle realistische UNet-Eingabe
                    h, w = size, size
                    
                    # Input und Target (wie im echten Training)
                    phase_input = CUDA.randn(Float32, h, w, 1, batch_size)
                    velocity_target = CUDA.randn(Float32, h, w, 2, batch_size)
                    
                    # Simuliere einfachen Forward Pass
                    # (ohne komplettes UNet, nur Speicher-Overhead)
                    temp_activations = []
                    
                    # Simuliere Encoder-Activations
                    current_data = phase_input
                    for layer in 1:4
                        # Verdopple Kanäle, halbiere Auflösung
                        current_h = max(4, h ÷ (2^(layer-1)))
                        current_w = max(4, w ÷ (2^(layer-1)))
                        current_ch = min(256, 32 * 2^(layer-1))
                        
                        activation = CUDA.randn(Float32, current_h, current_w, current_ch, batch_size)
                        push!(temp_activations, activation)
                    end
                    
                    max_working_batch = batch_size
                    
                    mem_used = (CUDA.total_memory() - CUDA.available_memory()) / 1e6
                    println("  Batch $batch_size: ✓ ($(round(mem_used, digits=1)) MB)")
                    
                    # Cleanup
                    temp_activations = nothing
                    phase_input = nothing
                    velocity_target = nothing
                    
                catch e
                    println("  Batch $batch_size: ✗ ($e)")
                    break
                finally
                    GC.gc(); CUDA.reclaim()
                end
            end
            
            # Sicherheitsfaktor: 50% der max getesteten Batch-Größe
            safe_batch = max(1, max_working_batch ÷ 2)
            recommendations[size] = safe_batch
            
            println("  → Empfohlen: Batch-Größe $safe_batch (sicher)")
            println("  → Maximum: Batch-Größe $max_working_batch (riskant)")
        end
        
        return recommendations
    end

    batch_recommendations = find_realistic_batch_sizes()
    function create_memory_monitor()
        return function monitor_gpu_memory(epoch, batch_idx)
            if CUDA.functional()
                total_gb = CUDA.total_memory() / 1e9
                available_gb = CUDA.available_memory() / 1e9
                used_gb = total_gb - available_gb
                usage_percent = (used_gb / total_gb) * 100
                
                if batch_idx % 10 == 0  # Alle 10 Batches
                    println("    GPU Memory: $(round(used_gb, digits=1))/$(round(total_gb, digits=1)) GB ($(round(usage_percent, digits=1))%)")
                end
                
                # Warnung bei hoher Speichernutzung
                if usage_percent > 90
                    println("    ⚠ GPU-Speicher > 90% - möglicher OOM!")
                    GC.gc()
                    CUDA.reclaim()
                end
            end
        end
    end

    # Verwendung im Training:
    memory_monitor = create_memory_monitor()
    # memory_monitor(epoch, batch_idx)  # In der Training-Loop aufrufen
    function print_adjusted_recommendations()
        println("\n" * "="^60)
        println("ANGEPASSTE EMPFEHLUNGEN NACH cuDNN-ANALYSE")
        println("="^60)
        
        println("EMPFOHLENE TRAININGS-KONFIGURATION:")
        println("\n1. Auflösungen:")
        println("   - Beginnen Sie mit 64x64 (sicher)")
        println("   - 128x128 nur mit Batch-Größe 1")
        println("   - 256x256 nur wenn absolut nötig, Batch-Größe 1")
        
        println("\n2. Batch-Größen (konservativ):")
        println("   - 64x64: Batch 2-4")
        println("   - 128x128: Batch 1-2") 
        println("   - 256x256: Batch 1")
        
        println("\n3. UNet-Architektur anpassen:")
        println("   - Weniger Pooling-Schritte für kleine Auflösungen")
        println("   - Adaptive Netzwerktiefe je nach Eingabeauflösung")
        println("   - Gradient Checkpointing für große Modelle")
        
        println("\n4. Trainings-Strategie:")
        println("   - Mixed-Precision Training (Float16)")
        println("   - Gradient Accumulation statt große Batches")
        println("   - Regelmäßige Memory-Cleanups")
        
        println("\n5. Debugging:")
        println("   - Immer mit kleinsten Parametern beginnen")
        println("   - Schrittweise erhöhen und Memory überwachen")
        println("   - Bei cuDNN-Fehlern: Dimensionen prüfen")
        
        println("="^60)
    end

    print_adjusted_recommendations()

    end
end