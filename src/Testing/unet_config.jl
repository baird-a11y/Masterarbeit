# =============================================================================
# UNET CONFIGURATION MODULE
# =============================================================================
# Speichern als: unet_config.jl

using Printf

"""
Konfiguration für ein adaptives UNet
"""
struct UNetConfig
    input_resolution::Int           
    input_channels::Int            
    output_channels::Int           
    depth::Int                     
    base_filters::Int              
    filter_progression::Vector{Int} 
    pooling_factor::Int            
    bottleneck_size::Int           
    dropout_rate::Float32          
    use_batchnorm::Bool           
    activation::Function           
    final_activation::Union{Function, Nothing}  
    use_checkpointing::Bool        
    mixed_precision::Bool          
end

"""
Zeigt UNet-Konfiguration übersichtlich an
"""
function Base.show(io::IO, config::UNetConfig)
    println(io, "UNetConfig für $(config.input_resolution)×$(config.input_resolution):")
    println(io, "  Tiefe: $(config.depth) Pooling-Schritte")
    println(io, "  Filter: $(config.filter_progression)")
    println(io, "  Bottleneck: $(config.bottleneck_size)×$(config.bottleneck_size)")
    println(io, "  Dropout: $(config.dropout_rate)")
    println(io, "  BatchNorm: $(config.use_batchnorm)")
    println(io, "  Checkpointing: $(config.use_checkpointing)")
end

"""
Berechnet resultierende Feature-Map-Größen nach Pooling-Schritten
"""
function calculate_feature_map_sizes(input_resolution::Int, depth::Int, pooling_factor::Int=2)
    sizes = [input_resolution]
    current_size = input_resolution
    
    for i in 1:depth
        current_size = current_size ÷ pooling_factor
        push!(sizes, current_size)
    end
    
    return sizes
end

"""
Validiert ob eine UNet-Konfiguration sinnvoll ist
"""
function validate_unet_config(input_resolution::Int, depth::Int, min_bottleneck_size::Int=4)
    if input_resolution & (input_resolution - 1) != 0
        return false, "Eingabeauflösung $input_resolution ist keine Potenz von 2"
    end
    
    if input_resolution < 64 || input_resolution > 512
        return false, "Eingabeauflösung $input_resolution außerhalb erlaubtem Bereich [64, 512]"
    end
    
    sizes = calculate_feature_map_sizes(input_resolution, depth)
    bottleneck_size = sizes[end]
    
    if bottleneck_size < min_bottleneck_size
        return false, "Bottleneck zu klein: $(bottleneck_size)×$(bottleneck_size) < $(min_bottleneck_size)×$(min_bottleneck_size)"
    end
    
    max_possible_depth = Int(floor(log2(input_resolution / min_bottleneck_size)))
    if depth > max_possible_depth
        return false, "Tiefe $depth zu groß für Auflösung $input_resolution (max: $max_possible_depth)"
    end
    
    return true, "Konfiguration gültig"
end

"""
Bestimmt optimale Tiefe für eine gegebene Auflösung
"""
function determine_optimal_depth(input_resolution::Int; min_bottleneck_size::Int=8, prefer_deeper::Bool=true)
    max_depth = Int(floor(log2(input_resolution / min_bottleneck_size)))
    
    if input_resolution <= 64
        return min(2, max_depth)  # 64→32→16
    elseif input_resolution <= 128
        return min(3, max_depth)  # 128→64→32→16
    elseif input_resolution <= 256
        return min(4, max_depth)  # 256→128→64→32→16
    else  # 512×512
        return min(5, max_depth)  # 512→256→128→64→32→16
    end
end

"""
Verschiedene Strategien für Filter-Anzahl pro Layer
"""
function create_filter_progression(base_filters::Int, depth::Int, strategy::Symbol=:exponential)
    if strategy == :exponential
        return [base_filters * (2^i) for i in 0:depth]
    elseif strategy == :linear
        return [base_filters + i * base_filters÷2 for i in 0:depth]
    elseif strategy == :conservative
        progression = [base_filters]
        for i in 1:depth
            next_filters = min(progression[end] * 2, 256)
            push!(progression, next_filters)
        end
        return progression
    elseif strategy == :custom_small
        if depth <= 3
            return [base_filters, base_filters*2, base_filters*4, base_filters*4]
        else
            return [base_filters * (2^min(i, 2)) for i in 0:depth]
        end
    else
        error("Unbekannte Filter-Strategie: $strategy")
    end
end

"""
Bestimmt optimale Base-Filter-Anzahl basierend auf Auflösung
"""
function determine_base_filters(input_resolution::Int; memory_efficient::Bool=false)
    if memory_efficient
        return 32
    else
        if input_resolution <= 128
            return 32
        else
            return 64
        end
    end
end

"""
Erstellt optimale UNet-Konfiguration für gegebene Auflösung
"""
function design_adaptive_unet(input_resolution::Int; 
                              input_channels::Int=1,
                              output_channels::Int=2,
                              prefer_deeper::Bool=true,
                              memory_efficient::Bool=false,
                              filter_strategy::Symbol=:exponential,
                              min_bottleneck_size::Int=8)
    
    println("=== ADAPTIVE UNET DESIGN ===")
    println("Zielauflösung: $(input_resolution)×$(input_resolution)")
    
    # Validiere Eingabeauflösung
    is_valid, message = validate_unet_config(input_resolution, 1, min_bottleneck_size)
    if !is_valid
        error("Ungültige Auflösung: $message")
    end
    
    # Bestimme optimale Tiefe
    optimal_depth = determine_optimal_depth(input_resolution, 
                                           min_bottleneck_size=min_bottleneck_size,
                                           prefer_deeper=prefer_deeper)
    
    # Validiere gewählte Tiefe
    is_valid, message = validate_unet_config(input_resolution, optimal_depth, min_bottleneck_size)
    if !is_valid
        error("Optimale Tiefe ungültig: $message")
    end
    
    # Bestimme Base-Filter
    base_filters = determine_base_filters(input_resolution, memory_efficient=memory_efficient)
    
    # Erstelle Filter-Progression
    filter_progression = create_filter_progression(base_filters, optimal_depth, filter_strategy)
    
    # Berechne finale Feature-Map-Größen
    feature_sizes = calculate_feature_map_sizes(input_resolution, optimal_depth)
    bottleneck_size = feature_sizes[end]
    
    # Bestimme Regularisierungs-Parameter
    dropout_rate = if input_resolution >= 256
        0.2f0
    elseif input_resolution >= 128
        0.1f0
    else
        0.05f0
    end
    
    # GPU-Optimierungen
    use_checkpointing = input_resolution >= 256 || memory_efficient
    mixed_precision = input_resolution >= 256
    
    # Erstelle finale Konfiguration
    config = UNetConfig(
        input_resolution, input_channels, output_channels,
        optimal_depth, base_filters, filter_progression,
        2, bottleneck_size,
        dropout_rate, true,
        relu, nothing,
        use_checkpointing, mixed_precision
    )
    
    # Debug-Ausgabe
    println("  Gewählte Tiefe: $optimal_depth")
    println("  Base-Filter: $base_filters")
    println("  Filter-Progression: $filter_progression")
    println("  Feature-Map-Größen: $feature_sizes")
    println("  Bottleneck: $(bottleneck_size)×$(bottleneck_size)")
    println("  Dropout: $(dropout_rate)")
    println("  Checkpointing: $use_checkpointing")
    println("  Mixed-Precision: $mixed_precision")
    
    return config
end

"""
Vergleicht Konfigurationen für verschiedene Auflösungen
"""
function compare_unet_configs(resolutions::Vector{Int}; kwargs...)
    println("=== UNET KONFIGURATIONSVERGLEICH ===")
    
    configs = Dict{Int, UNetConfig}()
    
    for res in resolutions
        println("\n" * "="^50)
        try
            config = design_adaptive_unet(res; kwargs...)
            configs[res] = config
        catch e
            println("Fehler für Auflösung $res: $e")
        end
    end
    
    # Zusammenfassungstabelle
    println("\n" * "="^80)
    println("ZUSAMMENFASSUNG:")
    println("="^80)
    println(@sprintf("%-12s %-6s %-6s %-20s %-12s %-8s", 
                     "Auflösung", "Tiefe", "Base", "Filter", "Bottleneck", "Dropout"))
    println("-"^80)
    
    for res in sort(collect(keys(configs)))
        config = configs[res]
        filter_str = join(string.(config.filter_progression), "→")
        if length(filter_str) > 18
            filter_str = filter_str[1:15] * "..."
        end
        
        println(@sprintf("%-12s %-6d %-6d %-20s %-12s %-8.1f%%", 
                        "$(res)×$(res)", 
                        config.depth, 
                        config.base_filters,
                        filter_str,
                        "$(config.bottleneck_size)×$(config.bottleneck_size)",
                        config.dropout_rate*100))
    end
    
    return configs
end

println("UNet Configuration Module geladen!")
println("Verfügbare Funktionen:")
println("  - design_adaptive_unet(resolution)")
println("  - compare_unet_configs([64, 128, 256])")