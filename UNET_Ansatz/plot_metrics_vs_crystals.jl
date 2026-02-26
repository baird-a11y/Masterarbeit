# Datei: plot_metrics_vs_crystals.jl
# Standalone-Wrapper – Plotting-Logik liegt in plots_eval.jl
#
# Nutzung:
#   julia plot_metrics_vs_crystals.jl                                   # sucht *_by_n.csv im aktuellen Ordner
#   julia plot_metrics_vs_crystals.jl --csv eval_psi_dataset_by_n.csv
#   julia plot_metrics_vs_crystals.jl --csv path/to/by_n.csv --out metrics.png
#   julia plot_metrics_vs_crystals.jl --search_dir Ergebnisse/UNET_Ergebnisse

include(joinpath(@__DIR__, "plots_eval.jl"))
using .UNetPlotsEval

# =============================================================================
# CLI-Argument-Parsing
# =============================================================================

csv_path   = nothing
out_path   = nothing
search_dir = nothing

let i = 1
    while i ≤ length(ARGS)
        if ARGS[i] == "--csv"        && i < length(ARGS); global csv_path   = ARGS[i+1]; i += 2; continue; end
        if ARGS[i] == "--out"        && i < length(ARGS); global out_path   = ARGS[i+1]; i += 2; continue; end
        if ARGS[i] == "--search_dir" && i < length(ARGS); global search_dir = ARGS[i+1]; i += 2; continue; end
        i += 1
    end
end

# =============================================================================
# Ausführung
# =============================================================================

if csv_path !== nothing
    kw = out_path !== nothing ? (; out_path) : NamedTuple()
    plot_metrics_vs_crystals(csv_path; kw...)
else
    root = search_dir !== nothing ? search_dir : "."
    csvs = String[]
    for (dirpath, _, files) in walkdir(root)
        for f in files
            endswith(f, "_by_n.csv") && push!(csvs, joinpath(dirpath, f))
        end
    end
    if isempty(csvs)
        @warn "Keine *_by_n.csv gefunden in: $root"
        @info "Nutzung: julia plot_metrics_vs_crystals.jl --csv eval_psi_dataset_by_n.csv"
    else
        @info "$(length(csvs)) CSV-Datei(en) gefunden:"
        for c in csvs; println("  ", c); end
        for c in csvs; plot_metrics_vs_crystals(c); end
    end
end
