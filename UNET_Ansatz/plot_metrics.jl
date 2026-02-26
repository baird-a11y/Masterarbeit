# Datei: plot_metrics.jl
# Erzeugt alle Eval-Plots aus bereits berechneten CSVs (kein Modell nötig).
# Nützlich um Plots nachträglich zu regenerieren ohne das Modell neu zu laden.
#
# Nutzung:
#   julia plot_metrics.jl --prefix eval_psi_dataset --plot_dir eval_plots
#   julia plot_metrics.jl --prefix eval_psi_dataset --plot_dir eval_plots --history training_history.csv

include(joinpath(@__DIR__, "plots_eval.jl"))
using .UNetPlotsEval

# =============================================================================
# CLI-Argumente
# =============================================================================

out_prefix   = "eval_psi_dataset"
plot_dir     = "eval_plots"
history_csv  = nothing   # optional: Pfad zur training_history.csv

for i in eachindex(ARGS)
    if ARGS[i] == "--prefix"   && i < length(ARGS); global out_prefix  = ARGS[i+1]; end
    if ARGS[i] == "--plot_dir" && i < length(ARGS); global plot_dir    = ARGS[i+1]; end
    if ARGS[i] == "--history"  && i < length(ARGS); global history_csv = ARGS[i+1]; end
end

mkpath(plot_dir)

# =============================================================================
# Metriken vs. Kristallanzahl
# =============================================================================

agg_csv = out_prefix * "_by_n.csv"
if isfile(agg_csv)
    plot_metrics_vs_crystals(agg_csv;
        out_path = joinpath(plot_dir, "metrics_vs_crystals.png"))
else
    @warn "CSV nicht gefunden: $agg_csv  →  Kein Metrics-Plot erzeugt."
end

# =============================================================================
# Training History
# =============================================================================

# Suche in: explizit angegeben, dann aktuellem Ordner
candidates = filter(!isnothing, [history_csv, "training_history.csv"])
for h_csv in candidates
    if isfile(h_csv)
        plot_training_history(h_csv;
            out_path = joinpath(plot_dir, "training_history.png"))
        break
    end
end

println("\nPlots gespeichert in: $plot_dir/")
