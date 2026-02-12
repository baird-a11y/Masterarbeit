# Datei: run_eval.jl
# Standalone-Evaluation: Lädt gespeichertes Modell und evaluiert auf einem Dataset.
#
# Nutzung:
#   julia run_eval.jl
#   julia run_eval.jl --checkpoint checkpoints/best_model.jld2 --data_dir data_val
#
# Was passiert:
#   1. Checkpoint laden (inkl. Architektur-Config)
#   2. Modell neu bauen + Gewichte laden
#   3. Dataset evaluieren
#   4. Ergebnisse speichern (CSV + optional Predictions als JLD2)
#   5. Plots erstellen

using Printf, Flux, JLD2

# ── Projekt laden ──
include("FNOPsi.jl")
using .FNOPsi
using .FNOPsi.FNOModel: build_fno
using .FNOPsi.TrainingPsi: load_checkpoint
using .FNOPsi.DatasetPsi: load_dataset, get_sample, dataset_summary
using .FNOPsi.EvalPsi: predict_sample, evaluate_dataset, aggregate_results,
                        save_results_csv, print_eval_summary

# =============================================================================
# 1. Argument-Parsing
# =============================================================================

checkpoint_path = "checkpoints/best_model.jld2"
data_dir        = "data_val"
out_dir         = "eval_output"
save_preds      = true

for i in eachindex(ARGS)
    if (ARGS[i] == "--checkpoint" || ARGS[i] == "--ckpt") && i < length(ARGS)
        global checkpoint_path = ARGS[i+1]
    elseif (ARGS[i] == "--data_dir" || ARGS[i] == "--data") && i < length(ARGS)
        global data_dir = ARGS[i+1]
    elseif (ARGS[i] == "--out" || ARGS[i] == "--out_dir") && i < length(ARGS)
        global out_dir = ARGS[i+1]
    elseif ARGS[i] == "--no_preds"
        global save_preds = false
    end
end

mkpath(out_dir)

@info """
Evaluation-Konfiguration:
  Checkpoint:  $checkpoint_path
  Daten:       $data_dir
  Ausgabe:     $out_dir
  Predictions: $(save_preds ? "ja" : "nein")
"""

# =============================================================================
# 2. Checkpoint laden + Modell rekonstruieren
# =============================================================================

@assert isfile(checkpoint_path) "Checkpoint nicht gefunden: $checkpoint_path"

# Architektur-Config aus Checkpoint lesen
ckpt_data = JLD2.load(checkpoint_path)
model_config = get(ckpt_data, "model_config", nothing)

if model_config !== nothing
    @info "Modell-Architektur aus Checkpoint: $model_config"
    model = build_fno(model_config.Cin;
                      width   = model_config.width,
                      depth   = model_config.depth,
                      modes_x = model_config.modes_x,
                      modes_z = model_config.modes_z,
                      Cout    = model_config.Cout)
else
    # Fallback: Config-Datei verwenden
    @warn "Keine model_config im Checkpoint – verwende config.jl"
    cfg = FNOPsi.load_config()
    # Cin aus Dataset bestimmen
    ds_tmp = load_dataset(data_dir; use_coords=true)
    X1, _, _ = get_sample(ds_tmp, 1)
    Cin = size(X1, 3)
    model = build_fno(Cin; width=cfg.width, depth=cfg.depth,
                      modes_x=cfg.modes, modes_z=cfg.modes, Cout=1)
end

# Gewichte laden
Flux.loadmodel!(model, ckpt_data["model_state"])
epoch = get(ckpt_data, "epoch", 0)
@info "Modell geladen (Epoch $epoch)"

n_params = sum(length, Flux.params(model))
@info "Parameter: $n_params"

# =============================================================================
# 3. Dataset laden
# =============================================================================

@info "Lade Evaluierungsdaten aus: $data_dir"
eval_ds = load_dataset(data_dir; use_coords=true, coord_range=:pm1)
dataset_summary(eval_ds; n=5)

# =============================================================================
# 4. Evaluierung
# =============================================================================

# dx/dz (Domain [-1,1])
cfg_eval = FNOPsi.load_config()
dx = Float64(2.0 / (cfg_eval.nx - 1))
dz = Float64(2.0 / (cfg_eval.nz - 1))

pred_dir = save_preds ? joinpath(out_dir, "predictions") : nothing

@info "Starte Evaluation ($(length(eval_ds)) Samples)..."
results = evaluate_dataset(model, eval_ds;
                           dx=dx, dz=dz, mask_width=2,
                           save_dir=pred_dir,
                           verbose=true)

# =============================================================================
# 5. Ergebnisse speichern
# =============================================================================

# Zusammenfassung ausgeben
print_eval_summary(results)

# CSV: Alle Samples
results_csv = joinpath(out_dir, "eval_results.csv")
save_results_csv(results_csv, results)

# CSV: Aggregiert nach Kristallanzahl
agg = aggregate_results(results)
agg_csv = joinpath(out_dir, "eval_aggregated.csv")
save_results_csv(agg_csv, agg)

# =============================================================================
# 6. Plots (optional)
# =============================================================================

try
    using .FNOPsi.PlotsEval

    plots_dir = joinpath(out_dir, "plots")
    mkpath(plots_dir)

    # Training History (suche CSV im aktuellen Verzeichnis)
    if isfile("training_history.csv")
        plot_training_history("training_history.csv";
                              out_path=joinpath(plots_dir, "training_history.png"))
    end
    # Metriken vs. Kristallanzahl
    plot_metrics_vs_crystals(agg_csv;
                              out_path=joinpath(plots_dir, "metrics_vs_crystals.png"))

    # Gallery: 8 schlechteste Samples
    if pred_dir !== nothing && isdir(pred_dir)
        make_eval_gallery(pred_dir, joinpath(plots_dir, "gallery");
                          n=8, pick=:worst, dx=dx, dz=dz)
    end

    @info "Plots gespeichert in $plots_dir"
catch e
    @warn "Plots konnten nicht erstellt werden: $e"
end

println("\n" * "=" ^ 60)
@printf("Evaluation fertig (Epoch %d, %d Samples)\n", epoch, length(results))
println("=" ^ 60)
println("  Ergebnisse:    $results_csv")
println("  Aggregiert:    $agg_csv")
pred_dir !== nothing && println("  Predictions:   $pred_dir")
println("=" ^ 60)
