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

using Printf, Flux, JLD2

# ── Projekt laden ──
include("UNetPsi.jl")
using .UNetPsi
using .UNetPsi.UNetPsiModel: build_unet
using .UNetPsi.TrainingPsi: load_checkpoint
using .UNetPsi.DatasetPsi: load_dataset, get_sample, dataset_summary
# using .UNetPsi.EvaluatePsi: evaluate_dataset  # TODO: wenn evaluate_psi.jl angepasst wurde

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
    model = build_unet(model_config.Cin, model_config.Cout;
                       base_channels=model_config.base_channels)
else
    # Fallback: Config-Datei verwenden
    @warn "Keine model_config im Checkpoint – verwende config.jl"
    cfg = UNetPsi.load_config()
    # Cin aus Dataset bestimmen
    ds_tmp = load_dataset(data_dir; use_coords=cfg.use_coords, coord_range=cfg.coord_range)
    X1, _, _ = get_sample(ds_tmp, 1)
    Cin = size(X1, 3)
    model = build_unet(Cin, 1; base_channels=cfg.base_channels)
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

cfg_eval = UNetPsi.load_config()

@info "Lade Evaluierungsdaten aus: $data_dir"
eval_ds = load_dataset(data_dir; use_coords=cfg_eval.use_coords,
                       coord_range=cfg_eval.coord_range)
dataset_summary(eval_ds; n=5)

# =============================================================================
# 4. Evaluierung
# =============================================================================

# dx/dz (Domain [-1,1])
dx = Float64(2.0 / (cfg_eval.nx - 1))
dz = Float64(2.0 / (cfg_eval.nz - 1))

# TODO: Hier würde evaluate_dataset aus evaluate_psi.jl verwendet werden
# Wenn evaluate_psi.jl angepasst wurde, kann man das hier einkommentieren:

# pred_dir = save_preds ? joinpath(out_dir, "predictions") : nothing
#
# @info "Starte Evaluation ($(length(eval_ds)) Samples)..."
# results = evaluate_dataset(model, eval_ds;
#                            dx=dx, dz=dz, mask_width=2,
#                            save_dir=pred_dir,
#                            verbose=true)
#
# # Zusammenfassung ausgeben
# print_eval_summary(results)
#
# # CSV: Alle Samples
# results_csv = joinpath(out_dir, "eval_results.csv")
# save_results_csv(results_csv, results)
#
# # CSV: Aggregiert nach Kristallanzahl
# agg = aggregate_results(results)
# agg_csv = joinpath(out_dir, "eval_aggregated.csv")
# save_results_csv(agg_csv, agg)

@warn "evaluate_psi.jl muss noch angepasst werden für vollständige Evaluation"
@info "Modell erfolgreich geladen und bereit für Evaluation"

println("\n" * "=" ^ 60)
@printf("Evaluation vorbereitet (Epoch %d, %d Samples)\n", epoch, length(eval_ds))
println("=" ^ 60)
println("  Ausgabe:       $out_dir")
println("=" ^ 60)
