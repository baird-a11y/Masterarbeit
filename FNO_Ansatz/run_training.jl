# Datei: run_training.jl
# Training + Evaluation Entry Point für ψ-FNO
#
# Nutzung:
#   julia run_training.jl                  # mit Default-Config
#   julia run_training.jl --epochs 200     # eigene Epochenzahl
#
# Oder interaktiv (empfohlen für Debugging):
#   include("FNOPsi.jl")
#   using .FNOPsi
#   include("run_training.jl")

using Printf, Flux

# ── Projekt laden ──
if !isdefined(Main, :FNOPsi)
    include("FNOPsi.jl")
end
using .FNOPsi
using .FNOPsi.FNOModel: build_fno
using .FNOPsi.DatasetPsi: load_dataset, get_sample, dataset_summary
using .FNOPsi.TrainingPsi: train!
using .FNOPsi.EvalPsi: evaluate_dataset, aggregate_results,
                        save_results_csv, print_eval_summary
# using CUDA
# @info "CUDA.functional() = $(CUDA.functional())"
# try
#     CUDA.versioninfo()
#     @info "CUDA.devices() = $(collect(CUDA.devices()))"
# catch e
#     @warn "CUDA.versioninfo/devices failed: $e"
# end
# =============================================================================
# 1. Konfiguration
# =============================================================================

cfg = FNOPsi.load_config()

# Override via CLI-Argumente (optional)
epochs     = cfg.epochs
batch_size = cfg.batch_size
lr         = cfg.lr
train_dir  = cfg.train_dir
val_dir    = cfg.val_dir

for i in eachindex(ARGS)
    if ARGS[i] == "--epochs"     && i < length(ARGS); global epochs     = parse(Int, ARGS[i+1]); end
    if ARGS[i] == "--batch"      && i < length(ARGS); global batch_size = parse(Int, ARGS[i+1]); end
    if ARGS[i] == "--lr"         && i < length(ARGS); global lr         = parse(Float64, ARGS[i+1]); end
    if ARGS[i] == "--train_dir"  && i < length(ARGS); global train_dir  = ARGS[i+1]; end
    if ARGS[i] == "--val_dir"    && i < length(ARGS); global val_dir    = ARGS[i+1]; end
end

@info """
Training-Konfiguration:
  Grid:        $(cfg.nx) × $(cfg.nz)
  Modell:      width=$(cfg.width), depth=$(cfg.depth), modes=$(cfg.modes)
  Training:    epochs=$epochs, batch=$batch_size, lr=$lr
  Train-Dir:   $train_dir
  Val-Dir:     $val_dir
  GPU:         $(cfg.use_gpu)
"""

# =============================================================================
# 2. Datasets laden
# =============================================================================

@info "Lade Trainingsdaten..."
train_ds = load_dataset(train_dir; use_coords=true, coord_range=:pm1)

@info "Lade Validierungsdaten..."
val_ds = load_dataset(val_dir; use_coords=true, coord_range=:pm1)

# Kurze Übersicht
dataset_summary(train_ds; n=5)
dataset_summary(val_ds; n=5)

# =============================================================================
# 3. Input-Kanäle bestimmen
# =============================================================================

# Erstes Sample laden um Cin zu bestimmen
X1, _, _ = get_sample(train_ds, 1)
Cin = size(X1, 3)
@info "Input-Kanäle: Cin=$Cin ($(Cin-2) aus JLD2 + 2 Koordinaten)"

# =============================================================================
# 4. Modell bauen
# =============================================================================

# Architektur-Config (wird im Checkpoint mitgespeichert)
model_cfg = (Cin=Cin, width=cfg.width, depth=cfg.depth,
             modes_x=cfg.modes, modes_z=cfg.modes, Cout=1)

model = build_fno(Cin;
                  width   = cfg.width,
                  depth   = cfg.depth,
                  modes_x = cfg.modes,
                  modes_z = cfg.modes,
                  Cout    = 1)

n_params = sum(length, Flux.params(model))
@info "Modell erstellt: $(n_params) Parameter"
println(model)

# =============================================================================
# 5. Training
# =============================================================================

@info "Starte Training..."

# dx/dz aus dem Gitter berechnen (für Gradienten-Loss)
# Domain [-1,1] mit nx Punkten → dx = 2/(nx-1)
dx = Float32(2.0 / (cfg.nx - 1))
dz = Float32(2.0 / (cfg.nz - 1))

model_trained, best_metrics = train!(;
    model       = model,
    train_ds    = train_ds,
    val_ds      = val_ds,
    epochs      = epochs,
    batch_size  = batch_size,
    lr          = lr,
    max_grad_norm = 1.0,
    α_grad_target = 0.1f0,     # Gradienten-Loss Ziel
    warmup_epochs = 5,          # Warmup für Gradienten-Loss
    α_bnd         = 0.01f0,     # Rand-Loss Koeffizient
    dx            = dx,
    dz            = dz,
    mask_width    = 2,
    save_dir      = "checkpoints",
    history_csv   = "training_history.csv",
    use_gpu       = cfg.use_gpu,
    model_config  = model_cfg,
)

@info "Training abgeschlossen!"
println("Best Validation Metrics: $best_metrics")

# =============================================================================
# 6. Evaluation auf Validierungsdaten
# =============================================================================

@info "Starte Evaluation..."

results = evaluate_dataset(model_trained, val_ds;
                           dx=dx, dz=dz, mask_width=2,
                           use_gpu=cfg.use_gpu,
                           save_dir="predictions",
                           verbose=true)

# Zusammenfassung
print_eval_summary(results)

# Ergebnisse speichern
save_results_csv("eval_results.csv", results)

# Aggregation nach Kristallanzahl
agg = aggregate_results(results)
save_results_csv("eval_aggregated.csv", agg)

# =============================================================================
# 7. Plots (optional)
# =============================================================================

# try
#     using .FNOPsi.PlotsEval

#     @info "Erstelle Plots..."
#     mkpath("plots")

#     # Training History
#     if isfile("training_history.csv")
#         plot_training_history("training_history.csv";
#                               out_path="plots/training_history.png")
#     end

#     # Metriken vs. Kristallanzahl
#     if isfile("eval_aggregated.csv")
#         plot_metrics_vs_crystals("eval_aggregated.csv";
#                                   out_path="plots/metrics_vs_crystals.png")
#     end

#     # Gallery: alle Samples (sortiert nach rel_l2 worst→best)
#     if isdir("predictions")
#         make_eval_gallery("predictions", "plots/gallery";
#                           n=0, pick=:worst, dx=dx, dz=dz)
#     end

#     @info "Plots gespeichert in plots/"

# catch e
#     @warn "Plots konnten nicht erstellt werden: $e"
# end

println("\n" * "=" ^ 60)
println("Fertig!")
println("=" ^ 60)
println("  Checkpoints:  checkpoints/")
println("  History CSV:  training_history.csv")
println("  Eval CSV:     eval_results.csv")
println("  Aggregiert:   eval_aggregated.csv")
println("  Predictions:  predictions/")
println("  Plots:        plots/")
println("=" ^ 60)
