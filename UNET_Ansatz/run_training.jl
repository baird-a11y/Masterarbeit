# Datei: run_training.jl
# Training Entry Point für ψ-U-Net
#
# Nutzung:
#   julia run_training.jl                  # mit Default-Config
#   julia run_training.jl --epochs 200     # eigene Epochenzahl
#
# Oder interaktiv (empfohlen für Debugging):
#   include("UNetPsi.jl")
#   using .UNetPsi
#   include("run_training.jl")

using Printf, Flux

# ── Projekt laden ──
if !isdefined(Main, :UNetPsi)
    include("UNetPsi.jl")
end
using .UNetPsi
using .UNetPsi.UNetPsiModel: build_unet
using .UNetPsi.DatasetPsi: load_dataset, get_sample, dataset_summary
using .UNetPsi.TrainingPsi: train!

# =============================================================================
# 1. Konfiguration
# =============================================================================

cfg = UNetPsi.load_config()

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
  Modell:      base_channels=$(cfg.base_channels), depth=$(cfg.depth)
  Training:    epochs=$epochs, batch=$batch_size, lr=$lr
  Train-Dir:   $train_dir
  Val-Dir:     $val_dir
  GPU:         $(cfg.use_gpu)
"""

# =============================================================================
# 2. Datasets laden
# =============================================================================

@info "Lade Trainingsdaten..."
train_ds = load_dataset(train_dir; use_coords=cfg.use_coords, coord_range=cfg.coord_range)

@info "Lade Validierungsdaten..."
val_ds = load_dataset(val_dir; use_coords=cfg.use_coords, coord_range=cfg.coord_range)

# Kurze Übersicht
dataset_summary(train_ds; n=5)
dataset_summary(val_ds; n=5)

# =============================================================================
# 3. Input-Kanäle bestimmen
# =============================================================================

# Erstes Sample laden um Cin zu bestimmen
X1, _, _ = get_sample(train_ds, 1)
Cin = size(X1, 3)
@info "Input-Kanäle: Cin=$Cin"

# =============================================================================
# 4. Modell bauen
# =============================================================================

# Architektur-Config (wird im Checkpoint mitgespeichert)
model_cfg = (Cin=Cin, base_channels=cfg.base_channels, depth=cfg.depth, Cout=1)

model = build_unet(Cin, 1; base_channels=cfg.base_channels)

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
    max_grad_norm = cfg.max_grad_norm,
    α_grad_target = cfg.α_grad,
    warmup_epochs = cfg.warmup_epochs,
    α_bnd         = cfg.α_bnd,
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
# 6. Optional: Evaluation auf Validierungsdaten
# =============================================================================

# Hier könnte man optional noch eine Evaluation durchführen
# using .UNetPsi.EvaluatePsi
# ...

println("\n" * "=" ^ 60)
println("Fertig!")
println("=" ^ 60)
println("  Checkpoints:  checkpoints/")
println("  History CSV:  training_history.csv")
println("=" ^ 60)
