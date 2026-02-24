# Datei: run_training.jl
# Training Entry Point für ψ-U-Net
#
# Nutzung:
#   julia run_training.jl
#   julia run_training.jl --epochs 300 --batch 8 --lr 5e-5
#   julia run_training.jl --train_dir data_train --save_path checkpoints/unet.bson --gpu

# =============================================================================
# 1. Default-Werte
# =============================================================================

epochs       = 300
batch_size   = 8
lr           = 5e-5
train_dir    = "/local/home/baselt/src/Daten/data_psi"
val_dir      = nothing   # nothing = keine Validierung
save_path    = "unet_psi.bson"
history_csv  = "training_history.csv"
use_gpu      = nothing   # nothing = auto, true = GPU, false = CPU

# =============================================================================
# 2. CLI-Argumente parsen
# =============================================================================

for i in eachindex(ARGS)
    if ARGS[i] == "--epochs"      && i < length(ARGS); global epochs      = parse(Int,     ARGS[i+1]); end
    if ARGS[i] == "--batch"       && i < length(ARGS); global batch_size  = parse(Int,     ARGS[i+1]); end
    if ARGS[i] == "--lr"          && i < length(ARGS); global lr          = parse(Float64, ARGS[i+1]); end
    if ARGS[i] == "--train_dir"   && i < length(ARGS); global train_dir   = ARGS[i+1]; end
    if ARGS[i] == "--val_dir"     && i < length(ARGS); global val_dir     = ARGS[i+1]; end
    if ARGS[i] == "--save_path"   && i < length(ARGS); global save_path   = ARGS[i+1]; end
    if ARGS[i] == "--history_csv" && i < length(ARGS); global history_csv = ARGS[i+1]; end
    if ARGS[i] == "--gpu"                             ; global use_gpu     = true; end
    if ARGS[i] == "--no_gpu"                          ; global use_gpu     = false; end
end

@info """
Training-Konfiguration (U-Net):
  epochs:      $epochs
  batch_size:  $batch_size
  lr:          $lr
  train_dir:   $train_dir
  val_dir:     $(val_dir === nothing ? "(keine Validierung)" : val_dir)
  save_path:   $save_path
  history_csv: $history_csv
  use_gpu:     $(use_gpu === nothing ? "auto" : use_gpu)
"""

# =============================================================================
# 3. Module laden
# =============================================================================

include("dataset_psi.jl")
include("unet_psi.jl")
include("training_psi.jl")

using .DatasetPsi
using .UNetPsi
using .TrainingPsi
using Random

# =============================================================================
# 4. Training
# =============================================================================

rng = MersenneTwister(42)

mkpath(dirname(save_path) == "" ? "." : dirname(save_path))

TrainingPsi.train_unet(;
    data_dir    = train_dir,
    val_dir     = val_dir,
    epochs      = epochs,
    batch_size  = batch_size,
    lr          = lr,
    rng         = rng,
    save_path   = save_path,
    history_csv = history_csv,
    use_gpu     = use_gpu,
)

best_path = let base = splitext(save_path); base[1] * "_best" * base[2]; end

println("\n" * "=" ^ 60)
println("Training fertig!")
println("=" ^ 60)
println("  Letztes Modell: $save_path")
val_dir !== nothing && println("  Bestes Modell:  $best_path")
println("  History CSV:    $history_csv")
println("=" ^ 60)
