# Datei: run_eval.jl
# Evaluation Entry Point für ψ-U-Net
#
# Nutzung:
#   julia run_eval.jl
#   julia run_eval.jl --model unet_psi.bson --data_dir data_eval --out eval_output
#   julia run_eval.jl --model checkpoints/unet.bson --data_dir data_eval --out eval_exp1 --save_plots --denorm

# =============================================================================
# 1. Default-Werte
# =============================================================================

model_path  = "unet_psi.bson"
data_dir    = "/local/home/baselt/src/Daten/data_psi"
out_prefix  = "eval_psi_dataset"
plot_dir    = "eval_plots"
save_plots  = false
denorm_psi  = false
use_gpu     = nothing   # nothing = auto, true = GPU, false = CPU

# =============================================================================
# 2. CLI-Argumente parsen
# =============================================================================

for i in eachindex(ARGS)
    if ARGS[i] == "--model"      && i < length(ARGS); global model_path = ARGS[i+1]; end
    if ARGS[i] == "--data_dir"   && i < length(ARGS); global data_dir   = ARGS[i+1]; end
    if ARGS[i] == "--out"        && i < length(ARGS); global out_prefix = ARGS[i+1]; end
    if ARGS[i] == "--plot_dir"   && i < length(ARGS); global plot_dir   = ARGS[i+1]; end
    if ARGS[i] == "--save_plots"                     ; global save_plots = true; end
    if ARGS[i] == "--denorm"                         ; global denorm_psi = true; end
    if ARGS[i] == "--gpu"                            ; global use_gpu    = true; end
    if ARGS[i] == "--no_gpu"                         ; global use_gpu    = false; end
end

@info """
Evaluation-Konfiguration (U-Net):
  model_path: $model_path
  data_dir:   $data_dir
  out_prefix: $out_prefix
  plot_dir:   $plot_dir
  save_plots: $save_plots
  denorm_psi: $denorm_psi
  use_gpu:    $(use_gpu === nothing ? "auto" : use_gpu)
"""

# =============================================================================
# 3. Module laden
# =============================================================================

include("dataset_psi.jl")
include("unet_psi.jl")
include("evaluate_psi.jl")

using .DatasetPsi
using .UNetPsi
using .EvaluatePsi

# =============================================================================
# 4. Evaluation
# =============================================================================

errors_by_n, all_results = EvaluatePsi.evaluate_dataset(;
    data_dir   = data_dir,
    model_path = model_path,
    out_prefix = out_prefix,
    save_plots = save_plots,
    plot_dir   = plot_dir,
    denorm_psi = denorm_psi,
    use_gpu    = use_gpu,
)

println("\n" * "=" ^ 60)
println("Evaluation fertig!")
println("=" ^ 60)
println("  Metriken: $(out_prefix)_by_n.csv")
save_plots && println("  Plots:    $plot_dir/")
println("=" ^ 60)
