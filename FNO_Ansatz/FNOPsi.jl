# Datei: FNOPsi.jl
# Parent-Modul – bündelt alle Submodule in der richtigen Reihenfolge.
#
# Nutzung:
#   include("FNOPsi.jl")
#   using .FNOPsi

module FNOPsi

# ── Config (kein eigenes Modul, wird direkt eingebunden) ──
include("config.jl")

# ── Basis-Utilities ──
include("utils_grids.jl")               # GridFDUtils
include("streamfunction_possion.jl")     # StreamFunctionPoisson

# ── Physik-Pipeline ──
include("lamem_interface.jl")            # LaMEMInterface  (→ StreamFunctionPoisson)
include("normalization.jl")              # Normalization

# ── Dataset ──
include("dataset_psi.jl")               # DatasetPsi (→ GridFDUtils)

# ── Modell ──
include("fno_layers.jl")                # FNOLayers
include("fno_model.jl")                 # FNOModel (→ FNOLayers)

# ── Training & Evaluation ──
include("losses.jl")                     # Losses
include("train.jl")                      # TrainingPsi (→ DatasetPsi, Losses)
include("eval.jl")                       # EvalPsi (→ DatasetPsi, Normalization, GridFDUtils, Losses)

# ── Plots (optional – nur laden wenn Plots.jl verfügbar) ──
try
    include("plots_eval.jl")             # PlotsEval (→ GridFDUtils)
catch e
    @warn "PlotsEval konnte nicht geladen werden (Plots.jl installiert?): $e"
end

# ── Re-Exports ──
using .GridFDUtils
using .StreamFunctionPoisson
using .LaMEMInterface
using .Normalization
using .DatasetPsi
using .FNOLayers
using .FNOModel
using .Losses
using .TrainingPsi
using .EvalPsi

end # module
