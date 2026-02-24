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

# ── Physik-Pipeline (optional – nur laden wenn LaMEM verfügbar) ──
try
    include("lamem_interface.jl")            # LaMEMInterface  (→ StreamFunctionPoisson)
catch e
    @warn "LaMEMInterface nicht geladen (LaMEM.jl installiert/kompatibel?): $e"
end
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
if isdefined(@__MODULE__, :LaMEMInterface)
    using .LaMEMInterface
end
using .Normalization
using .DatasetPsi
using .FNOLayers
using .FNOModel
using .Losses
using .TrainingPsi
using .EvalPsi

end # module
