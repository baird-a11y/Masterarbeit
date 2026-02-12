# Datei: UNetPsi.jl
# Parent-Modul – bündelt alle Submodule in der richtigen Reihenfolge.
#
# Nutzung:
#   include("UNetPsi.jl")
#   using .UNetPsi

module UNetPsi

# ── Config (kein eigenes Modul, wird direkt eingebunden) ──
include("config.jl")

# ── Basis-Utilities ──
include("utils_grids.jl")               # GridFDUtils
include("streamfunction_poisson.jl")     # StreamFunctionPoisson

# ── Physik-Pipeline ──
include("lamem_interface.jl")            # LaMEMInterface
include("normalization.jl")              # Normalization

# ── Dataset ──
include("dataset_psi.jl")               # DatasetPsi

# ── Modell ──
include("unet_psi.jl")                  # UNetPsi (Architektur)

# ── Training & Evaluation ──
include("losses.jl")                     # Losses
include("training_psi.jl")               # TrainingPsi
include("evaluate_psi.jl")               # EvaluatePsi

# ── Data Generation (optional) ──
include("data_generation_psi.jl")       # DataGenerationPsi

# ── Re-Exports ──
using .GridFDUtils
using .StreamFunctionPoisson
using .LaMEMInterface
using .Normalization
using .DatasetPsi
using .UNetPsiModel              # Das Modell-Modul
using .Losses
using .TrainingPsi
using .EvaluatePsi
using .DataGenerationPsi

end # module
