# Datei: fno_model.jl
# Komplettes 2D-FNO Modell für ψ-Vorhersage
#
# Architektur (Li et al., 2020 – 2010.08895):
#
#   x (nx, nz, Cin, B)
#     │
#   ┌─┴─┐  P  (Lift)
#   │Conv(1×1, Cin → width)
#   └─┬─┘
#     │
#   ┌─┴─┐  T₁ … T_depth  (FNO-Blöcke)
#   │FNOBlock × depth
#   └─┬─┘
#     │
#   ┌─┴─┐  Q  (Project)
#   │Conv(1×1, width → width) + gelu
#   │Conv(1×1, width → Cout)
#   └─┬─┘
#     │
#   ŷ (nx, nz, Cout, B)


# Komponente            Was	                                                                Code
# Lift (P)	        Conv((1,1), Cin → width)	                                            Hebt Input-Kanäle in den latenten Raum
# Blocks (T)	    Chain(FNOBlock × depth)	                                                depth FNO-Blöcke (spectral + local + gelu)
# Project (Q)	    Conv((1,1), width → width, gelu) + Conv((1,1), width → Cout)	        2-stufiges MLP zurück auf Cout=1
# FNOPsi	        ŷ = project(blocks(lift(x)))	                                        Parametrisches Struct mit @functor


# build_fno – zwei Varianten

# # Direkt mit Hyperparametern
# model = build_fno(4; width=64, depth=4, modes_x=16, modes_z=16)

# # Aus Config-Objekt
# cfg = load_config()
# model = build_fno(cfg; Cin=4)
# Datenfluss für dein Setup (256×256, Cin=4, width=64, depth=4, modes=16)

# x:  (256, 256, 4, B)
#  → lift:     (256, 256, 64, B)
#  → block×4:  (256, 256, 64, B)    je Block: rfft → 16×16 Moden → irfft + 1×1Conv + gelu
#  → project:  (256, 256, 1, B)     width→width→1




module FNOModel

using Flux
using Functors: @functor
using NNlib: gelu
using ..FNOLayers: SpectralConv2D, FNOBlock

export FNOPsi,
       build_fno

# =============================================================================
# M5 – Model Struct
# =============================================================================

"""
    FNOPsi

Komplettes 2D Fourier Neural Operator Modell.

Felder:
- `lift`    – P: Conv((1,1), Cin → width)
- `blocks`  – T₁…T_d: Chain von FNOBlocks
- `project` – Q: 2-stufiges MLP (width → width → Cout)

Forward: `ŷ = project(blocks(lift(x)))`
"""
struct FNOPsi{L, B, P}
    lift::L
    blocks::B
    project::P
end

@functor FNOPsi

function (m::FNOPsi)(x::AbstractArray{<:Real, 4})
    return m.project(m.blocks(m.lift(x)))
end

function Base.show(io::IO, m::FNOPsi)
    # Lift info
    lift_ch = "?"
    if m.lift isa Conv
        cin  = size(m.lift.weight, 3)
        cout = size(m.lift.weight, 4)
        lift_ch = "$cin => $cout"
    end

    # Block count
    n_blocks = 0
    if m.blocks isa Chain
        n_blocks = length(m.blocks.layers)
    end

    # Project info
    proj_out = "?"
    if m.project isa Chain && length(m.project.layers) ≥ 2
        last_conv = m.project.layers[end]
        if last_conv isa Conv
            proj_out = string(size(last_conv.weight, 4))
        end
    end

    print(io, "FNOPsi(lift=$lift_ch, blocks=$n_blocks, Cout=$proj_out)")
end

# =============================================================================
# M6 – Factory: build_fno
# =============================================================================

"""
    build_fno(Cin; width=64, depth=4, modes_x=16, modes_z=16,
              Cout=1, activation=gelu)

Baut ein komplettes FNO-2D Modell für ψ-Vorhersage.

Architektur:
1. **Lift** (P): `Conv((1,1), Cin → width)`
2. **Blocks** (T): `depth × FNOBlock(width, modes_x, modes_z)`
3. **Project** (Q): `Conv((1,1), width → width) + activation + Conv((1,1), width → Cout)`

# Beispiel
```julia
model = build_fno(4; width=64, depth=4, modes_x=16, modes_z=16)
x = rand(Float32, 256, 256, 4, 2)
ŷ = model(x)   # (256, 256, 1, 2)
```
"""
function build_fno(Cin::Int;
                   width::Int = 64,
                   depth::Int = 4,
                   modes_x::Int = 16,
                   modes_z::Int = 16,
                   Cout::Int = 1,
                   activation = gelu)
    # M2 – Lift: Cin → width
    lift = Conv((1, 1), Cin => width)

    # M3 – FNO Blocks
    blocks = Chain([FNOBlock(width, modes_x, modes_z; activation=activation)
                    for _ in 1:depth]...)

    # M4 – Project: width → Cout (2-stufig wie im Paper)
    project = Chain(
        Conv((1, 1), width => width, activation),
        Conv((1, 1), width => Cout),
    )

    return FNOPsi(lift, blocks, project)
end

"""
    build_fno(cfg::NamedTuple)
    build_fno(cfg)

Convenience: baut FNO aus einem Config-Objekt mit Feldern
`modes`, `width`, `depth` und optionalem `Cin`.
"""
function build_fno(cfg; Cin::Int, Cout::Int = 1, activation = gelu)
    modes = hasproperty(cfg, :modes) ? cfg.modes : 16
    width = hasproperty(cfg, :width) ? cfg.width : 64
    depth = hasproperty(cfg, :depth) ? cfg.depth : 4

    return build_fno(Cin; width=width, depth=depth,
                     modes_x=modes, modes_z=modes,
                     Cout=Cout, activation=activation)
end

end # module
