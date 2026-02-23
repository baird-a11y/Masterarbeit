# Datei: losses.jl
# Loss-Funktionen für ψ-FNO Training
# Shape-Konvention: ψ̂, ψ immer (nx, nz, 1, B) – konsistent mit Flux
#
# Funktion              Was sie tut
# mse_loss              Mean Squared Error (mit optionaler Maske)
# mae_loss              Mean Absolute Error
# rel_l2                Relative L2-Norm ‖ŷ−y‖/‖y‖
# max_abs_error         Maximaler Absolutfehler
# grad_mse_loss         MSE der Gradienten ∂ψ/∂x, ∂ψ/∂z (Zygote-kompatibel)
# boundary_mse_loss     Dirichlet ψ=0 am Rand erzwingen
# total_loss            L = MSE + α_grad·GradMSE + α_bnd·BndMSE → (loss, parts)
# compute_metrics       Alle Kennzahlen für Logging/CSV (nicht differenziert)


# Funktion	                Formel	                                                    Zygote-sicher
# mse_loss	                mean((ŷ−y)²) mit optionaler Maske	                        Ja
# mae_loss	                `mean(ŷ−y) `	                                            Ja
# rel_l2	                ‖ŷ−y‖₂ / (‖y‖₂ + ε)	                                        Ja
# max_abs_error	            `max(ŷ−y)`	                                                Ja
# grad_mse_loss	            MSE(∂ψ̂/∂x, ∂ψ/∂x) + MSE(∂ψ̂/∂z, ∂ψ/∂z)	                      Ja
# boundary_mse_loss	        MSE(ψ̂_rand, 0) für Dirichlet	                             Ja
# total_loss	            MSE + α_grad·GradMSE + α_bnd·BndMSE → (loss, parts)	        Ja
# compute_metrics	        Alle Kennzahlen → NamedTuple für CSV	                    Nein (nur Logging)



# Zentrale Design-Entscheidungen
# Gradienten-Loss via Slicing statt GridFDUtils.ddx/ddz — die Loop-basierten FD-Operatoren nutzen Array-Mutation (out[i,j] = ...), was Zygote nicht differenzieren kann. Stattdessen: (A[3:end,...] - A[1:end-2,...]) / 2dx — reine Slicing-Operationen, die Zygote nativ versteht
# Interior-Maske mutation-frei — _interior_mask_4d nutzt Outer-Product statt fill!, Zygote-sicher
# total_loss gibt Tuple zurück — (loss, parts) wobei parts = (; mse, grad, bnd). Im Training: loss, _ = total_loss(...) für Backprop, parts für Logging
# compute_metrics separat — wird nur außerhalb von Zygote aufgerufen (Validation), kann daher maximum etc. nutzen
# Nutzung im Training

# # Trainings-Loop
# grads = gradient(params) do
#     ψ̂ = model(Xb)
#     loss, _ = total_loss(ψ̂, Yb; dx=dx, dz=dz, α_grad=0.1f0)
#     loss
# end

# # Logging (keine Gradientenberechnung)
# metrics = compute_metrics(model(Xb), Yb; dx=dx, dz=dz)


module Losses

using Statistics
using Flux: gpu

export mse_loss,
       mae_loss,
       rel_l2,
       max_abs_error,
       grad_mse_loss,
       boundary_mse_loss,
       total_loss,
       compute_metrics

# =============================================================================
# Helpers
# =============================================================================

"""
Erzeugt eine Interior-Maske als Float32 (nx, nz, 1, 1) – broadcastbar mit 4D Tensoren.
Mutation-frei → Zygote-sicher.
"""
function _interior_mask_4d(nx::Int, nz::Int, width::Int)
    x_in = Float32.((1:nx) .> width .&& (1:nx) .<= nx - width)
    z_in = Float32.((1:nz) .> width .&& (1:nz) .<= nz - width)
    return reshape(x_in * z_in', nx, nz, 1, 1)
end

"""Bringt Array aufs selbe Device wie ref (CPU oder GPU)."""
_to_device_of(x, ref::Array) = x isa Array ? x : Array(x)
_to_device_of(x, ref) = x isa Array ? gpu(x) : x

"""
Masked mean: gewichteter Mittelwert mit optionaler Float32-Maske.
"""
function _masked_mean(x, mask)
    if mask === nothing
        return mean(x)
    else
        return sum(x .* mask) / (sum(mask) + 1f-12)
    end
end

# =============================================================================
# L1 – Basis-Losses
# =============================================================================

"""
    mse_loss(ŷ, y; mask=nothing)

Mean Squared Error. `mask` ist optional (nx, nz, 1, 1) Float32.
"""
function mse_loss(ŷ, y; mask = nothing)
    return _masked_mean((ŷ .- y) .^ 2, mask)
end

"""
    mae_loss(ŷ, y; mask=nothing)

Mean Absolute Error.
"""
function mae_loss(ŷ, y; mask = nothing)
    return _masked_mean(abs.(ŷ .- y), mask)
end

"""
    rel_l2(ŷ, y; mask=nothing, eps=1f-12)

Relative L2-Norm: ‖ŷ−y‖₂ / (‖y‖₂ + ε).
Gut für Vergleichbarkeit über Samples mit unterschiedlicher Amplitude.
"""
function rel_l2(ŷ, y; mask = nothing, eps = 1f-12)
    diff = ŷ .- y
    if mask === nothing
        return sqrt(sum(diff .^ 2)) / (sqrt(sum(y .^ 2)) + eps)
    else
        return sqrt(sum(diff .^ 2 .* mask)) / (sqrt(sum(y .^ 2 .* mask)) + eps)
    end
end

"""
    max_abs_error(ŷ, y; mask=nothing)

Maximaler Absolutfehler – guter Alarm-Indikator.
"""
function max_abs_error(ŷ, y; mask = nothing)
    ad = abs.(ŷ .- y)
    if mask === nothing
        return maximum(ad)
    else
        return maximum(ad .* mask)
    end
end

# =============================================================================
# L2 – Gradienten-Loss (Zygote-kompatibel via Slicing)
# =============================================================================

# Zentrale Differenzen nur auf Interior-Punkten (keine Rand-Stencils nötig).
# Slicing ist Zygote-sicher, im Gegensatz zu Loop-basierten FD-Operatoren.

function _ddx_interior(A, dx)
    # A: (nx, nz, 1, B) → (nx-2, nz, 1, B)
    return (A[3:end, :, :, :] .- A[1:end-2, :, :, :]) ./ (2f0 * Float32(dx))
end

function _ddz_interior(A, dz)
    # A: (nx, nz, 1, B) → (nx, nz-2, 1, B)
    return (A[:, 3:end, :, :] .- A[:, 1:end-2, :, :]) ./ (2f0 * Float32(dz))
end

"""
    grad_mse_loss(ψ̂, ψ, dx, dz)

MSE der räumlichen Gradienten:
`L = MSE(∂ψ̂/∂x, ∂ψ/∂x) + MSE(∂ψ̂/∂z, ∂ψ/∂z)`

Nutzt zentrale Differenzen auf Interior-Punkten (Rand automatisch ausgeschlossen).
Direkt proportional zum Geschwindigkeitsfehler (Vx = ∂ψ/∂z, Vz = −∂ψ/∂x).
"""
function grad_mse_loss(ψ̂, ψ, dx::Real, dz::Real)
    ψ̂x = _ddx_interior(ψ̂, dx)
    ψx  = _ddx_interior(ψ, dx)
    ψ̂z = _ddz_interior(ψ̂, dz)
    ψz  = _ddz_interior(ψ, dz)
    return mean((ψ̂x .- ψx) .^ 2) + mean((ψ̂z .- ψz) .^ 2)
end

# =============================================================================
# L3 – Rand-Loss
# =============================================================================

"""
    boundary_mse_loss(ψ̂; width=1, target=0f0)

MSE der Randwerte gegenüber `target` (default 0 für Dirichlet).
Verhindert Rand-Drift des Netzes.
"""
function boundary_mse_loss(ψ̂; width::Int = 1, target::Real = 0f0)
    w = width
    t = Float32(target)

    top    = ψ̂[1:w, :, :, :]
    bottom = ψ̂[end-w+1:end, :, :, :]
    left   = ψ̂[w+1:end-w, 1:w, :, :]
    right  = ψ̂[w+1:end-w, end-w+1:end, :, :]

    return (mean((top .- t) .^ 2) + mean((bottom .- t) .^ 2) +
            mean((left .- t) .^ 2) + mean((right .- t) .^ 2)) / 4f0
end

# =============================================================================
# L4 – Total Loss
# =============================================================================

"""
    total_loss(ψ̂, ψ; dx=1f0, dz=1f0, α_grad=0f0, α_bnd=0f0,
               bnd_width=1, mask_width=2)

Kombinierter Loss:
`L = MSE(ψ̂, ψ) + α_grad · GradMSE + α_bnd · BndMSE`

Gibt `(loss, parts)` zurück, wobei `parts = (; mse, grad, bnd)` für Logging.
MSE wird mit Interior-Maske (Breite `mask_width`) berechnet.
"""
function total_loss(ψ̂, ψ;
                    dx::Real = 1f0, dz::Real = 1f0,
                    α_grad::Real = 0f0, α_bnd::Real = 0f0,
                    bnd_width::Int = 1, mask_width::Int = 2)
    nx, nz = size(ψ̂, 1), size(ψ̂, 2)
    mask = _to_device_of(_interior_mask_4d(nx, nz, mask_width), ψ̂)

    l_mse  = mse_loss(ψ̂, ψ; mask=mask)
    l_grad = α_grad > 0 ? grad_mse_loss(ψ̂, ψ, dx, dz) : 0f0
    l_bnd  = α_bnd  > 0 ? boundary_mse_loss(ψ̂; width=bnd_width) : 0f0

    loss = l_mse + Float32(α_grad) * l_grad + Float32(α_bnd) * l_bnd

    parts = (; mse=l_mse, grad=l_grad, bnd=l_bnd)
    return loss, parts
end

# =============================================================================
# L5 – Metrics für Logging (NICHT differenziert – nur Eval)
# =============================================================================

"""
    compute_metrics(ψ̂, ψ; dx=1f0, dz=1f0, mask_width=2)

Berechnet alle Kennzahlen für Training-History/CSV.
Wird NICHT durch Zygote differenziert – nur für Logging.

Gibt ein NamedTuple zurück:
`(; loss_mse, loss_mae, rel_l2, max_error, grad_mse,
    pred_min, pred_max, target_min, target_max, nan_count)`
"""
function compute_metrics(ψ̂, ψ;
                         dx::Real = 1f0, dz::Real = 1f0,
                         mask_width::Int = 2)
    nx, nz = size(ψ̂, 1), size(ψ̂, 2)
    mask = _to_device_of(_interior_mask_4d(nx, nz, mask_width), ψ̂)

    return (
        loss_mse   = Float64(mse_loss(ψ̂, ψ; mask=mask)),
        loss_mae   = Float64(mae_loss(ψ̂, ψ; mask=mask)),
        rel_l2     = Float64(rel_l2(ψ̂, ψ; mask=mask)),
        max_error  = Float64(max_abs_error(ψ̂, ψ; mask=mask)),
        grad_mse   = Float64(grad_mse_loss(ψ̂, ψ, dx, dz)),
        pred_min   = Float64(minimum(ψ̂)),
        pred_max   = Float64(maximum(ψ̂)),
        target_min = Float64(minimum(ψ)),
        target_max = Float64(maximum(ψ)),
        nan_count  = count(isnan, ψ̂),
    )
end

end # module
# Datei: losses.jl
# Loss-Funktionen für ψ-FNO Training
# Shape-Konvention: ψ̂, ψ immer (nx, nz, 1, B) – konsistent mit Flux
#
# Funktion              Was sie tut
# mse_loss              Mean Squared Error (mit optionaler Maske)
# mae_loss              Mean Absolute Error
# rel_l2                Relative L2-Norm ‖ŷ−y‖/‖y‖
# max_abs_error         Maximaler Absolutfehler
# grad_mse_loss         MSE der Gradienten ∂ψ/∂x, ∂ψ/∂z (Zygote-kompatibel)
# boundary_mse_loss     Dirichlet ψ=0 am Rand erzwingen
# total_loss            L = MSE + α_grad·GradMSE + α_bnd·BndMSE → (loss, parts)
# compute_metrics       Alle Kennzahlen für Logging/CSV (nicht differenziert)


# Funktion	                Formel	                                                    Zygote-sicher
# mse_loss	                mean((ŷ−y)²) mit optionaler Maske	                        Ja
# mae_loss	                `mean(ŷ−y) `	                                            Ja
# rel_l2	                ‖ŷ−y‖₂ / (‖y‖₂ + ε)	                                        Ja
# max_abs_error	            `max(ŷ−y)`	                                                Ja
# grad_mse_loss	            MSE(∂ψ̂/∂x, ∂ψ/∂x) + MSE(∂ψ̂/∂z, ∂ψ/∂z)	                      Ja
# boundary_mse_loss	        MSE(ψ̂_rand, 0) für Dirichlet	                             Ja
# total_loss	            MSE + α_grad·GradMSE + α_bnd·BndMSE → (loss, parts)	        Ja
# compute_metrics	        Alle Kennzahlen → NamedTuple für CSV	                    Nein (nur Logging)



# Zentrale Design-Entscheidungen
# Gradienten-Loss via Slicing statt GridFDUtils.ddx/ddz — die Loop-basierten FD-Operatoren nutzen Array-Mutation (out[i,j] = ...), was Zygote nicht differenzieren kann. Stattdessen: (A[3:end,...] - A[1:end-2,...]) / 2dx — reine Slicing-Operationen, die Zygote nativ versteht
# Interior-Maske mutation-frei — _interior_mask_4d nutzt Outer-Product statt fill!, Zygote-sicher
# total_loss gibt Tuple zurück — (loss, parts) wobei parts = (; mse, grad, bnd). Im Training: loss, _ = total_loss(...) für Backprop, parts für Logging
# compute_metrics separat — wird nur außerhalb von Zygote aufgerufen (Validation), kann daher maximum etc. nutzen
# Nutzung im Training

# # Trainings-Loop
# grads = gradient(params) do
#     ψ̂ = model(Xb)
#     loss, _ = total_loss(ψ̂, Yb; dx=dx, dz=dz, α_grad=0.1f0)
#     loss
# end

# # Logging (keine Gradientenberechnung)
# metrics = compute_metrics(model(Xb), Yb; dx=dx, dz=dz)


module Losses

using Statistics
using Flux: gpu

export mse_loss,
       mae_loss,
       rel_l2,
       max_abs_error,
       grad_mse_loss,
       boundary_mse_loss,
       total_loss,
       compute_metrics

# =============================================================================
# Helpers
# =============================================================================

"""
Erzeugt eine Interior-Maske als Float32 (nx, nz, 1, 1) – broadcastbar mit 4D Tensoren.
Mutation-frei → Zygote-sicher.
"""
function _interior_mask_4d(nx::Int, nz::Int, width::Int)
    x_in = Float32.((1:nx) .> width .&& (1:nx) .<= nx - width)
    z_in = Float32.((1:nz) .> width .&& (1:nz) .<= nz - width)
    return reshape(x_in * z_in', nx, nz, 1, 1)
end

"""Bringt Array aufs selbe Device wie ref (CPU oder GPU)."""
_to_device_of(x, ref::Array) = x isa Array ? x : Array(x)
_to_device_of(x, ref) = x isa Array ? gpu(x) : x

"""
Masked mean: gewichteter Mittelwert mit optionaler Float32-Maske.
"""
function _masked_mean(x, mask)
    if mask === nothing
        return mean(x)
    else
        return sum(x .* mask) / (sum(mask) + 1f-12)
    end
end

# =============================================================================
# L1 – Basis-Losses
# =============================================================================

"""
    mse_loss(ŷ, y; mask=nothing)

Mean Squared Error. `mask` ist optional (nx, nz, 1, 1) Float32.
"""
function mse_loss(ŷ, y; mask = nothing)
    return _masked_mean((ŷ .- y) .^ 2, mask)
end

"""
    mae_loss(ŷ, y; mask=nothing)

Mean Absolute Error.
"""
function mae_loss(ŷ, y; mask = nothing)
    return _masked_mean(abs.(ŷ .- y), mask)
end

"""
    rel_l2(ŷ, y; mask=nothing, eps=1f-12)

Relative L2-Norm: ‖ŷ−y‖₂ / (‖y‖₂ + ε).
Gut für Vergleichbarkeit über Samples mit unterschiedlicher Amplitude.
"""
function rel_l2(ŷ, y; mask = nothing, eps = 1f-12)
    diff = ŷ .- y
    if mask === nothing
        return sqrt(sum(diff .^ 2)) / (sqrt(sum(y .^ 2)) + eps)
    else
        return sqrt(sum(diff .^ 2 .* mask)) / (sqrt(sum(y .^ 2 .* mask)) + eps)
    end
end

"""
    max_abs_error(ŷ, y; mask=nothing)

Maximaler Absolutfehler – guter Alarm-Indikator.
"""
function max_abs_error(ŷ, y; mask = nothing)
    ad = abs.(ŷ .- y)
    if mask === nothing
        return maximum(ad)
    else
        return maximum(ad .* mask)
    end
end

# =============================================================================
# L2 – Gradienten-Loss (Zygote-kompatibel via Slicing)
# =============================================================================

# Zentrale Differenzen nur auf Interior-Punkten (keine Rand-Stencils nötig).
# Slicing ist Zygote-sicher, im Gegensatz zu Loop-basierten FD-Operatoren.

function _ddx_interior(A, dx)
    # A: (nx, nz, 1, B) → (nx-2, nz, 1, B)
    return (A[3:end, :, :, :] .- A[1:end-2, :, :, :]) ./ (2f0 * Float32(dx))
end

function _ddz_interior(A, dz)
    # A: (nx, nz, 1, B) → (nx, nz-2, 1, B)
    return (A[:, 3:end, :, :] .- A[:, 1:end-2, :, :]) ./ (2f0 * Float32(dz))
end

"""
    grad_mse_loss(ψ̂, ψ, dx, dz)

MSE der räumlichen Gradienten:
`L = MSE(∂ψ̂/∂x, ∂ψ/∂x) + MSE(∂ψ̂/∂z, ∂ψ/∂z)`

Nutzt zentrale Differenzen auf Interior-Punkten (Rand automatisch ausgeschlossen).
Direkt proportional zum Geschwindigkeitsfehler (Vx = ∂ψ/∂z, Vz = −∂ψ/∂x).
"""
function grad_mse_loss(ψ̂, ψ, dx::Real, dz::Real)
    ψ̂x = _ddx_interior(ψ̂, dx)
    ψx  = _ddx_interior(ψ, dx)
    ψ̂z = _ddz_interior(ψ̂, dz)
    ψz  = _ddz_interior(ψ, dz)
    return mean((ψ̂x .- ψx) .^ 2) + mean((ψ̂z .- ψz) .^ 2)
end

# =============================================================================
# L3 – Rand-Loss
# =============================================================================

"""
    boundary_mse_loss(ψ̂; width=1, target=0f0)

MSE der Randwerte gegenüber `target` (default 0 für Dirichlet).
Verhindert Rand-Drift des Netzes.
"""
function boundary_mse_loss(ψ̂; width::Int = 1, target::Real = 0f0)
    w = width
    t = Float32(target)

    top    = ψ̂[1:w, :, :, :]
    bottom = ψ̂[end-w+1:end, :, :, :]
    left   = ψ̂[w+1:end-w, 1:w, :, :]
    right  = ψ̂[w+1:end-w, end-w+1:end, :, :]

    return (mean((top .- t) .^ 2) + mean((bottom .- t) .^ 2) +
            mean((left .- t) .^ 2) + mean((right .- t) .^ 2)) / 4f0
end

# =============================================================================
# L4 – Total Loss
# =============================================================================

"""
    total_loss(ψ̂, ψ; dx=1f0, dz=1f0, α_grad=0f0, α_bnd=0f0,
               bnd_width=1, mask_width=2)

Kombinierter Loss:
`L = MSE(ψ̂, ψ) + α_grad · GradMSE + α_bnd · BndMSE`

Gibt `(loss, parts)` zurück, wobei `parts = (; mse, grad, bnd)` für Logging.
MSE wird mit Interior-Maske (Breite `mask_width`) berechnet.
"""
function total_loss(ψ̂, ψ;
                    dx::Real = 1f0, dz::Real = 1f0,
                    α_grad::Real = 0f0, α_bnd::Real = 0f0,
                    bnd_width::Int = 1, mask_width::Int = 2)
    nx, nz = size(ψ̂, 1), size(ψ̂, 2)
    mask = _to_device_of(_interior_mask_4d(nx, nz, mask_width), ψ̂)

    l_mse  = mse_loss(ψ̂, ψ; mask=mask)
    l_grad = α_grad > 0 ? grad_mse_loss(ψ̂, ψ, dx, dz) : 0f0
    l_bnd  = α_bnd  > 0 ? boundary_mse_loss(ψ̂; width=bnd_width) : 0f0

    loss = l_mse + Float32(α_grad) * l_grad + Float32(α_bnd) * l_bnd

    parts = (; mse=l_mse, grad=l_grad, bnd=l_bnd)
    return loss, parts
end

# =============================================================================
# L5 – Metrics für Logging (NICHT differenziert – nur Eval)
# =============================================================================

"""
    compute_metrics(ψ̂, ψ; dx=1f0, dz=1f0, mask_width=2)

Berechnet alle Kennzahlen für Training-History/CSV.
Wird NICHT durch Zygote differenziert – nur für Logging.

Gibt ein NamedTuple zurück:
`(; loss_mse, loss_mae, rel_l2, max_error, grad_mse,
    pred_min, pred_max, target_min, target_max, nan_count)`
"""
function compute_metrics(ψ̂, ψ;
                         dx::Real = 1f0, dz::Real = 1f0,
                         mask_width::Int = 2)
    nx, nz = size(ψ̂, 1), size(ψ̂, 2)
    mask = _to_device_of(_interior_mask_4d(nx, nz, mask_width), ψ̂)

    return (
        loss_mse   = Float64(mse_loss(ψ̂, ψ; mask=mask)),
        loss_mae   = Float64(mae_loss(ψ̂, ψ; mask=mask)),
        rel_l2     = Float64(rel_l2(ψ̂, ψ; mask=mask)),
        max_error  = Float64(max_abs_error(ψ̂, ψ; mask=mask)),
        grad_mse   = Float64(grad_mse_loss(ψ̂, ψ, dx, dz)),
        pred_min   = Float64(minimum(ψ̂)),
        pred_max   = Float64(maximum(ψ̂)),
        target_min = Float64(minimum(ψ)),
        target_max = Float64(maximum(ψ)),
        nan_count  = count(isnan, ψ̂),
    )
end

end # module
