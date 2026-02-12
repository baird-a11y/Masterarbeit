# Datei: losses.jl
# Zentrale Loss-Funktionen und Metriken für ψ-U-Net
#
# Funktionen:
#   mse_loss           - Mean Squared Error
#   rel_l2             - Relative L2-Norm
#   max_abs_error      - Maximaler absoluter Fehler
#   gradient_loss      - Gradienten-basierter Loss (∂ψ/∂x, ∂ψ/∂z)
#   boundary_loss      - Rand-Constraint Loss
#   total_loss         - Kombinierter Loss
#   compute_metrics    - Alle Metriken für Logging

module Losses

using Statistics

export mse_loss,
       rel_l2,
       max_abs_error,
       gradient_loss,
       boundary_loss,
       total_loss,
       compute_metrics

# =============================================================================
# Basis-Metriken
# =============================================================================

"""
    mse_loss(ŷ, y)

Mean Squared Error zwischen Prediction ŷ und Ground Truth y.
"""
function mse_loss(ŷ::AbstractArray, y::AbstractArray)
    return mean((ŷ .- y) .^ 2)
end

"""
    rel_l2(ŷ, y)

Relative L2-Norm: ||ŷ - y||₂ / ||y||₂
"""
function rel_l2(ŷ::AbstractArray, y::AbstractArray)
    num = sqrt(sum((ŷ .- y) .^ 2))
    den = sqrt(sum(y .^ 2))
    return den > 1f-12 ? num / den : num
end

"""
    max_abs_error(ŷ, y)

Maximaler absoluter Fehler: max|ŷ - y|
"""
function max_abs_error(ŷ::AbstractArray, y::AbstractArray)
    return maximum(abs.(ŷ .- y))
end

# =============================================================================
# Physik-basierte Losses
# =============================================================================

"""
    gradient_loss(ŷ, y, dx, dz; mask_width=2)

Loss auf den Gradienten ∂ψ/∂x und ∂ψ/∂z.
Vergleicht die Gradienten der Prediction mit dem Ground Truth.

Argumente:
- ŷ, y: (nx, nz, 1, B) Arrays
- dx, dz: Gitterabstand
- mask_width: Randbreite die ignoriert wird
"""
function gradient_loss(ŷ::AbstractArray{T,4}, y::AbstractArray{T,4},
                       dx::Real, dz::Real;
                       mask_width::Int=2) where T
    nx, nz, _, B = size(ŷ)

    # Interior-Maske
    i1, i2 = mask_width + 1, nx - mask_width
    j1, j2 = mask_width + 1, nz - mask_width

    loss_grad = 0f0

    for b in 1:B
        # Zentrale Differenzen für ∂ψ/∂x
        ∂ŷ_∂x = (ŷ[i1+1:i2, j1:j2-1, 1, b] .- ŷ[i1-1:i2-2, j1:j2-1, 1, b]) ./ (2f0 * Float32(dx))
        ∂y_∂x = (y[i1+1:i2, j1:j2-1, 1, b] .- y[i1-1:i2-2, j1:j2-1, 1, b]) ./ (2f0 * Float32(dx))

        # Zentrale Differenzen für ∂ψ/∂z
        ∂ŷ_∂z = (ŷ[i1:i2-1, j1+1:j2, 1, b] .- ŷ[i1:i2-1, j1-1:j2-2, 1, b]) ./ (2f0 * Float32(dz))
        ∂y_∂z = (y[i1:i2-1, j1+1:j2, 1, b] .- y[i1:i2-1, j1-1:j2-2, 1, b]) ./ (2f0 * Float32(dz))

        # MSE der Gradienten
        loss_grad += mean((∂ŷ_∂x .- ∂y_∂x) .^ 2)
        loss_grad += mean((∂ŷ_∂z .- ∂y_∂z) .^ 2)
    end

    return loss_grad / Float32(B)
end

"""
    boundary_loss(ŷ; mask_width=2)

Bestraft Abweichungen von ψ=0 am Rand.

Argumente:
- ŷ: (nx, nz, 1, B) Prediction
- mask_width: Breite des Rand-Bereichs
"""
function boundary_loss(ŷ::AbstractArray{T,4}; mask_width::Int=2) where T
    nx, nz, _, B = size(ŷ)

    loss_bnd = 0f0

    for b in 1:B
        # Ränder extrahieren
        left   = ŷ[1:mask_width, :, 1, b]
        right  = ŷ[nx-mask_width+1:nx, :, 1, b]
        bottom = ŷ[:, 1:mask_width, 1, b]
        top    = ŷ[:, nz-mask_width+1:nz, 1, b]

        # Penalize non-zero values
        loss_bnd += mean(left .^ 2)
        loss_bnd += mean(right .^ 2)
        loss_bnd += mean(bottom .^ 2)
        loss_bnd += mean(top .^ 2)
    end

    return loss_bnd / (4f0 * Float32(B))
end

# =============================================================================
# Kombinierter Loss
# =============================================================================

"""
    total_loss(ŷ, y; α_grad=0.0f0, α_bnd=0.0f0, dx=nothing, dz=nothing, mask_width=2)

Kombinierter Loss:
    L = L_MSE + α_grad * L_grad + α_bnd * L_bnd

Gibt (loss_total, loss_dict) zurück, wobei loss_dict alle Komponenten enthält.
"""
function total_loss(ŷ::AbstractArray{T,4}, y::AbstractArray{T,4};
                    α_grad::Real = 0f0,
                    α_bnd::Real = 0f0,
                    dx::Union{Real, Nothing} = nothing,
                    dz::Union{Real, Nothing} = nothing,
                    mask_width::Int = 2) where T

    # Basis-Loss: MSE
    loss_mse = mse_loss(ŷ, y)

    loss_dict = Dict{Symbol, Float32}()
    loss_dict[:mse] = Float32(loss_mse)

    loss_total = loss_mse

    # Gradienten-Loss (optional)
    if α_grad > 0f0 && dx !== nothing && dz !== nothing
        loss_grad = gradient_loss(ŷ, y, dx, dz; mask_width=mask_width)
        loss_dict[:grad] = Float32(loss_grad)
        loss_total += Float32(α_grad) * loss_grad
    else
        loss_dict[:grad] = 0f0
    end

    # Rand-Loss (optional)
    if α_bnd > 0f0
        loss_bnd = boundary_loss(ŷ; mask_width=mask_width)
        loss_dict[:bnd] = Float32(loss_bnd)
        loss_total += Float32(α_bnd) * loss_bnd
    else
        loss_dict[:bnd] = 0f0
    end

    loss_dict[:total] = Float32(loss_total)

    return loss_total, loss_dict
end

# =============================================================================
# Metriken für Evaluation
# =============================================================================

"""
    compute_metrics(ŷ, y; dx=nothing, dz=nothing, mask_width=2)

Berechnet alle relevanten Metriken für Logging/Evaluation.

Returns: NamedTuple mit:
- mse, rel_l2, max_err
- (optional) grad_mse wenn dx/dz gegeben
"""
function compute_metrics(ŷ::AbstractArray, y::AbstractArray;
                         dx::Union{Real, Nothing} = nothing,
                         dz::Union{Real, Nothing} = nothing,
                         mask_width::Int = 2)

    mse = Float64(mse_loss(ŷ, y))
    rel = Float64(rel_l2(ŷ, y))
    max_err = Float64(max_abs_error(ŷ, y))

    metrics = (mse=mse, rel_l2=rel, max_err=max_err)

    # Optional: Gradienten-Metriken
    if dx !== nothing && dz !== nothing && ndims(ŷ) == 4
        grad_mse = Float64(gradient_loss(ŷ, y, dx, dz; mask_width=mask_width))
        metrics = merge(metrics, (grad_mse=grad_mse,))
    end

    return metrics
end

end # module
