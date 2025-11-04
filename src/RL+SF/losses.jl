# =============================================================================
# LOSS FUNCTIONS - RESIDUAL LEARNING
# =============================================================================
# Custom Loss Functions für Residual Learning mit optionaler Stream Function

using Flux
using Statistics

println("Loss Functions wird geladen...")

# =============================================================================
# RESIDUAL LEARNING LOSS
# =============================================================================

"""
    residual_loss(v_pred, v_stokes, Δv, v_target; kwargs...)

Haupt-Loss-Funktion für Residual Learning.

# Komponenten
1. **Velocity Loss**: MSE zwischen Vorhersage und Ground Truth
   - L_vel = ||v_pred - v_target||²
   
2. **Residual Penalty**: Bevorzugt kleine Residuen
   - L_res = λ_residual * ||Δv||²
   
3. **Divergence Loss** (nur ohne Stream Function)
   - L_div = λ_divergence * ||∇·v||²

# Arguments
- `v_pred::AbstractArray`: Vorhergesagtes Geschwindigkeitsfeld [H, W, 2, B]
- `v_stokes::AbstractArray`: Stokes-Baseline [H, W, 2, B]
- `Δv::AbstractArray`: Gelerntes Residuum [H, W, 2, B]
- `v_target::AbstractArray`: Ground Truth von LaMEM [H, W, 2, B]

# Keyword Arguments
- `λ_residual::Float32`: Gewicht für Residual-Penalty (default: 0.01)
- `λ_divergence::Float32`: Gewicht für Divergenz-Loss (default: 0.1)
- `use_stream_function::Bool`: Ob Stream Function genutzt wird (default: false)

# Returns
- `Tuple`: (total_loss, velocity_loss, residual_penalty, divergence_loss)
"""
function residual_loss(
    v_pred::AbstractArray{T,4},
    v_stokes::AbstractArray{T,4},
    Δv::AbstractArray{T,4},
    v_target::AbstractArray{T,4};
    λ_residual::Float32 = 0.01f0,
    λ_divergence::Float32 = 0.1f0,
    use_stream_function::Bool = false
) where T
    
    # 1. Velocity Loss (Haupt-Ziel)
    velocity_loss = Flux.mse(v_pred, v_target)
    
    # 2. Residual Penalty (kleine Korrekturen bevorzugen)
    residual_penalty = λ_residual * mean(abs2, Δv)
    
    # 3. Divergence Loss (nur ohne Stream Function)
    if !use_stream_function
        div = compute_divergence_loss(v_pred)
        divergence_loss = λ_divergence * div
    else
        divergence_loss = T(0)
    end
    
    # Total Loss
    total_loss = velocity_loss + residual_penalty + divergence_loss
    
    return total_loss, velocity_loss, residual_penalty, divergence_loss
end

# =============================================================================
# DIVERGENZ-BERECHNUNG
# =============================================================================

"""
    compute_divergence_loss(velocity::AbstractArray{T,4})

Berechnet Divergenz-Loss für inkompressiblen Fluss.

# Ziel
∇·v = ∂vx/∂x + ∂vz/∂z ≈ 0

# Returns
- `T`: Mean squared divergence
"""
function compute_divergence_loss(velocity::AbstractArray{T,4}) where T
    vx = velocity[:, :, 1:1, :]
    vz = velocity[:, :, 2:2, :]
    
    # Finite Differenzen
    ∂vx_∂x = vx[:, 2:end, :, :] .- vx[:, 1:end-1, :, :]
    ∂vz_∂z = vz[2:end, :, :, :] .- vz[1:end-1, :, :, :]
    
    # Match Dimensionen
    min_h = min(size(∂vx_∂x, 1), size(∂vz_∂z, 1))
    min_w = min(size(∂vx_∂x, 2), size(∂vz_∂z, 2))
    
    ∂vx_∂x = ∂vx_∂x[1:min_h, 1:min_w, :, :]
    ∂vz_∂z = ∂vz_∂z[1:min_h, 1:min_w, :, :]
    
    # Divergenz
    div = ∂vx_∂x .+ ∂vz_∂z
    
    # Mean squared divergence
    return mean(abs2, div)
end

# =============================================================================
# ALTERNATIVE LOSS COMPONENTS
# =============================================================================

"""
    sparsity_loss(Δv::AbstractArray)

L1-Sparsity Loss für Residuen.

Bevorzugt spärliche Korrekturen (viele Null-Werte).

# Formula
L_sparse = λ * Σ|Δv|
"""
function sparsity_loss(Δv::AbstractArray{T}) where T
    return mean(abs, Δv)
end

"""
    gradient_loss(v_pred::AbstractArray{T,4}, v_target::AbstractArray{T,4})

Loss auf räumlichen Gradienten.

Verbessert strukturelle Genauigkeit (Kanten, Strömungskanäle).

# Formula
L_grad = ||∇v_pred - ∇v_target||²
"""
function gradient_loss(v_pred::AbstractArray{T,4}, v_target::AbstractArray{T,4}) where T
    # Gradienten in x-Richtung
    ∇x_pred = v_pred[:, 2:end, :, :] .- v_pred[:, 1:end-1, :, :]
    ∇x_target = v_target[:, 2:end, :, :] .- v_target[:, 1:end-1, :, :]
    
    # Gradienten in z-Richtung
    ∇z_pred = v_pred[2:end, :, :, :] .- v_pred[1:end-1, :, :, :]
    ∇z_target = v_target[2:end, :, :, :] .- v_target[1:end-1, :, :]
    
    # Match Dimensionen
    min_h = min(size(∇x_pred, 1), size(∇z_pred, 1))
    min_w = min(size(∇x_pred, 2), size(∇z_pred, 2))
    
    ∇x_pred = ∇x_pred[1:min_h, 1:min_w, :, :]
    ∇x_target = ∇x_target[1:min_h, 1:min_w, :, :]
    ∇z_pred = ∇z_pred[1:min_h, 1:min_w, :, :]
    ∇z_target = ∇z_target[1:min_h, 1:min_w, :, :]
    
    # MSE auf Gradienten
    loss_x = mean(abs2, ∇x_pred .- ∇x_target)
    loss_z = mean(abs2, ∇z_pred .- ∇z_target)
    
    return loss_x + loss_z
end

"""
    stokes_consistency_loss(v_stokes::AbstractArray, Δv::AbstractArray)

Prüft ob Residuum konsistent mit Stokes ist.

Warnt wenn Residuum unrealistisch groß wird.

# Returns
- `Float32`: Ratio ||Δv|| / ||v_stokes||
"""
function stokes_consistency_loss(v_stokes::AbstractArray{T,4}, Δv::AbstractArray{T,4}) where T
    norm_stokes = sqrt(mean(abs2, v_stokes)) + T(1e-8)
    norm_residual = sqrt(mean(abs2, Δv))
    
    # Ratio
    ratio = norm_residual / norm_stokes
    
    return ratio
end

# =============================================================================
# ERWEITERTE LOSS MIT ZUSÄTZLICHEN KOMPONENTEN
# =============================================================================

"""
    residual_loss_extended(v_pred, v_stokes, Δv, v_target; kwargs...)

Erweiterte Loss-Funktion mit optionalen Komponenten.

Zusätzlich zu `residual_loss`:
- Sparsity Loss (L1 auf Δv)
- Gradient Loss (strukturelle Genauigkeit)
- Stokes Consistency Check

# Keyword Arguments (zusätzlich)
- `λ_sparsity::Float32`: Gewicht für Sparsity (default: 0.001)
- `λ_gradient::Float32`: Gewicht für Gradient Loss (default: 0.05)
- `warn_consistency::Bool`: Warnung bei großen Residuen (default: true)
"""
function residual_loss_extended(
    v_pred::AbstractArray{T,4},
    v_stokes::AbstractArray{T,4},
    Δv::AbstractArray{T,4},
    v_target::AbstractArray{T,4};
    λ_residual::Float32 = 0.01f0,
    λ_divergence::Float32 = 0.1f0,
    λ_sparsity::Float32 = 0.001f0,
    λ_gradient::Float32 = 0.05f0,
    use_stream_function::Bool = false,
    warn_consistency::Bool = true
) where T
    
    # Basis-Loss
    total, vel_loss, res_penalty, div_loss = residual_loss(
        v_pred, v_stokes, Δv, v_target;
        λ_residual=λ_residual,
        λ_divergence=λ_divergence,
        use_stream_function=use_stream_function
    )
    
    # Sparsity
    sparse_loss = λ_sparsity * sparsity_loss(Δv)
    
    # Gradient
    grad_loss = λ_gradient * gradient_loss(v_pred, v_target)
    
    # Consistency Check
    if warn_consistency
        ratio = stokes_consistency_loss(v_stokes, Δv)
        if ratio > 2.0
            @warn "Residuum sehr groß im Vergleich zu Stokes: $(round(ratio, digits=2))x"
        end
    end
    
    # Extended Total
    total_extended = total + sparse_loss + grad_loss
    
    return total_extended, vel_loss, res_penalty, div_loss, sparse_loss, grad_loss
end

# =============================================================================
# LAMBDA-SCHEDULE (für Training)
# =============================================================================

"""
    LambdaSchedule

Schedule für Lambda-Gewichte während des Trainings.

Beispiel: Erhöhe Divergenz-Gewicht über Zeit
- Frühe Epochen: Fokus auf Velocity Accuracy
- Späte Epochen: Fokus auf Physik (Divergenzfreiheit)
"""
struct LambdaSchedule
    λ_residual_start::Float32
    λ_residual_end::Float32
    λ_divergence_start::Float32
    λ_divergence_end::Float32
    total_epochs::Int
end

"""
    get_lambdas(schedule::LambdaSchedule, epoch::Int)

Berechnet aktuelle Lambda-Werte für gegebene Epoche.

# Returns
- `Tuple`: (λ_residual, λ_divergence)
"""
function get_lambdas(schedule::LambdaSchedule, epoch::Int)
    # Linear interpolation
    progress = min(epoch / schedule.total_epochs, 1.0f0)
    
    λ_residual = schedule.λ_residual_start + 
                 progress * (schedule.λ_residual_end - schedule.λ_residual_start)
    
    λ_divergence = schedule.λ_divergence_start + 
                   progress * (schedule.λ_divergence_end - schedule.λ_divergence_start)
    
    return λ_residual, λ_divergence
end

# =============================================================================
# METRIKEN FÜR MONITORING
# =============================================================================

"""
    compute_training_metrics(v_pred, v_stokes, Δv, v_target)

Berechnet zusätzliche Metriken für Monitoring.

# Returns
- `Dict`: Verschiedene Metriken
"""
function compute_training_metrics(
    v_pred::AbstractArray{T,4},
    v_stokes::AbstractArray{T,4},
    Δv::AbstractArray{T,4},
    v_target::AbstractArray{T,4}
) where T
    
    # MSE Komponenten
    mse_total = Flux.mse(v_pred, v_target)
    mse_stokes = Flux.mse(v_stokes, v_target)
    
    # Improvement durch Residual Learning
    improvement = (mse_stokes - mse_total) / mse_stokes * 100
    
    # Residual Statistiken
    mean_residual = mean(abs, Δv)
    max_residual = maximum(abs, Δv)
    
    # Divergenz
    div = compute_divergence_loss(v_pred)
    
    metrics = Dict(
        "mse_total" => mse_total,
        "mse_stokes" => mse_stokes,
        "improvement_percent" => improvement,
        "mean_residual" => mean_residual,
        "max_residual" => max_residual,
        "divergence" => div
    )
    
    return metrics
end

# =============================================================================
# MODUL-INFO
# =============================================================================

println("Loss Functions geladen!")
println("   - Residual Loss (Basis)")
println("   - Extended Loss (mit Sparsity & Gradient)")
println("   - Lambda Schedule Support")
println("   - Training Metrics")
println("")
println("Wichtige Funktionen:")
println("   - residual_loss(...) - Basis-Loss")
println("   - residual_loss_extended(...) - Mit zusätzlichen Komponenten")
println("   - compute_training_metrics(...) - Monitoring")
println("")
println("Empfohlene Gewichte:")
println("   - λ_residual = 0.01 (Start)")
println("   - λ_divergence = 0.1 (ohne Stream Function)")
println("   - λ_sparsity = 0.001 (optional)")
println("   - λ_gradient = 0.05 (optional)")