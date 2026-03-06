module TrainingPsi

using Flux
using Statistics
using Random
using Printf
using BSON
using Optimisers: Adam
using Flux: update!

using ..DatasetPsi: PsiDataset, load_dataset, batch_iterator, get_sample
using ..UNetPsi: build_unet
using Functors: fmap

# --- Optional CUDA support (kompatibel mit alten/neuen CUDA.jl) ---
const _HAS_CUDA = try
    @eval import CUDA
    true
catch
    false
end

function has_gpu()
    if !_HAS_CUDA
        return false
    end
    try
        return CUDA.has_cuda()
    catch
        return false
    end
end

function _to_gpu(x)
    if !_HAS_CUDA
        return x
    end
    try
        if isdefined(CUDA, :cuda)
            return CUDA.cuda(x)
        elseif isdefined(CUDA, :cu)
            return CUDA.cu(x)
        else
            return x
        end
    catch
        return x
    end
end

_to_cpu(x) = cpu(x)

function _select_mover(use_gpu::Union{Nothing,Bool})
    if use_gpu === false
        return _to_cpu, :cpu
    elseif use_gpu === true
        if has_gpu()
            return _to_gpu, :gpu
        else
            @warn "GPU angefordert, aber CUDA nicht verfügbar – nutze CPU."
            return _to_cpu, :cpu
        end
    else
        if has_gpu()
            return _to_gpu, :gpu
        else
            return _to_cpu, :cpu
        end
    end
end

"""
    mse_loss(y_pred, y_true)

Mean-Squared-Error über alle Dimensionen.
"""
mse_loss(y_pred, y_true) = mean((y_pred .- y_true).^2)

# =============================================================================
# Composite Loss (analog FNO losses.jl)
# =============================================================================

"""Erzeugt Interior-Maske (nx, nz, 1, 1) – broadcastbar, Zygote-sicher."""
function _interior_mask_4d(nx::Int, nz::Int, width::Int)
    x_in = Float32.((1:nx) .> width .&& (1:nx) .<= nx - width)
    z_in = Float32.((1:nz) .> width .&& (1:nz) .<= nz - width)
    return reshape(x_in * z_in', nx, nz, 1, 1)
end

"""Verschiebt x auf dasselbe Device wie ref (CPU Array bleibt, sonst gpu())."""
_to_device_of(x, ref::Array) = x isa Array ? x : Array(x)
_to_device_of(x, ref)        = x isa Array ? gpu(x) : x

"""Zentrale Differenzen in x (Slicing, Zygote-sicher)."""
_ddx(A, dx) = (A[3:end, :, :, :] .- A[1:end-2, :, :, :]) ./ (2f0 * Float32(dx))

"""Zentrale Differenzen in z (Slicing, Zygote-sicher)."""
_ddz(A, dz) = (A[:, 3:end, :, :] .- A[:, 1:end-2, :, :]) ./ (2f0 * Float32(dz))

"""
    grad_mse_loss(ŷ, y, dx, dz)

MSE der räumlichen Gradienten ∂ψ/∂x und ∂ψ/∂z.
Entspricht dem Gradienten-Term aus der FNO-Loss: α·‖∇ψ̂ − ∇ψ‖₂².
"""
function grad_mse_loss(ŷ, y, dx::Real, dz::Real)
    return mean((_ddx(ŷ, dx) .- _ddx(y, dx)) .^ 2) +
           mean((_ddz(ŷ, dz) .- _ddz(y, dz)) .^ 2)
end

"""
    boundary_mse_loss(ŷ; width=1)

MSE der Randwerte gegenüber 0 (Dirichlet-BC).
Entspricht dem Rand-Term: β·‖ψ̂|∂Ω‖₂².
"""
function boundary_mse_loss(ŷ; width::Int = 1)
    top    = ŷ[1:width, :, :, :]
    bottom = ŷ[end-width+1:end, :, :, :]
    left   = ŷ[width+1:end-width, 1:width, :, :]
    right  = ŷ[width+1:end-width, end-width+1:end, :, :]
    return (mean(top .^ 2) + mean(bottom .^ 2) +
            mean(left .^ 2) + mean(right .^ 2)) / 4f0
end

"""
    total_loss(ŷ, y; dx, dz, α_grad, α_bnd, mask_width)

L = MSE(ψ̂, ψ) + α_grad·GradMSE + α_bnd·BndMSE
MSE wird mit Interior-Maske berechnet. Gibt (loss, parts) zurück.
"""
function total_loss(ŷ, y;
                    dx::Real = 1f0, dz::Real = 1f0,
                    α_grad::Real = 0f0, α_bnd::Real = 0f0,
                    mask_width::Int = 2)
    nx, nz = size(ŷ, 1), size(ŷ, 2)
    mask = _to_device_of(_interior_mask_4d(nx, nz, mask_width), ŷ)

    l_mse  = sum((ŷ .- y) .^ 2 .* mask) / (sum(mask) + 1f-12)
    l_grad = α_grad > 0 ? grad_mse_loss(ŷ, y, dx, dz) : 0f0
    l_bnd  = α_bnd  > 0 ? boundary_mse_loss(ŷ) : 0f0

    loss = l_mse + Float32(α_grad) * l_grad + Float32(α_bnd) * l_bnd
    return loss, (; mse=l_mse, grad=l_grad, bnd=l_bnd)
end

"""
    grad_alpha(epoch; warmup_epochs=5, alpha_target=0.1)

Linearer Warmup für den Gradienten-Loss-Koeffizienten (analog FNO).
- Epochen 1…warmup_epochs: α = 0
- Danach: linear bis alpha_target über weitere warmup_epochs Epochen
"""
function grad_alpha(epoch::Int; warmup_epochs::Int = 5,
                    alpha_target::Real = 0.1f0)
    epoch ≤ warmup_epochs && return 0f0
    ramp = min(1f0, Float32(epoch - warmup_epochs) / Float32(warmup_epochs))
    return Float32(alpha_target) * ramp
end

# =============================================================================
# Validation
# =============================================================================

"""
    validate!(model, val_ds, move; batch_size=8, dx=1f0, dz=1f0)

Evaluiert das Modell auf dem Validierungsdatensatz ohne Gradienten.
Gibt `(val_mse, val_rel_l2, val_grad_mse)` zurück.
"""
function validate!(model, val_ds::PsiDataset, move; batch_size::Int = 8,
                   dx::Real = 1f0, dz::Real = 1f0)
    mses      = Float64[]
    rel_l2s   = Float64[]
    grad_mses = Float64[]

    Flux.testmode!(model)
    try
        for (Xb, Yb) in batch_iterator(val_ds; batch_size=batch_size, shuffle=false)
            Xb = move(Xb)
            Yb = move(Yb)
            ŷ  = model(Xb)

            push!(mses, Float64(mean((ŷ .- Yb).^2)))

            num   = sqrt(sum((ŷ .- Yb).^2))
            denom = sqrt(sum(Yb.^2)) + eps(Float32)
            push!(rel_l2s, Float64(num / denom))

            push!(grad_mses, Float64(grad_mse_loss(ŷ, Yb, dx, dz)))
        end
    finally
        Flux.trainmode!(model)
    end

    val_mse      = isempty(mses)      ? NaN : mean(mses)
    val_rel_l2   = isempty(rel_l2s)   ? NaN : mean(rel_l2s)
    val_grad_mse = isempty(grad_mses) ? NaN : mean(grad_mses)
    return (val_mse = val_mse, val_rel_l2 = val_rel_l2, val_grad_mse = val_grad_mse)
end

# =============================================================================
# CSV History Logger
# =============================================================================

"""
    append_history!(csv_path, row::NamedTuple)

Hängt eine Zeile an die Training-History CSV an.
Erstellt Header wenn Datei nicht existiert.
"""
function append_history!(csv_path::AbstractString, row::NamedTuple)
    header_needed = !isfile(csv_path)
    open(csv_path, "a") do io
        if header_needed
            println(io, join(string.(keys(row)), ","))
        end
        vals = [v isa AbstractFloat ? @sprintf("%.8e", v) : string(v)
                for v in values(row)]
        println(io, join(vals, ","))
    end
end

# =============================================================================
# Training
# =============================================================================

"""
    train_unet(; data_dir, val_dir=nothing, epochs, batch_size, lr, rng,
                 save_path, use_gpu, use_coords, coord_range,
                 history_csv="training_history.csv",
                 α_grad_target=0.1, warmup_epochs=5, α_bnd=0.01,
                 dx=nothing, dz=nothing, mask_width=2)

Trainiert ein U-Net auf den in `data_dir` liegenden .jld2-Samples.
Falls `val_dir` angegeben, wird nach jeder Epoche validiert und das
beste Modell separat gespeichert.

Loss: L = MSE + α_grad·GradMSE + α_bnd·BndMSE (analog FNO)
  - α_grad steigt per linearem Warmup von 0 auf α_grad_target
  - α_bnd ist konstant

- `use_gpu = nothing`  → auto (nutze GPU, wenn verfügbar)
- `use_gpu = true`     → erzwinge GPU (fällt bei Bedarf auf CPU zurück)
- `use_gpu = false`    → erzwinge CPU
- `dx/dz = nothing`   → wird aus erster Sample-Form abgeleitet (Domain [-1,1])
"""
function train_unet(; data_dir::String="data_psi",
                     val_dir::Union{String, Nothing}=nothing,
                     epochs::Int=10,
                     batch_size::Int=8,
                     lr::Float64=5e-5,
                     rng::AbstractRNG = Random.default_rng(),
                     save_path::String="unet_psi.bson",
                     use_gpu::Union{Nothing,Bool}=nothing,
                     use_coords::Bool=true,
                     coord_range::Symbol=:pm1,
                     history_csv::String="training_history.csv",
                     α_grad_target::Float64=0.1,
                     warmup_epochs::Int=5,
                     α_bnd::Float64=0.01,
                     dx::Union{Float64,Nothing}=nothing,
                     dz::Union{Float64,Nothing}=nothing,
                     mask_width::Int=2)

    move, devsym = _select_mover(use_gpu)
    @info "Trainiere auf Gerät: $(devsym == :gpu ? "GPU (CUDA)" : "CPU")"

    if devsym == :gpu && _HAS_CUDA
        try
            CUDA.allowscalar(false)
        catch
        end
    end

    # Trainingsdaten laden
    ds = load_dataset(data_dir; use_coords=use_coords, coord_range=coord_range)
    @info "Trainingsdaten: $(length(ds.files)) Samples aus $data_dir"

    # Validierungsdaten laden (optional)
    val_ds = if val_dir !== nothing
        vds = load_dataset(val_dir; use_coords=use_coords, coord_range=coord_range)
        @info "Validierungsdaten: $(length(vds.files)) Samples aus $val_dir"
        vds
    else
        @info "Kein val_dir angegeben – keine Validierung während des Trainings."
        nothing
    end

    # Eingabekanäle und Gittergröße aus erstem Sample bestimmen
    X1, _, _ = get_sample(ds, 1)
    in_channels = size(X1, 3)
    nx_sample, nz_sample = size(X1, 1), size(X1, 2)
    @info "Eingabekanäle erkannt: $in_channels"

    # dx/dz aus Gittergröße ableiten (Domain [-1,1]) falls nicht angegeben
    _dx = dx !== nothing ? Float32(dx) : Float32(2.0 / (nx_sample - 1))
    _dz = dz !== nothing ? Float32(dz) : Float32(2.0 / (nz_sample - 1))
    @info "Gitter: $(nx_sample)×$(nz_sample), dx=$(_dx), dz=$(_dz)"
    @info "Loss: α_grad_target=$α_grad_target (warmup=$warmup_epochs Epochen), α_bnd=$α_bnd"

    # Modell bauen
    model = build_unet(in_channels, 1)
    model = fmap(move, model)

    # Adam-Optimizer
    opt = Adam(Float32(lr))
    opt_state = Flux.setup(opt, model)

    # Best-Model Tracking
    best_val_loss = Inf
    best_path = let base = splitext(save_path)
        base[1] * "_best" * base[2]
    end

    @info "Starte Training: epochs=$epochs, batch_size=$batch_size, lr=$lr"
    val_ds !== nothing && @info "Best-Modell wird gespeichert unter: $best_path"

    for epoch in 1:epochs

        # Grad-Loss Warmup (analog FNO)
        α_grad = grad_alpha(epoch; warmup_epochs=warmup_epochs,
                            alpha_target=Float32(α_grad_target))

        # ── Training ──
        train_losses = Float64[]
        train_grad_norms = Float64[]
        for (x, y) in batch_iterator(ds; batch_size=batch_size, shuffle=true, rng=rng)
            x = move(x)
            y = move(y)

            loss, grads = Flux.withgradient(model) do m
                ŷ = m(x)
                l, _ = total_loss(ŷ, y; dx=_dx, dz=_dz,
                                  α_grad=α_grad, α_bnd=Float32(α_bnd),
                                  mask_width=mask_width)
                l
            end

            # Gradient Norm
            gn = 0f0
            fmap(grads[1]) do g
                if g isa AbstractArray; gn += sum(abs2, g); end
                g
            end
            gn = sqrt(gn)

            Flux.update!(opt_state, model, grads[1])
            push!(train_losses, float(loss))
            push!(train_grad_norms, float(gn))
        end

        train_mse  = isempty(train_losses) ? NaN : mean(train_losses)
        grad_norm  = isempty(train_grad_norms) ? NaN : mean(train_grad_norms)

        # ── Validation ──
        val_mse, val_rel_l2, val_grad_mse = if val_ds !== nothing
            vm = validate!(model, val_ds, move; batch_size=batch_size,
                           dx=_dx, dz=_dz)
            vm.val_mse, vm.val_rel_l2, vm.val_grad_mse
        else
            NaN, NaN, NaN
        end

        # ── Logging ──
        val_str = val_ds !== nothing ?
            @sprintf(" | val_mse=%.4e  val_rel_l2=%.4f", val_mse, val_rel_l2) : ""
        @info @sprintf("Epoch %3d/%d | train_mse=%.4e  grad_norm=%.2f  α=%.3f%s",
                       epoch, epochs, train_mse, grad_norm, α_grad, val_str)

        # ── History CSV ──
        row = (epoch=epoch, train_mse=train_mse, grad_norm=grad_norm, alpha=α_grad,
               val_mse=val_mse, val_rel_l2=val_rel_l2, val_grad_mse=val_grad_mse)
        append_history!(history_csv, row)

        # ── Best-Model speichern ──
        current_loss = val_ds !== nothing ? val_mse : train_mse
        if !isnan(current_loss) && current_loss < best_val_loss
            best_val_loss = current_loss
            model_cpu = fmap(cpu, model)
            BSON.@save best_path model=model_cpu
            @info "  ↳ Neues bestes Modell ($(val_ds !== nothing ? "val" : "train")_mse=$(round(current_loss; sigdigits=4))) → $best_path"
        end

        # NaN-Guard
        if isnan(train_mse)
            @error "NaN in Training Loss – Abbruch nach Epoch $epoch"
            break
        end
    end

    # Letztes Modell speichern
    model_cpu = fmap(cpu, model)
    BSON.@save save_path model=model_cpu
    @info "Training abgeschlossen. Letztes Modell → $save_path"

    return model
end

end # module
