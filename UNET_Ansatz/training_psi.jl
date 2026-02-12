# Datei: training_psi.jl
# Training-Pipeline für ψ-U-Net
# Analog zum FNO-Ansatz für Konsistenz

module TrainingPsi

using CUDA, cuDNN   # GPU-Backend laden BEVOR Flux gpu() aufgerufen wird
using Flux
using Flux: gradient, update!, setup
using Optimisers: OptimiserChain, ClipNorm, Adam
using Functors: fmap
using Statistics, Random, Printf
using JLD2

using ..DatasetPsi: PsiDataset, batch_iterator
using ..Losses: total_loss, compute_metrics

export train!,
       validate!,
       get_device,
       save_checkpoint,
       load_checkpoint,
       grad_alpha

# =============================================================================
# T1 – Device Handling
# =============================================================================

"""
    get_device(use_gpu::Bool)

Gibt `gpu` oder `cpu` als Device-Funktion zurück.
"""
function get_device(use_gpu::Bool)
    if use_gpu
        try
            test = gpu(ones(Float32, 2))
            if !(test isa Array)   # Liegt auf GPU → nicht mehr ein normales Array
                @info "GPU aktiv"
                return gpu
            end
        catch
        end
        @warn "GPU angefordert aber nicht verfügbar – Fallback auf CPU"
    end
    return cpu
end

# =============================================================================
# T2 – Grad-Loss Warmup
# =============================================================================

"""
    grad_alpha(epoch; warmup_epochs=5, alpha_target=0.1)

Linearer Warmup für den Gradienten-Loss-Koeffizienten.
- Epochen 1…warmup_epochs: α = 0
- Danach: linear bis alpha_target über warmup_epochs Epochen
"""
function grad_alpha(epoch::Int; warmup_epochs::Int = 5,
                    alpha_target::Real = 0.1f0)
    epoch ≤ warmup_epochs && return 0f0
    ramp = min(1f0, Float32(epoch - warmup_epochs) / Float32(warmup_epochs))
    return Float32(alpha_target) * ramp
end

# =============================================================================
# T3 – Training Step
# =============================================================================

"""
    train_step!(model, opt_state, Xb, Yb; loss_kwargs...)

Ein Trainingsschritt: Forward → Loss → Backward → Update.
Gibt `(loss_val, grad_norm)` zurück.
"""
dev(x, use_gpu) = use_gpu ? gpu(x) : x

function train_step!(model, opt_state, Xb, Yb; use_gpu::Bool, loss_kwargs...)
    Xb = dev(Xb, use_gpu)
    Yb = dev(Yb, use_gpu)

    loss_val, grads = Flux.withgradient(model) do m
        ŷ = m(Xb)
        l, _ = total_loss(ŷ, Yb; loss_kwargs...)
        l
    end

    gn = _grad_norm(grads[1])
    update!(opt_state, model, grads[1])
    return Float32(loss_val), Float32(gn)
end

"""Berechnet die globale L2-Norm aller Gradienten."""
function _grad_norm(gs)
    s = 0f0
    fmap(gs) do g
        if g isa AbstractArray
            s += sum(abs2, g)
        end
        g
    end
    return sqrt(s)
end

# =============================================================================
# T4 – Training Epoch
# =============================================================================

"""
    train_epoch!(model, opt_state, train_ds; batch_size=32, shuffle=true,
                 loss_kwargs...)

Trainiert eine Epoche. Gibt NamedTuple mit gemittelten Metriken zurück.
"""
function train_epoch!(model, opt_state, train_ds::PsiDataset;
                      batch_size::Int = 32, shuffle::Bool = true,
                      use_gpu::Bool = true, loss_kwargs...)
    losses = Float32[]
    grad_norms = Float32[]

    for (Xb, Yb) in batch_iterator(train_ds; batch_size=batch_size,
                                    shuffle=shuffle)
        l, gn = train_step!(model, opt_state, Xb, Yb; use_gpu=use_gpu, loss_kwargs...)

        if isnan(l)
            @warn "NaN loss! Batch übersprungen."
            continue
        end

        push!(losses, l)
        push!(grad_norms, gn)
    end

    return (
        loss_mean     = isempty(losses) ? NaN32 : mean(losses),
        loss_std      = isempty(losses) ? NaN32 : std(losses),
        grad_norm     = isempty(grad_norms) ? NaN32 : mean(grad_norms),
        n_batches     = length(losses),
    )
end

# =============================================================================
# T5 – Validation
# =============================================================================

"""
    validate!(model, val_ds; batch_size=32, loss_kwargs...)

Validation ohne Gradienten. Gibt gemittelte Metriken zurück.
"""
function validate!(model, val_ds::PsiDataset;
                   batch_size::Int = 32, use_gpu::Bool = true, loss_kwargs...)
    metrics_accum = Dict{Symbol, Vector{Float64}}()

    Flux.testmode!(model)
    try
        for (Xb, Yb) in batch_iterator(val_ds; batch_size=batch_size,
                                        shuffle=false, drop_last=false)
            Xb = dev(Xb, use_gpu)
            Yb = dev(Yb, use_gpu)
            ŷ = model(Xb)
            m = compute_metrics(ŷ, Yb; loss_kwargs...)

            for (k, v) in pairs(m)
                if v isa Number
                    push!(get!(metrics_accum, k, Float64[]), Float64(v))
                end
            end
        end
    finally
        Flux.trainmode!(model)
    end

    # Mittelwerte
    result = Dict{Symbol, Float64}()
    for (k, vals) in metrics_accum
        result[k] = isempty(vals) ? NaN : mean(vals)
    end
    return NamedTuple{Tuple(keys(result))}(values(result))
end

# =============================================================================
# T6 – CSV History Logger
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
# T7 – Checkpointing
# =============================================================================

"""
    save_checkpoint(path, model, opt_state, epoch, metrics; model_config=nothing)

Speichert Modell + Optimizer-State + Metriken als JLD2.
"""
function save_checkpoint(path::AbstractString, model, opt_state,
                         epoch::Int, metrics;
                         model_config::Union{NamedTuple, Nothing} = nothing)
    model_cpu = cpu(model)
    jldsave(path; model_state=Flux.state(model_cpu),
                  epoch=epoch, metrics=metrics,
                  model_config=model_config)
    @info "Checkpoint gespeichert: $path (Epoch $epoch)"
end

"""
    load_checkpoint(path, model)

Lädt Modellgewichte aus einem Checkpoint in ein bestehendes Modell.
Gibt `(model, epoch, metrics, model_config)` zurück.
"""
function load_checkpoint(path::AbstractString, model)
    data = JLD2.load(path)
    Flux.loadmodel!(model, data["model_state"])
    epoch = get(data, "epoch", 0)
    metrics = get(data, "metrics", nothing)
    model_config = get(data, "model_config", nothing)
    @info "Checkpoint geladen: $path (Epoch $epoch)"
    return model, epoch, metrics, model_config
end

# =============================================================================
# T8 – Training Orchestrator
# =============================================================================

"""
    train!(; model, train_ds, val_ds=nothing, epochs=100,
             batch_size=32, lr=1e-3, max_grad_norm=1.0,
             α_grad_target=0.0, warmup_epochs=5, α_bnd=0.0,
             dx=1f0, dz=1f0, mask_width=2,
             save_dir="checkpoints", history_csv="training_history.csv",
             use_gpu=false, model_config=nothing)

Kompletter Training-Loop mit Validation, Logging und Checkpointing.

Returns: `(model, best_metrics)`
"""
function train!(; model,
                  train_ds::PsiDataset,
                  val_ds::Union{PsiDataset, Nothing} = nothing,
                  epochs::Int = 100,
                  batch_size::Int = 32,
                  lr::Real = 1e-3,
                  max_grad_norm::Real = 1.0,
                  α_grad_target::Real = 0f0,
                  warmup_epochs::Int = 5,
                  α_bnd::Real = 0f0,
                  dx::Real = 1f0,
                  dz::Real = 1f0,
                  mask_width::Int = 2,
                  save_dir::AbstractString = "checkpoints",
                  history_csv::AbstractString = "training_history.csv",
                  use_gpu::Bool = false,
                  model_config::Union{NamedTuple, Nothing} = nothing)

    # Device
    dev_fn = get_device(use_gpu)
    model = dev_fn(model)

    # Optimizer mit Gradient Clipping
    opt = OptimiserChain(ClipNorm(Float32(max_grad_norm)), Adam(Float32(lr)))
    opt_state = setup(opt, model)

    # Best-Model Tracking
    mkpath(save_dir)
    best_val_loss = Inf
    best_metrics = nothing

    @info "Training Start: $(length(train_ds)) Samples, $epochs Epochen, batch=$batch_size"

    for epoch in 1:epochs
        # Grad-Loss Warmup
        α_grad = grad_alpha(epoch; warmup_epochs=warmup_epochs,
                            alpha_target=α_grad_target)

        # Loss-Config für diese Epoche
        loss_kw = (dx=Float32(dx), dz=Float32(dz),
                   α_grad=α_grad, α_bnd=Float32(α_bnd),
                   mask_width=mask_width)

        # ── Train ──
        train_metrics = train_epoch!(model, opt_state, train_ds;
                                     batch_size=batch_size,
                                     use_gpu=use_gpu,
                                     loss_kw...)

        # ── Validate ──
        val_metrics = if val_ds !== nothing
            validate!(model, val_ds; batch_size=batch_size, use_gpu=use_gpu,
                      dx=Float32(dx), dz=Float32(dz), mask_width=mask_width)
        else
            (mse=NaN,)
        end

        # ── History Row ──
        row = (
            epoch     = epoch,
            train_mse = train_metrics.loss_mean,
            train_std = train_metrics.loss_std,
            grad_norm = train_metrics.grad_norm,
            alpha     = α_grad,
            val_mse   = get(val_metrics, :mse, NaN),
            val_rel_l2 = get(val_metrics, :rel_l2, NaN),
            val_max_err = get(val_metrics, :max_err, NaN),
            val_grad_mse  = get(val_metrics, :grad_mse, NaN),
        )
        append_history!(history_csv, row)

        # ── Logging ──
        val_str = val_ds !== nothing ? @sprintf(" | val_mse=%.4e rel_l2=%.4f", row.val_mse, row.val_rel_l2) : ""
        @info @sprintf("Epoch %3d/%d | train_mse=%.4e grad_norm=%.2f α=%.3f%s",
                       epoch, epochs, row.train_mse, row.grad_norm, α_grad, val_str)

        # ── Checkpoint Best ──
        current_val = val_ds !== nothing ? row.val_mse : row.train_mse
        if !isnan(current_val) && current_val < best_val_loss
            best_val_loss = current_val
            best_metrics = val_metrics
            save_checkpoint(joinpath(save_dir, "best_model.jld2"),
                           model, opt_state, epoch, val_metrics;
                           model_config=model_config)
        end

        # ── NaN Guard ──
        if isnan(train_metrics.loss_mean)
            @error "NaN in Training Loss – Abbruch nach Epoch $epoch"
            break
        end
    end

    # Letzten Checkpoint speichern
    save_checkpoint(joinpath(save_dir, "last_model.jld2"),
                   model, opt_state, epochs, best_metrics;
                   model_config=model_config)

    return model, best_metrics
end

end # module
