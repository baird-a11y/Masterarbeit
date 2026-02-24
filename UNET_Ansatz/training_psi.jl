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
# Validation
# =============================================================================

"""
    validate!(model, val_ds, move; batch_size=8)

Evaluiert das Modell auf dem Validierungsdatensatz ohne Gradienten.
Gibt `(val_mse, val_rel_l2)` zurück.
"""
function validate!(model, val_ds::PsiDataset, move; batch_size::Int = 8)
    mses    = Float64[]
    rel_l2s = Float64[]

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
        end
    finally
        Flux.trainmode!(model)
    end

    val_mse    = isempty(mses)    ? NaN : mean(mses)
    val_rel_l2 = isempty(rel_l2s) ? NaN : mean(rel_l2s)
    return (val_mse = val_mse, val_rel_l2 = val_rel_l2)
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
                 history_csv="training_history.csv")

Trainiert ein U-Net auf den in `data_dir` liegenden .jld2-Samples.
Falls `val_dir` angegeben, wird nach jeder Epoche validiert und das
beste Modell separat gespeichert.

- `use_gpu = nothing`  → auto (nutze GPU, wenn verfügbar)
- `use_gpu = true`     → erzwinge GPU (fällt bei Bedarf auf CPU zurück)
- `use_gpu = false`    → erzwinge CPU
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
                     history_csv::String="training_history.csv")

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

    # Eingabekanäle aus erstem Sample bestimmen
    X1, _, _ = get_sample(ds, 1)
    in_channels = size(X1, 3)
    @info "Eingabekanäle erkannt: $in_channels"

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

        # ── Training ──
        train_losses = Float64[]
        for (x, y) in batch_iterator(ds; batch_size=batch_size, shuffle=true, rng=rng)
            x = move(x)
            y = move(y)

            loss, grads = Flux.withgradient(model) do m
                mse_loss(m(x), y)
            end

            Flux.update!(opt_state, model, grads[1])
            push!(train_losses, float(loss))
        end

        train_mse = isempty(train_losses) ? NaN : mean(train_losses)

        # ── Validation ──
        val_mse, val_rel_l2 = if val_ds !== nothing
            vm = validate!(model, val_ds, move; batch_size=batch_size)
            vm.val_mse, vm.val_rel_l2
        else
            NaN, NaN
        end

        # ── Logging ──
        val_str = val_ds !== nothing ?
            @sprintf(" | val_mse=%.4e  val_rel_l2=%.4f", val_mse, val_rel_l2) : ""
        @info @sprintf("Epoch %3d/%d | train_mse=%.4e%s", epoch, epochs, train_mse, val_str)

        # ── History CSV ──
        row = (epoch=epoch, train_mse=train_mse, val_mse=val_mse, val_rel_l2=val_rel_l2)
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
