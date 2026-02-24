module TrainingPsi

using Flux
using Statistics
using Random
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

"""
    train_unet(; data_dir, epochs, batch_size, lr, rng, save_path, use_gpu,
                 use_coords, coord_range)

Trainiert ein U-Net auf den in `data_dir` liegenden .jld2-Samples.
Verwendet dieselbe Datenpipeline wie das FNO (PsiDataset, batch_iterator).
Die Anzahl der Eingabekanäle wird automatisch aus dem ersten Sample bestimmt.

- `use_gpu = nothing`  → auto (nutze GPU, wenn verfügbar)
- `use_gpu = true`     → erzwinge GPU (fällt bei Bedarf auf CPU zurück)
- `use_gpu = false`    → erzwinge CPU

Speichert nach jeder Epoche ein CPU-Checkpoint unter `save_path`.
"""
function train_unet(; data_dir::String="data_psi",
                     epochs::Int=10,
                     batch_size::Int=8,
                     lr::Float64=5e-5,
                     rng::AbstractRNG = Random.default_rng(),
                     save_path::String="unet_psi.bson",
                     use_gpu::Union{Nothing,Bool}=nothing,
                     use_coords::Bool=true,
                     coord_range::Symbol=:pm1)

    move, devsym = _select_mover(use_gpu)
    @info "Trainiere auf Gerät: $(devsym == :gpu ? "GPU (CUDA)" : "CPU")"

    if devsym == :gpu && _HAS_CUDA
        try
            CUDA.allowscalar(false)
        catch
        end
    end

    # Dataset laden (identisches Interface zum FNO)
    ds = load_dataset(data_dir; use_coords=use_coords, coord_range=coord_range)

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

    @info "Starte Training: epochs=$epochs, batch_size=$batch_size, lr=$lr"

    for epoch in 1:epochs
        losses = Float64[]

        for (x, y) in batch_iterator(ds; batch_size=batch_size, shuffle=true, rng=rng)
            x = move(x)
            y = move(y)

            loss, grads = Flux.withgradient(model) do m
                mse_loss(m(x), y)
            end

            Flux.update!(opt_state, model, grads[1])
            push!(losses, float(loss))
        end

        avg_loss = mean(losses)
        @info "Epoch $epoch / $epochs   |   loss = $avg_loss"

        # Checkpoint: immer CPU-Version speichern
        model_cpu = fmap(cpu, model)
        BSON.@save save_path model=model_cpu
    end

    @info "Training abgeschlossen. Modell gespeichert unter $save_path"
    return model
end

end # module
