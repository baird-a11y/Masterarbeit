module TrainingResidual

using Flux
using Statistics
using Random
using BSON
using Functors: fmap

using ..DatasetResidual: load_dataset, make_batches
using ..UNetPsi: build_unet   # gleiche Architektur wie Ansatz 2

# --- Optional CUDA support (wie in TrainingPsi) ---

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

mse_loss(y_pred, y_true) = mean((y_pred .- y_true).^2)

"""
    train_unet_residual(; ...)

Trainiert ein U-Net auf Residual-Daten (Ansatz 3).
Standard-Checkpoint: "unet_residual.bson".
"""
function train_unet_residual(; data_dir::String="data_residual",
                             epochs::Int=10,
                             batch_size::Int=4,
                             lr::Float64=1e-4,
                             rng = Random.default_rng(),
                             save_path::String="unet_residual.bson",
                             use_gpu::Union{Nothing,Bool}=nothing)

    move, devsym = _select_mover(use_gpu)
    @info "Trainiere (Residual) auf Gerät: $(devsym == :gpu ? "GPU (CUDA)" : "CPU")"

    if devsym == :gpu && _HAS_CUDA
        try
            CUDA.allowscalar(false)
        catch
        end
    end

    ds = load_dataset(data_dir)

    model = build_unet(1, 1)    # 1 Input (Maske), 1 Output (Residual_norm)
    model = fmap(move, model)

    ps = Flux.params(model)

    @info "Starte Residual-Training: epochs=$epochs, batch_size=$batch_size, lr=$lr"

    for epoch in 1:epochs
        batches = make_batches(ds, batch_size; rng=rng)
        losses = Float64[]

        for (x, y) in batches
            x = move(x)
            y = move(y)

            loss, back = Flux.withgradient(ps) do
                y_pred = model(x)
                mse_loss(y_pred, y)
            end

            for p in ps
                g = back[p]
                if g === nothing
                    continue
                end
                @. p -= lr * g
            end

            push!(losses, float(loss))
        end

        avg_loss = mean(losses)
        @info "Epoch $epoch / $epochs   |   loss = $avg_loss"

        model_cpu = fmap(cpu, model)
        BSON.@save save_path model=model_cpu
    end

    @info "Residual-Training abgeschlossen. Modell gespeichert nach $save_path"
    return model
end

end # module
