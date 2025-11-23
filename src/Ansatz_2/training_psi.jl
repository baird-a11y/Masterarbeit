module TrainingPsi

using Flux
using Statistics
using Random
using BSON
# CUDA optional und versionssicher einbinden (kein Hard-Import oben!)

using ..DatasetPsi: load_dataset, make_batches
using ..UNetPsi: build_unet
using Functors: fmap

# --- Optional CUDA support (kompatibel mit alten/neuen CUDA.jl) ---
const _HAS_CUDA = try
    @eval import CUDA
    true
catch
    false
end

# Prüfe sicher, ob CUDA initialisierbar ist (Import kann klappen, Init aber scheitern)
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

# Universal-"to GPU": nutzt CUDA.cuda ODER CUDA.cu – je nach Version.
# Fällt bei Problemen still auf CPU zurück.
function _to_gpu(x)
    if !_HAS_CUDA
        return x
    end
    try
        if isdefined(CUDA, :cuda)
            return CUDA.cuda(x)          # neuere CUDA.jl
        elseif isdefined(CUDA, :cu)
            return CUDA.cu(x)            # ältere CUDA.jl
        else
            return x
        end
    catch
        return x
    end
end

_to_cpu(x) = cpu(x)

# Gerätemover auswählen
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
        # auto
        if has_gpu()
            return _to_gpu, :gpu
        else
            return _to_cpu, :cpu
        end
    end
end

# Huber-Loss (glatt zwischen L2 und L1)
function huber_loss(y_pred, y_true; δ::Float32 = 1.0f0)
    r = y_pred .- y_true
    abs_r = abs.(r)

    quad = 0.5f0 .* r.^2
    linear = δ .* (abs_r .- 0.5f0*δ)

    use_quad = abs_r .<= δ
    return mean(ifelse.(use_quad, quad, linear))
end

function loss(x, y)
    y_pred = model(x)
    return huber_loss(y_pred, y; δ=1.0f0)
end

"""
    weighted_mse_loss(y_pred, y_true, x_mask;
                      w_crystal=5.0, w_matrix=1.0)

Gewichteter MSE:
- höhere Gewichte im Kristall (x_mask ≈ 1),
- geringere Gewichte in der Matrix (x_mask ≈ 0).
x_mask muss dieselbe Shape wie y_true haben: (nx, nz, 1, B).
"""
function weighted_mse_loss(y_pred, y_true, x_mask;
                           w_crystal::Float32 = 5.0f0,
                           w_matrix::Float32  = 1.0f0)

    @assert size(x_mask) == size(y_true) "x_mask und y_true müssen gleiche Shape haben"

    mask = x_mask .> 0.5f0

    w = similar(x_mask)
    w[mask]   .= w_crystal
    w[.!mask] .= w_matrix

    diff2    = (y_pred .- y_true).^2
    weighted = w .* diff2

    return sum(weighted) / sum(w)
end


"""
    weighted_mse_loss(y_pred, y_true, x;
                      w_crystal=5.0, w_matrix=1.0)

Gewichteter MSE:
- höhere Gewichte im Kristall (x ≈ 1),
- geringere Gewichte in der Matrix (x ≈ 0).
"""
function weighted_mse_loss(y_pred, y_true, x;
                           w_crystal::Float32 = 5.0f0,
                           w_matrix::Float32  = 1.0f0)

    @assert size(x) == size(y_true) "x und y_true müssen gleiche Shape haben"

    # Maske: Kristall = true, Matrix = false
    mask = x .> 0.5f0

    # Gewichte pro Pixel
    w = similar(x)
    w[mask]    .= w_crystal
    w[.!mask]  .= w_matrix

    diff2 = (y_pred .- y_true).^2
    weighted = w .* diff2

    # normiert: Mittelwert mit Gewichtung
    return sum(weighted) / sum(w)
end

"""
    train_unet(; data_dir, epochs, batch_size, lr, rng, save_path, use_gpu)

Trainiert ein U-Net auf den in `data_dir` liegenden .jld2-Samples.
Unterstützt optional GPU-Training (CUDA) mit automatischer Fallback-Logik.
- `use_gpu = nothing`  → auto (nutze GPU, wenn verfügbar)
- `use_gpu = true`     → erzwinge GPU (fällt bei Bedarf auf CPU zurück)
- `use_gpu = false`    → erzwinge CPU

Speichert nach jeder Epoche ein **CPU-Checkpoint** unter `save_path`.
"""
function train_unet(; data_dir::String="data_psi",
                     epochs::Int=10,
                     batch_size::Int=4,
                     lr::Float64=1e-4,
                     rng = Random.default_rng(),
                     save_path::String="unet_psi.bson",
                     use_gpu::Union{Nothing,Bool}=nothing)

    # Gerät auswählen
    move, devsym = _select_mover(use_gpu)
    @info "Trainiere auf Gerät: $(devsym == :gpu ? "GPU (CUDA)" : "CPU")"

    # (optional) Scalar-Indexing global verbieten, wenn GPU aktiv
    if devsym == :gpu && _HAS_CUDA
        try
            CUDA.allowscalar(false)
        catch
        end
    end

    # Dataset laden
    ds = load_dataset(data_dir)

    # Modell bauen und auf Zielgerät verschieben (rekursiv!)
    model = build_unet(1, 1)                 # 1 Input-Kanal (Maske), 1 Output-Kanal (ψ_norm)
    model = fmap(move, model)                 # <— wichtig: alle Gewichte/Buffer verschieben

    # Parameter *nach* dem Verschieben ziehen
    ps = Flux.params(model)

    @info "Starte Training: epochs=$epochs, batch_size=$batch_size, lr=$lr"

    for epoch in 1:epochs
        batches = make_batches(ds, batch_size; rng=rng)
        losses = Float64[]

        for (x, y) in batches
            # Daten auf Zielgerät
            x = move(x)
            y = move(y)
            
            # x hat Shape (nx, nz, C, B), wir nehmen Kanal 1 als Kristallmaske
            x_mask = x[:, :, 1:1, :]  # Shape (nx, nz, 1, B)

            loss, back = Flux.withgradient(ps) do
                y_pred = model(x)
                # gewichtete Loss: Kristallbereiche stärker gewichten
                weighted_mse_loss(y_pred, y, x_mask)
            end

            # Manuelles SGD-Update
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

        # Checkpoint: immer CPU-Version speichern für Portabilität
        model_cpu = fmap(cpu, model)
        BSON.@save save_path model=model_cpu
    end

    @info "Training abgeschlossen. Modell gespeichert nach $save_path"
    return model
end

end # module
module TrainingPsi

using Flux
using Statistics
using Random
using BSON
# CUDA optional und versionssicher einbinden (kein Hard-Import oben!)

using ..DatasetPsi: load_dataset, make_batches
using ..UNetPsi: build_unet
using Functors: fmap

# --- Optional CUDA support (kompatibel mit alten/neuen CUDA.jl) ---
const _HAS_CUDA = try
    @eval import CUDA
    true
catch
    false
end

# Prüfe sicher, ob CUDA initialisierbar ist (Import kann klappen, Init aber scheitern)
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

# Universal-"to GPU": nutzt CUDA.cuda ODER CUDA.cu – je nach Version.
# Fällt bei Problemen still auf CPU zurück.
function _to_gpu(x)
    if !_HAS_CUDA
        return x
    end
    try
        if isdefined(CUDA, :cuda)
            return CUDA.cuda(x)          # neuere CUDA.jl
        elseif isdefined(CUDA, :cu)
            return CUDA.cu(x)            # ältere CUDA.jl
        else
            return x
        end
    catch
        return x
    end
end

_to_cpu(x) = cpu(x)

# Gerätemover auswählen
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
        # auto
        if has_gpu()
            return _to_gpu, :gpu
        else
            return _to_cpu, :cpu
        end
    end
end

"""
    mse_loss(y_pred, y_true)

Einfacher Mean-Squared-Error über alle Dimensionen.
"""
mse_loss(y_pred, y_true) = mean((y_pred .- y_true).^2)

"""
    train_unet(; data_dir, epochs, batch_size, lr, rng, save_path, use_gpu)

Trainiert ein U-Net auf den in `data_dir` liegenden .jld2-Samples.
Unterstützt optional GPU-Training (CUDA) mit automatischer Fallback-Logik.
- `use_gpu = nothing`  → auto (nutze GPU, wenn verfügbar)
- `use_gpu = true`     → erzwinge GPU (fällt bei Bedarf auf CPU zurück)
- `use_gpu = false`    → erzwinge CPU

Speichert nach jeder Epoche ein **CPU-Checkpoint** unter `save_path`.
"""
function train_unet(; data_dir::String="data_psi",
                     epochs::Int=10,
                     batch_size::Int=4,
                     lr::Float64=1e-4,
                     rng = Random.default_rng(),
                     save_path::String="unet_psi.bson",
                     use_gpu::Union{Nothing,Bool}=nothing)

    # Gerät auswählen
    move, devsym = _select_mover(use_gpu)
    @info "Trainiere auf Gerät: $(devsym == :gpu ? "GPU (CUDA)" : "CPU")"

    # (optional) Scalar-Indexing global verbieten, wenn GPU aktiv
    if devsym == :gpu && _HAS_CUDA
        try
            CUDA.allowscalar(false)
        catch
        end
    end

    # Dataset laden
    ds = load_dataset(data_dir)

    # Modell bauen und auf Zielgerät verschieben (rekursiv!)
    model = build_unet(2, 1)  # 2 Input-Kanäle (Maske + Distanz), 1 Output-Kanal (ψ_norm)
    model = fmap(move, model)                 # <— wichtig: alle Gewichte/Buffer verschieben

    # Parameter *nach* dem Verschieben ziehen
    ps = Flux.params(model)

    @info "Starte Training: epochs=$epochs, batch_size=$batch_size, lr=$lr"

    for epoch in 1:epochs
        batches = make_batches(ds, batch_size; rng=rng)
        losses = Float64[]

        for (x, y) in batches
            # Daten auf Zielgerät
            x = move(x)
            y = move(y)

            # Gradient über die Parameter ps
            loss, back = Flux.withgradient(ps) do
                y_pred = model(x)
                mse_loss(y_pred, y)
            end

            # Manuelles SGD-Update: p .= p .- lr * grad
            for p in ps
                g = back[p]
                if g === nothing
                    continue
                end
                @. p -= lr * g
            end

            # Verlust als Float64 loggen (unabhängig vom Gerät)
            push!(losses, float(loss))
        end

        avg_loss = mean(losses)
        @info "Epoch $epoch / $epochs   |   loss = $avg_loss"

        # Checkpoint: immer CPU-Version speichern für Portabilität
        model_cpu = fmap(cpu, model)
        BSON.@save save_path model=model_cpu
    end

    @info "Training abgeschlossen. Modell gespeichert nach $save_path"
    return model
end

end # module
