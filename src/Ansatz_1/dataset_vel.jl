module DatasetVel

using Random
using JLD2
using Printf

"""
    VelDataset

Speichert Pfade zu den .jld2-Samples für Ansatz 1 (v_x, v_z).
"""
struct VelDataset
    files::Vector{String}
end

"""
    load_dataset(dir; max_samples=nothing)

Liest alle .jld2-Dateien im Verzeichnis `dir` ein.
"""
function load_dataset(dir::String; max_samples::Union{Int,Nothing}=nothing)
    all_files = filter(f -> endswith(f, ".jld2"),
                       readdir(dir; join=true))
    sort!(all_files)
    if max_samples !== nothing && length(all_files) > max_samples
        all_files = all_files[1:max_samples]
    end
    @info "Vel-Dataset geladen: $(length(all_files)) Samples aus $dir"
    return VelDataset(all_files)
end

"""
    get_sample(ds, idx)

Lädt ein einzelnes Sample:
- input   :: (nx, nz, 1)    Float32 (Kristallmaske)
- v_norm  :: (nx, nz, 2)    Float32 (Vx_norm, Vz_norm)
"""
function get_sample(ds::VelDataset, idx::Int)
    file = ds.files[idx]
    input  = nothing
    v_norm = nothing
    @load file input v_norm

    x = Float32.(input)        # (nx, nz, 1)
    y = Float32.(v_norm)       # (nx, nz, 2)

    return x, y
end

"""
    make_batches(ds, batch_size; rng)

Gibt Vektor von (x_batch, y_batch) zurück:
- x_batch: (nx, nz, 1, B)
- y_batch: (nx, nz, 2, B)
"""
function make_batches(ds::VelDataset, batch_size::Int; rng=Random.default_rng())
    N = length(ds.files)
    idxs = collect(1:N)
    Random.shuffle!(rng, idxs)

    batches = Vector{Tuple{Array{Float32,4}, Array{Float32,4}}}()

    i = 1
    while i <= N
        batch_idxs = idxs[i:min(i + batch_size - 1, N)]
        x1, y1 = get_sample(ds, batch_idxs[1])
        nx, nz, nc_in  = size(x1)
        _,  _, nc_out = size(y1)
        B = length(batch_idxs)

        x_batch = Array{Float32}(undef, nx, nz, nc_in,  B)
        y_batch = Array{Float32}(undef, nx, nz, nc_out, B)

        for (b, idx) in enumerate(batch_idxs)
            x, y = get_sample(ds, idx)
            x_batch[:, :, :, b] .= x
            y_batch[:, :, :, b] .= y
        end

        push!(batches, (x_batch, y_batch))
        i += batch_size
    end

    return batches
end

end # module
