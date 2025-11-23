module DatasetPsi

using Random
using JLD2
using Printf

"""
    PsiDataset

Speichert nur die Pfade zu den .jld2-Samples.
Die Daten werden beim Bauen der Batches von Platte geladen.
"""
struct PsiDataset
    files::Vector{String}
end

"""
    load_dataset(dir; max_samples=nothing)

Liest alle .jld2-Dateien im Verzeichnis `dir` und baut ein PsiDataset.
Optional: auf max_samples beschränken.
"""
function load_dataset(dir::String; max_samples::Union{Int,Nothing}=nothing)
    all_files = filter(f -> endswith(f, ".jld2"),
                       readdir(dir; join=true))
    sort!(all_files)
    if max_samples !== nothing && length(all_files) > max_samples
        all_files = all_files[1:max_samples]
    end
    @info "Dataset geladen: $(length(all_files)) Samples aus $dir"
    return PsiDataset(all_files)
end

"""
    get_sample(ds, idx)

Lädt ein einzelnes Sample aus der .jld2-Datei:
- input  :: (nx, nz, C)  Float32  (Maske + weitere Kanäle)
- target :: (nx, nz, 1)  Float32  (ψ_norm)
"""
function get_sample(ds::PsiDataset, idx::Int)
    file = ds.files[idx]
    input = nothing
    ψ_norm = nothing
    @load file input ψ_norm

    x = Float32.(input)           # (nx, nz, C)
    y = Float32.(ψ_norm)
    nx, nz = size(y)
    y = reshape(y, nx, nz, 1)     # (nx, nz, 1), passend zum U-Net-Output

    return x, y
end


"""
    make_batches(ds, batch_size; rng)

Erstellt eine Vector{Tuple{Array{Float32,4}, Array{Float32,4}}}
mit Batches der Form (x_batch, y_batch):
- x_batch: (nx, nz, 1, B)
- y_batch: (nx, nz, 1, B)
"""
function make_batches(ds::PsiDataset, batch_size::Int; rng=Random.default_rng())
    N = length(ds.files)
    idxs = collect(1:N)
    Random.shuffle!(rng, idxs)

    batches = Vector{Tuple{Array{Float32,4}, Array{Float32,4}}}()

    i = 1
    while i <= N
        batch_idxs = idxs[i:min(i + batch_size - 1, N)]
        # Erstes Sample zum Dimensionen-Abgreifen
        x1, y1 = get_sample(ds, batch_idxs[1])
        nx, nz, nc = size(x1)
        B = length(batch_idxs)

        x_batch = Array{Float32}(undef, nx, nz, nc, B)
        y_batch = Array{Float32}(undef, nx, nz, 1, B)

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
