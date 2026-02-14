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

Lädt ein einzelnes Sample aus einer .jld2-Datei und bereitet es für das Training auf.

Eingaben:
- Aus der Datei werden geladen:
    • input  :: (nx, nz, 2)  Float32      — Binärmaske (Kristall) + SDF-Distanzfeld
    • ψ_norm :: (nx, nz)     Float32      — normierte Stromfunktion
    • meta   :: NamedTuple                — enthält x_vec_1D und z_vec_1D

Zusätzliche Kanäle:
- Beim Laden werden zwei weitere Eingabekanäle erzeugt:
    • x-Koordinate normiert auf [-1, 1]
    • z-Koordinate normiert auf [-1, 1]

Ausgabe:
- input_aug :: (nx, nz, 4) Float32   — [Maske, SDF, x_norm, z_norm]
- target    :: (nx, nz, 1) Float32   — ψ_norm als 3D-Tensor

Diese Funktion ermöglicht es, zusätzliche Positionsinformation ohne neue Datengenerierung zu nutzen.
"""

function get_sample(ds::PsiDataset, idx::Int)
    file = ds.files[idx]

    input  = nothing
    ψ_norm = nothing
    meta   = nothing
    @load file input ψ_norm meta  # meta enthält x_vec_1D, z_vec_1D

    # Basis-Eingabe (Maske + SDF)
    x = Float32.(input)           # (nx, nz, C_in), C_in = 2
    y = Float32.(ψ_norm)          # (nx, nz)
    nx, nz = size(y)
    y = reshape(y, nx, nz, 1)     # (nx, nz, 1)

    # --- Koordinatenkanäle aufbauen (x, z) ---
    x_vec = Float32.(meta.x_vec_1D)  # Länge nx, in km
    z_vec = Float32.(meta.z_vec_1D)  # Länge nz, in km

    # zur Sicherheit auf [-1, 1] normieren (falls LaMEM-Domäne mal angepasst wird)
    x_min, x_max = minimum(x_vec), maximum(x_vec)
    z_min, z_max = minimum(z_vec), maximum(z_vec)

    x_norm_1D = 2f0 .* (x_vec .- x_min) ./ (x_max - x_min) .- 1f0  # ∈ [-1, 1]
    z_norm_1D = 2f0 .* (z_vec .- z_min) ./ (z_max - z_min) .- 1f0  # ∈ [-1, 1]

    # 2D-Gitter aus 1D-Vektoren
    x_coord = Array{Float32}(undef, nx, nz)
    z_coord = Array{Float32}(undef, nx, nz)
    @inbounds for ix in 1:nx
        for iz in 1:nz
            x_coord[ix, iz] = x_norm_1D[ix]
            z_coord[ix, iz] = z_norm_1D[iz]
        end
    end

    # --- Kanäle anhängen: [Maske, SDF, x, z] ---
    nx_in, nz_in, nc_in = size(x)
    @assert nx_in == nx && nz_in == nz "Input- und Zielgröße passen nicht zusammen"

    x_aug = Array{Float32}(undef, nx, nz, nc_in + 2)
    @inbounds begin
        x_aug[:, :, 1:nc_in] .= x
        x_aug[:, :, nc_in + 1] .= x_coord
        x_aug[:, :, nc_in + 2] .= z_coord
    end

    return x_aug, y
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
