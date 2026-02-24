module DatasetPsi

# Dataset-Modul für U-Net Training.
# Interface ist identisch zum FNO-Dataset (DatasetPsi in FNO_Ansatz/),
# damit beide Modelle dieselben .jld2-Dateien und dieselbe Datenpipeline nutzen.
#
# Funktion              Was sie tut
# list_jld2_files       Rekursive Suche in Unterordnern, sortiert
# PsiDataset            Struct mit files, use_coords, coord_range
# load_dataset          Ein oder mehrere Dirs → PsiDataset
# read_jld2             JLD2 laden mit Key-Varianten + Dimensionschecks
# build_input_channels  JLD2-Input + optional Xn/Zn Koordinatenkanäle → (nx,nz,Cin)
# build_target          ψ_norm → (nx,nz,1) Float32
# get_sample            Zentraler Zugriff: ds[i] → (X, Y, meta)
# batch_iterator        Iterator mit shuffle, drop_last → Batches (nx,nz,Cin,B)

using Random
using JLD2
using Statistics

export PsiDataset,
       list_jld2_files,
       load_dataset,
       read_jld2,
       build_input_channels,
       build_target,
       get_sample,
       batch_iterator

# =============================================================================
# D0 – Koordinaten-Normalisierung (inline, ohne GridFDUtils-Abhängigkeit)
# =============================================================================

function _normalize_coords(x_vec_1D::AbstractVector, z_vec_1D::AbstractVector;
                           range::Symbol = :pm1)
    xmin, xmax = extrema(x_vec_1D)
    zmin, zmax = extrema(z_vec_1D)
    @assert xmax > xmin && zmax > zmin "Degenerate coordinate ranges"

    nx = length(x_vec_1D)
    nz = length(z_vec_1D)

    if range == :pm1
        xnorm(v) = 2*(v - xmin)/(xmax - xmin) - 1
        znorm(v) = 2*(v - zmin)/(zmax - zmin) - 1
    elseif range == :unit
        xnorm(v) = (v - xmin)/(xmax - xmin)
        znorm(v) = (v - zmin)/(zmax - zmin)
    else
        error("Unknown range=$range (use :pm1 or :unit)")
    end

    X = [xnorm(x_vec_1D[ix]) for ix in 1:nx, iz in 1:nz]
    Z = [znorm(z_vec_1D[iz]) for ix in 1:nx, iz in 1:nz]
    return Float32.(X), Float32.(Z)
end

# =============================================================================
# D1 – File Listing
# =============================================================================

"""
    list_jld2_files(dir; recursive=true)

Sammelt alle `*.jld2`-Dateien in `dir`, optional rekursiv in Unterordnern.
Gibt sortierte Liste absoluter Pfade zurück.
"""
function list_jld2_files(dir::AbstractString; recursive::Bool = true)
    if !isdir(dir)
        @warn "Verzeichnis existiert nicht: $dir"
        return String[]
    end

    files = String[]
    if recursive
        for (root, _, filenames) in walkdir(dir)
            for f in filenames
                if endswith(f, ".jld2")
                    push!(files, joinpath(root, f))
                end
            end
        end
    else
        for f in readdir(dir; join=true)
            if isfile(f) && endswith(f, ".jld2")
                push!(files, f)
            end
        end
    end

    sort!(files)
    return files
end

# =============================================================================
# D2 – Dataset Container
# =============================================================================

"""
    PsiDataset

Container für ein ψ-Dataset. Identisch zum FNO-PsiDataset.

Felder:
- `files`       – sortierte Liste der JLD2-Pfade
- `use_coords`  – ob Koordinatenkanäle (Xn, Zn) angehängt werden
- `coord_range` – Normalisierungsbereich für Koordinaten (:pm1 oder :unit)
"""
struct PsiDataset
    files::Vector{String}
    use_coords::Bool
    coord_range::Symbol
end

Base.length(ds::PsiDataset) = length(ds.files)
Base.getindex(ds::PsiDataset, i::Int) = get_sample(ds, i)
Base.lastindex(ds::PsiDataset) = length(ds)

function Base.show(io::IO, ds::PsiDataset)
    print(io, "PsiDataset($(length(ds.files)) samples, use_coords=$(ds.use_coords))")
end

# =============================================================================
# D7 – Dataset Loader
# =============================================================================

"""
    load_dataset(dirs; use_coords=true, coord_range=:pm1)
    load_dataset(dir::String; kwargs...)

Erstellt ein `PsiDataset` aus einem oder mehreren Verzeichnissen.
Durchsucht rekursiv nach .jld2-Dateien.
"""
function load_dataset(dirs::AbstractVector{<:AbstractString};
                      use_coords::Bool = true,
                      coord_range::Symbol = :pm1)
    all_files = String[]
    for d in dirs
        append!(all_files, list_jld2_files(d))
    end
    sort!(all_files)

    n = length(all_files)
    if n == 0
        @warn "Keine JLD2-Dateien gefunden in: $dirs"
    else
        @info "Dataset geladen: $n Samples aus $(length(dirs)) Verzeichnis(sen)"
    end

    return PsiDataset(all_files, use_coords, coord_range)
end

function load_dataset(dir::AbstractString; kwargs...)
    return load_dataset([dir]; kwargs...)
end

# =============================================================================
# D3 – Einzelnes Sample aus JLD2 lesen
# =============================================================================

# Key-Varianten für Rückwärtskompatibilität (identisch zum FNO)
const PSI_KEYS   = ["ψ_norm", "psi_norm", "psi", "ψ"]
const INPUT_KEYS = ["input", "X", "inputs"]
const SCALE_KEYS = ["scale", "norm_scale", "psi_scale"]
const META_KEYS  = ["meta", "metadata", "norm_meta"]

"""
    read_jld2(filepath)

Liest ein Sample aus einer JLD2-Datei. Gibt ein NamedTuple zurück mit:
`(; input, ψ_norm, scale, meta, filepath)`

Handhabt verschiedene Key-Varianten für Rückwärtskompatibilität.
"""
function read_jld2(filepath::AbstractString)
    @assert isfile(filepath) "Datei nicht gefunden: $filepath"

    data = JLD2.load(filepath)

    input  = _find_key(data, INPUT_KEYS, filepath, "input")
    ψ_norm = _find_key(data, PSI_KEYS,   filepath, "ψ_norm")
    scale  = _find_key(data, SCALE_KEYS, filepath, "scale")
    meta   = _find_key(data, META_KEYS,  filepath, "meta")

    # Dimensionschecks
    if ndims(ψ_norm) == 2
        nx, nz = size(ψ_norm)
    else
        error("ψ_norm hat unerwartete Dimensionen: $(size(ψ_norm)) in $filepath")
    end

    if ndims(input) == 3
        @assert size(input, 1) == nx && size(input, 2) == nz "input-Shape $(size(input)) passt nicht zu ψ_norm ($nx, $nz) in $filepath"
    else
        error("input hat unerwartete Dimensionen: $(size(input)) in $filepath")
    end

    return (; input, ψ_norm, scale, meta, filepath)
end

function _find_key(data::Dict, candidates::Vector{String}, filepath, label)
    for k in candidates
        if haskey(data, k)
            return data[k]
        end
    end
    available = join(collect(keys(data)), ", ")
    error("Key '$label' nicht gefunden in $filepath (versucht: $candidates, vorhanden: $available)")
end

# =============================================================================
# D4 – Input Tensor X bauen
# =============================================================================

"""
    build_input_channels(sample; use_coords=true, coord_range=:pm1)

Baut den Input-Tensor X als `Array{Float32,3}` mit Shape `(nx, nz, Cin)`.

Kanäle:
1. Vorbereitete Kanäle aus JLD2 (Mask, SDF, ggf. weitere)
2. Optional: normierte Koordinaten Xn, Zn (aus meta.x_vec_1D, meta.z_vec_1D)
"""
function build_input_channels(sample::NamedTuple;
                              use_coords::Bool = true,
                              coord_range::Symbol = :pm1)
    input = Float32.(sample.input)
    nx, nz, cin = size(input)

    if use_coords && hasproperty(sample.meta, :x_vec_1D) && hasproperty(sample.meta, :z_vec_1D)
        Xn, Zn = _normalize_coords(sample.meta.x_vec_1D, sample.meta.z_vec_1D;
                                    range=coord_range)
        X = Array{Float32}(undef, nx, nz, cin + 2)
        X[:, :, 1:cin]   .= input
        X[:, :, cin + 1]  .= Xn
        X[:, :, cin + 2]  .= Zn
    else
        X = input
    end

    return X
end

# =============================================================================
# D5 – Target Y bauen
# =============================================================================

"""
    build_target(sample)

Baut den Target-Tensor Y als `Array{Float32,3}` mit Shape `(nx, nz, 1)`.
"""
function build_target(sample::NamedTuple)
    ψ = Float32.(sample.ψ_norm)
    nx, nz = size(ψ)
    Y = reshape(ψ, nx, nz, 1)

    n_nan = count(isnan, Y)
    n_nan > 0 && @warn "Target enthält $n_nan NaN-Werte! ($(sample.filepath))"

    return Y
end

# =============================================================================
# D6 – get_sample
# =============================================================================

"""
    get_sample(ds::PsiDataset, i::Int)

Lädt Sample `i` und gibt `(X, Y, meta)` zurück.
- X: `Array{Float32,3}` (nx, nz, Cin)
- Y: `Array{Float32,3}` (nx, nz, 1)
- meta: NamedTuple mit scale, meta, filepath
"""
function get_sample(ds::PsiDataset, i::Int)
    @assert 1 ≤ i ≤ length(ds) "Index $i außerhalb [1, $(length(ds))]"

    sample = read_jld2(ds.files[i])

    X = build_input_channels(sample; use_coords=ds.use_coords,
                             coord_range=ds.coord_range)
    Y = build_target(sample)

    sample_meta = (
        scale    = sample.scale,
        meta     = sample.meta,
        filepath = sample.filepath,
    )

    return X, Y, sample_meta
end

# =============================================================================
# D8 – Batch Iterator
# =============================================================================

"""
    batch_iterator(ds; batch_size=8, shuffle=true, rng=Random.GLOBAL_RNG,
                       drop_last=false)

Erzeugt einen Iterator über Batches.
Jeder Batch ist `(Xb, Yb)` mit:
- Xb: `Array{Float32,4}` (nx, nz, Cin, B)
- Yb: `Array{Float32,4}` (nx, nz, 1, B)

`drop_last=true` verwirft den letzten unvollständigen Batch.
"""
function batch_iterator(ds::PsiDataset;
                        batch_size::Int = 8,
                        shuffle::Bool = true,
                        rng::AbstractRNG = Random.GLOBAL_RNG,
                        drop_last::Bool = false)
    n = length(ds)
    @assert n > 0 "Dataset ist leer"

    indices = collect(1:n)
    shuffle && Random.shuffle!(rng, indices)

    batches = Vector{UnitRange{Int}}()
    for start in 1:batch_size:n
        stop = min(start + batch_size - 1, n)
        if drop_last && (stop - start + 1) < batch_size
            break
        end
        push!(batches, start:stop)
    end

    return BatchIterator(ds, indices, batches)
end

struct BatchIterator
    ds::PsiDataset
    indices::Vector{Int}
    batches::Vector{UnitRange{Int}}
end

Base.length(bi::BatchIterator) = length(bi.batches)

function Base.iterate(bi::BatchIterator, state=1)
    state > length(bi.batches) && return nothing

    batch_range = bi.batches[state]
    batch_indices = bi.indices[batch_range]

    X1, Y1, _ = get_sample(bi.ds, batch_indices[1])
    nx, nz, cin = size(X1)
    B = length(batch_indices)

    Xb = Array{Float32}(undef, nx, nz, cin, B)
    Yb = Array{Float32}(undef, nx, nz, 1, B)

    Xb[:, :, :, 1] .= X1
    Yb[:, :, :, 1] .= Y1

    for (b, idx) in enumerate(batch_indices[2:end])
        X, Y, _ = get_sample(bi.ds, idx)
        Xb[:, :, :, b+1] .= X
        Yb[:, :, :, b+1] .= Y
    end

    return (Xb, Yb), state + 1
end

end # module
