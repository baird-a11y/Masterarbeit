#!/usr/bin/env julia

# Skript zur Analyse:
# Zusammenhang zwischen Fehler in ψ und Position der Kristalle.
# - Beliebig viele Kristalle pro Sample
# - Abstand der Kristallzentren zum Domänenmittelpunkt
# - Abstände zu den vier Ecken der Domäne
#
# CSV + Plots werden im Ordner "Auswertung" gespeichert (unterhalb von data_dir).

using JLD2
using BSON
using Statistics
using Printf
using CairoMakie
using Functors: fmap
using Base.Filesystem: mkpath

# Eigene Module einbinden (wie in main.jl)
include("dataset_psi.jl")
include("unet_psi.jl")

using .DatasetPsi: load_dataset, get_sample
using .UNetPsi: build_unet

# ------------------------------------------------------------
# Optional: CUDA-Support (an EvaluatePsi angelehnt)
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Hauptfunktion: Fehler vs. Abstände (pro Kristall)
# ------------------------------------------------------------
function compute_error_vs_center_distance(; 
        data_dir::String,
        model_path::String,
        out_dir::String = joinpath(data_dir, "Auswertung"),
        out_csv_name::String = "error_vs_center_distance.csv",
        denorm_psi::Bool = true,
        use_gpu::Union{Nothing,Bool} = nothing)

    # Auswertung-Ordner anlegen
    mkpath(out_dir)
    out_csv = joinpath(out_dir, out_csv_name)

    move, devsym = _select_mover(use_gpu)
    @info "Gerät für Vorhersage: $(devsym == :gpu ? "GPU (CUDA)" : "CPU")"
    @info "Auswertung wird geschrieben nach: $out_dir"

    if devsym == :gpu && _HAS_CUDA
        try
            CUDA.allowscalar(false)
        catch
        end
    end

    # Modell laden und ggf. auf GPU verschieben
    @info "Lade Modell aus $model_path"
    model_bson = BSON.load(model_path)
    model_cpu  = model_bson[:model]
    model      = fmap(move, model_cpu)

    # Datensatz laden
    ds = load_dataset(data_dir)
    n_samples = length(ds.files)
    @info "Dataset mit $n_samples Samples geladen aus $data_dir"

    results = NamedTuple[]

    for (i, filepath) in enumerate(ds.files)
        # Input/Target laden
        x, y_true = get_sample(ds, i)
        nx, nz, ch = size(x)

        # Batch-Dimension hinzufügen
        x_batch = reshape(x, nx, nz, ch, 1)
        x_dev   = move(x_batch)

        # Vorhersage im (normalisierten) Raum
        y_pred_batch_dev = model(x_dev)
        y_pred_norm = dropdims(Array(y_pred_batch_dev)[:, :, :, 1], dims=3)
        y_true_norm = dropdims(Array(y_true), dims=3)

        # Metadaten & Skala laden
        filedata = JLD2.load(filepath)
        meta  = filedata["meta"]
        scale = get(filedata, "scale", 1.0)

        # Anzahl Kristalle & Zentren
        n_crystals = haskey(meta, :n_crystals) ? meta[:n_crystals] : meta.n_crystals
        centers_2D = haskey(meta, :centers_2D) ? meta[:centers_2D] : meta.centers_2D

        # Physikalische Koordinaten (km)
        xcoords = haskey(meta, :x_vec_1D) ? meta[:x_vec_1D] : collect(1:nx)
        zcoords = haskey(meta, :z_vec_1D) ? meta[:z_vec_1D] : collect(1:nz)

        # Domänenmittelpunkt
        x_center = (xcoords[1] + xcoords[end]) / 2
        z_center = (zcoords[1] + zcoords[end]) / 2

        # Ecken der Domäne
        x1, xN = xcoords[1], xcoords[end]
        z1, zN = zcoords[1], zcoords[end]

        # ψ-Raum wählen (normalisiert vs. physikalisch)
        if denorm_psi
            y_true_eval = y_true_norm ./ scale
            y_pred_eval = y_pred_norm ./ scale
        else
            y_true_eval = y_true_norm
            y_pred_eval = y_pred_norm
        end

        # Fehler ψ (global pro Sample)
        mse_psi = mean((y_pred_eval .- y_true_eval).^2)

        num_psi   = sqrt(sum((y_pred_eval .- y_true_eval).^2))
        denom_psi = sqrt(sum(y_true_eval.^2)) + eps()
        rel_l2_psi = num_psi / denom_psi

        # Für jeden Kristall im Sample einen Eintrag erzeugen
        for (k, c) in enumerate(centers_2D)
            cx, cz = c

            # Abstand zum Domänenmittelpunkt
            dist_center = sqrt((cx - x_center)^2 + (cz - z_center)^2)

            # Abstände zu den vier Ecken
            dist_c1 = sqrt((cx - x1)^2 + (cz - z1)^2)  # unten links
            dist_c2 = sqrt((cx - xN)^2 + (cz - z1)^2)  # unten rechts
            dist_c3 = sqrt((cx - x1)^2 + (cz - zN)^2)  # oben links
            dist_c4 = sqrt((cx - xN)^2 + (cz - zN)^2)  # oben rechts

            push!(results, (; 
                sample_idx    = i,
                crystal_idx   = k,
                n_crystals    = n_crystals,
                cx            = cx,
                cz            = cz,
                x_center      = x_center,
                z_center      = z_center,
                dist_center   = dist_center,
                dist_corner_1 = dist_c1,
                dist_corner_2 = dist_c2,
                dist_corner_3 = dist_c3,
                dist_corner_4 = dist_c4,
                mse_psi       = mse_psi,
                rel_l2_psi    = rel_l2_psi,
            ))
        end
    end

    # CSV schreiben (pro Kristall eine Zeile)
    open(out_csv, "w") do io
        println(io,
            "sample_idx,crystal_idx,n_crystals," *
            "cx,cz,x_center,z_center," *
            "dist_center,dist_corner_1,dist_corner_2,dist_corner_3,dist_corner_4," *
            "mse_psi,relL2_psi"
        )
        for r in results
            println(io, @sprintf(
                "%d,%d,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e",
                r.sample_idx, r.crystal_idx, r.n_crystals,
                r.cx, r.cz,
                r.x_center, r.z_center,
                r.dist_center,
                r.dist_corner_1, r.dist_corner_2, r.dist_corner_3, r.dist_corner_4,
                r.mse_psi,
                r.rel_l2_psi,
            ))
        end
    end

    @info "Ergebnisse nach $out_csv geschrieben (n=$(length(results)) Kristalle über alle Samples)."
    return results, out_dir
end

# ------------------------------------------------------------
# Plot: rel. L2-Fehler vs. Abstand (pro Kristall, farblich nach N)
# ------------------------------------------------------------
function plot_error_vs_distance_byN(
        results;
        out_dir::String,
        kind::Symbol = :center,
        outfile_name::Union{Nothing,String} = nothing)

    mkpath(out_dir)

    # Distanz-Feld und Achsentext je nach Art
    dist_accessor, x_label, title =
        if kind == :center
            (r -> r.dist_center,
             "Abstand vom Domänenmittelpunkt (km)",
             "Fehler vs. Abstand vom Zentrum (pro Kristall)")
        elseif kind == :corner1
            (r -> r.dist_corner_1,
             "Abstand zu Ecke 1 (unten links) (km)",
             "Fehler vs. Abstand zu Ecke 1")
        elseif kind == :corner2
            (r -> r.dist_corner_2,
             "Abstand zu Ecke 2 (unten rechts) (km)",
             "Fehler vs. Abstand zu Ecke 2")
        elseif kind == :corner3
            (r -> r.dist_corner_3,
             "Abstand zu Ecke 3 (oben links) (km)",
             "Fehler vs. Abstand zu Ecke 3")
        elseif kind == :corner4
            (r -> r.dist_corner_4,
             "Abstand zu Ecke 4 (oben rechts) (km)",
             "Fehler vs. Abstand zu Ecke 4")
        else
            error("Unbekanntes kind = $kind. Erlaubt: :center, :corner1..:corner4")
        end

    # Dateiname automatisch wählen, falls nicht gesetzt
    if outfile_name === nothing
        suffix = kind == :center ? "center" :
                 kind == :corner1 ? "corner1" :
                 kind == :corner2 ? "corner2" :
                 kind == :corner3 ? "corner3" : "corner4"
        outfile_name = "error_vs_$(suffix)_distance_byN.png"
    end

    outfile = joinpath(out_dir, outfile_name)

    # Einzigartige Kristallzahlen
    all_N = sort(unique([r.n_crystals for r in results]))

    fig = Figure(resolution = (800, 500))
    ax  = Axis(fig[1, 1];
        xlabel = x_label,
        ylabel = "rel. L2-Fehler ψ",
        title  = title,
    )

    # Pro Kristallzahl eigene Farbe, Label für Legend
    for N in all_N
        group = filter(r -> r.n_crystals == N, results)
        d   = [dist_accessor(r) for r in group]
        err = [r.rel_l2_psi    for r in group]

        scatter!(ax, d, err; label = "N = $N")
    end

    axislegend(ax; position = :rb)  # Legende unten rechts

    save(outfile, fig)
    @info "Plot gespeichert: $outfile"

    return fig
end

# ------------------------------------------------------------
# Direkt aufrufbar machen (wie main.jl)
# ------------------------------------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    @info "Running error_vs_center_distance.jl as script."
    @info "Working directory: $(pwd())"

    # Default-Pfade wie in main.jl anpassen
    data_dir   = "/local/home/baselt/src/Daten/data_psi_vali"
    model_path = "unet_psi.bson"

    results, out_dir = compute_error_vs_center_distance(; 
        data_dir   = data_dir,
        model_path = model_path,
        denorm_psi = true,   # physikalischer ψ-Raum
        use_gpu    = nothing # auto
    )

    # 1) Fehler vs. Abstand vom Zentrum, farblich nach N
    plot_error_vs_distance_byN(results; out_dir = out_dir, kind = :center)

    # 2) Fehler vs. Abstand zu den vier Ecken, farblich nach N
    plot_error_vs_distance_byN(results; out_dir = out_dir, kind = :corner1)
    plot_error_vs_distance_byN(results; out_dir = out_dir, kind = :corner2)
    plot_error_vs_distance_byN(results; out_dir = out_dir, kind = :corner3)
    plot_error_vs_distance_byN(results; out_dir = out_dir, kind = :corner4)
end
