module EvaluatePsi

using JLD2
using BSON
using CairoMakie            # CairoMakie ist für PNGs oft stressfreier als GLMakie
using Statistics
using Printf

using ..DatasetPsi: load_dataset, get_sample
using CairoMakie: DataAspect, Figure, Axis, heatmap!, Colorbar

# ------------------------------------------------------------
# Helper: Kristall-Umrisse plotten
# ------------------------------------------------------------
"""
    plot_crystal_outlines!(ax, centers_2D, radii; n_pts=200, ...)

Zeichnet Kristall-Umrisse (Kreise) in physikalischen Koordinaten
auf das gegebene Axis-Objekt `ax`.

- `centers_2D` : Vector{Tuple{Float64,Float64}} mit (cx, cz)
- `radii`      : Vector{Float64} mit passenden Radien
"""
function plot_crystal_outlines!(
    ax,
    centers_2D,
    radii;
    n_pts::Int = 200,
    linewidth = 2,
    color = :black,
    linestyle = :solid,
)
    θ = range(0, 2π; length = n_pts)

    for ((cx, cz), r) in zip(centers_2D, radii)
        xs = cx .+ r .* cos.(θ)
        zs = cz .+ r .* sin.(θ)
        lines!(ax, xs, zs;
               linewidth = linewidth,
               color     = color,
               linestyle = linestyle)
    end

    return ax
end

# ------------------------------------------------------------
# Datensatz-Evaluierung (mit optionalen Plots + Overlays)
# ------------------------------------------------------------
"""
    evaluate_dataset(; data_dir, model_path, out_prefix="eval_psi_dataset",
                      save_plots=false, plot_dir="eval_plots",
                      denorm_psi=false)

Lädt ein trainiertes Modell und einen kompletten Datensatz und
berechnet Fehlermetriken für jedes Sample. Die Ergebnisse werden
nach Kristallanzahl `meta.n_crystals` gruppiert und als Übersicht
im Log ausgegeben.

- Wenn `denorm_psi = false`: Fehler im normalisierten ψ-Raum (ψ_norm)
- Wenn `denorm_psi = true` : Fehler im physikalischen ψ-Raum

Bei `save_plots = true` werden für jedes Sample Plots mit Kristall-
Overlays in Unterordnern je Kristallanzahl gespeichert.
"""
function evaluate_dataset(; data_dir::String,
                          model_path::String,
                          out_prefix::String = "eval_psi_dataset",
                          save_plots::Bool = false,
                          plot_dir::String = "eval_plots",
                          denorm_psi::Bool = false)

    @info "Lade Modell aus $model_path"
    model_bson = BSON.load(model_path)
    model = model_bson[:model]

    ds = load_dataset(data_dir)
    n_samples = length(ds.files)
    @info "Datensatz mit $n_samples Samples geladen aus $data_dir"

    if denorm_psi
        @info "Evaluierung im PHYSIKALISCHEN ψ-Raum (de-normalisiert mit 'scale')."
    else
        @info "Evaluierung im NORMALISIERTEN ψ-Raum (ψ_norm)."
    end

    # Plot-Basisordner
    if save_plots
        mkpath(plot_dir)
        @info "Speichere Plots nach: $plot_dir"
    end

    errors_by_n = Dict{Int, Vector{NamedTuple}}()

    for (i, filepath) in enumerate(ds.files)
        # --- Daten als (nx,nz,1) laden ---
        x, y_true = get_sample(ds, i)

        nx, nz, ch = size(x)
        x_batch = reshape(x, nx, nz, ch, 1)

        # Vorhersage (normalisierter Raum)
        y_pred_batch = model(x_batch)
        y_pred_norm = dropdims(y_pred_batch[:, :, :, 1], dims=3)   # (nx,nz)
        y_true_norm = dropdims(y_true, dims=3)                     # (nx,nz)

        # --- Zusatzinfos (meta + scale) aus der Datei ---
        filedata = JLD2.load(filepath)
        meta  = filedata["meta"]
        scale = get(filedata, "scale", 1.0)  # fallback 1.0 falls älterer Datensatz

        # Robust auf NamedTuple vs. struct
        n = haskey(meta, :n_crystals) ? meta[:n_crystals] : 1

        centers_2D = haskey(meta, :centers_2D) ? meta[:centers_2D] : meta.centers_2D
        radii      = haskey(meta, :radii)      ? meta[:radii]      : meta.radii

        # --- Entscheiden, in welchem Raum evaluiert wird ---
        if denorm_psi
            y_true_eval = y_true_norm ./ scale
            y_pred_eval = y_pred_norm ./ scale
        else
            y_true_eval = y_true_norm
            y_pred_eval = y_pred_norm
        end

        # --- Fehlermetrik im gewählten Raum ---
        mse = mean((y_pred_eval .- y_true_eval).^2)

        num = sqrt(sum((y_pred_eval .- y_true_eval).^2))
        denom = sqrt(sum(y_true_eval.^2)) + eps()
        rel_l2 = num / denom

        stat = (; mse=mse, rel_l2=rel_l2)
        if !haskey(errors_by_n, n)
            errors_by_n[n] = NamedTuple[]
        end
        push!(errors_by_n[n], stat)

        # --- Optional: Plot speichern (ebenfalls im gewählten Raum) ---
                if save_plots
            subdir = joinpath(plot_dir, @sprintf("n_%02d", n))
            mkpath(subdir)

            filename = joinpath(subdir, @sprintf("sample_%04d.png", i))

            # etwas flacheres, breites Format
            fig = Figure(resolution = (1500, 450))

            # gemeinsame Skala für ψ_true / ψ_pred im gewählten Raum
            global_min = min(minimum(y_true_eval), minimum(y_pred_eval))
            global_max = max(maximum(y_true_eval), maximum(y_pred_eval))

            diff = y_pred_eval .- y_true_eval
            max_abs_diff = maximum(abs.(diff))

            # physikalische Koordinaten (wie in der Datengenerierung)
            nxp, nzp = size(y_true_eval)
            xcoords = range(-0.5, 0.5; length = nxp)
            zcoords = range(-0.5, 0.5; length = nzp)

            # --- Layout: 3 Spalten, in jeder Spalte: Axis + Colorbar rechts ---
            gl1 = fig[1, 1] = GridLayout()
            gl2 = fig[1, 2] = GridLayout()
            gl3 = fig[1, 3] = GridLayout()

            # Panel 1: ψ_true
            ax1 = Axis(gl1[1, 1],
                       title  = "ψ LaMEM (n = $n, idx = $i)",
                       xlabel = "x",
                       ylabel = "z",
                       aspect = DataAspect())
            hm1 = heatmap!(ax1, xcoords, zcoords, y_true_eval'; colorrange = (global_min, global_max))
            plot_crystal_outlines!(ax1, centers_2D, radii)
            Colorbar(gl1[1, 2], hm1, label = "ψ_true")

            # Panel 2: ψ_pred
            ax2 = Axis(gl2[1, 1],
                       title  = "ψ U-Net",
                       xlabel = "x",
                       ylabel = "z",
                       aspect = DataAspect())
            hm2 = heatmap!(ax2, xcoords, zcoords, y_pred_eval'; colorrange = (global_min, global_max))
            plot_crystal_outlines!(ax2, centers_2D, radii)
            Colorbar(gl2[1, 2], hm2, label = "ψ_pred")

            # Panel 3: Δψ
            ax3 = Axis(gl3[1, 1],
                       title  = "Δψ = ψ_pred − ψ_true",
                       xlabel = "x",
                       ylabel = "z",
                       aspect = DataAspect())
            hm3 = heatmap!(ax3, xcoords, zcoords, diff'; colorrange = (-max_abs_diff, max_abs_diff))
            plot_crystal_outlines!(ax3, centers_2D, radii)
            Colorbar(gl3[1, 2], hm3, label = "Fehler")

            save(filename, fig)
            @info "Plot gespeichert: $filename"
        end

    end

    @info "=== Auswertung nach Kristallanzahl ($(denorm_psi ? "physikalisches ψ" : "ψ_norm")) ==="
    for (n, stats) in sort(collect(errors_by_n); by = first)
        mses   = [s.mse for s in stats]
        rels   = [s.rel_l2 for s in stats]
        N      = length(stats)

        mse_mean = mean(mses)
        mse_std  = std(mses)
        rel_mean = mean(rels)
        rel_std  = std(rels)

        msg = @sprintf("n_crystals=%d | N=%3d | MSE: %.4e ± %.4e | relL2: %.4e ± %.4e",
                       n, N, mse_mean, mse_std, rel_mean, rel_std)
        @info msg
    end

    return errors_by_n
end

end # module
