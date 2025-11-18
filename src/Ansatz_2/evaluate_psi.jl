module EvaluatePsi

using JLD2
using BSON
using CairoMakie
using Statistics
using Printf

using ..DatasetPsi: load_dataset, get_sample
using CairoMakie: DataAspect, Figure, Axis, heatmap!, Colorbar

# ------------------------------------------------------------
# Helper: Kristall-Umrisse plotten (physikalische Koordinaten)
# ------------------------------------------------------------
"""
    plot_crystal_outlines!(ax, centers_2D, radii; ...)

Zeichnet Kristall-Umrisse (Kreise) auf das Axis-Objekt `ax`
in denselben physikalischen Koordinaten wie xcoords/zcoords (z.B. in km).

- `centers_2D` : (cx, cz) in physikalischen Koordinaten
- `radii`      : Radien in physikalischen Koordinaten
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
# Helper: Ableitungen ψ_x, ψ_z berechnen
# ------------------------------------------------------------
"""
    grad_psi(ψ, dx, dz)

Berechnet numerisch
- ψ_x = ∂ψ/∂x
- ψ_z = ∂ψ/∂z

mit zentrierten Differenzen im Inneren.
Randpunkte bleiben 0.
"""
function grad_psi(ψ::AbstractMatrix{<:Real}, dx::Real, dz::Real)
    nx, nz = size(ψ)
    ψ_x = zeros(Float64, nx, nz)
    ψ_z = zeros(Float64, nx, nz)

    for i in 2:nx-1, j in 2:nz-1
        ψ_x[i, j] = (ψ[i+1, j] - ψ[i-1, j]) / (2 * dx)
        ψ_z[i, j] = (ψ[i, j+1] - ψ[i, j-1]) / (2 * dz)
    end

    return ψ_x, ψ_z
end

# ------------------------------------------------------------
# Datensatz-Evaluierung
# ------------------------------------------------------------
"""
    evaluate_dataset(; data_dir, model_path, out_prefix="eval_psi_dataset",
                      save_plots=false, plot_dir="eval_plots",
                      denorm_psi=false)

- Evaluierung von ψ (MSE, relL2)
- Zusätzlich: ψ_x, ψ_z (ebenfalls MSE, relL2)
- Gruppierung nach Kristallanzahl
- Plots in physikalischen Koordinaten (z.B. km, aus x_vec_1D/z_vec_1D)
- Wenn `save_plots = true`:
    - pro Sample: ψ-Plot (3 Panels)
    - pro Sample: ψ_x / ψ_z-Plot (2×3 Panels)
- Aggregierte Metriken werden als CSV gespeichert:
    `<out_prefix>_by_n.csv`
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

    if save_plots
        mkpath(plot_dir)
        @info "Speichere Plots nach: $plot_dir"
    end

    # pro Kristallanzahl: Vector von NamedTuples mit allen Metriken
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
        scale = get(filedata, "scale", 1.0)

        n = haskey(meta, :n_crystals) ? meta[:n_crystals] : 1
        centers_2D = haskey(meta, :centers_2D) ? meta[:centers_2D] : meta.centers_2D
        radii      = haskey(meta, :radii)      ? meta[:radii]      : meta.radii

        # Physikalische Koordinaten (z.B. km)
        if haskey(meta, :x_vec_1D) && haskey(meta, :z_vec_1D)
            xcoords = meta[:x_vec_1D]
            zcoords = meta[:z_vec_1D]
        else
            # Fallback: Indizes als "Koordinaten"
            xcoords = collect(1:nx)
            zcoords = collect(1:nz)
        end

        # --- ψ-Raum wählen (normalisiert oder physikalisch) ---
        if denorm_psi
            y_true_eval = y_true_norm ./ scale
            y_pred_eval = y_pred_norm ./ scale
        else
            y_true_eval = y_true_norm
            y_pred_eval = y_pred_norm
        end

        # --- Fehler für ψ ---
        mse_psi = mean((y_pred_eval .- y_true_eval).^2)

        num_psi = sqrt(sum((y_pred_eval .- y_true_eval).^2))
        denom_psi = sqrt(sum(y_true_eval.^2)) + eps()
        rel_l2_psi = num_psi / denom_psi

        # --- ψ_x und ψ_z berechnen ---
        if length(xcoords) > 1 && length(zcoords) > 1
            dx = (xcoords[2] - xcoords[1]) * 1000.0   # km → m
            dz = (zcoords[2] - zcoords[1]) * 1000.0   # km → m

        else
            dx = 1.0
            dz = 1.0
        end

        ψx_true, ψz_true = grad_psi(y_true_eval, dx, dz)
        ψx_pred, ψz_pred = grad_psi(y_pred_eval, dx, dz)

        # Fehler ψ_x
        mse_psix = mean((ψx_pred .- ψx_true).^2)
        num_psix = sqrt(sum((ψx_pred .- ψx_true).^2))
        denom_psix = sqrt(sum(ψx_true.^2)) + eps()
        rel_l2_psix = num_psix / denom_psix

        # Fehler ψ_z
        mse_psiz = mean((ψz_pred .- ψz_true).^2)
        num_psiz = sqrt(sum((ψz_pred .- ψz_true).^2))
        denom_psiz = sqrt(sum(ψz_true.^2)) + eps()
        rel_l2_psiz = num_psiz / denom_psiz

        stat = (; mse_psi, rel_l2_psi,
                 mse_psix, rel_l2_psix,
                 mse_psiz, rel_l2_psiz)

        if !haskey(errors_by_n, n)
            errors_by_n[n] = NamedTuple[]
        end
        push!(errors_by_n[n], stat)

        # --- Optional: Plots speichern ---
        if save_plots
            subdir = joinpath(plot_dir, @sprintf("n_%02d", n))
            mkpath(subdir)

            # ---------- ψ-Plot ----------
            filename_psi = joinpath(subdir, @sprintf("sample_%04d_psi.png", i))

            fig = Figure(resolution = (1500, 450))

            global_min = min(minimum(y_true_eval), minimum(y_pred_eval))
            global_max = max(maximum(y_true_eval), maximum(y_pred_eval))

            diff = y_pred_eval .- y_true_eval
            max_abs_diff = maximum(abs.(diff))

            gl1 = fig[1, 1] = GridLayout()
            gl2 = fig[1, 2] = GridLayout()
            gl3 = fig[1, 3] = GridLayout()

            ax1 = Axis(gl1[1, 1],
                       title  = "ψ LaMEM (n = $n, idx = $i, nx = $nx, nz = $nz)",
                       xlabel = "x (km)",
                       ylabel = "z (km)",
                       aspect = DataAspect())
            hm1 = heatmap!(ax1, xcoords, zcoords, y_true_eval; colorrange = (global_min, global_max))
            plot_crystal_outlines!(ax1, centers_2D, radii)
            Colorbar(gl1[1, 2], hm1, label = "ψ_true")

            ax2 = Axis(gl2[1, 1],
                       title  = "ψ U-Net",
                       xlabel = "x (km)",
                       ylabel = "z (km)",
                       aspect = DataAspect())
            hm2 = heatmap!(ax2, xcoords, zcoords, y_pred_eval; colorrange = (global_min, global_max))
            plot_crystal_outlines!(ax2, centers_2D, radii)
            Colorbar(gl2[1, 2], hm2, label = "ψ_pred")

            ax3 = Axis(gl3[1, 1],
                       title  = "Δψ = ψ_pred − ψ_true",
                       xlabel = "x (km)",
                       ylabel = "z (km)",
                       aspect = DataAspect())
            hm3 = heatmap!(ax3, xcoords, zcoords, diff; colorrange = (-max_abs_diff, max_abs_diff))
            plot_crystal_outlines!(ax3, centers_2D, radii)
            Colorbar(gl3[1, 2], hm3, label = "Fehler")

            save(filename_psi, fig)
            @info "ψ-Plot gespeichert: $filename_psi"

            # ---------- ψ_x / ψ_z-Plot ----------
            filename_grad = joinpath(subdir, @sprintf("sample_%04d_grad.png", i))

            fig_grad = Figure(resolution = (1500, 900))

            # ψ_x
            gx1 = fig_grad[1, 1] = GridLayout()
            gx2 = fig_grad[1, 2] = GridLayout()
            gx3 = fig_grad[1, 3] = GridLayout()

            min_psix = min(minimum(ψx_true), minimum(ψx_pred))
            max_psix = max(maximum(ψx_true), maximum(ψx_pred))
            diff_psix = ψx_pred .- ψx_true
            max_abs_diff_psix = maximum(abs.(diff_psix))

            axx1 = Axis(gx1[1, 1],
                        title  = "ψ_x true",
                        xlabel = "x (km)",
                        ylabel = "z (km)",
                        aspect = DataAspect())
            hmx1 = heatmap!(axx1, xcoords, zcoords, ψx_true; colorrange = (min_psix, max_psix))
            plot_crystal_outlines!(axx1, centers_2D, radii)
            Colorbar(gx1[1, 2], hmx1, label = "ψ_x true")

            axx2 = Axis(gx2[1, 1],
                        title  = "ψ_x pred",
                        xlabel = "x (km)",
                        ylabel = "z (km)",
                        aspect = DataAspect())
            hmx2 = heatmap!(axx2, xcoords, zcoords, ψx_pred; colorrange = (min_psix, max_psix))
            plot_crystal_outlines!(axx2, centers_2D, radii)
            Colorbar(gx2[1, 2], hmx2, label = "ψ_x pred")

            axx3 = Axis(gx3[1, 1],
                        title  = "Δψ_x",
                        xlabel = "x (km)",
                        ylabel = "z (km)",
                        aspect = DataAspect())
            hmx3 = heatmap!(axx3, xcoords, zcoords, diff_psix;
                            colorrange = (-max_abs_diff_psix, max_abs_diff_psix))
            plot_crystal_outlines!(axx3, centers_2D, radii)
            Colorbar(gx3[1, 2], hmx3, label = "Fehler ψ_x")

            # ψ_z
            gz1 = fig_grad[2, 1] = GridLayout()
            gz2 = fig_grad[2, 2] = GridLayout()
            gz3 = fig_grad[2, 3] = GridLayout()

            min_psiz = min(minimum(ψz_true), minimum(ψz_pred))
            max_psiz = max(maximum(ψz_true), maximum(ψz_pred))
            diff_psiz = ψz_pred .- ψz_true
            max_abs_diff_psiz = maximum(abs.(diff_psiz))

            axz1 = Axis(gz1[1, 1],
                        title  = "ψ_z true",
                        xlabel = "x (km)",
                        ylabel = "z (km)",
                        aspect = DataAspect())
            hmz1 = heatmap!(axz1, xcoords, zcoords, ψz_true; colorrange = (min_psiz, max_psiz))
            plot_crystal_outlines!(axz1, centers_2D, radii)
            Colorbar(gz1[1, 2], hmz1, label = "ψ_z true")

            axz2 = Axis(gz2[1, 1],
                        title  = "ψ_z pred",
                        xlabel = "x (km)",
                        ylabel = "z (km)",
                        aspect = DataAspect())
            hmz2 = heatmap!(axz2, xcoords, zcoords, ψz_pred; colorrange = (min_psiz, max_psiz))
            plot_crystal_outlines!(axz2, centers_2D, radii)
            Colorbar(gz2[1, 2], hmz2, label = "ψ_z pred")

            axz3 = Axis(gz3[1, 1],
                        title  = "Δψ_z",
                        xlabel = "x (km)",
                        ylabel = "z (km)",
                        aspect = DataAspect())
            hmz3 = heatmap!(axz3, xcoords, zcoords, diff_psiz;
                            colorrange = (-max_abs_diff_psiz, max_abs_diff_psiz))
            plot_crystal_outlines!(axz3, centers_2D, radii)
            Colorbar(gz3[1, 2], hmz3, label = "Fehler ψ_z")

            save(filename_grad, fig_grad)
            @info "Gradienten-Plot gespeichert: $filename_grad"
        end
    end

    @info "=== Auswertung nach Kristallanzahl ($(denorm_psi ? "physikalisches ψ" : "ψ_norm")) ==="

    # sortierte Gruppen, einmal berechnen
    sorted_groups = sort(collect(errors_by_n); by = first)

    # CSV schreiben
    csv_path = out_prefix * "_by_n.csv"
    open(csv_path, "w") do io
        println(io,
            "n_crystals,N," *
            "mse_psi_mean,mse_psi_std,relL2_psi_mean,relL2_psi_std," *
            "mse_psix_mean,mse_psix_std,relL2_psix_mean,relL2_psix_std," *
            "mse_psiz_mean,mse_psiz_std,relL2_psiz_mean,relL2_psiz_std"
        )

        for (n, stats) in sorted_groups
            N = length(stats)

            mse_psi_vals    = [s.mse_psi    for s in stats]
            rel_psi_vals    = [s.rel_l2_psi for s in stats]

            mse_psix_vals   = [s.mse_psix    for s in stats]
            rel_psix_vals   = [s.rel_l2_psix for s in stats]

            mse_psiz_vals   = [s.mse_psiz    for s in stats]
            rel_psiz_vals   = [s.rel_l2_psiz for s in stats]

            mse_psi_mean  = mean(mse_psi_vals)
            mse_psi_std   = std(mse_psi_vals)
            rel_psi_mean  = mean(rel_psi_vals)
            rel_psi_std   = std(rel_psi_vals)

            mse_psix_mean = mean(mse_psix_vals)
            mse_psix_std  = std(mse_psix_vals)
            rel_psix_mean = mean(rel_psix_vals)
            rel_psix_std  = std(rel_psix_vals)

            mse_psiz_mean = mean(mse_psiz_vals)
            mse_psiz_std  = std(mse_psiz_vals)
            rel_psiz_mean = mean(rel_psiz_vals)
            rel_psiz_std  = std(rel_psiz_vals)

            println(io, @sprintf(
                "%d,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e",
                n, N,
                mse_psi_mean,  mse_psi_std,  rel_psi_mean,  rel_psi_std,
                mse_psix_mean, mse_psix_std, rel_psix_mean, rel_psix_std,
                mse_psiz_mean, mse_psiz_std, rel_psiz_mean, rel_psiz_std
            ))
        end
    end
    @info "Metriken als CSV gespeichert: $csv_path"

    # Log-Ausgabe
    for (n, stats) in sorted_groups
        N = length(stats)

        mse_psi_vals    = [s.mse_psi    for s in stats]
        rel_psi_vals    = [s.rel_l2_psi for s in stats]

        mse_psix_vals   = [s.mse_psix    for s in stats]
        rel_psix_vals   = [s.rel_l2_psix for s in stats]

        mse_psiz_vals   = [s.mse_psiz    for s in stats]
        rel_psiz_vals   = [s.rel_l2_psiz for s in stats]

        msg = @sprintf(
            "n_crystals=%d | N=%3d | ψ:   MSE = %.4e, relL2 = %.4e | ψ_x: MSE = %.4e, relL2 = %.4e | ψ_z: MSE = %.4e, relL2 = %.4e",
            n, N,
            mean(mse_psi_vals),  mean(rel_psi_vals),
            mean(mse_psix_vals), mean(rel_psix_vals),
            mean(mse_psiz_vals), mean(rel_psiz_vals),
        )
        @info msg
    end

    return errors_by_n
end

end # module
