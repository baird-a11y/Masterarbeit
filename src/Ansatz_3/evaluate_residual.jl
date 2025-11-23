module EvaluateResidual

using JLD2
using BSON
using CairoMakie
using Statistics
using Printf

using ..DatasetResidual: load_dataset, get_sample
using CairoMakie: DataAspect, Figure, Axis, heatmap!, Colorbar

# ------------------------------------------------------------
# Helper: Kristall-Umrisse plotten (wie in EvaluatePsi)
# ------------------------------------------------------------
"""
    plot_crystal_outlines!(ax, centers_2D, radii; ...)

Zeichnet Kristall-Umrisse (Kreise) auf das Axis-Objekt `ax`
in denselben physikalischen Koordinaten wie xcoords/zcoords (z. B. in km).
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
# Hauptfunktion: Datensatz evaluieren + Plots erzeugen
# ------------------------------------------------------------
"""
    evaluate_dataset_residual(; data_dir, model_path,
                               out_prefix="eval_residual_dataset",
                               save_plots=false,
                               plot_dir="eval_residual_plots",
                               denorm_residual=true)

Evaluierung von Ansatz 3 (Residual-Learning):

- Modell-Target: residual_norm (normalisiertes Residuum)
- De-Normierung über `scale` aus den .jld2-Dateien
- Berechnet:
    - MSE & relL2 des Residuals r
    - MSE & relL2 von ψ_corr = ψ_LaMEM + r_pred  vs ψ_analytic
- Gruppiert Fehler nach Kristallanzahl
- Optional: speichert Plots pro Sample
    1) Residuum: r_true, r_pred, r_diff
    2) ψ: ψ_analytic, ψ_corr_pred, ψ_diff
- Schreibt CSV: `<out_prefix>_by_n.csv`
"""
function evaluate_dataset_residual(; data_dir::String,
                                   model_path::String,
                                   out_prefix::String = "eval_residual_dataset",
                                   save_plots::Bool = false,
                                   plot_dir::String = "eval_residual_plots",
                                   denorm_residual::Bool = true)

    @info "Lade Residual-Modell aus $model_path"
    model_bson = BSON.load(model_path)
    model = model_bson[:model]

    ds = load_dataset(data_dir)
    n_samples = length(ds.files)
    @info "Residual-Datensatz mit $n_samples Samples geladen aus $data_dir"

    if denorm_residual
        @info "Evaluierung im PHYSIKALISCHEN Residual-Raum (de-normalisiert mit 'scale')."
    else
        @info "Evaluierung im NORMALISIERTEN Residual-Raum (residual_norm)."
    end

    if save_plots
        mkpath(plot_dir)
        @info "Speichere Residual-Plots nach: $plot_dir"
    end

    # pro Kristallanzahl: Vector von NamedTuples mit allen Metriken
    errors_by_n = Dict{Int, Vector{NamedTuple}}()

    for (i, filepath) in enumerate(ds.files)
        # --- x, y_true (residual_norm als (nx,nz,1)) laden ---
        x, y_true = get_sample(ds, i)
        nx, nz, nc = size(x)
        x_batch = reshape(x, nx, nz, nc, 1)

        # Vorhersage im normalisierten Raum
        y_pred_batch = model(x_batch)
        y_pred_norm = dropdims(y_pred_batch[:, :, :, 1], dims=3)   # (nx,nz)
        r_true_norm = dropdims(y_true, dims=3)                     # (nx,nz)

        # --- Zusätzliche Daten aus der Datei laden ---
        filedata   = JLD2.load(filepath)
        residual_norm_file = filedata["residual_norm"]  # zur Sicherheit vorhanden
        scale      = get(filedata, "scale", 1.0)
        ψ_lamem    = filedata["ψ_lamem"]
        ψ_analytic = filedata["ψ_analytic"]
        meta       = filedata["meta"]

        n = haskey(meta, :n_crystals) ? meta[:n_crystals] : 1
        centers_2D = haskey(meta, :centers_2D) ? meta[:centers_2D] : meta.centers_2D
        radii      = haskey(meta, :radii)      ? meta[:radii]      : meta.radii

        xcoords = haskey(meta, :x_vec_1D) ? meta[:x_vec_1D] : collect(1:nx)
        zcoords = haskey(meta, :z_vec_1D) ? meta[:z_vec_1D] : collect(1:nz)

        # --- Residual-Raum (normalisiert oder physikalisch) ---
        if denorm_residual && scale != 0.0
            r_true_eval = r_true_norm ./ scale
            r_pred_eval = y_pred_norm ./ scale
        else
            r_true_eval = r_true_norm
            r_pred_eval = y_pred_norm
        end

        # --- ψ_corr bilden (IMMER physikalisch) ---
        # Residuum physikalisch:
        r_pred_phys = y_pred_norm ./ (scale == 0.0 ? 1.0 : scale)
        ψ_lamem_phys = Float64.(ψ_lamem)
        ψ_analytic_phys = Float64.(ψ_analytic)
        ψ_corr_pred = ψ_lamem_phys .+ r_pred_phys

        # --- Metriken: Residual ---
        mse_r = mean((r_pred_eval .- r_true_eval).^2)
        num_r   = sqrt(sum((r_pred_eval .- r_true_eval).^2))
        denom_r = sqrt(sum(r_true_eval.^2)) + eps()
        relL2_r = num_r / denom_r

        # relative Pixel-Fehler für Residuum
        rel_err_r = abs.(r_pred_eval .- r_true_eval) ./ (abs.(r_true_eval) .+ eps())
        eps01_r = mean(rel_err_r .> 0.01)
        eps05_r = mean(rel_err_r .> 0.05)
        eps10_r = mean(rel_err_r .> 0.10)

        # --- Metriken: ψ_corr vs ψ_analytic ---
        mse_psi = mean((ψ_corr_pred .- ψ_analytic_phys).^2)
        num_psi   = sqrt(sum((ψ_corr_pred .- ψ_analytic_phys).^2))
        denom_psi = sqrt(sum(ψ_analytic_phys.^2)) + eps()
        relL2_psi = num_psi / denom_psi

        stat = (; mse_r, relL2_r,
                 mse_psi, relL2_psi,
                 eps01_r, eps05_r, eps10_r)

        if !haskey(errors_by_n, n)
            errors_by_n[n] = NamedTuple[]
        end
        push!(errors_by_n[n], stat)

        # ----------------------------------------------------
        # Optional: Plots speichern
        # ----------------------------------------------------
        if save_plots
            subdir = joinpath(plot_dir, @sprintf("n_%02d", n))
            mkpath(subdir)

            # ---------- Residual-Plot ----------
            filename_res = joinpath(subdir, @sprintf("sample_%04d_residual.png", i))

            fig_res = Figure(resolution = (1500, 450))

            global_min_r = min(minimum(r_true_eval), minimum(r_pred_eval))
            global_max_r = max(maximum(r_true_eval), maximum(r_pred_eval))
            diff_r = r_pred_eval .- r_true_eval
            max_abs_diff_r = maximum(abs.(diff_r))

            gl1 = fig_res[1, 1] = GridLayout()
            gl2 = fig_res[1, 2] = GridLayout()
            gl3 = fig_res[1, 3] = GridLayout()

            ax1 = Axis(gl1[1, 1],
                       title  = "Residuum true (n = $n, idx = $i)",
                       xlabel = "x (km)",
                       ylabel = "z (km)",
                       aspect = DataAspect())
            hm1 = heatmap!(ax1, xcoords, zcoords, r_true_eval; colorrange = (global_min_r, global_max_r))
            plot_crystal_outlines!(ax1, centers_2D, radii)
            Colorbar(gl1[1, 2], hm1, label = "r_true")

            ax2 = Axis(gl2[1, 1],
                       title  = "Residuum pred",
                       xlabel = "x (km)",
                       ylabel = "z (km)",
                       aspect = DataAspect())
            hm2 = heatmap!(ax2, xcoords, zcoords, r_pred_eval; colorrange = (global_min_r, global_max_r))
            plot_crystal_outlines!(ax2, centers_2D, radii)
            Colorbar(gl2[1, 2], hm2, label = "r_pred")

            ax3 = Axis(gl3[1, 1],
                       title  = "Δr = r_pred − r_true",
                       xlabel = "x (km)",
                       ylabel = "z (km)",
                       aspect = DataAspect())
            hm3 = heatmap!(ax3, xcoords, zcoords, diff_r; colorrange = (-max_abs_diff_r, max_abs_diff_r))
            plot_crystal_outlines!(ax3, centers_2D, radii)
            Colorbar(gl3[1, 2], hm3, label = "Fehler r")

            save(filename_res, fig_res)
            @info "Residual-Plot gespeichert: $filename_res"

            # ---------- ψ_corr-Plot ----------
            filename_psi = joinpath(subdir, @sprintf("sample_%04d_psi_corr.png", i))

            fig_psi = Figure(resolution = (1500, 450))

            global_min_psi = min(minimum(ψ_analytic_phys), minimum(ψ_corr_pred))
            global_max_psi = max(maximum(ψ_analytic_phys), maximum(ψ_corr_pred))

            diff_psi = ψ_corr_pred .- ψ_analytic_phys
            max_abs_diff_psi = maximum(abs.(diff_psi))

            glp1 = fig_psi[1, 1] = GridLayout()
            glp2 = fig_psi[1, 2] = GridLayout()
            glp3 = fig_psi[1, 3] = GridLayout()

            axp1 = Axis(glp1[1, 1],
                        title  = "ψ_analytic (n = $n, idx = $i)",
                        xlabel = "x (km)",
                        ylabel = "z (km)",
                        aspect = DataAspect())
            hmp1 = heatmap!(axp1, xcoords, zcoords, ψ_analytic_phys; colorrange = (global_min_psi, global_max_psi))
            plot_crystal_outlines!(axp1, centers_2D, radii)
            Colorbar(glp1[1, 2], hmp1, label = "ψ_analytic")

            axp2 = Axis(glp2[1, 1],
                        title  = "ψ_corr = ψ_LaMEM + r_pred",
                        xlabel = "x (km)",
                        ylabel = "z (km)",
                        aspect = DataAspect())
            hmp2 = heatmap!(axp2, xcoords, zcoords, ψ_corr_pred; colorrange = (global_min_psi, global_max_psi))
            plot_crystal_outlines!(axp2, centers_2D, radii)
            Colorbar(glp2[1, 2], hmp2, label = "ψ_corr_pred")

            axp3 = Axis(glp3[1, 1],
                        title  = "Δψ = ψ_corr_pred − ψ_analytic",
                        xlabel = "x (km)",
                        ylabel = "z (km)",
                        aspect = DataAspect())
            hmp3 = heatmap!(axp3, xcoords, zcoords, diff_psi;
                            colorrange = (-max_abs_diff_psi, max_abs_diff_psi))
            plot_crystal_outlines!(axp3, centers_2D, radii)
            Colorbar(glp3[1, 2], hmp3, label = "Fehler ψ")

            save(filename_psi, fig_psi)
            @info "ψ_corr-Plot gespeichert: $filename_psi"
        end
    end

    @info "=== Auswertung nach Kristallanzahl (Residual-Ansatz) ==="

    # sortierte Gruppen
    sorted_groups = sort(collect(errors_by_n); by = first)

    # CSV schreiben
    csv_path = out_prefix * "_by_n.csv"
    open(csv_path, "w") do io
        println(io, "n_crystals,N," *
           "mse_r_mean,mse_r_std,relL2_r_mean,relL2_r_std," *
           "mse_psi_mean,mse_psi_std,relL2_psi_mean,relL2_psi_std," *
           "eps01_r_mean,eps01_r_std,eps05_r_mean,eps05_r_std,eps10_r_mean,eps10_r_std")

        for (n, stats) in sorted_groups
            N = length(stats)

            mse_r_vals   = [s.mse_r   for s in stats]
            rel_r_vals   = [s.relL2_r for s in stats]

            mse_psi_vals = [s.mse_psi   for s in stats]
            rel_psi_vals = [s.relL2_psi for s in stats]

            eps01_vals = [s.eps01_r for s in stats]
            eps05_vals = [s.eps05_r for s in stats]
            eps10_vals = [s.eps10_r for s in stats]

            mse_r_mean   = mean(mse_r_vals)
            mse_r_std    = std(mse_r_vals)
            rel_r_mean   = mean(rel_r_vals)
            rel_r_std    = std(rel_r_vals)

            mse_psi_mean = mean(mse_psi_vals)
            mse_psi_std  = std(mse_psi_vals)
            rel_psi_mean = mean(rel_psi_vals)
            rel_psi_std  = std(rel_psi_vals)

            eps01_mean = mean(eps01_vals)
            eps01_std  = std(eps01_vals)
            eps05_mean = mean(eps05_vals)
            eps05_std  = std(eps05_vals)
            eps10_mean = mean(eps10_vals)
            eps10_std  = std(eps10_vals)

            println(io, @sprintf(
                "%d,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e",
                n, N,
                mse_r_mean,   mse_r_std,   rel_r_mean,   rel_r_std,
                mse_psi_mean, mse_psi_std, rel_psi_mean, rel_psi_std,
                eps01_mean, eps01_std, eps05_mean, eps05_std, eps10_mean, eps10_std
            ))
        end
    end
    @info "Residual-Metriken als CSV gespeichert: $csv_path"

    # Log-Ausgabe
    for (n, stats) in sorted_groups
        N = length(stats)

        mse_r_vals   = [s.mse_r   for s in stats]
        rel_r_vals   = [s.relL2_r for s in stats]

        mse_psi_vals = [s.mse_psi   for s in stats]
        rel_psi_vals = [s.relL2_psi for s in stats]

        msg = @sprintf(
            "n_crystals=%d | N=%3d | r:   MSE = %.4e, relL2 = %.4e | ψ_corr: MSE = %.4e, relL2 = %.4e",
            n, N,
            mean(mse_r_vals),   mean(rel_r_vals),
            mean(mse_psi_vals), mean(rel_psi_vals),
        )
        @info msg
    end

    return errors_by_n
end

end # module
