
module EvaluatePsi

using JLD2
using BSON
using CairoMakie
using Statistics
using Printf
using Functors: fmap

using ..DatasetPsi: load_dataset, get_sample
using CairoMakie: DataAspect, Figure, Axis, heatmap!, Colorbar

# ============================================================
# Optional CUDA-Support (wie im Training)
# ============================================================
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
        # auto
        if has_gpu()
            return _to_gpu, :gpu
        else
            return _to_cpu, :cpu
        end
    end
end

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
# Helper: Gemeinsame symmetrische Farbskala (wie im FNO)
# ------------------------------------------------------------
"""
    common_clim(A, B; q=0.995)

Bestimmt gemeinsame Farbskala robust via Quantile (statt min/max).
Gibt `(-cmax, cmax)` für symmetrische Skala zurück.
Identisch mit der Implementierung in FNO plots_eval.jl.
"""
function common_clim(A::AbstractMatrix, B::AbstractMatrix; q::Real=0.995)
    all_vals = vcat(vec(A), vec(B))
    all_vals = filter(!isnan, all_vals)
    isempty(all_vals) && return (-1.0, 1.0)
    sorted = sort(abs.(all_vals))
    idx = min(length(sorted), max(1, round(Int, q * length(sorted))))
    cmax = sorted[idx]
    cmax = cmax > 0 ? cmax : 1.0
    return (-cmax, cmax)
end

# ------------------------------------------------------------
# Velocity-Helpers (identisch zu FNO GridFDUtils)
# Vx =  ∂ψ/∂z,  Vz = -∂ψ/∂x
# ------------------------------------------------------------

function _ddx(A::AbstractMatrix, dx::Real)
    nx, nz = size(A)
    out = similar(A, Float64)
    for j in 1:nz
        out[1, j] = (A[2, j] - A[1, j]) / dx
        for i in 2:nx-1
            out[i, j] = (A[i+1, j] - A[i-1, j]) / (2dx)
        end
        out[nx, j] = (A[nx, j] - A[nx-1, j]) / dx
    end
    return out
end

function _ddz(A::AbstractMatrix, dz::Real)
    nx, nz = size(A)
    out = similar(A, Float64)
    for i in 1:nx
        out[i, 1] = (A[i, 2] - A[i, 1]) / dz
        for j in 2:nz-1
            out[i, j] = (A[i, j+1] - A[i, j-1]) / (2dz)
        end
        out[i, nz] = (A[i, nz] - A[i, nz-1]) / dz
    end
    return out
end

"""
    velocity_from_streamfunction(ψ, dx, dz)

Vx = ∂ψ/∂z,  Vz = -∂ψ/∂x — identisch zur FNO-Konvention (GridFDUtils).
"""
function velocity_from_streamfunction(ψ::AbstractMatrix, dx::Real, dz::Real)
    Vx =  _ddz(ψ, dz)
    Vz = -_ddx(ψ, dx)
    return Vx, Vz
end

function _divergence(Vx::AbstractMatrix, Vz::AbstractMatrix, dx::Real, dz::Real)
    return _ddx(Vx, dx) .+ _ddz(Vz, dz)
end

"""
    interior_mask(nx, nz; width=2)

Boolesche Maske: true im Inneren (Randbereich ausgeschlossen).
Identisch zu FNO GridFDUtils.interior_mask.
"""
function interior_mask(nx::Int, nz::Int; width::Int = 2)
    mask = trues(nx, nz)
    if width > 0
        mask[1:width, :]         .= false
        mask[end-width+1:end, :] .= false
        mask[:, 1:width]         .= false
        mask[:, end-width+1:end] .= false
    end
    return mask
end

# ------------------------------------------------------------
# Zusammenfassung (wie FNO print_eval_summary)
# ------------------------------------------------------------

"""
    print_eval_summary(results; n_show=5)

Gibt eine kompakte Zusammenfassung aus: Overall-Stats + Best/Worst Samples.
Analog zu FNO EvalPsi.print_eval_summary.
"""
function print_eval_summary(results::Vector{<:NamedTuple}; n_show::Int = 5)
    n = length(results)
    n == 0 && return @warn "Keine Ergebnisse"

    rel_l2s     = [Float64(r.rel_l2_psi)  for r in results]
    mses        = [Float64(r.mse_psi)      for r in results]
    v_rel_l2s = filter(!isnan, [Float64(r.v_rel_l2) for r in results])
    div_rmss  = filter(!isnan, [Float64(r.div_rms)  for r in results])
    eps01s    = [Float64(r.eps01_psi) for r in results]
    eps05s    = [Float64(r.eps05_psi) for r in results]
    eps10s    = [Float64(r.eps10_psi) for r in results]

    println("=" ^ 60)
    @printf("Evaluation Summary (%d Samples)\n", n)
    println("=" ^ 60)
    @printf("  ψ  rel_l2:    mean=%.4f  std=%.4f  min=%.4f  max=%.4f\n",
            mean(rel_l2s), std(rel_l2s), minimum(rel_l2s), maximum(rel_l2s))
    @printf("  ψ  MSE:       mean=%.4e  max=%.4e\n", mean(mses), maximum(mses))
    if !isempty(v_rel_l2s)
        @printf("  v  rel_l2:    mean=%.4f  std=%.4f\n", mean(v_rel_l2s), std(v_rel_l2s))
    end
    if !isempty(div_rmss)
        @printf("  div RMS:      mean=%.4e\n", mean(div_rmss))
    end
    @printf("  ε > 1%%:       mean=%.2f%%\n",  100 * mean(eps01s))
    @printf("  ε > 5%%:       mean=%.2f%%\n",  100 * mean(eps05s))
    @printf("  ε > 10%%:      mean=%.2f%%\n",  100 * mean(eps10s))

    sorted = sortperm(rel_l2s)
    n_s = min(n_show, n)

    println("\n  Best $n_s:")
    for idx in sorted[1:n_s]
        r = results[idx]
        @printf("    #%d  rel_l2=%.4f  n_cryst=%d  %s\n",
                r.sample_idx, r.rel_l2_psi, r.n_crystals, basename(string(r.filepath)))
    end

    println("  Worst $n_s:")
    for idx in sorted[end-n_s+1:end]
        r = results[idx]
        @printf("    #%d  rel_l2=%.4f  n_cryst=%d  %s\n",
                r.sample_idx, r.rel_l2_psi, r.n_crystals, basename(string(r.filepath)))
    end
    println("=" ^ 60)
end

# ------------------------------------------------------------
# Datensatz-Evaluierung
# ------------------------------------------------------------
"""
    evaluate_dataset(; data_dir, model_path, out_prefix="eval_psi_dataset",
                      save_plots=false, plot_dir="eval_plots",
                      denorm_psi=false, use_gpu=nothing)

- Evaluierung von ψ (MSE, relL2)
- Zusätzlich: ψ_x, ψ_z (ebenfalls MSE, relL2)
- Gruppierung nach Kristallanzahl
- Plots in physikalischen Koordinaten (z.B. km, aus x_vec_1D/z_vec_1D)
- Wenn `save_plots = true`:
    - pro Sample: ψ-Plot (3 Panels)
    - pro Sample: ψ_x / ψ_z-Plot (2×3 Panels)
- `use_gpu` wie im Training:
    - nothing → auto
    - true    → GPU (falls möglich, sonst CPU)
    - false   → CPU
"""
function evaluate_dataset(; data_dir::String,
                          model_path::String,
                          out_prefix::String = "eval_psi_dataset",
                          save_plots::Bool = false,
                          plot_dir::String = "eval_plots",
                          denorm_psi::Bool = false,
                          use_gpu::Union{Nothing,Bool} = nothing)

    # Gerät auswählen (für Modell + Forward-Pass)
    move, devsym = _select_mover(use_gpu)
    @info "Evaluierung auf Gerät: $(devsym == :gpu ? "GPU (CUDA)" : "CPU")"

    if devsym == :gpu && _HAS_CUDA
        try
            CUDA.allowscalar(false)
        catch
        end
    end

    @info "Lade Modell aus $model_path"
    model_bson = BSON.load(model_path)
    model_cpu = model_bson[:model]
    model = fmap(move, model_cpu)  # Modell ggf. auf GPU

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

    # Alle Einzelergebnisse (flach, für Summary + Per-Sample CSV)
    all_results = NamedTuple[]

    # pro Kristallanzahl: Vector von NamedTuples mit allen Metriken
    errors_by_n = Dict{Int, Vector{NamedTuple}}()

    for (i, filepath) in enumerate(ds.files)
        # --- Daten als (nx,nz,Channels) laden ---
        x, y_true = get_sample(ds, i)
        nx, nz, ch = size(x)               # ch = 1 oder 2 (Maske + SDF)
        x_batch = reshape(x, nx, nz, ch, 1)

        # Auf Ausführungsgerät verschieben
        x_dev = move(x_batch)

        # Vorhersage (normalisierter Raum) auf Gerät
        y_pred_batch_dev = model(x_dev)

        # zurück auf CPU, Shape (nx,nz)
        y_pred_norm = dropdims(Array(y_pred_batch_dev)[:, :, :, 1], dims=3)
        y_true_norm = dropdims(Array(y_true), dims=3)

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

        # Pixel-Fehlerraten ε_α für ψ (relativer Fehler pro Pixel)
        rel_err_psi = abs.(y_pred_eval .- y_true_eval) ./ (abs.(y_true_eval) .+ eps())
        eps01_psi = mean(rel_err_psi .> 0.01)   # > 1 %
        eps05_psi = mean(rel_err_psi .> 0.05)   # > 5 %
        eps10_psi = mean(rel_err_psi .> 0.10)   # > 10 %

        # --- Velocities (identisch zu FNO: Vx = ∂ψ/∂z, Vz = -∂ψ/∂x) ---
        if length(xcoords) > 1 && length(zcoords) > 1
            dx = (xcoords[2] - xcoords[1]) * 1000.0   # km → m
            dz = (zcoords[2] - zcoords[1]) * 1000.0   # km → m
        else
            dx = 1.0
            dz = 1.0
        end

        Vx_true, Vz_true = velocity_from_streamfunction(Float64.(y_true_eval), dx, dz)
        Vx_pred, Vz_pred = velocity_from_streamfunction(Float64.(y_pred_eval), dx, dz)

        speed_true = sqrt.(Vx_true .^ 2 .+ Vz_true .^ 2)
        speed_pred = sqrt.(Vx_pred .^ 2 .+ Vz_pred .^ 2)

        # v_rel_l2 auf Innenpunkten (wie FNO eval.jl)
        mask      = interior_mask(nx, nz; width=2)
        denom_v   = sqrt(mean(speed_true[mask] .^ 2))
        v_rel_l2  = denom_v > 0 ?
                    sqrt(mean((speed_pred[mask] .- speed_true[mask]) .^ 2)) / denom_v : NaN

        # Divergenz (wie FNO)
        div_pred = _divergence(Vx_pred, Vz_pred, dx, dz)
        div_rms  = sqrt(mean(div_pred[mask] .^ 2))

        stat = (; mse_psi, rel_l2_psi, v_rel_l2, div_rms,
                 eps01_psi, eps05_psi, eps10_psi)

        if !haskey(errors_by_n, n)
            errors_by_n[n] = NamedTuple[]
        end
        push!(errors_by_n[n], stat)

        # Flache Ergebnisliste für Summary + Per-Sample CSV
        push!(all_results, (; stat..., sample_idx=i, filepath=filepath, n_crystals=n))

        # Progress-Logging (wie FNO: alle ~10%)
        if i % max(1, n_samples ÷ 10) == 0 || i == n_samples
            @info @sprintf("  [%d/%d] n=%d  rel_l2=%.4f  mse=%.4e  ε>1%%=%.1f%%",
                           i, n_samples, n, rel_l2_psi, mse_psi, 100 * eps01_psi)
        end

        # --- Optional: Plots speichern ---
        if save_plots
            subdir = joinpath(plot_dir, @sprintf("n_%02d", n))
            mkpath(subdir)

            # ---------- ψ-Plot ----------
            filename_psi = joinpath(subdir, @sprintf("sample_%04d_rel%.4f_psi.png", i, rel_l2_psi))

            fig = Figure(resolution = (1500, 450))

            # Robuste symmetrische Farbskala (wie FNO)
            cl = common_clim(y_true_eval, y_pred_eval)

            diff = y_pred_eval .- y_true_eval
            max_abs_diff = maximum(abs.(diff))

            gl1 = fig[1, 1] = GridLayout()
            gl2 = fig[1, 2] = GridLayout()
            gl3 = fig[1, 3] = GridLayout()

            ax1 = Axis(gl1[1, 1],
                       title  = @sprintf("ψ true | n=%d | idx=%d | rel_l2=%.4f", n, i, rel_l2_psi),
                       xlabel = "x (km)",
                       ylabel = "z (km)",
                       aspect = DataAspect())
            hm1 = heatmap!(ax1, xcoords, zcoords, y_true_eval;
                           colorrange = cl, colormap = :RdBu)
            plot_crystal_outlines!(ax1, centers_2D, radii)
            Colorbar(gl1[1, 2], hm1, label = "ψ_true")

            ax2 = Axis(gl2[1, 1],
                       title  = "ψ pred (U-Net)",
                       xlabel = "x (km)",
                       ylabel = "z (km)",
                       aspect = DataAspect())
            hm2 = heatmap!(ax2, xcoords, zcoords, y_pred_eval;
                           colorrange = cl, colormap = :RdBu)
            plot_crystal_outlines!(ax2, centers_2D, radii)
            Colorbar(gl2[1, 2], hm2, label = "ψ_pred")

            ax3 = Axis(gl3[1, 1],
                       title  = "Δψ = ψ_pred − ψ_true",
                       xlabel = "x (km)",
                       ylabel = "z (km)",
                       aspect = DataAspect())
            hm3 = heatmap!(ax3, xcoords, zcoords, diff;
                           colorrange = (-max_abs_diff, max_abs_diff), colormap = :RdBu)
            plot_crystal_outlines!(ax3, centers_2D, radii)
            Colorbar(gl3[1, 2], hm3, label = "Fehler")

            save(filename_psi, fig)
            @info "ψ-Plot gespeichert: $filename_psi"

            # ---------- Velocity Comparison (wie FNO plot_velocity_comparison) ----------
            filename_vel = joinpath(subdir, @sprintf("sample_%04d_rel%.4f_vel.png", i, rel_l2_psi))

            fig_vel = Figure(resolution = (1500, 450))

            vmax = max(maximum(filter(!isnan, vec(speed_true))),
                       maximum(filter(!isnan, vec(speed_pred))))
            vmax = vmax > 0 ? vmax : 1.0

            speed_err    = speed_pred .- speed_true
            speed_emax   = maximum(abs.(filter(!isnan, vec(speed_err))))
            speed_emax   = speed_emax > 0 ? speed_emax : 1.0

            gv1 = fig_vel[1, 1] = GridLayout()
            gv2 = fig_vel[1, 2] = GridLayout()
            gv3 = fig_vel[1, 3] = GridLayout()

            av1 = Axis(gv1[1, 1],
                       title  = @sprintf("|v| true | n=%d | idx=%d | v_rel_l2=%.4f", n, i, v_rel_l2),
                       xlabel = "x (km)", ylabel = "z (km)", aspect = DataAspect())
            hv1 = heatmap!(av1, xcoords, zcoords, speed_true;
                           colorrange = (0, vmax), colormap = :viridis)
            plot_crystal_outlines!(av1, centers_2D, radii)
            Colorbar(gv1[1, 2], hv1, label = "|v| true")

            av2 = Axis(gv2[1, 1],
                       title  = "|v| pred (U-Net)",
                       xlabel = "x (km)", ylabel = "z (km)", aspect = DataAspect())
            hv2 = heatmap!(av2, xcoords, zcoords, speed_pred;
                           colorrange = (0, vmax), colormap = :viridis)
            plot_crystal_outlines!(av2, centers_2D, radii)
            Colorbar(gv2[1, 2], hv2, label = "|v| pred")

            av3 = Axis(gv3[1, 1],
                       title  = "Δ|v| = |v|_pred − |v|_true",
                       xlabel = "x (km)", ylabel = "z (km)", aspect = DataAspect())
            hv3 = heatmap!(av3, xcoords, zcoords, speed_err;
                           colorrange = (-speed_emax, speed_emax), colormap = :RdBu)
            plot_crystal_outlines!(av3, centers_2D, radii)
            Colorbar(gv3[1, 2], hv3, label = "Fehler")

            save(filename_vel, fig_vel)
            @info "Velocity-Plot gespeichert: $filename_vel"

            # ---------- Divergence Map (wie FNO plot_divergence_map) ----------
            filename_div = joinpath(subdir, @sprintf("sample_%04d_rel%.4f_div.png", i, rel_l2_psi))

            fig_div = Figure(resolution = (600, 500))

            dmax = maximum(abs.(filter(!isnan, vec(div_pred))))
            dmax = dmax > 0 ? dmax : 1.0

            div_title = @sprintf("Divergenz | n=%d | idx=%d | RMS_int=%.3e", n, i, div_rms)
            gd1 = fig_div[1, 1] = GridLayout()
            ad1 = Axis(gd1[1, 1],
                       title  = div_title,
                       xlabel = "x (km)", ylabel = "z (km)", aspect = DataAspect())
            hd1 = heatmap!(ad1, xcoords, zcoords, div_pred;
                           colorrange = (-dmax, dmax), colormap = :RdBu)
            plot_crystal_outlines!(ad1, centers_2D, radii)
            Colorbar(gd1[1, 2], hd1, label = "div v")

            save(filename_div, fig_div)
            @info "Divergenz-Plot gespeichert: $filename_div"
        end
    end

    @info "=== Auswertung nach Kristallanzahl ($(denorm_psi ? "physikalisches ψ" : "ψ_norm")) ==="

    sorted_groups = sort(collect(errors_by_n); by = first)

    # CSV schreiben (wie FNO eval_aggregated.csv)
    csv_path = out_prefix * "_by_n.csv"
    open(csv_path, "w") do io
        println(io, "n_crystals,N," *
           "psi_mse_mean,psi_mse_std,psi_rel_l2_mean,psi_rel_l2_std," *
           "v_rel_l2_mean,v_rel_l2_std,div_rms_mean,div_rms_std," *
           "eps01_psi_mean,eps01_psi_std,eps05_psi_mean,eps05_psi_std,eps10_psi_mean,eps10_psi_std")

        for (n, stats) in sorted_groups
            N = length(stats)

            mse_psi_vals  = [s.mse_psi    for s in stats]
            rel_psi_vals  = [s.rel_l2_psi for s in stats]
            v_rel_l2_vals = filter(!isnan, [s.v_rel_l2 for s in stats])
            div_rms_vals  = filter(!isnan, [s.div_rms  for s in stats])
            eps01_vals    = [s.eps01_psi  for s in stats]
            eps05_vals    = [s.eps05_psi  for s in stats]
            eps10_vals    = [s.eps10_psi  for s in stats]

            v_mean  = isempty(v_rel_l2_vals) ? NaN : mean(v_rel_l2_vals)
            v_std   = isempty(v_rel_l2_vals) ? NaN : (length(v_rel_l2_vals) > 1 ? std(v_rel_l2_vals) : 0.0)
            dr_mean = isempty(div_rms_vals)  ? NaN : mean(div_rms_vals)
            dr_std  = isempty(div_rms_vals)  ? NaN : (length(div_rms_vals)  > 1 ? std(div_rms_vals)  : 0.0)

            println(io, @sprintf(
                "%d,%d,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e",
                n, N,
                mean(mse_psi_vals), (N > 1 ? std(mse_psi_vals) : 0.0),
                mean(rel_psi_vals), (N > 1 ? std(rel_psi_vals) : 0.0),
                v_mean, v_std, dr_mean, dr_std,
                mean(eps01_vals), (N > 1 ? std(eps01_vals) : 0.0),
                mean(eps05_vals), (N > 1 ? std(eps05_vals) : 0.0),
                mean(eps10_vals), (N > 1 ? std(eps10_vals) : 0.0)
            ))
        end
    end
    @info "Aggregierte Metriken gespeichert: $csv_path"

    # Per-Sample CSV (wie FNO eval_results.csv)
    sample_csv = out_prefix * "_samples.csv"
    open(sample_csv, "w") do io
        println(io, "sample_idx,filepath,n_crystals," *
                    "rel_l2_psi,mse_psi,v_rel_l2,div_rms," *
                    "eps01_psi,eps05_psi,eps10_psi")
        for r in all_results
            @printf(io, "%d,%s,%d,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e,%.8e\n",
                    r.sample_idx, r.filepath, r.n_crystals,
                    r.rel_l2_psi, r.mse_psi, r.v_rel_l2, r.div_rms,
                    r.eps01_psi, r.eps05_psi, r.eps10_psi)
        end
    end
    @info "Per-Sample Metriken gespeichert: $sample_csv"

    # Log-Ausgabe nach Kristallanzahl (wie FNO)
    for (n, stats) in sorted_groups
        N = length(stats)

        rel_psi_vals  = [s.rel_l2_psi for s in stats]
        mse_psi_vals  = [s.mse_psi    for s in stats]
        v_rel_l2_vals = filter(!isnan, [s.v_rel_l2 for s in stats])

        v_str = isempty(v_rel_l2_vals) ? "" :
                @sprintf(" | v_rel_l2 = %.4e", mean(v_rel_l2_vals))

        @info @sprintf(
            "n_crystals=%d | N=%3d | ψ: MSE = %.4e  rel_l2 = %.4e%s",
            n, N, mean(mse_psi_vals), mean(rel_psi_vals), v_str)
    end

    # Gesamt-Summary (wie FNO print_eval_summary)
    print_eval_summary(all_results)

    return errors_by_n, all_results
end

end # module
