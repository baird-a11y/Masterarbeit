# ============================================================
# plot_frequency_spectrum.jl
# Frequency spectrum analysis of the FNO residual (ψ_pred − ψ_true)
#
# Motivation: FNO truncates Fourier modes at modes_x = modes_z = 16
# on a 256×256 grid. High-frequency structures (k > 16) are
# structurally lost. This script shows whether the error is
# concentrated in high frequencies — and how this changes across
# the 4 experiments.
#
# Usage:
#   julia plot_frequency_spectrum.jl
#
# Packages: JLD2, FFTW, CairoMakie, Statistics
# ============================================================

using JLD2, FFTW, Statistics, CairoMakie, Printf

const SCRIPT_DIR = @__DIR__
const RES        = dirname(SCRIPT_DIR)
const OUT        = SCRIPT_DIR

const FNO_MODES = 16
const GRID_N    = 256

const C = Makie.wong_colors()

# Exp-1: best config = lr=5e-4, bs=16 → eval_output_exp1_16_4
const EXPERIMENTS = [
    (label = "Exp-1 (n=1)",
     dir   = joinpath(RES, "FNO_Ergebnisse", "One_Crystal",
                      "eval_output_exp1_16_4", "predictions")),
    (label = "Exp-2 (n=1–10)",
     dir   = joinpath(RES, "FNO_Ergebnisse", "Experiment_2",
                      "eval_output_exp2_16", "predictions")),
    (label = "Exp-3 (n=1–25)",
     dir   = joinpath(RES, "FNO_Ergebnisse", "Experiment_3",
                      "eval_output_exp3_16", "predictions")),
    (label = "Exp-4 (crystal size)",
     dir   = joinpath(RES, "FNO_Ergebnisse", "Experiment_4",
                      "eval_output_exp4_16", "predictions")),
]

# ─── Helper functions ─────────────────────────────────────────

function load_predictions(pred_dir::AbstractString)
    files = sort(filter(f -> startswith(basename(f), "pred_") && endswith(f, ".jld2"),
                        readdir(pred_dir; join=true)))
    isempty(files) && error("No pred_*.jld2 found in $pred_dir")

    out = NamedTuple[]
    for f in files
        d = load(f)
        m = d["metrics"]
        nc = haskey(m, :n_crystals) ? Int(m.n_crystals) : -1
        ψ_true = Float64.(d["ψ_true_norm"])
        ψ_pred = Float64.(d["ψ_pred_norm"])
        push!(out, (ψ_true=ψ_true, ψ_pred=ψ_pred, n_crystals=nc))
    end
    @info "Loaded: $(length(out)) samples from $(basename(dirname(pred_dir)))"
    return out
end

# Normalized 2D power spectrum, fftshift-centered (zero freq at center)
function power_spectrum_2d(field::AbstractMatrix{<:Real})
    N = size(field, 1)
    F = fft(field)
    P = abs2.(F) ./ N^2
    return fftshift(P)
end

# Radially averaged 1D spectrum of a centered NxN power spectrum.
# Returns (k_values, P_radial) with k in grid frequency units (0..N/2).
function radial_average(P2d::AbstractMatrix{<:Real})
    N = size(P2d, 1)
    center = N ÷ 2 + 1

    k_max  = N ÷ 2
    P_sum  = zeros(k_max + 1)
    counts = zeros(Int, k_max + 1)

    for j in 1:N, i in 1:N
        kx = i - center
        kz = j - center
        k  = round(Int, sqrt(kx^2 + kz^2))
        if 0 ≤ k ≤ k_max
            P_sum[k+1]  += P2d[i, j]
            counts[k+1] += 1
        end
    end

    k_vals   = collect(0:k_max)
    P_radial = P_sum ./ max.(counts, 1)
    return k_vals, P_radial
end

function spectral_analysis(samples;
                           n_crystals_min::Union{Int,Nothing} = nothing,
                           n_crystals_max::Union{Int,Nothing} = nothing)
    filtered = if n_crystals_min !== nothing || n_crystals_max !== nothing
        filter(s -> begin
            nc = s.n_crystals
            (n_crystals_min === nothing || nc >= n_crystals_min) &&
            (n_crystals_max === nothing || nc <= n_crystals_max)
        end, samples)
    else
        samples
    end
    isempty(filtered) && error("No samples remaining after filter")

    N = size(first(filtered).ψ_true, 1)
    P2d_true_sum = zeros(N, N)
    P2d_res_sum  = zeros(N, N)

    for s in filtered
        residual = s.ψ_pred .- s.ψ_true
        P2d_true_sum .+= power_spectrum_2d(s.ψ_true)
        P2d_res_sum  .+= power_spectrum_2d(residual)
    end

    n = length(filtered)
    P2d_true = P2d_true_sum ./ n
    P2d_res  = P2d_res_sum  ./ n

    k_true, P_true_radial = radial_average(P2d_true)
    k_res,  P_res_radial  = radial_average(P2d_res)

    return (
        P2d_true = P2d_true,
        P2d_res  = P2d_res,
        k        = k_true,
        P_true   = P_true_radial,
        P_res    = P_res_radial,
        n        = n,
    )
end

# ─── Load data ────────────────────────────────────────────────

@info "Loading predictions for all 4 experiments..."
all_data = map(EXPERIMENTS) do exp
    samples = load_predictions(exp.dir)
    spec    = spectral_analysis(samples)
    (label=exp.label, spec=spec, samples=samples)
end

# ─── Plot 1: 2D power spectra (all 4 experiments) ─────────────
#
# 2 rows × 4 columns:
#   Row 1: Ground truth (ψ_true)
#   Row 2: Residual (ψ_pred − ψ_true)
# Zoomed to k ∈ [−64, 64] so the FNO cutoff box is visible.

@info "Creating Plot 1: 2D power spectra..."

let
    fig = Figure(size=(1400, 720), fontsize=12)

    N    = GRID_N
    # Zoom: only show k in [-K_SHOW, K_SHOW]
    K_SHOW = 64
    k_ax   = range(-N÷2, N÷2; length=N)
    # Index range for zoomed view
    center = N ÷ 2 + 1
    i_lo   = center - K_SHOW
    i_hi   = center + K_SHOW
    k_zoom = k_ax[i_lo:i_hi]

    m = Float64(FNO_MODES)

    # Shared color scale across all experiments
    all_true_vals = vcat([vec(log10.(max.(d.spec.P2d_true[i_lo:i_hi, i_lo:i_hi], 1e-12))) for d in all_data]...)
    clim_true = (quantile(all_true_vals, 0.01), quantile(all_true_vals, 0.999))

    all_res_vals = vcat([vec(log10.(max.(d.spec.P2d_res[i_lo:i_hi, i_lo:i_hi], 1e-12))) for d in all_data]...)
    clim_res = (quantile(all_res_vals, 0.01), quantile(all_res_vals, 0.999))

    for (col, d) in enumerate(all_data)
        P_true_log = log10.(max.(d.spec.P2d_true[i_lo:i_hi, i_lo:i_hi], 1e-12))
        P_res_log  = log10.(max.(d.spec.P2d_res[i_lo:i_hi,  i_lo:i_hi], 1e-12))

        # Row 1: Ground truth
        ax_true = Axis(fig[1, col];
            title         = d.label,
            xlabel        = "kₓ",
            ylabel        = col == 1 ? "k_z" : "",
            aspect        = DataAspect(),
            yticksvisible = col == 1,
            yticklabelsvisible = col == 1)

        heatmap!(ax_true, k_zoom, k_zoom, P_true_log;
                 colormap=:inferno, colorrange=clim_true)

        lines!(ax_true, [-m, m, m, -m, -m], [-m, -m, m, m, -m];
               color=:cyan, linewidth=2.0, linestyle=:dash)

        # Row 2: Residual
        ax_res = Axis(fig[2, col];
            xlabel        = "kₓ",
            ylabel        = col == 1 ? "k_z" : "",
            aspect        = DataAspect(),
            yticksvisible = col == 1,
            yticklabelsvisible = col == 1)

        heatmap!(ax_res, k_zoom, k_zoom, P_res_log;
                 colormap=:inferno, colorrange=clim_res)

        lines!(ax_res, [-m, m, m, -m, -m], [-m, -m, m, m, -m];
               color=:cyan, linewidth=2.0, linestyle=:dash)
    end

    # Row labels
    Label(fig[1, 0], "Ground truth (ψ_true)";
          rotation=π/2, tellheight=false, fontsize=12, font=:bold)
    Label(fig[2, 0], "Residual (ψ_pred − ψ_true)";
          rotation=π/2, tellheight=false, fontsize=12, font=:bold)

    # Colorbars
    Colorbar(fig[1, 5]; colormap=:inferno, colorrange=clim_true,
             label="log₁₀(power)", width=15)
    Colorbar(fig[2, 5]; colormap=:inferno, colorrange=clim_res,
             label="log₁₀(power)", width=15)

    Label(fig[3, 1:4],
          "Dashed cyan box: FNO truncation boundary (k = ±$(FNO_MODES))";
          fontsize=11, halign=:left)

    Label(fig[0, 1:4],
          "2D Power Spectrum: Ground Truth and Residual (all experiments, bs=16, zoomed to k ∈ [−$(K_SHOW), $(K_SHOW)])";
          fontsize=13, font=:bold)

    out = joinpath(OUT, "fig_freq_spectrum_2d.pdf")
    save(out, fig)
    @info "Saved: $out"
    save(replace(out, ".pdf" => ".png"), fig; px_per_unit=2)
end

# ─── Plot 2: Radially averaged 1D spectrum (all 4 experiments) ─

@info "Creating Plot 2: Radial 1D spectrum..."

let
    fig = Figure(size=(900, 500), fontsize=13)

    ax = Axis(fig[1,1];
        title        = "Radially Averaged Power Spectrum: Ground Truth vs. Residual",
        xlabel       = "Wavenumber k (grid units)",
        ylabel       = "Mean power",
        yscale       = log10,
        xticks       = 0:4:64,
        xgridvisible = true,
        ygridvisible = true)

    k_max_show = 64

    for (i, d) in enumerate(all_data)
        mask = d.spec.k .≤ k_max_show

        lines!(ax, d.spec.k[mask], d.spec.P_true[mask];
               color=C[i], linewidth=2.5, linestyle=:solid,
               label=d.label * " (ground truth)")

        lines!(ax, d.spec.k[mask], d.spec.P_res[mask];
               color=(C[i], 0.5), linewidth=1.5, linestyle=:dash,
               label=d.label * " (residual)")
    end

    vlines!(ax, [FNO_MODES];
            color=:black, linewidth=2.0, linestyle=:dash,
            label="FNO truncation (k=$(FNO_MODES))")
    vspan!(ax, FNO_MODES, k_max_show;
           color=(:gray, 0.12), label="High-frequency region (not learned)")

    axislegend(ax; position=:lb, framevisible=true, labelsize=10, nbanks=2)

    Label(fig[0, 1],
          "Radially Averaged Power Spectrum – All FNO Experiments (bs=16)";
          fontsize=14, font=:bold)

    out = joinpath(OUT, "fig_freq_spectrum_radial.pdf")
    save(out, fig)
    @info "Saved: $out"
    save(replace(out, ".pdf" => ".png"), fig; px_per_unit=2)
end

# ─── Plot 3: Relative error fraction P_res(k) / P_true(k) ────

@info "Creating Plot 3: Relative error fraction..."

let
    fig = Figure(size=(900, 480), fontsize=13)

    ax = Axis(fig[1,1];
        title        = "Relative Error Fraction per Frequency",
        xlabel       = "Wavenumber k (grid units)",
        ylabel       = "P_residual(k) / P_ground truth(k)",
        yscale       = log10,
        xticks       = 0:4:64,
        xgridvisible = true,
        ygridvisible = true)

    k_max_show = 64

    for (i, d) in enumerate(all_data)
        mask  = d.spec.k .≤ k_max_show
        ratio = d.spec.P_res[mask] ./ max.(d.spec.P_true[mask], 1e-20)
        lines!(ax, d.spec.k[mask], ratio;
               color=C[i], linewidth=2.2, label=d.label)
    end

    vlines!(ax, [FNO_MODES];
            color=:black, linewidth=2.0, linestyle=:dash,
            label="FNO truncation (k=$(FNO_MODES))")
    vspan!(ax, FNO_MODES, k_max_show;
           color=(:gray, 0.12), label="High-frequency region (not learned)")

    axislegend(ax; position=:rt, framevisible=true, labelsize=11)

    Label(fig[0, 1],
          "Relative Error Fraction P_residual / P_ground truth – All FNO Experiments";
          fontsize=14, font=:bold)

    out = joinpath(OUT, "fig_freq_error_ratio.pdf")
    save(out, fig)
    @info "Saved: $out"
    save(replace(out, ".pdf" => ".png"), fig; px_per_unit=2)
end

# ─── Plot 4: Exp-3 split by crystal count ────────────────────

@info "Creating Plot 4: Exp-3 by crystal count..."

let
    samples_exp3 = all_data[3].samples

    groups = [
        (label="n = 1–5",   kw=(n_crystals_min=1,  n_crystals_max=5)),
        (label="n = 6–10",  kw=(n_crystals_min=6,  n_crystals_max=10)),
        (label="n = 11–15", kw=(n_crystals_min=11, n_crystals_max=15)),
        (label="n = 16–25", kw=(n_crystals_min=16, n_crystals_max=25)),
    ]

    fig = Figure(size=(900, 480), fontsize=13)

    ax = Axis(fig[1,1];
        title        = "Exp-3: Relative Error Fraction by Crystal Count",
        xlabel       = "Wavenumber k (grid units)",
        ylabel       = "P_residual(k) / P_ground truth(k)",
        yscale       = log10,
        xticks       = 0:4:64,
        xgridvisible = true,
        ygridvisible = true)

    k_max_show = 64

    for (i, g) in enumerate(groups)
        spec  = spectral_analysis(samples_exp3; g.kw...)
        mask  = spec.k .≤ k_max_show
        ratio = spec.P_res[mask] ./ max.(spec.P_true[mask], 1e-20)
        lines!(ax, spec.k[mask], ratio;
               color=C[i], linewidth=2.2,
               label="$(g.label)  (n=$(spec.n) samples)")
    end

    vlines!(ax, [FNO_MODES];
            color=:black, linewidth=2.0, linestyle=:dash,
            label="FNO truncation (k=$(FNO_MODES))")
    vspan!(ax, FNO_MODES, k_max_show;
           color=(:gray, 0.12), label="High-frequency region (not learned)")

    axislegend(ax; position=:rt, framevisible=true, labelsize=11)

    Label(fig[0, 1],
          "Exp-3: Does more crystals mean more high-frequency error?";
          fontsize=13, font=:bold)

    out = joinpath(OUT, "fig_freq_exp3_by_crystals.pdf")
    save(out, fig)
    @info "Saved: $out"
    save(replace(out, ".pdf" => ".png"), fig; px_per_unit=2)
end

println("\n" * "=" ^ 60)
println("Frequency spectrum analysis complete. Outputs:")
println("  fig_freq_spectrum_2d       – 2D spectra (4 experiments, 2 rows)")
println("  fig_freq_spectrum_radial   – 1D radial spectrum (all 4 experiments)")
println("  fig_freq_error_ratio       – P_residual/P_true (all 4 experiments)")
println("  fig_freq_exp3_by_crystals  – Exp-3 split by crystal count")
println("=" ^ 60)
