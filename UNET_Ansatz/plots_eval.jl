# Datei: plots_eval.jl
# Plotting-Modul für ψ-U-Net Evaluation
#
# Funktion                    Was sie tut
# plot_metrics_vs_crystals    Metriken vs. Kristallanzahl (4-Panel, aus *_by_n.csv)
# plot_training_history       Training-CSV → Loss-Kurven (MSE, rel_l2)

module UNetPlotsEval

using Plots
using Plots.Measures
using Printf

export plot_metrics_vs_crystals, plot_training_history

# =============================================================================
# Helper: CSV lesen
# =============================================================================

function _read_csv(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && error("CSV ist leer: $path")
    header = Symbol.(strip.(split(lines[1], ",")))
    data   = Dict{Symbol, Vector{Float64}}(h => Float64[] for h in header)
    for line in lines[2:end]
        isempty(strip(line)) && continue
        vals = strip.(split(line, ","))
        for (j, h) in enumerate(header)
            v = tryparse(Float64, vals[j])
            push!(data[h], v === nothing ? NaN : v)
        end
    end
    return data
end

# =============================================================================
# P1 – Metriken vs. Kristallanzahl (4-Panel)
# =============================================================================

"""
    plot_metrics_vs_crystals(agg_csv; out_path="metrics_vs_crystals.png")

Plottet aggregierte U-Net-Metriken vs. Kristallanzahl mit Fehlerbalken.
Liest die `*_by_n.csv` aus `evaluate_psi.jl`.

Spalten erwartet: n_crystals, N (oder n_samples),
  psi_rel_l2_mean/std, psi_mse_mean/std, v_rel_l2_mean/std
"""
function plot_metrics_vs_crystals(agg_csv::AbstractString;
                                  out_path::AbstractString = joinpath(dirname(agg_csv),
                                                                      "metrics_vs_crystals.png"))
    data = _read_csv(agg_csv)

    n_cryst = haskey(data, :n_crystals) ? data[:n_crystals] :
              collect(1.0:length(first(values(data))))

    # ── Panel 1: ψ rel_l2 ──
    p1 = plot(; title="ψ rel_l2 vs. Crystals", xlabel="n_crystals",
              ylabel="rel_l2 (mean ± std)", legend=:topleft, grid=true,
              left_margin=8mm, bottom_margin=6mm)
    if haskey(data, :psi_rel_l2_mean)
        y    = data[:psi_rel_l2_mean]
        yerr = haskey(data, :psi_rel_l2_std) ? data[:psi_rel_l2_std] : zeros(length(y))
        scatter!(p1, n_cryst, y; yerror=yerr, label="rel_l2", ms=6, color=:blue)
        plot!(p1, n_cryst, y; lw=1.5, color=:blue, label="")
    end

    # ── Panel 2: ψ MSE (log-Skala) ──
    p2 = plot(; title="ψ MSE vs. Crystals", xlabel="n_crystals",
              ylabel="MSE (mean)", yscale=:log10, legend=:topleft, grid=true,
              left_margin=8mm, bottom_margin=6mm)
    if haskey(data, :psi_mse_mean)
        y = data[:psi_mse_mean]
        scatter!(p2, n_cryst, y; label="MSE", ms=6, color=:red)
        plot!(p2, n_cryst, y; lw=1.5, color=:red, label="")
        if haskey(data, :psi_mse_std)
            y_lo = max.(y .- data[:psi_mse_std], y .* 0.1)
            y_hi = y .+ data[:psi_mse_std]
            plot!(p2, n_cryst, y_lo; fillrange=y_hi, fillalpha=0.2,
                  color=:red, label="±1σ", lw=0)
        end
    end

    # ── Panel 3: v rel_l2 ──
    p3 = plot(; title="v rel_l2 vs. Crystals", xlabel="n_crystals",
              ylabel="v_rel_l2 (mean)", legend=:topleft, grid=true,
              left_margin=8mm, bottom_margin=6mm)
    if haskey(data, :v_rel_l2_mean)
        y     = data[:v_rel_l2_mean]
        valid = .!isnan.(y)
        if any(valid)
            yerr = haskey(data, :v_rel_l2_std) ? data[:v_rel_l2_std] : zeros(length(y))
            scatter!(p3, n_cryst[valid], y[valid]; yerror=yerr[valid],
                     label="v_rel_l2", ms=6, color=:green)
            plot!(p3, n_cryst[valid], y[valid]; lw=1.5, color=:green, label="")
        end
    end

    # ── Panel 4: Samples pro Gruppe ──
    # UNet-CSV: Spalte "N"; FNO-CSV: "n_samples" – beide unterstützt
    ns_key = haskey(data, :N) ? :N : (haskey(data, :n_samples) ? :n_samples : nothing)
    p4 = plot(; title="Samples per Group", xlabel="n_crystals",
              ylabel="n_samples", legend=false, grid=true,
              left_margin=8mm, bottom_margin=6mm)
    if ns_key !== nothing
        ns = data[ns_key]
        bar!(p4, n_cryst, ns; color=:gray, alpha=0.7)
        ylims!(p4, (0, maximum(ns) * 1.3))
    end

    fig = plot(p1, p2, p3, p4; layout=(2, 2), size=(1200, 900), margin=5mm)
    savefig(fig, out_path)
    @info "Metrics vs. Crystals gespeichert: $out_path"
    return fig
end

# =============================================================================
# P2 – Training History (2-Panel)
# =============================================================================

"""
    plot_training_history(history_csv; out_path="training_history.png")

Liest `training_history.csv` und erzeugt Loss-Kurven.
Panels: train/val MSE (log-Skala), val rel_l2 mit Best-Epoch-Markierung.
"""
function plot_training_history(csv_path::AbstractString;
                               out_path::AbstractString = joinpath(dirname(csv_path),
                                                                   "training_history.png"))
    data   = _read_csv(csv_path)
    epochs = haskey(data, :epoch) ? data[:epoch] :
             collect(1.0:length(first(values(data))))

    # ── Panel 1: MSE (log-Skala) ──
    p1 = plot(; title="Loss (MSE)", xlabel="Epoch", ylabel="MSE",
              yscale=:log10, legend=:topright, grid=true,
              left_margin=8mm, bottom_margin=6mm)
    if haskey(data, :train_mse)
        plot!(p1, epochs, data[:train_mse]; label="Train MSE", lw=2, color=:steelblue)
    end
    if haskey(data, :val_mse)
        plot!(p1, epochs, data[:val_mse]; label="Val MSE", lw=2, ls=:dash, color=:darkorange)
    end

    # ── Panel 2: rel_l2 mit Best-Markierung ──
    p2 = plot(; title="Relative L2 Error (Validation)", xlabel="Epoch", ylabel="rel. L2",
              legend=:topright, grid=true,
              left_margin=8mm, bottom_margin=6mm)
    if haskey(data, :val_rel_l2)
        plot!(p2, epochs, data[:val_rel_l2]; label="Val rel. L2", lw=2, color=:crimson)
        best_idx = argmin(data[:val_rel_l2])
        best_ep  = epochs[best_idx]
        best_val = data[:val_rel_l2][best_idx]
        scatter!(p2, [best_ep], [best_val];
                 label="Best (ep $(Int(best_ep)), $(round(best_val; sigdigits=4)))",
                 marker=:star5, markersize=8, color=:gold, markerstrokecolor=:black)
    end

    fig = plot(p1, p2; layout=(1, 2), size=(1100, 450),
               plot_title=basename(dirname(csv_path)))
    savefig(fig, out_path)
    @info "Training-History gespeichert: $out_path"
    return fig
end

end # module UNetPlotsEval
