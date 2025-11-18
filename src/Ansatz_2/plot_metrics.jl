using CSV
using DataFrames
using CairoMakie
using Printf

# ---------------------------------------------------------
# CSV laden
# ---------------------------------------------------------
df = CSV.read("eval_psi_by_n.csv", DataFrame)

println("Spalten im DataFrame:")
println(names(df))

# ---------------------------------------------------------
# Hilfsfunktion: Plotten + Speichern
# - log10_y = true  → y-Werte werden als log10(y) geplottet
# - ytickformat:
#     * linear: wissenschaftliche Notation (1.2e-25)
#     * log10 : normale Zahlen (z.B. -25.3)
# ---------------------------------------------------------
function plot_metric(df::DataFrame, xcol::Symbol, ycol::Symbol, yerrcol::Symbol,
                     ylabel::String, title::String, outfile::String;
                     log10_y::Bool=false)

    ax_ytickformat = log10_y ?
        (xs -> [@sprintf("%.1f", x) for x in xs]) :
        (xs -> [@sprintf("%.1e", x) for x in xs])

    fig = Figure(resolution = (700, 450))
    ax = Axis(fig[1, 1];
        xlabel = "Kristallanzahl",
        ylabel = ylabel,
        title  = title,
        ytickformat = ax_ytickformat,
    )

    x    = df[!, xcol]
    y    = df[!, ycol]
    yerr = df[!, yerrcol]

    if log10_y
        # y → log10(y), Fehler via einfacher Fehlerfortpflanzung
        y_log    = similar(y)
        yerr_log = similar(y)
        for i in eachindex(y)
            yi = y[i]
            σ  = yerr[i]
            if yi <= 0
                error("log10 nicht definiert für nicht-positive Werte in $(ycol) bei Index $i (Wert = $yi).")
            end
            y_log[i] = log10(yi)
            # σ_log10 ≈ σ / (y * ln(10))
            yerr_log[i] = σ > 0 ? σ / (yi * log(10)) : 0.0
        end
        y    = y_log
        yerr = yerr_log
    end

    errorbars!(ax, x, y, yerr)
    scatter!(ax, x, y)
    lines!(ax, x, y)

    save(outfile, fig)
    println("✔ gespeichert: $outfile")
end

# ---------------------------------------------------------
# 1) ψ: MSE im log10-Raum, relL2 linear
# ---------------------------------------------------------
plot_metric(df, :n_crystals, :mse_psi_mean,  :mse_psi_std,
            "log₁₀(MSE(ψ))",
            "MSE(ψ) vs. Kristallanzahl (log₁₀)",
            "mse_psi_vs_crystals_log10.png";
            log10_y = true)

plot_metric(df, :n_crystals, :relL2_psi_mean, :relL2_psi_std,
            "relative L2-Norm(ψ)",
            "rel. L2(ψ) vs. Kristallanzahl",
            "relL2_psi_vs_crystals.png")

# ---------------------------------------------------------
# 2) ψₓ: MSE log10, relL2 linear
# ---------------------------------------------------------
plot_metric(df, :n_crystals, :mse_psix_mean,  :mse_psix_std,
            "log₁₀(MSE(ψₓ))",
            "MSE(ψₓ) vs. Kristallanzahl (log₁₀)",
            "mse_psix_vs_crystals_log10.png";
            log10_y = true)

plot_metric(df, :n_crystals, :relL2_psix_mean, :relL2_psix_std,
            "relative L2-Norm(ψₓ)",
            "rel. L2(ψₓ) vs. Kristallanzahl",
            "relL2_psix_vs_crystals.png")

# ---------------------------------------------------------
# 3) ψ_z: MSE log10, relL2 linear
# ---------------------------------------------------------
plot_metric(df, :n_crystals, :mse_psiz_mean,  :mse_psiz_std,
            "log₁₀(MSE(ψ_z))",
            "MSE(ψ_z) vs. Kristallanzahl (log₁₀)",
            "mse_psiz_vs_crystals_log10.png";
            log10_y = true)

plot_metric(df, :n_crystals, :relL2_psiz_mean, :relL2_psiz_std,
            "relative L2-Norm(ψ_z)",
            "rel. L2(ψ_z) vs. Kristallanzahl",
            "relL2_psiz_vs_crystals.png")

println("Fertig – alle 6 Plots erzeugt.")
