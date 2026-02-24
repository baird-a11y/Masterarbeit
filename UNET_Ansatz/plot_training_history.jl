# Datei: plot_training_history.jl
# Erzeugt training_history.png aus training_history.csv für ψ-U-Net Experimente.
#
# Nutzung:
#   julia plot_training_history.jl                              # alle exp_* in aktuellem Ordner
#   julia plot_training_history.jl --csv path/to/training_history.csv
#   julia plot_training_history.jl --exp_dir path/to/exp_*     # ein Experiment-Ordner
#   julia plot_training_history.jl --search_dir Ergebnisse/UNET_Ergebnisse/One_Crystal

using Plots
using Plots.Measures

# =============================================================================
# Hilfsfunktion: CSV lesen
# =============================================================================

function read_csv(path::AbstractString)
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
# Hauptfunktion: Plot erzeugen
# =============================================================================

function plot_training_history(csv_path::AbstractString;
                               out_path::AbstractString = joinpath(dirname(csv_path), "training_history.png"))

    data   = read_csv(csv_path)
    epochs = haskey(data, :epoch) ? data[:epoch] : collect(1.0:length(first(values(data))))

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

    # ── Panel 2: Relative L2 ──
    p2 = plot(; title="Relative L2 Error (Validation)", xlabel="Epoch", ylabel="rel. L2",
              legend=:topright, grid=true,
              left_margin=8mm, bottom_margin=6mm)

    if haskey(data, :val_rel_l2)
        plot!(p2, epochs, data[:val_rel_l2]; label="Val rel. L2", lw=2, color=:crimson)
    end

    # Bestes Modell markieren (Minimum val_rel_l2)
    if haskey(data, :val_rel_l2)
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
    @info "Gespeichert: $out_path"
    return fig
end

# =============================================================================
# CLI-Argument-Parsing
# =============================================================================

csv_path    = nothing
exp_dir     = nothing
search_dir  = nothing

let i = 1
    while i ≤ length(ARGS)
        if ARGS[i] == "--csv"        && i < length(ARGS); global csv_path   = ARGS[i+1]; i += 2; continue; end
        if ARGS[i] == "--exp_dir"    && i < length(ARGS); global exp_dir    = ARGS[i+1]; i += 2; continue; end
        if ARGS[i] == "--search_dir" && i < length(ARGS); global search_dir = ARGS[i+1]; i += 2; continue; end
        i += 1
    end
end

# =============================================================================
# Ausführung
# =============================================================================

if csv_path !== nothing
    # Einzelne CSV
    plot_training_history(csv_path)

elseif exp_dir !== nothing
    # Einzelner Experiment-Ordner
    csv = joinpath(exp_dir, "training_history.csv")
    isfile(csv) || error("Keine training_history.csv in $exp_dir")
    plot_training_history(csv)

else
    # Suche alle training_history.csv in search_dir (oder aktuellem Ordner)
    root = search_dir !== nothing ? search_dir : "."
    csvs = String[]
    for (dirpath, dirs, files) in walkdir(root)
        if "training_history.csv" in files
            push!(csvs, joinpath(dirpath, "training_history.csv"))
        end
    end

    if isempty(csvs)
        @warn "Keine training_history.csv gefunden in: $root"
    else
        @info "$(length(csvs)) CSV-Dateien gefunden:"
        for c in csvs
            println("  ", c)
        end
        for csv in csvs
            plot_training_history(csv)
        end
    end
end
