module EvaluatePsi

using JLD2
using BSON
using CairoMakie            # CairoMakie ist für PNGs oft stressfreier als GLMakie
using Statistics

using ..DatasetPsi: load_dataset, get_sample
using CairoMakie: DataAspect

"""
    evaluate_single(; data_dir, model_path, sample_idx, out_prefix)

Lädt ein trainiertes Modell + Datensatz, nimmt ein Sample,
berechnet ψ_pred und speichert Vergleichsplots:
- ψ_true
- ψ_pred
- Fehler = ψ_pred - ψ_true
"""
function evaluate_single(; data_dir::String = "data_psi",
                          model_path::String = "unet_psi.bson",
                          sample_idx::Int = 1,
                          out_prefix::String = "eval_psi")

    # Dataset laden und Sample holen
    ds = load_dataset(data_dir)
    @assert 1 <= sample_idx <= length(ds.files) "sample_idx liegt außerhalb des Datensatzes"

    x, y = get_sample(ds, sample_idx)  # x, y: (nx, nz, 1)
    nx, nz, _ = size(x)

    # Modell laden
    model = nothing
    BSON.@load model_path model
    @assert model !== nothing "Konnte Modell aus $model_path nicht laden"

    # Vorhersage: Batch-Dimension hinzufügen
    x_batch = reshape(x, nx, nz, 1, 1)     # (nx, nz, 1, 1)
    y_pred  = model(x_batch)               # (nx, nz, 1, 1)

    ψ_true = reshape(y, nx, nz)            # (nx, nz)
    ψ_pred = Array(y_pred[:, :, 1, 1])     # (nx, nz)
    err    = ψ_pred .- ψ_true

    @info "Eval-Sample $sample_idx: nx=$nx, nz=$nz"
    @info "ψ_true range: $(extrema(ψ_true)), ψ_pred range: $(extrema(ψ_pred))"
    @info "Fehler (ψ_pred - ψ_true) range: $(extrema(err))"

    # Gemeinsame Farbskalen
    cr_ψ_true = extrema(ψ_true)
    cr_ψ_pred = extrema(ψ_pred)
    cr_ψ = (min(cr_ψ_true[1], cr_ψ_pred[1]),
            max(cr_ψ_true[2], cr_ψ_pred[2]))

    max_err = maximum(abs.(err))
    cr_err = (-max_err, max_err)

    xcoords = 1:nx
    zcoords = 1:nz

    CairoMakie.activate!()

    # Haupt-Grid: 1 Zeile, 3 Spalten
    fig = Figure(resolution = (1500, 500))
    # Jede Haupt-Zelle bekommt ein Nested-Grid (links Heatmap, rechts Colorbar)
    for j in 1:3
        fig[1, j] = GridLayout()
    end

    # Panel 1: ψ Ground Truth
    ax1 = Axis(fig[1, 1][1, 1],
               title = "ψ LaMEM (Ground Truth)",
               xlabel = "x", ylabel = "z",
               aspect = DataAspect())
    hm1 = heatmap!(ax1, xcoords, zcoords, ψ_true'; colorrange = cr_ψ)
    Colorbar(fig[1, 1][1, 2], hm1, label = "ψ_true")

    # Panel 2: ψ predicted
    ax2 = Axis(fig[1, 2][1, 1],
               title = "ψ predicted (Model)",
               xlabel = "x", ylabel = "z",
               aspect = DataAspect())
    hm2 = heatmap!(ax2, xcoords, zcoords, ψ_pred'; colorrange = cr_ψ)
    Colorbar(fig[1, 2][1, 2], hm2, label = "ψ_pred")

    # Panel 3: Fehler Δψ
    ax3 = Axis(fig[1, 3][1, 1],
               title = "Δψ = ψ_pred − ψ_true",
               xlabel = "x", ylabel = "z",
               aspect = DataAspect())
    hm3 = heatmap!(ax3, xcoords, zcoords, err'; colorrange = cr_err)
    Colorbar(fig[1, 3][1, 2], hm3, label = "Fehler")

    # Ticks ausdünnen
    for ax in (ax1, ax2, ax3)
        ax.xticks = 0:50:nx
        ax.yticks = 0:50:nz
    end

    filename = "$(out_prefix)_sample$(sample_idx).png"
    save(filename, fig)
    @info "Eval-Plot gespeichert als $filename"

    display(fig)

    return ψ_true, ψ_pred, err
end

end # module
