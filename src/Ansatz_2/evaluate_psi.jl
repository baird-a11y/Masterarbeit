module EvaluatePsi

using JLD2
using BSON
using GLMakie
using Statistics

using ..DatasetPsi: load_dataset, get_sample

"""
    evaluate_single(; data_dir, model_path, sample_idx, out_prefix)

Lädt ein trainiertes Modell + einen Datensatz, nimmt ein Sample,
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
    x_batch = reshape(x, nx, nz, 1, 1)       # (nx, nz, 1, 1)
    y_pred  = model(x_batch)                # (nx, nz, 1, 1)

    ψ_true = reshape(y, nx, nz)             # (nx, nz)
    ψ_pred = Array(y_pred[:, :, 1, 1])      # (nx, nz)
    err    = ψ_pred .- ψ_true

    @info "Eval-Sample $sample_idx: nx=$nx, nz=$nz"
    @info "ψ_true range: $(extrema(ψ_true)), ψ_pred range: $(extrema(ψ_pred))"
    @info "Fehler (ψ_pred - ψ_true) range: $(extrema(err))"

    # Plot
    fig = Figure(resolution = (1200, 400))

    ax1 = Axis(fig[1, 1], title = "ψ_true", xlabel = "x", ylabel = "z")
    hm1 = heatmap!(ax1, ψ_true')
    Colorbar(fig[2, 1], hm1, label = "ψ_true")

    ax2 = Axis(fig[1, 2], title = "ψ_pred", xlabel = "x", ylabel = "z")
    hm2 = heatmap!(ax2, ψ_pred')
    Colorbar(fig[2, 2], hm2, label = "ψ_pred")

    ax3 = Axis(fig[1, 3], title = "Fehler (ψ_pred - ψ_true)", xlabel = "x", ylabel = "z")
    hm3 = heatmap!(ax3, err')
    Colorbar(fig[2, 3], hm3, label = "Fehler")

    # Datei speichern
    filename = "$(out_prefix)_sample$(sample_idx).png"
    save(filename, fig)
    @info "Eval-Plot gespeichert als $filename"

    display(fig)

    return ψ_true, ψ_pred, err
end

end # module
