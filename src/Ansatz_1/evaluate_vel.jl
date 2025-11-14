module EvaluateVel

using JLD2
using BSON
using Statistics
using Printf

using ..DatasetVel: load_dataset, get_sample

"""
    evaluate_dataset(; data_dir, model_path, denorm_vel=false)

Evaluiert den kompletten Datensatz für Ansatz 1.

- `denorm_vel = false`: Fehler im normalisierten v-Raum (v_norm).
- `denorm_vel = true` : Fehler im physikalischen v-Raum (mit scale_v zurückskaliert).
"""
function evaluate_dataset(; data_dir::String,
                          model_path::String,
                          denorm_vel::Bool=false)

    @info "Lade Ansatz-1-Modell aus $model_path"
    model_bson = BSON.load(model_path)
    model = model_bson[:model]

    ds = load_dataset(data_dir)
    n_samples = length(ds.files)
    @info "Datensatz mit $n_samples Samples geladen aus $data_dir"

    if denorm_vel
        @info "Evaluierung im PHYSIKALISCHEN v-Raum (de-normalisiert mit 'scale_v')."
    else
        @info "Evaluierung im NORMALISIERTEN v-Raum (v_norm)."
    end

    errors_by_n = Dict{Int, Vector{NamedTuple}}()

    for (i, filepath) in enumerate(ds.files)
        x, y_true = get_sample(ds, i)
        nx, nz, ch_in = size(x)
        _,  _, ch_out = size(y_true)

        x_batch = reshape(x, nx, nz, ch_in, 1)
        y_pred_batch = model(x_batch)

        y_pred_norm = dropdims(y_pred_batch[:, :, :, 1]; dims=4)
        y_true_norm = y_true

        filedata = JLD2.load(filepath)
        meta    = filedata["meta"]
        scale_v = get(filedata, "scale_v", 1.0)

        n = haskey(meta, :n_crystals) ? meta[:n_crystals] : 1

        if denorm_vel
            y_true_eval = y_true_norm ./ scale_v
            y_pred_eval = y_pred_norm ./ scale_v
        else
            y_true_eval = y_true_norm
            y_pred_eval = y_pred_norm
        end

        # Fehler über beide Komponenten gemeinsam
        diff = y_pred_eval .- y_true_eval
        mse  = mean(diff.^2)

        num   = sqrt(sum(diff.^2))
        denom = sqrt(sum(y_true_eval.^2)) + eps()
        rel_l2 = num / denom

        stat = (; mse, rel_l2)
        if !haskey(errors_by_n, n)
            errors_by_n[n] = NamedTuple[]
        end
        push!(errors_by_n[n], stat)
    end

    @info "=== Ansatz 1 – Auswertung nach Kristallanzahl ($(denorm_vel ? "physikalisches v" : "v_norm")) ==="
    for (n, stats) in sort(collect(errors_by_n); by=first)
        mses = [s.mse for s in stats]
        rels = [s.rel_l2 for s in stats]
        N    = length(stats)

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
