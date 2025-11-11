module TrainingPsi

using Flux
using Statistics
using Random
using BSON

using ..DatasetPsi: load_dataset, make_batches
using ..UNetPsi: build_unet

"""
    mse_loss(y_pred, y_true)

Einfacher Mean-Squared-Error über alle Dimensionen.
"""
mse_loss(y_pred, y_true) = mean((y_pred .- y_true).^2)

"""
    train_unet(; data_dir, epochs, batch_size, lr, rng, save_path)

Trainiert ein U-Net auf den in `data_dir` liegenden .jld2-Samples.
"""
function train_unet(; data_dir::String="data_psi",
                     epochs::Int=10,
                     batch_size::Int=4,
                     lr::Float64=1e-4,
                     rng = Random.default_rng(),
                     save_path::String="unet_psi.bson")

    # Dataset laden
    ds = load_dataset(data_dir)

    # Modell bauen
    model = build_unet(1, 1)   # 1 Input-Kanal (Maske), 1 Output-Kanal (ψ_norm)

    # Optimizer nach neuem Flux-API-Stil
    opt = Flux.setup(Flux.Adam(lr), model)

    @info "Starte Training: epochs=$epochs, batch_size=$batch_size, lr=$lr"

    for epoch in 1:epochs
        batches = make_batches(ds, batch_size; rng=rng)
        losses = Float64[]

        for (x, y) in batches
            # x, y: (H, W, C, B)
            loss, back = Flux.withgradient(model) do m
                y_pred = m(x)
                mse_loss(y_pred, y)
            end

            Flux.update!(opt, model, back)
            push!(losses, loss)
        end

        avg_loss = mean(losses)
        @info "Epoch $epoch / $epochs   |   loss = $avg_loss"

        # Einfaches Checkpointing: überschreibt immer dieselbe Datei
        BSON.@save save_path model
    end

    @info "Training abgeschlossen. Modell gespeichert nach $save_path"
    return model
end

end # module
