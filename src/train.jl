module Train

export train_model, train_model_cuda

include("dataloader.jl")
include("model.jl")
using Flux
using Flux.Optimise: update!
using .DataLoader
using .Model

# Common training function
function train!(model, train_data, val_data; epochs, learning_rate, use_cuda=false)
    opt_state = Flux.setup(Flux.ADAM(learning_rate), model)
    loss(m, x, y) = Flux.logitcrossentropy(m(x), y)

    for epoch in 1:epochs
        @info "Epoch $epoch/$epochs"
        epoch_loss = 0.0

        for (x_batch, y_batch) in train_data
            x_batch = Float32.(x_batch)
            y_batch = Float32.(y_batch) ./ maximum(y_batch)

            grads = Flux.gradient(m -> loss(m, x_batch, y_batch), model)
            Flux.update!(opt_state, model, grads[1])

            epoch_loss += loss(model, x_batch, y_batch)
        end

        @info "Training Loss: $(epoch_loss / length(train_data))"

        val_loss = 0.0
        for (x_val, y_val) in val_data
            y_val = Float32.(y_val) ./ maximum(y_val)
            val_loss += loss(model, x_val, y_val)
        end

        @info "Validation Loss: $(val_loss / length(val_data))"
    end

    @info "Training complete. Saving model..."
    return model
end

# Training function without CUDA
function train_model(model, train_data, val_data; epochs, learning_rate)
    return train!(model, train_data, val_data; epochs=epochs, learning_rate=learning_rate, use_cuda=false)
end

# Training function with CUDA
function train_model_cuda(model, train_data, val_data; epochs, learning_rate)
    return train!(model, train_data, val_data; epochs=epochs, learning_rate=learning_rate, use_cuda=true)
end

end # module
