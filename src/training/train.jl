using Flux
using MLUtils: DataLoader

function train_model!(model, train_X, train_y, training_config)
    loss(x, y) = Flux.crossentropy(model(x), y)
    opt = Flux.Adam(training_config["learning_rate"])
    batches = DataLoader((train_X, train_y), batchsize=training_config["batch_size"], shuffle=true)

    for epoch in 1:training_config["epochs"]
        for (x, y) in batches
            Flux.train!(loss, Flux.params(model), [(x, y)], opt)
        end
        println("Epoch $epoch: Loss = ", loss(train_X, train_y))
    end
end
