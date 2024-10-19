using Flux

function train!(model, train_X, train_y, config)
    loss(x, y) = Flux.crossentropy(model(x), Flux.onehotbatch(y, 0:9))
    
    opt = ADAM(config[:learning_rate])
    
    batches = Flux.Data.DataLoader(train_X, train_y, batchsize=config[:batch_size], shuffle=true)
    
    for epoch in 1:config[:epochs]
        for (x, y) in batches
            Flux.train!(loss, params(model), [(x, y)], opt)
        end
        
        println("Epoch $epoch: Loss = ", loss(train_X, Flux.onehotbatch(train_y, 0:9)))
    end
end
