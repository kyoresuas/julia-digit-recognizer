using MLDatasets

function load_data()
    train = MNIST(split=:train)[:]
    test = MNIST(split=:test)[:]

    train_X = Flux.flatten(train.data) ./ 255.0
    train_y = train.targets
    test_X = Flux.flatten(test.data) ./ 255.0
    test_y = test.targets

    return train_X, train_y, test_X, test_y
end
