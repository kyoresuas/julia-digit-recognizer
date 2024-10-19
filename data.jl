using MLDatasets

function load_data()
    data_dir = joinpath(pwd(), "datasets")

    mkpath(data_dir)

    train = MNIST(split=:train, dir=data_dir)[:]
    test = MNIST(split=:test, dir=data_dir)[:]

    train_X = Flux.flatten(train.data) ./ 255.0
    train_y = train.targets
    test_X = Flux.flatten(test.data) ./ 255.0
    test_y = test.targets

    return train_X, train_y, test_X, test_y
end
