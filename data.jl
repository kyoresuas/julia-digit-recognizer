using MLDatasets

function load_data()
    data_dir = joinpath(pwd(), "datasets")

    mkpath(data_dir)

    train = MNIST(:train, dir=data_dir)
    test = MNIST(:test, dir=data_dir)

    train_X, train_y = train.features, train.targets
    test_X, test_y = test.features, test.targets

    train_X = Flux.flatten(train_X) ./ 255.0 |> x -> Float32.(x)
    test_X = Flux.flatten(test_X) ./ 255.0 |> x -> Float32.(x)

    train_y = Flux.onehotbatch(train_y, 0:9)
    test_y = Flux.onehotbatch(test_y, 0:9)

    return train_X, train_y, test_X, test_y
end
