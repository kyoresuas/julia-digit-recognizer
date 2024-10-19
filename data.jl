using MLDatasets

function load_data()
    train_X, train_y = MNIST.traindata()
    test_X, test_y = MNIST.testdata()

    train_X = Flux.flatten(train_X) ./ 255.0
    test_X = Flux.flatten(test_X) ./ 255.0

    return train_X, train_y, test_X, test_y
end
