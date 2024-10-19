using Flux

function accuracy(model, test_X, test_y)
    predictions = Flux.onecold(model(test_X), 0:9)
    return mean(predictions .== test_y)
end
