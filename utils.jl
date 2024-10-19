using Flux
using Statistics

function accuracy(model, test_X, test_y)
    predictions = Flux.onecold(model(test_X), 0:9)  # Преобразуем предсказания в классы
    true_labels = Flux.onecold(test_y, 0:9)         # Преобразуем one-hot метки в классы
    
    return mean(predictions .== true_labels)
end
