using Flux

function build_model()
    return Chain(
        Dense(784, 128, relu),  # Первый слой: 784 входа (28x28 пикселей), 128 нейронов
        Dense(128, 64, relu),   # Второй слой: 128 входов, 64 нейрона
        Dense(64, 10),          # Выходной слой: 64 входа, 10 выходов (для классов 0-9)
        softmax                 # Softmax для классификации
    )
end
