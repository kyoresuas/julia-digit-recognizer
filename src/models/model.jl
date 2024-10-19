using Flux

function build_model()
    return Chain(
        Dense(784, 512, relu),   # Увеличенный первый слой
        Dropout(0.5),            # Dropout для регуляризации
        Dense(512, 256, relu),   # Второй слой
        Dropout(0.5),            # Еще один Dropout
        Dense(256, 128, elu),    # Третий слой с ELU
        Dense(128, 10),          # Выходной слой
        softmax                  # Softmax для классификации
    )
end
