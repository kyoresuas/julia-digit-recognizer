include("model.jl")
include("train.jl")
include("data.jl")
include("utils.jl")

config = Dict(
    :epochs => 20,
    :batch_size => 64,
    :learning_rate => 0.001
)

# Загружаем данные
train_X, train_y, test_X, test_y = load_data()

# Создаем модель
model = build_model()

# Обучаем модель
train_model!(model, train_X, train_y, config)

# Проверяем точность
println("Точность на тестовом наборе: ", accuracy(model, test_X, test_y))

# Сохраняем модель
using BSON: @save
@save "model.bson" model
