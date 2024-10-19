include("src/models/model.jl")
include("src/training/train.jl")
include("src/data/data.jl")
include("src/utils/utils.jl")
using YAML

# Загружаем конфигурации
config = YAML.load_file("config.yaml")

# Загружаем данные
train_X, train_y, test_X, test_y = load_data()

# Создаем модель
model = build_model()

# Обучаем модель
train_model!(model, train_X, train_y, config["training"])

# Проверяем точность
println("Точность на тестовом наборе: ", accuracy(model, test_X, test_y))

# Сохраняем модель
using BSON: @save
@save config["paths"]["model_save_path"] model
