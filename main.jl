include("model.jl")
include("train.jl")
include("data.jl")
include("utils.jl")

config = Dict(
    :epochs => 20,
    :batch_size => 64,
    :learning_rate => 0.001
)

train_X, train_y, test_X, test_y = load_data()

model = build_model()

train!(model, train_X, train_y, config)

println("Точность на тестовом наборе: ", accuracy(model, test_X, test_y))
