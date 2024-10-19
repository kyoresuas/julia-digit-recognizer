using Images, Flux
using BSON: @load
using YAML

# Загружаем конфигурации
config = YAML.load_file("config.yaml")

# Загружаем модель
@load config["paths"]["model_save_path"] model

# Загрузим изображение и подготовим его
function load_custom_image(path)
    img = load(path)
    img = Gray.(img)
    img = imresize(img, (28, 28))
    img_array = Float32.(img)
    img_array = reshape(img_array, 784, 1)
    return img_array
end

function predict_digit(model, img_array)
    pred = model(img_array)
    return Flux.onecold(pred, 0:9)
end

custom_image = load_custom_image(config["paths"]["image_path"])
predicted_digit = predict_digit(model, custom_image)

println("Цифра на фото: ", predicted_digit)
