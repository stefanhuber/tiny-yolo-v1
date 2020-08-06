from tensorflow import keras


def Conv2D(filters, kernel_size, strides=1):
    return keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='same',
        activation=keras.layers.LeakyReLU(.01)
    )


def TinyBackboneModel(input_image_shape=(448, 448, 3)):
    inputs = keras.Input(shape=input_image_shape)

    layers = Conv2D(16, 3)(inputs)
    layers = keras.layers.BatchNormalization()(layers)
    layers = keras.layers.MaxPool2D(pool_size=2, strides=2)(layers)

    layers = Conv2D(32, 3)(layers)
    layers = keras.layers.BatchNormalization()(layers)
    layers = keras.layers.MaxPool2D(pool_size=2, strides=2)(layers)

    layers = Conv2D(64, 3)(layers)
    layers = keras.layers.BatchNormalization()(layers)
    layers = keras.layers.MaxPool2D(pool_size=2, strides=2)(layers)

    layers = Conv2D(128, 3)(layers)
    layers = keras.layers.BatchNormalization()(layers)
    layers = keras.layers.MaxPool2D(pool_size=2, strides=2)(layers)

    layers = Conv2D(256, 3)(layers)
    layers = keras.layers.BatchNormalization()(layers)
    layers = keras.layers.MaxPool2D(pool_size=2, strides=2)(layers)

    layers = Conv2D(512, 3)(layers)
    layers = keras.layers.BatchNormalization()(layers)
    layers = keras.layers.MaxPool2D(pool_size=2, strides=2)(layers)

    layers = Conv2D(1024, 3)(layers)
    layers = keras.layers.BatchNormalization()(layers)
    layers = keras.layers.MaxPool2D(pool_size=2, strides=2)(layers)

    layers = Conv2D(256, 3)(layers)
    layers = keras.layers.BatchNormalization()(layers)
    layers = keras.layers.MaxPool2D(pool_size=2, strides=2)(layers)

    return keras.Model(inputs=inputs, outputs=layers, name="tiny_yolo_v1_backbone")


def TinyYoloModel(input_image_shape=(200, 200, 3), num_cells=7, num_classes=20, num_boxes_per_cell=2):
    yolo_model = keras.Sequential()
    yolo_model.add(TinyBackboneModel(input_image_shape))
    yolo_model.add(keras.layers.BatchNormalization())
    yolo_model.add(Conv2D(1024, 3))
    yolo_model.add(keras.layers.BatchNormalization())
    yolo_model.add(Conv2D(1024, 3))
    yolo_model.add(keras.layers.BatchNormalization())
    yolo_model.add(keras.layers.Flatten())
    yolo_model.add(keras.layers.Dense(units=4096, activation=keras.layers.LeakyReLU(.01)))
    yolo_model.add(keras.layers.BatchNormalization())
    yolo_model.add(keras.layers.Dense(units=num_cells * num_cells * ((5 * num_boxes_per_cell) + num_classes)))

    return yolo_model
