import tensorflow as tf
from model import TinyYoloModel
from data import generate_training_batch
from loss import YoloLoss

yolo_model = TinyYoloModel(input_image_shape=(448, 448, 3), num_classes=3, num_boxes_per_cell=1)

loss_object = YoloLoss()
loss_object.config(num_classes=3, num_boxes_per_cell=1)

optimizer = tf.keras.optimizers.Adam()

epochs = 500

for epoch in range(epochs):
    batch = next(generate_training_batch(image_size=(448, 448), batch_size=32))
    with tf.GradientTape() as tape:
        result = yolo_model(batch[0])
        loss_value = loss_object(batch[1], result)
        grads = tape.gradient(loss_value, yolo_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, yolo_model.trainable_variables))
        print("Epoch {} loss {}".format(epoch, loss_value.numpy().mean()))
