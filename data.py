import tensorflow as tf
import numpy as np
from image import generate_image


def get_generator(image_size=(200, 200), batch_size=32):
    def generate_training_batch():
        while True:
            x = np.zeros((batch_size, image_size[1], image_size[0], 3), dtype=np.float32)
            y_true = []

            for i in range(batch_size):
                image, data = generate_image(image_size)
                # transform (x1, y1, x2, y2, class) to (class, x, y, w, h)
                data = map(lambda item: [item[4], item[0], item[1], item[2] - item[0], item[3] - item[1]], data)
                x[i] = image / 255.
                y_true.append(transform_to_cell_representation(data, image_size, num_classes=3))

            yield x, tf.constant(y_true)

    return generate_training_batch


def transform_to_cell_representation(training_data=[], image_size=[], num_cells=7,  num_classes=20):
    cells = [0 for _ in range((num_classes + 5) * num_cells * num_cells)] # 5 = confidence,x,y,w,h
    yolo_data = from_training_data(training_data, image_size, num_cells)

    for key, entry in yolo_data.items():
        cell = [int(i) for i in key.split("_")]
        cell_index = ((cell[1] * num_cells) + cell[0]) * (num_classes + 5)
        cells[cell_index] = 1 # confidence
        cells[cell_index + 1] = entry['x']
        cells[cell_index + 2] = entry['y']
        cells[cell_index + 3] = entry['w']
        cells[cell_index + 4] = entry['h']
        cells[cell_index + 5 + entry['c']] = 1 # set class index, entry['c'] ranges from 0 until num_classes - 1

    return cells


def from_training_data(training_data=[], image_size=(200, 200), num_cells=7):
    """
    transform raw training data representation into yolo training representation
    a training entry is specified like this: <object-class> <x> <y> <w> <h>

    :param training_data:
    :param image_size:
    :param num_cells:
    :return: dictionary with occupied cells
    """
    cell_width = image_size[0] / num_cells
    cell_height = image_size[1] / num_cells

    occupied_cells = {}
    for training_entry in training_data:
        center_x = (training_entry[1] + (training_entry[3] / 2))
        center_y = (training_entry[2] + (training_entry[4] / 2))
        cell_x = center_x // cell_width
        cell_y = center_y // cell_height

        occupied_cells["{:.0f}_{:.0f}".format(cell_x, cell_y)] = {
            "c": training_entry[0],
            "x": (center_x - (cell_x * cell_width)) / cell_width,
            "y": (center_y - (cell_y * cell_height)) / cell_height,
            "w": training_entry[3] / image_size[0],
            "h": training_entry[4] / image_size[1],
        }

    return occupied_cells
