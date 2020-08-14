import random
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_bounding_boxes_from_tensor(image_array, tensor, num_cells=7, num_classes=3, num_boxes_per_cell=2, min_confidence_treshold=0.01, classes=["sqaure", "triangle", "circle"]):
    bounding_boxes = []
    class_names = []
    data = tf.reshape(tensor, (num_cells * num_cells, num_classes + (5 * num_boxes_per_cell))).numpy()
    image_size = image_array.shape[0]
    cell_size = image_size / num_cells

    for cell in range(data.shape[0]):
        class_index = tf.argmax(data[cell, (5 * num_boxes_per_cell):])

        for box in range(num_boxes_per_cell):
            confidence_index = (5 * box) + 0
            x_index = (5 * box) + 1
            y_index = (5 * box) + 2
            w_index = (5 * box) + 3
            h_index = (5 * box) + 4

            if data[cell, confidence_index] > min_confidence_treshold:
                cell_x = cell % num_cells
                cell_y = cell // num_cells
                w = image_size * data[cell, w_index]
                h = image_size * data[cell, h_index]
                x1 = (cell_x * cell_size) + cell_size * data[cell, x_index] - w/2
                y1 = (cell_y * cell_size) + cell_size * data[cell, y_index] - h/2
                x2 = x1 + w
                y2 = y1 + h
                class_names.append(classes[class_index])
                bounding_boxes.append([x1, y1, x2, y2])

    return draw_bounding_boxes(image_array, bounding_boxes, class_names)


def draw_bounding_boxes(image_array, bounding_boxes=[], class_names=[]):
    assert len(bounding_boxes) == len(class_names)

    image = Image.fromarray(image_array.astype('uint8'))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:\\Windows\\Fonts\\Arial.ttf", 40)

    for bounding_box, class_name in zip(bounding_boxes, class_names):
        draw.rectangle(bounding_box[0:4], outline="black", width=4)
        draw.text(xy=(bounding_box[0], bounding_box[1]-40), text=class_name, font=font, fill="black")

    return np.array(image)


def generate_random_box(image_size=(200, 200), min_size=0.1, max_size=0.25, square=True):
    # generate width/height
    w = random.randint(int(image_size[0] * min_size), int(image_size[0] * max_size))
    h = w if square else random.randint(int(image_size[1] * min_size), int(image_size[1] * max_size))

    # generate x/y position
    x = random.randint(0, image_size[0] - w)
    y = random.randint(0, image_size[1] - h)

    return [x, y, x+w, y+h]


def get_random_color():
    return random.choice(["red", "green", "blue", "yellow", "orange", "brown", "cyan", "magenta", "black", "grey"])


def draw_random_form(image, boxes=[]):
    draw = ImageDraw.Draw(image)
    form_index = random.randint(0, 2)

    count = 0
    while True:
        bounding_box = generate_random_box(image.size)
        box_intersects = False

        for box in boxes:
            if overlap(bounding_box, box):
                box_intersects = True

        if not box_intersects:
            break

        # if no space left for form, stop search process
        if count >= 10:
            return None

        count += 1


    if form_index == 0:
        draw.rectangle(bounding_box, get_random_color())
    elif form_index == 1:
        draw.polygon((
            int(((bounding_box[2] - bounding_box[0])/2) + bounding_box[0]),
            bounding_box[1],
            bounding_box[2],
            bounding_box[3],
            bounding_box[0],
            bounding_box[3]), get_random_color())
    else:
        draw.ellipse(bounding_box, get_random_color())

    bounding_box.append(form_index)

    # returns bounding box including form_index
    # [x1, y1, x2, y2, form_index]

    return bounding_box


def generate_image(image_size=(200, 200), max_form_count=6):
    '''

    function returns (1) image as numpy array and (2) metadata
    metadata has the structure: (class is an index for the form)
    [
      [x1, y1, x2, y2, class],
      [x1, y1, x2, y2, class],
      [x1, y1, x2, y2, class]
      ...
    ]

    :param image_size:
    :param max_form_count:
    :return:
    '''
    image = Image.new("RGB", image_size, "white")
    boxes = []

    form_count = random.randint(1, max_form_count)
    for i in range(form_count):
        result = draw_random_form(image, boxes)

        if result is not None:
            boxes.append(result)

    return np.array(image), boxes


def draw_training_data(image, training_data, num_cells=7):
    cell_width = image.size[0] / num_cells
    cell_height = image.size[1] / num_cells
    draw = ImageDraw.Draw(image)

    # draw grid
    for cell_x in range(1, num_cells):
        draw.line((cell_x * cell_width, 0, cell_x * cell_width, image.size[1]), fill=(0, 0, 0))
    for cell_y in range(1, num_cells):
        draw.line((0, cell_y * cell_height, image.size[0], cell_y * cell_height), fill=(0, 0, 0))

    for entry in training_data:
        w = entry[2] - entry[0]
        h = entry[3] - entry[1]

        # draw center
        center_x = entry[0] + w / 2
        center_y = entry[1] + h / 2
        draw.ellipse((center_x-2, center_y-2, center_x+2, center_y+2), fill=(255, 0, 0))

        #draw bounding box
        draw.rectangle((entry[0], entry[1], entry[2], entry[3]), outline=(255, 0, 0))

    return image



def overlap(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if (x2 - x1) > 0 or (y2 - y1) > 0:
        return True
    else:
        return False
