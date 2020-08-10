import random
import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw


def draw_bounding_boxes_from_tensor(image_array, tensor, num_cells=7, num_classes=3):
    bounding_boxes = []
    data = tf.reshape(tensor, (num_cells * num_cells, num_classes + 5)).numpy()
    image_size = image_array.shape[0]
    cell_size = image_size / num_cells

    for i in range(data.shape[0]):
        if data[i, 0] > 0:
            cell_x = i % num_cells
            cell_y = i // num_cells
            w = image_size * data[i, 3]
            h = image_size * data[i, 4]
            x1 = (cell_x * cell_size) + cell_size * data[i, 1] - w/2
            y1 = (cell_y * cell_size) + cell_size * data[i, 2] - h/2
            x2 = x1 + w
            y2 = y1 + h
            bounding_boxes.append([x1, y1, x2, y2])

    return draw_bounding_boxes(image_array, bounding_boxes)


def draw_bounding_boxes(image_array, bounding_boxes=[]):
    image = Image.fromarray(image_array.astype('uint8'))
    draw = ImageDraw.Draw(image)

    for bounding_box in bounding_boxes:
        draw.rectangle(bounding_box[0:4], outline="black", width=4)

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


def overlap(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    if (x2 - x1) > 0 or (y2 - y1) > 0:
        return True
    else:
        return False
