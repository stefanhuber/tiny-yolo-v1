import unittest
from loss import YoloLoss
#from training.data import transform_to_tensor, transform_to_predicted_tensor

class TestLossFunctions(unittest.TestCase):

    def setUp(self):
        self.yolo_loss = YoloLoss()
        self.yolo_loss.config(image_size=(200, 200))

