import tensorflow as tf


class YoloLoss(tf.keras.losses.Loss):

    def config(self, image_size=(448, 448), num_cells=7, num_classes=20, num_boxes_per_cell=2, lambda_coord=5, lambda_noobj=.5):
        self.__image_size = image_size
        self.__num_cells = num_cells
        self.__num_classes = num_classes
        self.__boxes_per_cell = num_boxes_per_cell
        self.__lambda_coord = lambda_coord
        self.__lambda_noobj = lambda_noobj

    @tf.autograph.experimental.do_not_convert
    def call(self, y1, y2):
        y_true = tf.reshape(y1, (-1, self.__num_cells * self.__num_cells, self.__num_classes + 5))
        y_pred = tf.clip_by_value(tf.reshape(y2, (-1, self.__num_cells * self.__num_cells, self.__num_classes + 5)), clip_value_min=0, clip_value_max=1)

        loss_xy = tf.multiply(
            self.__lambda_coord,
            tf.reduce_sum(
                tf.multiply(
                    y_true[..., 0],
                    tf.add(
                        tf.square(y_true[..., 1] - y_pred[..., 1]),
                        tf.square(y_true[..., 2] - y_pred[..., 2])
                    )
                )
            )
        )

        loss_wh = tf.multiply(
            self.__lambda_coord,
            tf.reduce_sum(
                tf.multiply(
                    y_true[..., 0],
                    tf.add(
                        tf.square(tf.sqrt(y_true[..., 3]) - tf.sqrt(y_pred[..., 3])),
                        tf.square(tf.sqrt(y_true[..., 4]) - tf.sqrt(y_pred[..., 4]))
                    )
                )
            )
        )

        loss_confidence = tf.reduce_sum(
            tf.square(y_true[..., 0] - y_pred[..., 0])
        )

        loss_classes = tf.reduce_sum(
            tf.multiply(
                y_true[..., 0],
                tf.reduce_sum(
                    tf.square(
                        y_true[..., 5:] - y_pred[..., 5:]
                    )
                )
            )
        )

        return loss_xy + loss_wh + loss_confidence + loss_classes