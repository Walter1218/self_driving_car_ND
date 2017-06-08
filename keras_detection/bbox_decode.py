import keras.backend
import numpy as np
import tensorflow as tf
#cpu version
def bbox_transform_inv_cpu(shifted, boxes):
    if boxes.shape[0] == 0:
        return np.zeros((0, boxes.shape[1]), dtype=boxes.dtype)

    a = shifted[:, 2] - shifted[:, 0] + 1.0
    b = shifted[:, 3] - shifted[:, 1] + 1.0

    ctr_x = shifted[:, 0] + 0.5 * a
    ctr_y = shifted[:, 1] + 0.5 * b

    dx = boxes[:, 0::4]
    dy = boxes[:, 1::4]
    dw = boxes[:, 2::4]
    dh = boxes[:, 3::4]

    pred_ctr_x = dx * a[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * b[:, np.newaxis] + ctr_y[:, np.newaxis]

    pred_w = np.exp(dw) * a[:, np.newaxis]
    pred_h = np.exp(dh) * b[:, np.newaxis]

    pred_boxes = [
        pred_ctr_x - 0.5 * pred_w,
        pred_ctr_y - 0.5 * pred_h,
        pred_ctr_x + 0.5 * pred_w,
        pred_ctr_y + 0.5 * pred_h
    ]

    return np.concatenate(pred_boxes)

#gpu version
def bbox_transform_inv(shifted, boxes):
    if boxes.shape[0] == 0:
        return tf.zeros((0, boxes.shape[1]), dtype=boxes.dtype)

    a = shifted[:, 2] - shifted[:, 0] + 1.0
    b = shifted[:, 3] - shifted[:, 1] + 1.0

    ctr_x = shifted[:, 0] + 0.5 * a
    ctr_y = shifted[:, 1] + 0.5 * b

    dx = boxes[:, 0::4]
    dy = boxes[:, 1::4]
    dw = boxes[:, 2::4]
    dh = boxes[:, 3::4]

    pred_ctr_x = dx * a[:, tf.newaxis] + ctr_x[:, tf.newaxis]
    pred_ctr_y = dy * b[:, tf.newaxis] + ctr_y[:, tf.newaxis]

    pred_w = tf.exp(dw) * a[:, tf.newaxis]
    pred_h = tf.exp(dh) * b[:, tf.newaxis]

    pred_boxes = [
        pred_ctr_x - 0.5 * pred_w,
        pred_ctr_y - 0.5 * pred_h,
        pred_ctr_x + 0.5 * pred_w,
        pred_ctr_y + 0.5 * pred_h
    ]

    return keras.backend.concatenate(pred_boxes)