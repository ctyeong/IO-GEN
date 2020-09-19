import tensorflow.keras as keras
import numpy as np


def smooth_accuracy(y_true, y_pred):
    y_true = keras.backend.round(y_true)
    y_pred = keras.backend.round(y_pred)
    correct = keras.backend.cast(keras.backend.equal(y_true, y_pred), dtype='float32')
    return keras.backend.mean(correct)


def feat_matching_loss(y_true, y_pred):
    mean_true_feat = keras.backend.mean(y_true, axis=0)
    mean_pred_feat = keras.backend.mean(y_pred, axis=0)
    return keras.backend.sum(keras.backend.square(mean_pred_feat - mean_true_feat))


def euclidean_distance_square_loss(c_vec, v_vec):
    return keras.backend.sum(keras.backend.square(v_vec - c_vec), axis=-1)


def score(y_true, y_pred):
    
    n1 = len(y_true)
    n2 = len(y_pred)
    
    y_true = np.reshape(y_true, (n1, -1))
    y_pred = np.reshape(y_pred, (n2, -1))
    
    return np.sum(np.square(y_true - y_pred), axis=-1)
