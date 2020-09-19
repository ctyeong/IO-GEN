import argparse
import os
import tensorflow as tf
import numpy as np 
import tensorflow.keras as keras
from utils import load_of_data
from metrics import euclidean_distance_square_loss, smooth_accuracy, score 
from sklearn.metrics import roc_curve, auc

# parse arguments 
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--split_dir', help='Directory for split')
parser.add_argument('-m', '--m', default=2, type=int, help='Number of optical flow pairs per input (default=2)')
parser.add_argument('-d', '--model_dir', default='./saved_models', help='Directory to save trained models')
parser.add_argument('-n', '--model_name', help='Model name to test, e.g.) DCAE, DSVDD, IO-GEN')
parser.add_argument('-v', '--verbose', default=1, help='verbose option, either 0 or 1')
options = parser.parse_args()

split_dir = options.split_dir
m = options.m
model_dir = options.model_dir
model_name = options.model_name
verbose = options.verbose

# necessary arguments 
assert split_dir != None, 'Please specify the directory of split to use. Use "-s" argument in execution' 
assert model_name != None, 'Please specify the directory of split to use. Use "-s" argument in execution' 

# load data
train_x, test_stable_x, test_unstable_x = load_of_data(split_dir, m)

# unstable_x locations to confine in time   
n_test_samples = [0, 666, 1333, 4000, 6666, 9333, len(test_unstable_x)]
days = ['D+1', 'D+2', 'D+3 - D+6', 'D+7 - D+10', 'D+11 - D+14', 'D+15 - D+18']  

# test for different models 
print('AUC Scores') 
if model_name == 'DCAE':

    # load model
    ae = keras.models.load_model('./{}/DCAE.h5'.format(model_dir))
    encoder = keras.Model(inputs=ae.input, outputs=ae.get_layer('encoded').output)                   
    decoder = keras.Model(inputs=ae.input, outputs=ae.get_layer('decoded').output)                   
    y_test_stable_hat = score(ae.predict(test_stable_x), test_stable_x)

    for n_test_i in range(1, len(n_test_samples)):
        y_test_unstable_hat = score(ae.predict(test_unstable_x[n_test_samples[n_test_i-1]:n_test_samples[n_test_i]]), \
                              test_unstable_x[n_test_samples[n_test_i-1]:n_test_samples[n_test_i]])
        true_labels = [0.] * len(y_test_stable_hat) + [1.] * len(y_test_unstable_hat)    

        fpr, tpr, th = roc_curve(true_labels, np.concatenate([y_test_stable_hat, y_test_unstable_hat], axis=-1))
        auc_score = auc(fpr, tpr)
        print('{}: {}'.format(days[n_test_i-1], auc_score))
  
    # test with all 
    y_test_unstable_hat = score(ae.predict(test_unstable_x), test_unstable_x)
    true_labels = [0.] * len(y_test_stable_hat) + [1.] * len(y_test_unstable_hat)    
    fpr, tpr, th = roc_curve(true_labels, np.concatenate([y_test_stable_hat, y_test_unstable_hat], axis=-1))
    auc_score = auc(fpr, tpr)
    print('ALL: {}'.format(auc_score))

elif model_name == 'DSVDD': 

    # load model
    ae = keras.models.load_model('./{}/DCAE.h5'.format(model_dir))
    encoder = keras.Model(inputs=ae.input, outputs=ae.get_layer('encoded').output)                   
    dsvdd = keras.models.load_model('./{}/DSVDD.h5'.format(model_dir), \
           custom_objects={'euclidean_distance_square_loss':euclidean_distance_square_loss})

    # Compute Center Feature
    initial_outputs = encoder.predict(train_x)
    center_feat = np.mean(initial_outputs, axis=0)
    target_feat = np.expand_dims(center_feat, 0) 

    y_test_stable_hat = score(dsvdd.predict(test_stable_x), target_feat)
    for n_test_i in range(1, len(n_test_samples)):
        y_test_unstable_hat = score(dsvdd.predict(test_unstable_x[n_test_samples[n_test_i-1]:n_test_samples[n_test_i]]), \
                              target_feat)
        true_labels = [0.] * len(y_test_stable_hat) + [1.] * len(y_test_unstable_hat)    

        fpr, tpr, th = roc_curve(true_labels, np.concatenate([y_test_stable_hat, y_test_unstable_hat], axis=-1))
        auc_score = auc(fpr, tpr)
        print('{}: {}'.format(days[n_test_i-1], auc_score))
  
    # test with all 
    y_test_unstable_hat = score(dsvdd.predict(test_unstable_x), target_feat)
    true_labels = [0.] * len(y_test_stable_hat) + [1.] * len(y_test_unstable_hat)    
    fpr, tpr, th = roc_curve(true_labels, np.concatenate([y_test_stable_hat, y_test_unstable_hat], axis=-1))
    auc_score = auc(fpr, tpr)
    print('ALL: {}'.format(auc_score))
    
elif model_name == 'IO-GEN': 

    # load model
    cls = keras.models.load_model('./{}/CLASSIFIER.h5'.format(model_dir), \
          custom_objects={'smooth_accuracy': smooth_accuracy, 'keras': keras})

    y_test_stable_hat = cls.predict(test_stable_x).flatten()
    for n_test_i in range(1, len(n_test_samples)):
        y_test_unstable_hat = cls.predict(test_unstable_x[n_test_samples[n_test_i-1]:n_test_samples[n_test_i]]).flatten()
        true_labels = [0.] * len(y_test_stable_hat) + [1.] * len(y_test_unstable_hat)    

        fpr, tpr, th = roc_curve(true_labels, np.concatenate([y_test_stable_hat, y_test_unstable_hat], axis=-1))
        auc_score = auc(fpr, tpr)
        print('{}: {}'.format(days[n_test_i-1], auc_score))
  
    # test with all 
    y_test_unstable_hat = cls.predict(test_unstable_x).flatten() 
    true_labels = [0.] * len(y_test_stable_hat) + [1.] * len(y_test_unstable_hat)    
    fpr, tpr, th = roc_curve(true_labels, np.concatenate([y_test_stable_hat, y_test_unstable_hat], axis=-1))
    auc_score = auc(fpr, tpr)
    print('ALL: {}'.format(auc_score))

else:
    print('Not appropriate model name') 
