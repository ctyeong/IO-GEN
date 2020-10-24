import argparse
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
from utils import load_of_data
from models import build_DCAE, build_IO_GEN, build_classifier
from metrics import feat_matching_loss

# parse arguments 
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--split_dir', help='Directory for split')
parser.add_argument('-m', '--m', default=2, type=int, help='Number of optical flow pairs per input (default=2)')
parser.add_argument('-p', '--model_path', help='Path to IO-GEN model')
parser.add_argument('-f', '--fake_dir', default='./fake_imgs', help='Directory to save synthesized images')
parser.add_argument('-b', '--n_fakes', default=10, type=int, help='Number of fake optical flow pairs')
parser.add_argument('-c', '--color_map', default='Spectral', help='Colormap of matplotlib')

options = parser.parse_args()

split_dir = options.split_dir
m = options.m
model_path = options.model_path
fake_dir = options.fake_dir
n_fakes = options.n_fakes
cmap = options.color_map

# necessary arguments 
assert split_dir != None, 'Please specify the directory of split to use. Use "-s" argument in execution' 
assert model_path != None, 'Please specify the path to the IO-GEN model to deploy. Use "-p" argument in execution' 

if not os.path.isdir(fake_dir):
    os.makedirs(fake_dir)

# load data
train_x, test_stable_x, test_unstable_x = load_of_data(split_dir, m)

# load IO-GEN 
print('\n==================================')
print('Loading IO-GEN')

model = keras.models.load_model(model_path, \
        custom_objects={'feat_matching_loss': feat_matching_loss})
gen = keras.Model(inputs=model.get_layer('gen').input, outputs=model.get_layer('gen').output)  
latent_dim = (100,)

# generate synthetic data 
for i in range(n_fakes): 
    print('Fake {}'.format(i))
    plt.figure(figsize=(12,6))
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                        hspace = 0, wspace = 0)
    z = np.random.normal(loc=0, scale=1., size=(1,) + latent_dim)
    flows = gen.predict(z)[0, ..., :2] 

    ax = plt.subplot(1, 2, 1)
    ax.imshow(-abs(flows[..., 0]), cmap=cmap)
    ax.set_title('X')
    ax.set_xticks([], [])
    ax.set_yticks([], [])

    ax = plt.subplot(1, 2, 2)
    ax.imshow(-abs(flows[..., 1]), cmap=cmap)
    ax.set_title('Y')
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    plt.savefig(os.path.join(fake_dir, '{}.jpg'.format(i)))


