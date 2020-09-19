import argparse
import os
import tensorflow as tf
import numpy as np 
import tensorflow.keras as keras
from utils import load_of_data
from models import build_DCAE, build_IO_GEN, build_classifier
from metrics import euclidean_distance_square_loss, smooth_accuracy, feat_matching_loss

# parse arguments 
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--split_dir', help='Directory for split')
parser.add_argument('-m', '--m', default=2, type=int, help='Number of optical flow pairs per input (default=2)')
parser.add_argument('-d', '--model_dir', default='./saved_models', help='Directory to save trained models')
parser.add_argument('-t', '--tensorboard_dir', default='./tb_logs', help='Directory to save tensorboard logs')
parser.add_argument('-v', '--verbose', default=1, help='verbose option, either 0 or 1')
options = parser.parse_args()

split_dir = options.split_dir
m = options.m
model_dir = options.model_dir
tb_dir = options.tensorboard_dir
verbose = options.verbose

# necessary arguments 
assert split_dir != None, 'Please specify the directory of split to use. Use "-s" argument in execution' 

# make directories if no
if not os.path.isdir(model_dir):
    os.makedirs(model_dir)
    
if not os.path.isdir(tb_dir):
    os.makedirs(tb_dir)

# load data
train_x, test_stable_x, test_unstable_x = load_of_data(split_dir, m)

# DCAE 
print('\n==================================')
print('DCAE')

ae = build_DCAE(m)
if verbose:
    print(ae.summary())

lr = 0.00005 * 10. 
n_epochs = 2 #750
batch_size = 16
noise_level = 0.02

saved_path = './{}/DCAE.h5'.format(model_dir)
cp_callback = keras.callbacks.ModelCheckpoint(filepath=saved_path, save_weights_only=False, 
              verbose=1, monitor='val_loss', mode='min', save_best_only=True)
log_dir = "./{}/DCAE".format(tb_dir)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False,
                       profile_batch=0)
ae.compile(loss=['mse'], optimizer=keras.optimizers.Adam(learning_rate=lr))

for e in range(n_epochs):
    x_train_z = train_x + np.random.normal(scale=noise_level, size=train_x.shape)    
    history = ae.fit(x_train_z, x_train_z, validation_data=(test_stable_x, test_stable_x), 
              batch_size=batch_size, initial_epoch=e, 
              epochs=e+1, callbacks=[cp_callback, tensorboard_callback], verbose=2)

# DSVDD
print('\n==================================')
print('DSVDD')

model = keras.models.load_model('./{}/DCAE.h5'.format(model_dir))
encoder = keras.Model(inputs=model.input, outputs=model.get_layer('encoded').output)                   

if verbose:
    print(encoder.summary())

# Compute Center Feature
batch_size = 16
initial_outputs = encoder.predict(train_x)
center_feat = np.mean(initial_outputs, axis=0)
target_feat = np.expand_dims(center_feat, 0) 
target_feat_train = np.repeat(target_feat, len(train_x), axis=0)
target_feat_val = np.repeat(target_feat, len(test_stable_x), axis=0)

n_epochs = 2 #420 #160 * 2
saved_path = './{}/DSVDD.h5'.format(model_dir)
cp_callback = keras.callbacks.ModelCheckpoint(filepath=saved_path, save_weights_only=False, verbose=1, 
                                              monitor='val_loss', mode='min', save_best_only=True)
log_dir = "./{}/DSVDD".format(tb_dir) 
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False,
                       profile_batch=0)

lr = 0.00005 
encoder.compile(loss=[euclidean_distance_square_loss], optimizer=keras.optimizers.Adam(learning_rate=lr))

for e in range(n_epochs):
    print('***********\nEpoch {}/{}'.format(e+1, n_epochs))
    encoder.fit(train_x, target_feat_train, validation_data=(test_stable_x, target_feat_val), 
                  batch_size=batch_size, initial_epoch=e, epochs=e+1, callbacks=[cp_callback, tensorboard_callback], 
                  verbose=2)        

# IO-GEN
print('\n==================================')
print('IO-GEN')

lr = 0.000005 * .5 * .5 
latent_dim = (100,)

ae = keras.models.load_model('./{}/DCAE.h5'.format(model_dir))
dsvdd = keras.models.load_model('./{}/DSVDD.h5'.format(model_dir), \
       custom_objects={'euclidean_distance_square_loss':euclidean_distance_square_loss})

gan, gen, dsc = build_IO_GEN(ae, dsvdd, latent_dim, lr, m)
if verbose:
    print(gan.summary())

# prepare E(v)
encoder = keras.Model(inputs=ae.input, outputs=ae.get_layer('encoded').output)
initial_outputs = encoder.predict(train_x)
center_feat = np.mean(initial_outputs, axis=0)
target_feat = np.expand_dims(center_feat, 0) 
target_feat_train = np.repeat(target_feat, len(train_x), axis=0)
print(target_feat_train.shape)

n_epochs = 2 #20000
batch_size = 16 #16
noise_level = 0
v_mean, v_std = 0, 1 

log_dir = "./{}/IO-GEN".format(tb_dir)
tb_writer = tf.summary.create_file_writer(log_dir)
tb_writer.set_as_default()

for e in range(n_epochs):
    print('***********\nEpoch {}/{}'.format(e+1, n_epochs))
    
    #####################
    # DSC
    #####################
    # Real
    #####################
    real_train_idx = np.random.randint(0, high=len(train_x), size=batch_size)
    real_train_x = train_x[real_train_idx] \
                    + np.random.normal(scale=noise_level, size=train_x[:batch_size].shape)
    real_labels = np.ones((batch_size, 1)) - np.random.uniform(0., .1, size=(batch_size, 1))
        
    dsc_real_loss, dsc_real_acc = dsc.train_on_batch(real_train_x, real_labels)

    tf.summary.scalar('DSC/Real_Ent', data=dsc_real_loss, step=e)
    tf.summary.scalar('DSC/Real_Acc', data=dsc_real_acc, step=e)    

    #####################
    # Fake
    #####################
    z = np.random.normal(loc=v_mean, scale=v_std, size=(batch_size,) + latent_dim) 
    fake_train_x = gen.predict(z) \
                    + np.random.normal(scale=noise_level, size=train_x[:batch_size].shape)
    fake_labels = np.zeros((batch_size, 1)) + np.random.uniform(0., .1, size=(batch_size, 1))
        
    dsc_fake_loss, dsc_fake_acc = dsc.train_on_batch(fake_train_x, fake_labels)

    tf.summary.scalar('DSC/Fake_Ent', data=dsc_fake_loss, step=e)    
    tf.summary.scalar('DSC/Fake_Acc', data=dsc_fake_acc, step=e)    

    #####################
    # GEN
    #####################
    z = np.random.normal(loc=v_mean, scale=v_std, size=(batch_size,) + latent_dim)
    loss, feat_loss, gen_loss, gen_acc = gan.train_on_batch(z, \
                                         [target_feat_train[:batch_size], np.ones((batch_size, 1))])
    tf.summary.scalar('GEN_FM/DSVDD_Loss', data=feat_loss, step=e)    
    tf.summary.scalar('GEN/Loss', data=gen_loss, step=e)    
    tf.summary.scalar('GEN/Acc', data=gen_acc, step=e)    

    print('Dsc')
    print('RealLoss={}, FakeLoss={}'.format(dsc_real_loss, dsc_fake_loss))
    print('RealAcc={}, FakeAcc={}'.format(dsc_real_acc, dsc_fake_acc))
    
    print('\nIO-GEN')
    print('Loss={}'.format(gen_loss))
    print('InvAcc={}'.format(gen_acc))
    print('DSVDD_Loss={}\n'.format(feat_loss))
    
    if e % 500 == 0:
        z = np.random.normal(loc=v_mean, scale=v_std, size=(batch_size,) + latent_dim) 
        
        x_star = np.asarray(gen.predict(z))
        x_star = (x_star - np.min(x_star))/(np.max(x_star) - np.min(x_star))
        tf.summary.image('{} x^* Ch1'.format(batch_size), x_star[:][..., 0:1], max_outputs=batch_size, step=e)
        tf.summary.image('{} x^* Ch2'.format(batch_size), x_star[:][..., 1:2], max_outputs=batch_size, step=e)
        
    tf.summary.flush()
    
saved_path = './{}/IO-GEN.h5'.format(model_dir)
gan.save(saved_path)

# Classifier
print('\n==================================')
print('CLASSIFIER')

model = keras.models.load_model('./{}/IO-GEN.h5'.format(model_dir), \
        custom_objects={'feat_matching_loss': feat_matching_loss})
gen = keras.Model(inputs=model.get_layer('gen').input, outputs=model.get_layer('gen').output)     

dsvdd = keras.models.load_model('./{}/DSVDD.h5'.format(model_dir), \
       custom_objects={'euclidean_distance_square_loss':euclidean_distance_square_loss})
dsvdd = keras.Model(inputs=dsvdd.input, outputs=dsvdd.output, name='DSVDD')

cls = build_classifier(dsvdd)

lr = 0.005 * .05 * .5
cls.compile(loss=['binary_crossentropy'], metrics=[smooth_accuracy], optimizer=keras.optimizers.Adam(learning_rate=lr))

if verbose:
    print(cls.summary())

n_epochs = 2 #40 
batch_size = 32
saved_path = './{}/CLASSIFIER.h5'.format(model_dir)

cp_callback = keras.callbacks.ModelCheckpoint(filepath=saved_path, save_weights_only=False, verbose=1, 
                                              monitor='val_smooth_accuracy', mode='max', save_best_only=True)
log_dir = "./{}/CLASSIFIER".format(tb_dir) 
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, write_graph=False,
                       profile_batch=0)
    
for e in range(n_epochs):
    print('***********\nEpoch {}/{}'.format(e+1, n_epochs))

    ########
    # Train
    ########
    z =  np.random.normal(loc=0, scale=1, size=(train_x.shape[0], gen.input.shape[-1]))
    fake_train_x = gen.predict(z) #+ np.random.normal(scale=noise_level, size=train_x.shape)
    real_train_x = train_x #
    fake_y_train = np.zeros((len(fake_train_x), 1))
    real_y_train = np.ones((len(train_x), 1)) 
    train_x_samples = np.concatenate([real_train_x, fake_train_x], 0)
    train_x_labels = np.concatenate([real_y_train, fake_y_train], 0)


    ########
    # Val
    ########
    z =  np.random.normal(loc=0, scale=1, size=(test_unstable_x.shape[0], gen.input.shape[-1]))
    fake_test_unstable_x = gen.predict(z) #+ np.random.normal(scale=noise_level, size=test_unstable_x.shape)
    real_test_unstable_x = test_unstable_x #+ np.random.normal(scale=noise_level, size=test_unstable_x.shape)
    fake_y_val = np.zeros((len(fake_test_unstable_x), 1))
    real_y_val = np.ones((len(test_unstable_x), 1))
    test_unstable_x_samples = np.concatenate([real_test_unstable_x, fake_test_unstable_x], 0)
    test_unstable_x_labels = np.concatenate([real_y_val, fake_y_val], 0)
    
    # Fit
    cls.fit(train_x_samples, train_x_labels, validation_data=(test_unstable_x_samples, test_unstable_x_labels), 
            batch_size=batch_size, initial_epoch=e, epochs=e+1, 
            callbacks=[cp_callback, tensorboard_callback], verbose=2)


