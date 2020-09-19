import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from metrics import smooth_accuracy, feat_matching_loss
import numpy as np


def build_classifier(dsvdd):

    filter_size = 3 
    n_filters_factor = 2

    dsvdd.trainable = False

    c_x = keras.Input(shape=dsvdd.input.shape[1:], name='c_x')
    y = dsvdd(c_x)

    y = layers.Lambda(lambda x: keras.backend.expand_dims(x, -1))(y)
    y = layers.Conv1D(4*n_filters_factor, filter_size, padding='same')(y)
    y = layers.LeakyReLU(.3)(y)
    y = layers.AveragePooling1D(padding='same')(y)
    y = layers.Conv1D(8*n_filters_factor, filter_size, padding='same')(y)
    y = layers.LeakyReLU(.3)(y)
    y = layers.AveragePooling1D(padding='same')(y)
    y = layers.Conv1D(12*n_filters_factor, filter_size, padding='same')(y)
    y = layers.LeakyReLU(.3)(y)
    y = layers.AveragePooling1D(padding='same')(y)
    y = layers.Conv1D(24*n_filters_factor, filter_size, padding='same')(y)
    y = layers.LeakyReLU(.3)(y)
    y = layers.AveragePooling1D(padding='same')(y)
    y = layers.Conv1D(24*n_filters_factor, filter_size, padding='same')(y)
    y = layers.LeakyReLU(.3)(y)
    y = layers.AveragePooling1D(padding='same')(y)

    y = layers.Flatten()(y)
    y = layers.Dense(1)(y) # relu?
    y = layers.Activation(keras.activations.sigmoid)(y)

    cls = keras.Model(inputs=c_x, outputs=y, name='cls')
    
    return cls


def build_IO_GEN(ae, dsvdd, latent_dim,  lr, m):

    gen_lr = lr * 2 * 2

    #################### 
    # Autoencoder and
    # Encoder (Pre-trained)
    ####################

    encoder_layer_name = 'encoded'
    gen_dim = ae.get_layer(encoder_layer_name).input.shape[1:] # test, small noise input to Dense

    #################### 
    # DSC_v
    #################### 

    encoder = keras.Model(inputs=ae.input, outputs=ae.get_layer('encoded').output, name='DCAE_Encoder')
    encoder = keras.models.clone_model(encoder) # re-initialize weights 

    l2_norm = 1e-4
    d_x = keras.Input(shape=(64, 64) + (2 * m,), name='d_x')
    y = encoder(d_x) # pre-trained encoder
    y = layers.Flatten()(y)
    y = layers.Dense(1, kernel_regularizer=keras.regularizers.l2(l2_norm), activation='sigmoid')(y)
    dsc = keras.Model(inputs=d_x, outputs=y, name='DSC')
    dsc.compile(loss=['binary_crossentropy'], metrics=[smooth_accuracy], 
                optimizer=keras.optimizers.Adam(learning_rate=lr, beta_1=0.5))
    print(dsc.summary())
    dsc.trainable = False

    #################### 
    # REG - SVDD
    ####################

    dsvdd = keras.Model(inputs=dsvdd.input, outputs=dsvdd.output, name='DSVDD')
    dsvdd.trainable = False

    #################### 
    # GEN
    ####################

    y = g_x = keras.Input(shape=latent_dim, name='g_x')
    flag = False
    y = layers.Dense(np.prod(gen_dim), activation='relu')(y)

    for i, l in enumerate(ae.layers):
        if l.name == encoder_layer_name:
            flag = True

        if flag: 
            y = l(y) 

    gen = keras.Model(inputs=g_x, outputs=y, name='gen')
    print(gen.summary())

    #################### 
    # GAN
    ####################

    gan_opt = keras.optimizers.Adam(learning_rate=gen_lr, beta_1=.5)

    g_x = keras.Input(shape=latent_dim, name='g_x')
    x_star = gen(g_x)

    y = dsc(x_star)
    feat = dsvdd(x_star)
    gan = keras.Model(g_x, [feat, y], name="gan")
    gan.compile(loss={'DSVDD': feat_matching_loss, 'DSC': 'binary_crossentropy'}, 
                metrics={'DSC': 'accuracy'}, loss_weights={'DSVDD':10., 'DSC': 1.},
                optimizer=gan_opt)    

    return gan, gen, dsc   


def build_DCAE(m, img_size=(64,64)): 
    
    use_bias = True 
    l2_norm = 0
    x = keras.Input(shape=img_size + (2 * m,))
    y = layers.Conv2D(32, 3, padding='same', activation="relu", use_bias=use_bias,
                     kernel_regularizer=keras.regularizers.l2(l2_norm))(x)
    y = layers.MaxPooling2D(2, padding='same')(y)
    y = layers.Conv2D(64, 3, padding='same', activation="relu", use_bias=use_bias,
                     kernel_regularizer=keras.regularizers.l2(l2_norm))(y)
    y = layers.MaxPooling2D(2, padding='same')(y)
    y = layers.Conv2D(128, 3, padding='same', activation="relu", use_bias=use_bias,
                     kernel_regularizer=keras.regularizers.l2(l2_norm))(y)
    y = layers.MaxPooling2D(2, padding='same')(y)

    y = layers.Conv2D(32, 3, padding='same', activation="relu", use_bias=use_bias,
                     kernel_regularizer=keras.regularizers.l2(l2_norm))(y) # 32
    featmap_shape = y.shape[1:]
    y = encoded = layers.Flatten(name='encoded')(y)
    y = layers.Reshape(featmap_shape)(y)
    y = layers.UpSampling2D(size=(2, 2))(y)
    y = layers.Conv2D(128, 3, padding='same', activation="relu", use_bias=use_bias,
                     kernel_regularizer=keras.regularizers.l2(l2_norm))(y)
    y = layers.UpSampling2D(size=(2, 2))(y)
    y = layers.Conv2D(64, 3, padding='same', activation="relu", use_bias=use_bias,
                     kernel_regularizer=keras.regularizers.l2(l2_norm))(y)
    y = layers.UpSampling2D(size=(2, 2))(y)
    y = layers.Conv2D(32, 3, padding='same', activation="relu", use_bias=use_bias,
                     kernel_regularizer=keras.regularizers.l2(l2_norm))(y)
    decoded = layers.Conv2D(2 * m, 3, padding='same', activation="tanh", use_bias=use_bias, name='decoded')(y)
    ae = keras.Model(x, decoded, name="DCAE")
    
    return ae
