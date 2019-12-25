'''
Variational autoencoder and some of its variants, such as beta-VAE.
The user provides the path to an image dataset that is used to train
the VAE and optionally specifies the VAE architecture.

By default, the architecture is such that 

    - the encoder matches the DQN 2015 architecture except the FC 
      layer has 256 units instead of 512
    - the latent layer has 32 units and
    - the decoder is the reverse of the encoder.

Some uses include:

    1. Learn low-dimensional latent representations
       and use these as features to accomplish some task

    2. Gain intuition on how different types of VAEs and
       different parameters affect the latent representations 
       learned from a user-provided image dataset.

    3. Explore the limits of using VAEs reliably for a
       user-provided dataset. For example, identify parameter 
       bounds to avoid latent variable collapse.

build_vae adapted from keras/examples/variational_autoencoder_deconv.py.

New or improved functionality includes:

    1. Support of arbitrary image type 
       -instead of just MNIST

    2. Support of color images
       -instead of just primarily B/W grayscale

    3. Support of flexible, user-specific VAE architecture
       -instead of just hard-coded architecture intended for MNIST

    4. (ongoing) Shape matching between encoder and decoder sections
       -previously none, which led to shape mismatch errors for
        non-MNIST datasets

    5. (ongoing) View reconstructions from latent factors as one
       latent activation is varied while all others are fixed.

    6. (ongoing) Return M most disentangled latent factors along
       with images of (5.) for these factors.
'''


from keras import backend as K
from keras import layers
from keras import models
from keras import losses

import os
import numpy as np
import argparse
import cv2

seed=0


# Load and prepare image data
def load_data(data_path, is_color=True):

    data = [cv2.imread(os.path.join(data_path, f), is_color) for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]
    data = np.array(data)
    img_shape = data[0].shape
    if is_color:
        data = np.reshape(data, [-1, img_shape[0], img_shape[1], img_shape[2]])
    else:
        data = np.reshape(data, [-1, img_shape[0], img_shape[1], 1])

    data = data.astype('float32') / 255.0   
        
    return data


# Sample means and log variances for reparametrization trick
def sample_params(params):
    
    z_mean, z_logvar = params
    batch_size = K.shape(z_mean)[0]
    latent_size = K.int_shape(z_mean)[1]
    eps = K.random_normal(shape=(batch_size, latent_size), mean=0.0, stddev=1.0)
    z = z_mean + K.exp(0.5 * z_logvar) * eps
    return z


# Build VAE graph
def build_vae(data, n_filters=[32, 64, 64], kernel_sizes=[8, 4, 3], strides=[4, 2, 1],
                    fc_sizes=[256], latent_size=32, batch_size=32, epochs=10):

    input_shape = data[0].shape

    enc_input = layers.Input(shape=input_shape, name='encoder_input')
    x = enc_input

    for filters, kernel_size, stride in zip(n_filters, kernel_sizes, strides):
        x = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride, padding='same', activation='relu')(x)
    
    pre_fc_shape = K.int_shape(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    for fc_size in fc_sizes:
        x = layers.Dense(fc_size, activation='relu')(x)

    z_mean = layers.Dense(latent_size, name='z_mean')(x)
    z_logvar = layers.Dense(latent_size, name='z_logvar')(x)
    z = layers.Lambda(sample_params, output_shape=(latent_size,), name='z')([z_mean, z_logvar])

    encoder = models.Model(enc_input, [z_mean, z_logvar, z], name='encoder')
    encoder.summary()

    latent_input = layers.Input(shape=(latent_size,), name='latent')
    x = latent_input

    for fc_size in fc_sizes[::-1]:
        x = layers.Dense(fc_size, activation='relu')(x)
    
    x = layers.Dense(pre_fc_shape[1] * pre_fc_shape[2] * pre_fc_shape[3], activation='relu')(x)
    x = layers.Reshape((pre_fc_shape[1], pre_fc_shape[2], pre_fc_shape[3]))(x)

#     n_filters = [3, 32, 64]
    for filters, kernel_size, stride in zip(n_filters[::-1], kernel_sizes[::-1], strides[::-1]):
        x = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=stride, padding='same', activation='relu')(x)

    dec_output = layers.Conv2DTranspose(filters=3, kernel_size=1, strides=1, padding='same', activation='sigmoid')(x)
    dec_output = layers.Cropping2D((3,0), name='decoder_output')(dec_output)

    decoder = models.Model(latent_input, dec_output, name='decoder')
    decoder.summary()

    input = enc_input
    output = decoder(encoder(input)[2])
    vae = models.Model(input, output, name='vae')

    return vae, encoder, decoder, input, output, z_mean, z_logvar


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m", "--mse", help=help_, action='store_true')
    parser.add_argument("-e", "--epochs", default=5, type=int)
    parser.add_argument("-bs", "--batch_size", default=32, type=int)
    help_ = "Path to images. All must have same format and same dimensions."
    parser.add_argument("-p", "--path", help=help_)
    help_ = "Save h5 model trained weights with --save <filename>"
    parser.add_argument("-s", "--save", help=help_)
    help_ = "Value of beta in beta-VAE"
    parser.add_argument("-b", "--beta", help=help_, default=0, type=float)
    args = parser.parse_args()

    data = load_data(os.path.join(os.getcwd(), args.path))
    vae, encoder, decoder, input, output, z_mean, z_logvar = build_vae(data, fc_sizes=[])

    # VAE loss = mse_loss or xent_loss + kl_loss
    if args.mse:
        reconstruction_loss = losses.mse(K.flatten(input), K.flatten(output))
    else:
        reconstruction_loss = losses.binary_crossentropy(K.flatten(input),
                                                  K.flatten(output))

    image_shape = data[0].shape
    reconstruction_loss *= np.prod(image_shape)
    kl_loss = 1 + z_logvar - K.square(z_mean) - K.exp(z_logvar)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + args.beta * kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='rmsprop')
    vae.summary()

    if args.weights:
        vae.load_weights(args.weights)
    else:
        # train the autoencoder
        vae.fit(data,
                epochs=args.epochs,
                batch_size=args.batch_size)
        vae.save_weights(args.save)
    
    frame = np.expand_dims(data[600], axis=0)
    z_mean, _ , _ = encoder.predict(frame)
    
    for latent_num in range(32):
        temp = z_mean
        offsets = np.linspace(-2.5, 2.5, 11)
        offset_recons = []
        for offset in offsets:
            temp[0, latent_num] = z_mean[0, latent_num] + offset
            recons_from_offsets.append(decoder.predict(temp)[0] * 255.0)
            temp = z_mean
        offset_recons_cat = np.hstack(latent_w_offsets)
        cv2.imwrite('mse_latent_{}.png'.format(latent_num), latent_all_offsets)
