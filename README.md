# vae-viewer
### Quickly get started
Portable implementation of variational autoencoders (and some variants). This tool can be used to quickly view learned latent representations of user-provided images.

Try it now on 1000 frames from the Atari game Beam Rider (i.e. sample_dataset) by running <code> python main.py --mse --beta 2 --epochs 50 --path sample_dataset -s vae_beam.h5 </code>

Here are reconstructions from varying latent factor 22

<img src="https://raw.githubusercontent.com/helloworldexpert/vae-viewer/master/mse_latent_22_beam.png" width="1000" />

### More info
Reasons to use this include:
1. Learn low-dimensional latent representations and use these as features to accomplish some task
2. Gain intuition on how different types of VAEs and different parameters affect the latent representations learned from a user-provided image dataset.
3. Explore the limits of using VAEs reliably for a user-provided dataset. For example, identify parameter bounds to avoid latent variable collapse.

New or improved functionality includes:
1. Support of arbitrary image type\
--instead of just MNIST
2. Support of color images\
--instead of just primarily B/W grayscale
3. Support of flexible, user-specific VAE architecture\
--instead of just hard-coded architecture intended for just MNIST
4. (ongoing) Shape matching between encoder and decoder sections\
--previously none, which led to shape mismatch errors for non-MNIST datasets
5. (ongoing) View reconstructions from latent factors as one latent activation is varied while all others are fixed\
--new
6. (ongoing) Return M most disentangled latent factors along with images of (5.) for these factors\
--new
