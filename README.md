# vae-viewer
Portable implementation of variational autoencoders (and some variants). This tool can be used to quickly view learned latent representations of user-provided images.

Try it now on 1000 frames from the Atari game Beam Rider (i.e. sample_dataset) by running python main.py --mse --beta 2 --epochs 50 --path sample_dataset -s vae_beam.h5

Reconstructions from varying latent factor 22
<img src="https://raw.githubusercontent.com/helloworldexpert/vae-viewer/master/mse_latent_22_beam.png" width="1000" />
