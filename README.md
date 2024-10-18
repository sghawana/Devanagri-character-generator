# Devnagri-character-generator
Generate alphabets and numbers from devnagri script using GAN and its different variations

This repository contains a Generative Adversarial Network (GAN) implemented in PyTorch to generate handwritten Devanagari character images, using the Devanagari MNIST dataset. The goal of this project is to train a GAN model capable of generating realistic images of Devanagari numerals.

---
### Dataset

The Devanagari MNIST dataset consists of 90,000 grayscale images of handwritten Devanagari numerals and characters each of size 32x32 pixels. The dataset is structured similarly to the well-known MNIST dataset but represents the numerals in the Devanagari script, commonly used in Hindi and other Indian languages.

You can download the dataset from the following link: [Devanagari MNIST dataset](https://www.kaggle.com/datasets/berlinsweird/devanagari).

---

## Model Overview

This project implements a standard GAN with the following components:

	•	Generator: A neural network that generates images from random noise (latent space).
	•	Discriminator: A neural network that tries to distinguish between real images from the dataset and fake images produced by the generator.

Both networks are trained in an adversarial manner, with the generator trying to fool the discriminator and the discriminator learning to identify real vs. fake images.



**Discriminator Loss:** <br>
$$
L_D() = - E_{x \sim p_{data}(x)}[\log(D_{\theta}(x))] - E_{z \sim p_z(z)}[\log(1 - D_{\theta}(G_{\phi}(z)))]
$$ <br>

**Generator Loss:** <br>
$$
L_G = - \mathbb{E}_{z \sim p_z(z)} \left[ \log (D(G(z))) \right]
$$



### GAN Architecture

The Architecture used is [DCGAN](https://arxiv.org/pdf/1511.06434).



	•	Generator: Fully connected layers followed by transposed convolutions (deconvolutions) to upsample the noise vector into a 28x28 image.
	•	Discriminator: Convolutional layers followed by fully connected layers to classify images as real or fake.

