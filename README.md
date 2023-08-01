# Image Denoising and the Generative Accumulation of Photons
This repository contains the code for the paper: [Image Denoising and the Generative Accumulation of Photons](https://arxiv.org/abs/2307.06607)

Shot noise is a fundamental property of many imaging applications, especially in fluorescence microscopy. Removing this noise is an ill-defined problem since many clean solutions exist for a single noisy image. Traditional approaches aiming to map a noisy image to its clean counterpart usually find the minimum mean square error (MMSE) solution,i.e. , they predict the expected value of the posterior distribution of possible clean images. We present a fresh perspective on shot noise corrupted images and noise removal.
We refer to this as the Generative Acuumumulation of Photons (GAP).
By viewing image formation as the sequential accumulation of photons on a detector grid, we show that a network trained to predict the where the next photon could arrive is in fact solving the traditional MMSE denoising task. This new perspective allows us to make three contributions:
* (i) We present a new strategy for self-supervised denoising.
* (ii) We present a new method for sampling from the posterior of possible solutions by iteratively sampling and adding small numbers of photons to the image.
* (iii) We derive a full generative model by starting this process from an empty canvas.

We evaluate our method quantitatively and qualitatively on 4 new fluorescence microscopy datasets, which will be made available to the community. We find that it outperforms supervised, self-supervised and unsupervised baselines or performs on-par. Additionally, we present preliminary results of our generative model applied to natural images and achieve visually convincing results making us hopeful that our new perspective might be applicable in areas beyond microscopy.

## Self-supervised MMSE denoising *(i)*
We can use GAP for MMSE denoising by training a network to predict where the next photon will land.
Such a network can be trained in a self-supervised way, requiring only noisy data, by randomly removing individual photons and using them as target during training.
The images below show our results on various datasets. Our method was trained using purely noisy data.
![image](https://github.com/krulllab/GAP/assets/1193459/e87abde8-8e74-469b-a43c-68652f2be0ae)

## Diversity denoising *(ii)*
GAP models can be used to remove shot noise by taking the noisy image as starting point and sequentially adding additional photons, until a clean image is produced.
By repeating the process, we can obtain multiple diverse samples form the posterior distribution of possible clean images.
The images below show diversity denoising results usinginput images with different levels of noise.
Less noisy inputs lead to less diverse predictions as more information about the clean image becomes available.
Arrows and insets highlight structural differences in the samples.
![image](https://github.com/krulllab/GAP/assets/1193459/c4898439-afc6-46ea-affe-a0aa402ce14f)

## Image generation *(iii)*
By applying the same proces as in *(ii)*, but starting with an empty image we are able to use GAP as a full generative model.
The videos below show the generative process for the Conv-PC dataset and the FFHQ-256x256 dataset.
The **left panel shows the accumulating photons**, with intensity corresponding to the corrent photon count at the pixel.
The **right panel shows the MMSE-denoised version, i.e., predicted next photon position**.

https://github.com/krulllab/GAP/assets/1193459/1d7d5334-ef71-466f-a93a-fd928d6297db

https://github.com/krulllab/GAP/assets/1193459/fcbc0286-338f-4e6d-bdc3-ed103b9fed27


# Data 
## Conv-PC Dataset
The convallaria photon counting dataset is available [here](https://figshare.com/articles/dataset/Convallaria_Photon_Counting_Dataset_Conv-PC_/23675334).

## Other Datasets
Our other datasets will be made available shortly.
Links will be provided here.
You can use this code with your own data, provided that integer pixel values correspond to photon counts, or other counts of independently occuring events.

# Code
We tested our code using **pytorch 1.12.0**, **torchvision 0.13.0** and **pytorch-lightning 1.6.5** on a Ubuntu 22.04.1 LTS system.
We also provide gap.yml, which should contian all dependencies.
We provide example notebooks in the **examples** subfolder.


