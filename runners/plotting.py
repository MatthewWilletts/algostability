from pathlib import Path
import numpy as np
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

import seaborn as sns; sns.set()

import io
from PIL import Image


def plot_recons_and_originals(recons, images, save_dir, epoch, writer=None):

    assets = Path(save_dir)
    plot_assets = assets
    plot_assets.mkdir(exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    image_grid = utils.make_grid(((images) * 255).cpu().long())
    image_grid = image_grid.numpy()

    ax1.set_title("Original", fontsize="large")
    ax1 = ax1.imshow(np.transpose(image_grid, (1, 2, 0)), interpolation='nearest')
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)

    output_grid = utils.make_grid(((recons+0.5)).cpu())
    output_grid = output_grid.numpy()

    ax2.set_title("Reconstructed", fontsize="large")
    ax2 = ax2.imshow(np.transpose(output_grid, (1, 2, 0)), interpolation='nearest')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)

    fig.tight_layout()
    filename = "reconstructions_" + str(epoch) + ".pdf"
    print('filepath', filename)
    fig.savefig(plot_assets / filename, bbox_inches="tight", pad_inches=0.5)
    if writer is not None:
        writer.add_figure("inputs and recons", fig, epoch)


def plot_recons_originals_samples(recons, images, samples, save_dir, epoch, writer=None, nrow=8):
    assets = Path(save_dir)
    plot_assets = assets
    plot_assets.mkdir(exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))

    image_grid = utils.make_grid(((images + 0.5) * 255).cpu().long(), nrow=nrow)
    image_grid = image_grid.numpy()

    ax1.set_title("Original", fontsize="large")
    ax1 = ax1.imshow(np.transpose(image_grid, (1, 2, 0)), interpolation='nearest')
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)

    output_grid = utils.make_grid(((recons + 0.5)).cpu(), nrow=nrow)
    output_grid = output_grid.numpy()

    ax2.set_title("Reconstructed", fontsize="large")
    ax2 = ax2.imshow(np.transpose(output_grid, (1, 2, 0)), interpolation='nearest')
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)

    samples_grid = utils.make_grid(((samples + 0.5)).cpu(), nrow=nrow)
    samples_grid = samples_grid.numpy()

    ax3.set_title("Samples", fontsize="large")
    ax3 = ax3.imshow(np.transpose(samples_grid, (1, 2, 0)), interpolation='nearest')
    ax3.axes.get_xaxis().set_visible(False)
    ax3.axes.get_yaxis().set_visible(False)

    fig.tight_layout()
    filename = "input_recon_sample_" + str(epoch) + ".pdf"
    fig.savefig(plot_assets / filename, bbox_inches="tight", pad_inches=0.5)
    if writer is not None:
        writer.add_figure("inputs recons samples", fig, epoch)


def sift_most_confident_images(y_pred, x, save_dir, n_plot=10, good_dims=None, index=None):
    assets = Path(save_dir)
    plot_assets = assets
    plot_assets.mkdir(exist_ok=True)

    n_classes = y_pred.shape[1]
    x = x[:len(y_pred)]
    if good_dims is not None:
        # pad with zeros and reshape to 28x28x1
        x_new = np.zeros([x.shape[0],784])
        x_new[:, good_dims] = x
        x = x_new
        x = np.reshape(x, [-1, 28, 28])
        depth = 1
        width = 28
        height = 28
    else:
        # put channels in pytorch position
        if x.shape[-1] in [1, 3]:
            x = np.transpose(x, [0, 3, 1, 2])
        depth = x.shape[-3]
        width = x.shape[-1]
        height = x.shape[-2]
    most_confident = []
    for i in range(n_classes):
        idx = np.where(y_pred.argmax(1) == i)[0]
        y_class = y_pred[idx, i]
        x_class = x[idx]
        sort_idx = np.argsort(y_class)[-n_plot:][::-1]
        sort_idx = sort_idx.copy()
        x_most_confident = x_class[sort_idx]
        if len(x_most_confident) < n_plot:
            black_image = np.zeros_like(x[0])[np.newaxis,:]
            while len(x_most_confident) < n_plot:
                x_most_confident = np.vstack([x_most_confident, black_image])
        most_confident.append(x_most_confident)
    #
    images = np.array(most_confident)
    if good_dims is not None:
        images = np.transpose(images, [1, 0, 2, 3])
        images = np.reshape(images, (n_plot, n_classes, width, height))
        images = np.transpose(images, [0, 2, 1, 3])
        images = np.reshape(images, [n_plot * width, n_classes * height])
    elif len(images.shape) == 4:
        images = np.transpose(images, [1, 0, 2, 3])
        images = np.reshape(images, (n_plot, n_classes, width, height))
        images = np.transpose(images, [0, 2, 1, 3])
        images = np.reshape(images, [n_plot * width, n_classes * height])
    else:
        images = np.transpose(images, [1, 0, 3, 4, 2])
        images = np.reshape(images, (n_plot, n_classes, width, height, depth))
        images = np.transpose(images, [0, 2, 1, 3, 4])
        images = np.reshape(images, [n_plot * width, n_classes * height, depth])
    img = Image.fromarray(images)
    img.save(save_dir + '/most_confident.png')
    # utils.save_image(images, args.checkpoints + '/most_confident.png')