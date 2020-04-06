'''
ref: https://github.com/eriklindernoren/PyTorch-Deep-Dream
'''
import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
import cv2
import sys
import torch
import numpy as np
import scipy.ndimage as nd
from tqdm import tqdm
from torch.optim import Adam
from _model import Model, make_layers
from _dataset import ImgDataset, data_transforms
from _utils import get_paths_labels, normalize, img_process

# set environment
matplotlib.use('Agg')

# set args
args = {
    'ckptpath': './model.ckpt',
    'dataset_dir': sys.argv[1],
    'output_dir': sys.argv[2],
    'img_indices': [373, 413, 428, 468],
    'cnnid': 26,
    'iterations': 100,
    'lr': 0.01,
    'octave_scale': 1.2,
    'num_octaves': 10,
    'device': 'cuda'
}
args = argparse.Namespace(**args)

# build model
model = Model(
    make_layers([
        32, 32, 32, 'M',
        64, 64, 64, 'M',
        128, 128, 128, 'M',
        256, 256, 256, 256, 'M',
        512, 512, 512, 512, 'M'
    ])
).to(args.device)

# load checkpoint
checkpoint = torch.load(args.ckptpath)
model.load_state_dict(checkpoint['model_state_dict'])

# prepare dataset
valid_paths, valid_labels = get_paths_labels(os.path.join(args.dataset_dir, 'validation'))
valid_set = ImgDataset(valid_paths, valid_labels, 512, data_transforms['test'])

# dream & deep_dream
layer_activations = None
def dream(image, model, iterations, lr):
    model.eval()
    image = torch.FloatTensor(image).to(args.device)
    image.requires_grad_()
    optimizer = Adam([image], lr=lr)
    for _ in range(iterations):
        optimizer.zero_grad()
        out = model(image)
        objective = layer_activations.norm()
        objective.backward()
        optimizer.step()
    return image.cpu().data.numpy()

def deep_dream(image, model, cnnid, iterations, lr, octave_scale, num_octaves):
    def hook(model, input, output):
        global layer_activations
        layer_activations = output
    hook_handle = model.features[26].register_forward_hook(hook)

    image = image.numpy()
    octaves = [image]
    for _ in range(num_octaves - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1, 1 / octave_scale, 1 / octave_scale), order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(tqdm(octaves[::-1], desc="Dreaming")):
        if octave > 0:
            detail = nd.zoom(detail, np.array(octave_base.shape) / np.array(detail.shape), order=1)
        input_image = octave_base + detail
        dreamed_image = dream(input_image, model, iterations, lr)
        detail = dreamed_image - octave_base

    hook_handle.remove()
    return dreamed_image

# dreaming
images, labels = valid_set.getbatch(args.img_indices)
fig, axs = plt.subplots(2, len(args.img_indices), figsize=(15, 8))
for i, (image, label) in enumerate(zip(images, labels)):
    dreamed_image = deep_dream(
        image=image.unsqueeze(0),
        model=model,
        cnnid=args.cnnid,
        iterations=args.iterations,
        lr=args.lr,
        octave_scale=args.octave_scale,
        num_octaves=args.num_octaves
    )
    axs[0][i].imshow(img_process(image))
    axs[1][i].imshow(img_process(torch.FloatTensor(dreamed_image[0])))

plt.savefig(os.path.join(args.output_dir, 'p4.jpg'), bbox_inches='tight')
