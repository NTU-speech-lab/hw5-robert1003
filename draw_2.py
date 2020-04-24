import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
import cv2
import sys
import torch
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
    'img_indices': [1429, 762, 3384, 210, 442],
    'cnnid': [13, 23, 33, 43],
    'filterid': [1, 1, 1, 1],
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
valid_set = ImgDataset(valid_paths, valid_labels, 128, data_transforms['test'])

# layer activation
layer_activations = None
def filter_explaination(x, model, cnnid, filterid, iteration=100, lr=1):
    model.eval()

    def hook(model, input, output):
        global layer_activations
        layer_activations = output

    hook_handle = model.features[cnnid].register_forward_hook(hook)
    model(x.to(args.device))
    filter_activations = layer_activations[:, filterid, :, :].detach().cpu()

    x = x.to(args.device)
    x.requires_grad_()
    optimizer = Adam([x], lr=lr)
    for _ in range(iteration):
        optimizer.zero_grad()
        model(x)
        objective = -layer_activations[:, filterid, :, :].sum()
        objective.backward()
        optimizer.step()

    filter_visualization = x.detach().cpu().squeeze()[0]
    hook_handle.remove()

    return filter_activations, filter_visualization

# draw filter activations and visualization
for j, (cnnid, filterid) in enumerate(zip(args.cnnid, args.filterid)):
    # draw filter activations and visualization
    images, labels = valid_set.getbatch(args.img_indices)
    filter_activations, filter_visualization = filter_explaination(
        images, model, cnnid=cnnid, filterid=filterid, iteration=100, lr=0.1
    )
    
    plt.imshow(normalize(filter_visualization.permute(1, 2, 0)))
    plt.savefig(os.path.join(args.output_dir, f'p2-{j}-1.jpg'), bbox_inches='tight')
    plt.close()

    fig, axs = plt.subplots(2, len(args.img_indices), figsize=(15, 8))
    for i, img in enumerate(images):
        axs[0][i].imshow(img_process(img))
    for i, img in enumerate(filter_activations):
        axs[1][i].imshow(normalize(img))
    
    plt.savefig(os.path.join(args.output_dir, f'p2-{j}-2.jpg'), bbox_inches='tight')
    plt.close()
