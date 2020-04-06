import argparse
import os
import sys
import torch
import matplotlib
import matplotlib.pyplot as plt
import cv2
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
    'img_indices': [900, 1282, 2554, 3232],
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

# def saliency function
def compute_saliency_maps(x, y, model):
    model.eval()
    x = x.to(args.device)
    x.requires_grad_()

    y_pred = model(x)
    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(y_pred, y.to(args.device))
    loss.backward()

    saliencies = x.grad.abs().detach().cpu()
    saliencies = torch.stack([normalize(item) for item in saliencies])
    return saliencies

# draw pic
images, labels = valid_set.getbatch(args.img_indices)
saliencies = compute_saliency_maps(images, labels, model)

fig, axs = plt.subplots(2, len(args.img_indices), figsize=(15, 8))
for row, target in enumerate([images, saliencies]):
    for column, img in enumerate(target):
        axs[row][column].imshow(img_process(img))

plt.savefig(os.path.join(args.output_dir, 'p1.jpg'), bbox_inches='tight')
