import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
import torch
from lime import lime_image
from torch.optim import Adam
from skimage.segmentation import slic
from _model import Model, make_layers
from _dataset import ImgDataset, data_transforms
from _utils import get_paths_labels, normalize, img_process

# set environment
matplotlib.use('Agg')
np.random.seed(1003)

# set args
args = {
    'ckptpath': './model.ckpt',
    'dataset_dir': sys.argv[1],
    'output_dir': sys.argv[2],
    'noodles_pasta': [2431, 2352, 2379, 2466, 2383],
    'rice': [2533, 2504, 2520, 2495, 2553],
    'soup': [3292, 3159, 3094, 3161, 3278],
    'vegetable_fruit': [523, 550, 507, 532, 561],
    'num_features': 11,
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

# helper functions
def predict(input):
    # input: numpy array, (batches, height, width, channels)
    model.eval()
    input = torch.FloatTensor(input).permute(0, 3, 1, 2) # (batch, channels, height, width)
    output = model(input.to(args.device))
    return output.detach().cpu().numpy()

def segmentation(input):
    return slic(input, n_segments=100, compactness=1, sigma=1)

# explain function
def lime_explain(img_indices, file_name):
    images, labels = valid_set.getbatch(img_indices)
    fig, axs = plt.subplots(1, len(img_indices), figsize=(15, 8))

    for i, (x, y) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):
        explainer = lime_image.LimeImageExplainer()
        explaination = explainer.explain_instance(image=x, labels=[y.item()], classifier_fn=predict, segmentation_fn=segmentation)
        lime_img, mask = explaination.get_image_and_mask(
            label=y.item(),
            positive_only=False,
            hide_rest=False,
            num_features=args.num_features,
            min_weight=0.05
        )
        
        axs[i].imshow(img_process(lime_img))
    
    plt.savefig(os.path.join(args.output_dir, file_name), bbox_inches='tight')

# plot!
# my classifier with highest accuracy:
# Noodles/Pasta (6): 95%
# Rice (7): 93%
# Soup (9) 94%
# Vegetable/Fruit (10): 91%

lime_explain(args.noodles_pasta, 'p3-noodles_pasta.jpg')
lime_explain(args.rice, 'p3-rice.jpg')
lime_explain(args.soup, 'p3-soup.jpg')
lime_explain(args.vegetable_fruit, 'p3-vegetable_fruit.jpg')
