import os
import cv2
import torch
import numpy as np

def get_paths_labels(path):
    imgnames = os.listdir(path)
    imgnames.sort()
    imgpaths = []
    labels = []
    for name in imgnames:
        imgpaths.append(os.path.join(path, name))
        labels.append(int(name.split('_')[0]))
    return imgpaths, labels
        
def normalize(image):
    return (image - image.min()) / (image.max() - image.min())

def img_process(image):
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
