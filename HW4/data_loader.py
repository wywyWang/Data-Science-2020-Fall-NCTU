import os
import glob

import cv2
import imageio
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import xml.etree.ElementTree as ET
from PIL import Image
from albumentations.pytorch import ToTensor

import torch
from torch.utils.data import Dataset, DataLoader


def load_bbox(file, breed_map, root_annots):
    """ Gets bounding box in annotation file. """

    file = str(breed_map[file.split('_')[0]]) + '/' + str(file.split('.')[0])
    path = os.path.join(root_annots, file)
    tree = ET.parse(path)
    root = tree.getroot()
    objects = root.findall('object')
    for o in objects:
        bndbox = o.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

    return (xmin, ymin, xmax, ymax)


def get_resized_bbox(height, width, bbox):
    """ Adjusts bounding box from annotation, since input
        data should be squares.
    """
    xmin, ymin, xmax, ymax = bbox
    xlen = xmax - xmin
    ylen = ymax - ymin

    if xlen > ylen:
        diff = xlen - ylen
        min_pad = min(ymin, diff // 2)
        max_pad = min(height - ymax, diff - min_pad)
        ymin = ymin - min_pad
        ymax = ymax + max_pad

    elif ylen > xlen:
        diff = ylen - xlen
        min_pad = min(xmin, diff // 2)
        max_pad = min(width - xmax, diff - min_pad)
        xmin = xmin - min_pad
        xmax = xmax + max_pad

    return xmin, ymin, xmax, ymax


def load_bboxcrop_resized_image(file, bbox, root_images):
    """ Crops and resizes images according to bounding box. """

    img = cv2.imread(os.path.join(root_images, file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    xmin, ymin, xmax, ymax = bbox
    img = img[ymin:ymax, xmin:xmax]

    transform = A.Compose([
        A.Resize(64, 64, interpolation=cv2.INTER_AREA),
        A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = transform(image=img)['image']

    return img


class DogDataset(Dataset):
    def __init__(self, images):
        super().__init__()
        self.images = images
        self.transform = A.Compose([ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = self.transform(image=img)['image']

        return img


def prepare_loader(root_images, root_annots, batch_size):
    all_files = os.listdir(root_images)

    breeds = glob.glob(root_annots + '*')
    annotations = []
    for breed in breeds:
        annotations += glob.glob(breed + '/*')

    breed_map = {}
    for annotation in annotations:
        breed = annotation.split('/')[-2] # ex.. 'n02110185-Siberian_husky'
        index = breed.split('-')[0] # ex.. 'n02110185'
        breed_map.setdefault(index, breed) # ex.. {'n02110185':'n02110185-Siberian_husky'}

    all_bboxes = [
        load_bbox(file, breed_map, root_annots) for file in all_files
    ] # ex.. [(177, 160, 314, 252), (11, 79, 221, 380),...]

    print('Total files       : {}'.format(len(all_files)))
    print('Total bboxes      : {}'.format(len(all_bboxes)))
    print('Total annotations : {}'.format(len(annotations)))

    resized_bboxes = []
    for file, bbox in zip(all_files, all_bboxes):
        img = Image.open(os.path.join(root_images, file))
        width, height = img.size
        xmin, ymin, xmax, ymax = get_resized_bbox(height, width, bbox)
        resized_bboxes.append((xmin, ymin, xmax, ymax))
    # ex.. [(177, 138, 314, 275), (0, 79, 301, 380),...]

    all_images = [
        load_bboxcrop_resized_image(f, b, root_images)
        for f, b in zip(all_files, resized_bboxes)
    ]
    all_images = np.array(all_images)

    train_dataset = DogDataset(all_images)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)

    return train_dataloader
