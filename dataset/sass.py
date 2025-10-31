import numpy as np
import matplotlib.pyplot as plt
from dataset.transform import *
import math
import os
from PIL import Image
import random
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as tf
from tqdm import tqdm
from util.utils import *
import torch
from torch.utils.data import DataLoader
from model.tools import *

def random_crop(image,mask,crop_size=(512,512), ignore_label=5):
    not_valid = True
    n = 0
    cls_label = np.unique(np.asarray(mask))
    while not_valid:
        i, j, h, w = transforms.RandomCrop.get_params(image,output_size=crop_size)
        image_crop = tf.crop(image,i,j,h,w)
        mask_crop = tf.crop(mask,i,j,h,w)
            
        label = np.asarray(mask_crop, np.float32)       
        n=n+1

        if np.sum(label!=ignore_label)>1:
            # print(f'cls_label is {cls_label}')
            not_valid = False

    return image_crop,mask_crop

######################################### This set for training and val(only for scribble mode) #######################################
class TreeDataset(Dataset):
    def __init__(self, name, root, mode, size, aug=True):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.aug = aug
        self.ignore_class = 2  # tree

        self.img_path = root + '/images/'
        self.true_mask_path = root + '/mask_results_255/'

        if mode == 'val':
            self.label_path = self.true_mask_path
            id_path = 'dataset/splits/%s/val.txt' % name
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

        else:
            id_path = 'dataset/splits/%s/train.txt' % name
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

            if mode == 'full':
                self.label_path = self.true_mask_path
            elif mode == 'point':
                self.label_path = root + '/point'
            elif mode == 'scribble':
                self.label_path = root + '/3_sparse_label'
            else:
                self.label_path = root + '/3_sparse_label'

    def get_cls_label(self, cls_label):
        cls_label_set = list(cls_label)

        if self.ignore_class in cls_label_set:
            cls_label_set.remove(self.ignore_class)
        if 255 in cls_label_set:
            cls_label_set.remove(255)

        cls_label = np.zeros(self.ignore_class)
        for i in cls_label_set:
            cls_label[i] += 1
        cls_label = torch.from_numpy(cls_label).float()
        return cls_label

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.img_path, id.split(' ')[0]))
        mask = Image.open(os.path.join(self.label_path, id.split(' ')[1]))

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id

        cls_label = np.unique(np.asarray(mask))

        # basic augmentation on all training images
        img, mask = resize([img, mask], (0.5, 2.0))
        img, mask = crop([img, mask], self.size)
        img, mask = hflip(img, mask, p=0.5)

        # # strong augmentation
        if self.aug:
            if random.random() < 0.5:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.1)(img)
            img = blur(img, p=0.5)
            img = random_bright(img, p=0.5)

        img, mask = normalize(img, mask)
        cls_label = self.get_cls_label(cls_label)
        return img, mask, cls_label, id

    def __len__(self):
        return len(self.ids)


######################################### This set for predict and val(only for scribble mode) ########################################
#########################################             Tree  Test                               ########################################
class TreeTestDataset(Dataset):
    def __init__(self, name, root, mode, size, aug=True):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.aug = aug
        self.ignore_class = 2  # tree

        self.img_path = root + '/test_images/'
        self.true_mask_path = root + '/labels_v2/'

        if mode == 'val':
            self.label_path = self.true_mask_path
            id_path = 'dataset/splits/%s/val.txt' % name
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

    def get_cls_label(self, cls_label):
        cls_label_set = list(cls_label)

        if self.ignore_class in cls_label_set:
            cls_label_set.remove(self.ignore_class)
        if 255 in cls_label_set:
            cls_label_set.remove(255)

        cls_label = np.zeros(self.ignore_class)
        for i in cls_label_set:
            cls_label[i] += 1
        cls_label = torch.from_numpy(cls_label).float()
        return cls_label

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.img_path, id.split(' ')[0]))
        mask = Image.open(os.path.join(self.label_path, id.split(' ')[1]))

        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id

        cls_label = np.unique(np.asarray(mask))

        # basic augmentation on all training images
        img, mask = resize([img, mask], (0.5, 2.0))
        img, mask = crop([img, mask], self.size)
        img, mask = hflip(img, mask, p=0.5)

        # # strong augmentation
        if self.aug:
            if random.random() < 0.5:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.1)(img)
            img = blur(img, p=0.5)
            img = random_bright(img, p=0.5)

        img, mask = normalize(img, mask)
        cls_label = self.get_cls_label(cls_label)
        return img, mask, cls_label, id

    def __len__(self):
        return len(self.ids)
    


class ISPRSDataset(Dataset):
    def __init__(self, name, root, mode, size, sup_id, aug=True):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.aug = aug
        self.ignore_class = 5  # ['impervious surface','Building','Low vegetation','Tree','Car']
        self.img_path = root + '/img/'
        self.true_mask_path = root + '/gt/'
        self.sup_id = sup_id

        if mode == 'val':
            self.label_path = self.true_mask_path
            id_path = 'dataset/splits/%s/val.txt' % name
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

        else:
            id_path = 'dataset/splits/%s/train.txt' % name
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

            if mode == 'full':
                self.label_path = self.true_mask_path
            elif mode == 'point':
                self.label_path = root + '/point/' + self.sup_id
            elif mode == 'line':
                self.label_path = root + '/line/' + self.sup_id
            else:
                self.label_path = root + '/polygon/' + self.sup_id

    def get_cls_label(self, cls_label):
        cls_label_set = list(cls_label)

        if self.ignore_class in cls_label_set:
            cls_label_set.remove(self.ignore_class)
        if 255 in cls_label_set:
            cls_label_set.remove(255)

        cls_label = np.zeros(self.ignore_class)
        for i in cls_label_set:
            cls_label[i] += 1
        cls_label = torch.from_numpy(cls_label).float()
        return cls_label

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.img_path, id))
        if self.mode == 'val':
            mask_gt = Image.open(os.path.join(self.label_path, id))
            img, mask_gt = normalize(img, mask_gt)
            # print(f"Val Image shape: {img.shape}")
            # print(f"Val Label shape: {mask_gt.shape}")
            return img, mask_gt, id
        elif self.mode == 'full':
            mask = Image.open(os.path.join(self.label_path, id))
        else: 
            mask = Image.open(os.path.join(self.label_path, 'mask_'+id))

        cls_label = np.unique(np.asarray(mask))
        # basic augmentation on all training images
        # img, mask = resize([img, mask], (0.235,0.5))
        img, mask = random_crop(img, mask, (self.size,self.size), 5)
        img, mask = hflip(img, mask, p=0.5)

        # # strong augmentation
        if self.aug:
            if random.random() < 0.5:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.1)(img)
            img = blur(img, p=0.5)
            img = random_bright(img, p=0.5)

        img, mask = normalize(img, mask)
        cls_label = self.get_cls_label(cls_label)
        # print(f"Image shape: {img.shape}")
        # print(f"Label shape: {mask.shape}")
        # print(f"class Label is: {cls_label}")
        return img, mask, cls_label, id

    def __len__(self):
        return len(self.ids)

class ZurichDataset(Dataset):
    def __init__(self, name, root, mode, size, sup_id, aug=True):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.aug = aug
        self.ignore_class = 9  # [Road','Build','Tree','Grass','Bardland','Water','Railway','Pool','Bg']
        self.img_path = root + '/img/'
        self.true_mask_path = root + '/gt/'
        self.sup_id = sup_id

        if mode == 'val':
            self.label_path = self.true_mask_path
            id_path = 'dataset/splits/%s/val.txt' % name
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

        else:
            id_path = 'dataset/splits/%s/train.txt' % name
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()

            if mode == 'full':
                self.label_path = self.true_mask_path
            elif mode == 'point':
                self.label_path = root + '/point/' + self.sup_id
            elif mode == 'line':
                self.label_path = root + '/line/' + self.sup_id
            else:
                self.label_path = root + '/polygon/' + self.sup_id

    def get_cls_label(self, cls_label):
        cls_label_set = list(cls_label)

        if self.ignore_class in cls_label_set:
            cls_label_set.remove(self.ignore_class)
        if 255 in cls_label_set:
            cls_label_set.remove(255)
        cls_label = np.zeros(self.ignore_class)
        for i in cls_label_set:
            cls_label[i] += 1
        cls_label = torch.from_numpy(cls_label).float()
        return cls_label

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.img_path, id))
        if self.mode == 'val':
            mask_gt = Image.open(os.path.join(self.label_path, id))
            img, mask_gt = normalize(img, mask_gt)
            # print(f"Val Image shape: {img.shape}")
            # print(f"Val Label shape: {mask_gt.shape}")
            return img, mask_gt, id
        elif self.mode == 'full':
            mask = Image.open(os.path.join(self.label_path, id))
        else: 
            mask = Image.open(os.path.join(self.label_path, 'mask_'+id))
        # basic augmentation on all training images
        # img, mask = resize([img, mask], (0.8,2.0))
        # mask_numpy = np.array(mask)
        # mask_numpy[mask_numpy>7] = 255
        # mask = Image.fromarray(mask_numpy)
        cls_label = np.unique(np.asarray(mask))
        # print(f"class Label is: {cls_label}")
        img, mask = random_crop(img, mask, (self.size,self.size), 9)
        img, mask = hflip(img, mask, p=0.5)

        # # strong augmentation
        if self.aug:
            if random.random() < 0.5:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.1)(img)
            img = blur(img, p=0.5)
            img = random_bright(img, p=0.5)

        img, mask = normalize(img, mask)
        cls_label = self.get_cls_label(cls_label)
        # print(f"Image shape: {img.shape}")
        # print(f"class Label is: {cls_label}")
        return img, mask, cls_label, id

    def __len__(self):
        return len(self.ids)

######################################### These set below for model predict ###############################################
#########################################            Sao_Paulo              ###############################################
class Sao_PauloDataset(Dataset):
    def __init__(self, name, root, mode, size, aug=True):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.aug = aug
        self.ignore_class = 2  # tree

        self.img_path = root + '/Sao_Paulo_sub448/'
        # self.true_mask_path = root + '/mask_results_255/'

        #if mode == 'val':
            # self.label_path = self.true_mask_path
        id_path = 'dataset/splits/%s/val.txt' % name
        with open(id_path, 'r') as f:
            self.ids = f.read().splitlines()

        #else:
        #    id_path = 'dataset/splits/%s/train.txt' % name
        #    with open(id_path, 'r') as f:
        #        self.ids = f.read().splitlines()
        #
        #    if mode == 'full':
        #        self.label_path = self.true_mask_path
        #    elif mode == 'point':
        #        self.label_path = root + '/point'
        #    elif mode == 'scribble':
        #        self.label_path = root + '/3_sparse_label'
        #    else:
        #        self.label_path = root + '/3_sparse_label'

    def get_cls_label(self, cls_label):
        cls_label_set = list(cls_label)

        if self.ignore_class in cls_label_set:
            cls_label_set.remove(self.ignore_class)
        if 255 in cls_label_set:
            cls_label_set.remove(255)

        cls_label = np.zeros(self.ignore_class)
        for i in cls_label_set:
            cls_label[i] += 1
        cls_label = torch.from_numpy(cls_label).float()
        return cls_label

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.img_path, id.split(' ')[0]))
        # mask = Image.open(os.path.join(self.label_path, id.split(' ')[1]))

        if self.mode == 'val':
            img = normalize(img)
            return img, id

        # cls_label = np.unique(np.asarray(mask))

        # basic augmentation on all training images
        img = resize([img], (0.5, 2.0))
        img = crop([img], self.size)
        img = hflip(img, p=0.5)

        # # strong augmentation
        if self.aug:
            if random.random() < 0.5:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.1)(img)
            img = blur(img, p=0.5)
            img = random_bright(img, p=0.5)

        img = normalize(img)
        cls_label = self.get_cls_label(cls_label)
        return img, cls_label, id

    def __len__(self):
        return len(self.ids)

######################################### These set below for model predict ###############################################
#########################################            Vaihingen                ###############################################
class VaihingenDataset(Dataset):
    def __init__(self, name, root, mode, size, aug=True):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.aug = aug
        self.ignore_class = 5  # impervious surface, Building, Low vegetation, Tree, Car

        self.img_path = root + '/img/'
        id_path = 'dataset/splits/%s/val.txt' % name

        with open(id_path, 'r') as f:
            self.ids = f.read().splitlines()

    def get_cls_label(self, cls_label):
        cls_label_set = list(cls_label)

        if self.ignore_class in cls_label_set:
            cls_label_set.remove(self.ignore_class)
        if 255 in cls_label_set:
            cls_label_set.remove(255)

        cls_label = np.zeros(self.ignore_class)
        for i in cls_label_set:
            cls_label[i] += 1
        cls_label = torch.from_numpy(cls_label).float()
        return cls_label

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.img_path, id.split(' ')[0]))
        # mask = Image.open(os.path.join(self.label_path, id.split(' ')[1]))

        if self.mode == 'val':
            img = normalize(img)
            return img, id

        # cls_label = np.unique(np.asarray(mask))

        # basic augmentation on all training images
        img = resize([img], (0.5, 2.0))
        img = crop([img], self.size)
        img = hflip(img, p=0.5)

        # # strong augmentation
        if self.aug:
            if random.random() < 0.5:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.1)(img)
            img = blur(img, p=0.5)
            img = random_bright(img, p=0.5)

        img = normalize(img)
        cls_label = self.get_cls_label(cls_label)
        return img, cls_label, id

    def __len__(self):
        return len(self.ids)
      
#########################################            Santiago               ###############################################
class SantiagoDataset(Dataset):
    def __init__(self, name, root, mode, size, aug=True):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.aug = aug
        self.ignore_class = 2  # tree

        self.img_path = root + '/Santiago_sub448/'

        id_path = 'dataset/splits/%s/val.txt' % name
        with open(id_path, 'r') as f:
            self.ids = f.read().splitlines()

    def get_cls_label(self, cls_label):
        cls_label_set = list(cls_label)

        if self.ignore_class in cls_label_set:
            cls_label_set.remove(self.ignore_class)
        if 255 in cls_label_set:
            cls_label_set.remove(255)

        cls_label = np.zeros(self.ignore_class)
        for i in cls_label_set:
            cls_label[i] += 1
        cls_label = torch.from_numpy(cls_label).float()
        return cls_label

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.img_path, id.split(' ')[0]))
        # mask = Image.open(os.path.join(self.label_path, id.split(' ')[1]))

        if self.mode == 'val':
            img = normalize(img)
            return img, id

        # cls_label = np.unique(np.asarray(mask))

        # basic augmentation on all training images
        img = resize([img], (0.5, 2.0))
        img = crop([img], self.size)
        img = hflip(img, p=0.5)

        # # strong augmentation
        if self.aug:
            if random.random() < 0.5:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.1)(img)
            img = blur(img, p=0.5)
            img = random_bright(img, p=0.5)

        img = normalize(img)
        cls_label = self.get_cls_label(cls_label)
        return img, cls_label, id

    def __len__(self):
        return len(self.ids)
      
######################################### These set below for model predict ###############################################
#########################################            Manaus                 ###############################################
class ManausDataset(Dataset):
    def __init__(self, name, root, mode, size, aug=True):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.aug = aug
        self.ignore_class = 2  # tree

        self.img_path = root + '/Manaus_sub448/'
        id_path = 'dataset/splits/%s/val.txt' % name

        with open(id_path, 'r') as f:
            self.ids = f.read().splitlines()

    def get_cls_label(self, cls_label):
        cls_label_set = list(cls_label)

        if self.ignore_class in cls_label_set:
            cls_label_set.remove(self.ignore_class)
        if 255 in cls_label_set:
            cls_label_set.remove(255)

        cls_label = np.zeros(self.ignore_class)
        for i in cls_label_set:
            cls_label[i] += 1
        cls_label = torch.from_numpy(cls_label).float()
        return cls_label

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.img_path, id.split(' ')[0]))
        # mask = Image.open(os.path.join(self.label_path, id.split(' ')[1]))

        if self.mode == 'val':
            img = normalize(img)
            return img, id

        # cls_label = np.unique(np.asarray(mask))

        # basic augmentation on all training images
        img = resize([img], (0.5, 2.0))
        img = crop([img], self.size)
        img = hflip(img, p=0.5)

        # # strong augmentation
        if self.aug:
            if random.random() < 0.5:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.1)(img)
            img = blur(img, p=0.5)
            img = random_bright(img, p=0.5)

        img = normalize(img)
        cls_label = self.get_cls_label(cls_label)
        return img, cls_label, id

    def __len__(self):
        return len(self.ids)
      
######################################### These set below for model predict ###############################################
#########################################            Caracas                ###############################################
class CaracasDataset(Dataset):
    def __init__(self, name, root, mode, size, aug=True):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.aug = aug
        self.ignore_class = 2  # tree

        self.img_path = root + '/Caracas_sub448/'

        id_path = 'dataset/splits/%s/val.txt' % name
        with open(id_path, 'r') as f:
            self.ids = f.read().splitlines()

    def get_cls_label(self, cls_label):
        cls_label_set = list(cls_label)

        if self.ignore_class in cls_label_set:
            cls_label_set.remove(self.ignore_class)
        if 255 in cls_label_set:
            cls_label_set.remove(255)

        cls_label = np.zeros(self.ignore_class)
        for i in cls_label_set:
            cls_label[i] += 1
        cls_label = torch.from_numpy(cls_label).float()
        return cls_label

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.img_path, id.split(' ')[0]))
        # mask = Image.open(os.path.join(self.label_path, id.split(' ')[1]))

        if self.mode == 'val':
            img = normalize(img)
            return img, id

        # cls_label = np.unique(np.asarray(mask))

        # basic augmentation on all training images
        img = resize([img], (0.5, 2.0))
        img = crop([img], self.size)
        img = hflip(img, p=0.5)

        # # strong augmentation
        if self.aug:
            if random.random() < 0.5:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.1)(img)
            img = blur(img, p=0.5)
            img = random_bright(img, p=0.5)

        img = normalize(img)
        cls_label = self.get_cls_label(cls_label)
        return img, cls_label, id

    def __len__(self):
        return len(self.ids)
      
######################################### These set below for model predict ###############################################
#########################################            Lima                   ###############################################
class LimaDataset(Dataset):
    def __init__(self, name, root, mode, size, aug=True):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.aug = aug
        self.ignore_class = 2  # tree

        self.img_path = root + '/Lima_sub448/'

        id_path = 'dataset/splits/%s/val.txt' % name
        with open(id_path, 'r') as f:
            self.ids = f.read().splitlines()

    def get_cls_label(self, cls_label):
        cls_label_set = list(cls_label)

        if self.ignore_class in cls_label_set:
            cls_label_set.remove(self.ignore_class)
        if 255 in cls_label_set:
            cls_label_set.remove(255)

        cls_label = np.zeros(self.ignore_class)
        for i in cls_label_set:
            cls_label[i] += 1
        cls_label = torch.from_numpy(cls_label).float()
        return cls_label

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.img_path, id.split(' ')[0]))
        # mask = Image.open(os.path.join(self.label_path, id.split(' ')[1]))

        if self.mode == 'val':
            img = normalize(img)
            return img, id

        # cls_label = np.unique(np.asarray(mask))

        # basic augmentation on all training images
        img = resize([img], (0.5, 2.0))
        img = crop([img], self.size)
        img = hflip(img, p=0.5)

        # # strong augmentation
        if self.aug:
            if random.random() < 0.5:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.1)(img)
            img = blur(img, p=0.5)
            img = random_bright(img, p=0.5)

        img = normalize(img)
        cls_label = self.get_cls_label(cls_label)
        return img, cls_label, id

    def __len__(self):
        return len(self.ids)

######################################### These set below for model predict ###############################################
#########################################            Buenos_Aires           ###############################################
class Buenos_AiresDataset(Dataset):
    def __init__(self, name, root, mode, size, aug=True):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.aug = aug
        self.ignore_class = 2  # tree

        self.img_path = root + '/Buenos_Aires_sub448/'

        id_path = 'dataset/splits/%s/val.txt' % name
        with open(id_path, 'r') as f:
            self.ids = f.read().splitlines()

    def get_cls_label(self, cls_label):
        cls_label_set = list(cls_label)

        if self.ignore_class in cls_label_set:
            cls_label_set.remove(self.ignore_class)
        if 255 in cls_label_set:
            cls_label_set.remove(255)

        cls_label = np.zeros(self.ignore_class)
        for i in cls_label_set:
            cls_label[i] += 1
        cls_label = torch.from_numpy(cls_label).float()
        return cls_label

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.img_path, id.split(' ')[0]))
        # mask = Image.open(os.path.join(self.label_path, id.split(' ')[1]))

        if self.mode == 'val':
            img = normalize(img)
            return img, id

        # cls_label = np.unique(np.asarray(mask))

        # basic augmentation on all training images
        img = resize([img], (0.5, 2.0))
        img = crop([img], self.size)
        img = hflip(img, p=0.5)

        # # strong augmentation
        if self.aug:
            if random.random() < 0.5:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.1)(img)
            img = blur(img, p=0.5)
            img = random_bright(img, p=0.5)

        img = normalize(img)
        cls_label = self.get_cls_label(cls_label)
        return img, cls_label, id

    def __len__(self):
        return len(self.ids)

######################################### These set below for model predict ###############################################
#########################################            Brasilia               ###############################################
class BrasiliaDataset(Dataset):
    def __init__(self, name, root, mode, size, aug=True):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.aug = aug
        self.ignore_class = 2  # tree

        self.img_path = root + '/Brasilia_sub448/'

        id_path = 'dataset/splits/%s/val.txt' % name
        with open(id_path, 'r') as f:
            self.ids = f.read().splitlines()

    def get_cls_label(self, cls_label):
        cls_label_set = list(cls_label)

        if self.ignore_class in cls_label_set:
            cls_label_set.remove(self.ignore_class)
        if 255 in cls_label_set:
            cls_label_set.remove(255)

        cls_label = np.zeros(self.ignore_class)
        for i in cls_label_set:
            cls_label[i] += 1
        cls_label = torch.from_numpy(cls_label).float()
        return cls_label

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.img_path, id.split(' ')[0]))
        # mask = Image.open(os.path.join(self.label_path, id.split(' ')[1]))

        if self.mode == 'val':
            img = normalize(img)
            return img, id

        # cls_label = np.unique(np.asarray(mask))

        # basic augmentation on all training images
        img = resize([img], (0.5, 2.0))
        img = crop([img], self.size)
        img = hflip(img, p=0.5)

        # # strong augmentation
        if self.aug:
            if random.random() < 0.5:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.1)(img)
            img = blur(img, p=0.5)
            img = random_bright(img, p=0.5)

        img = normalize(img)
        cls_label = self.get_cls_label(cls_label)
        return img, cls_label, id

    def __len__(self):
        return len(self.ids)
 
######################################### These set below for model predict ###############################################
#########################################            Bogota                 ###############################################
class BogotaDataset(Dataset):
    def __init__(self, name, root, mode, size, aug=True):
        """
        :param name: dataset name, pascal or cityscapes
        :param root: root path of the dataset.
        :param mode: train: supervised learning only with labeled images, no unlabeled images are leveraged.
                     label: pseudo labeling the remaining unlabeled images.
                     semi_train: semi-supervised learning with both labeled and unlabeled images.
                     val: validation.

        :param size: crop size of training images.
        """
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        self.aug = aug
        self.ignore_class = 2  # tree

        self.img_path = root + '/Bogota_sub448/'

        id_path = 'dataset/splits/%s/val.txt' % name
        with open(id_path, 'r') as f:
            self.ids = f.read().splitlines()

    def get_cls_label(self, cls_label):
        cls_label_set = list(cls_label)

        if self.ignore_class in cls_label_set:
            cls_label_set.remove(self.ignore_class)
        if 255 in cls_label_set:
            cls_label_set.remove(255)

        cls_label = np.zeros(self.ignore_class)
        for i in cls_label_set:
            cls_label[i] += 1
        cls_label = torch.from_numpy(cls_label).float()
        return cls_label

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.img_path, id.split(' ')[0]))
        # mask = Image.open(os.path.join(self.label_path, id.split(' ')[1]))

        if self.mode == 'val':
            img = normalize(img)
            return img, id

        # cls_label = np.unique(np.asarray(mask))

        # basic augmentation on all training images
        img = resize([img], (0.5, 2.0))
        img = crop([img], self.size)
        img = hflip(img, p=0.5)

        # # strong augmentation
        if self.aug:
            if random.random() < 0.5:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.1)(img)
            img = blur(img, p=0.5)
            img = random_bright(img, p=0.5)

        img = normalize(img)
        cls_label = self.get_cls_label(cls_label)
        return img, cls_label, id

    def __len__(self):
        return len(self.ids)
