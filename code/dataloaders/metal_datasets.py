import torch
import math
import numpy as np
import cv2
import random
from torch.utils.data import Dataset
from torchvision import transforms
from copy import deepcopy
from dataloaders.transform import *

class LabeledDataset(Dataset):
    def __init__(self, base_dir=None, dataset_name=None, labeled_proportion=1.0, transform=None):
        self._base_dir = base_dir + '/' + dataset_name
        self.sample_list = []
        self.dataset_name = dataset_name
        self.transform = transform

        with open(self._base_dir + '/labeled_list.txt', 'r') as f1:
            all_list = [item.replace('\n', '') for item in f1.readlines()]
            if dataset_name == 'MetalDAM':
                val_length = 10
            else:
                val_length = 4

            val_list = all_list[: val_length]
            train_list = [x for x in all_list if x not in val_list]
            train_num = math.floor(len(train_list) * labeled_proportion)
            self.sample_list = random.sample(train_list, train_num)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        img_path = self._base_dir + '/labeled_img/{}.jpg'.format(self.sample_list[idx])
        label_path = self._base_dir + '/GT/{}.png'.format(self.sample_list[idx])

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if self.dataset_name == 'MetalDAM':
            image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (label.shape[1]//2, label.shape[0]//2), interpolation=cv2.INTER_NEAREST)

        sample = {'image': image, 'label': label}
        sample = self.transform(sample)
        sample["idx"] = idx
        sample["name"] = self.sample_list[idx]
        return sample


class ValDataset(Dataset):
    def __init__(self, base_dir=None, dataset_name=None, transform=None):
        self.dataset_name = dataset_name
        self._base_dir = base_dir + '/' + dataset_name
        self.sample_list = []
        self.transform = transform

        with open(self._base_dir + '/labeled_list.txt', 'r') as f1:
            all_list = [item.replace('\n', '') for item in f1.readlines()]
            if dataset_name == 'MetalDAM':
                val_length = 10
            else:
                val_length = 4

            val_list = all_list[: val_length]
            self.sample_list = val_list

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        img_path = self._base_dir + '/labeled_img/{}.jpg'.format(self.sample_list[idx])
        label_path = self._base_dir + '/GT/{}.png'.format(self.sample_list[idx])

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        sample = {'image': image, 'label': label}
        sample = self.transform(sample)
        sample["idx"] = idx
        sample["name"] = self.sample_list[idx]
        sample["origin_gt"] = torch.from_numpy(label.astype(np.uint8))
        return sample


class UnlabeledDataset(Dataset):
    def __init__(self, base_dir=None, dataset_name=None, transform=None):
        self.dataset_name = dataset_name
        self._base_dir = base_dir + '/' + dataset_name
        self.sample_list = []
        self.transform = transform

        with open(self._base_dir + '/unlabeled_list.txt', 'r') as f1:
            self.sample_list = [item.replace('\n', '') for item in f1.readlines()]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        img_path = self._base_dir + '/unlabeled_img/{}.jpg'.format(self.sample_list[idx])

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if self.dataset_name == 'MetalDAM':
            image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2), interpolation=cv2.INTER_LINEAR)

        sample = {'image': image}
        sample = self.transform(sample)
        sample['idx'] = idx
        sample["name"] = self.sample_list[idx]
        return sample


def random_flip(image, label):

    flip_way = np.random.randint(-1, 3)
    if flip_way < 2:
        image = cv2.flip(image, flip_way)
        label = cv2.flip(label, flip_way)
    return image, label

def random_flip_unlabeled(image):
    flip_way = np.random.randint(-1, 3)
    if flip_way < 2:
        image = cv2.flip(image, flip_way)
    return image

def rand_crop(image, label, output_size):
    width1 = random.randint(0, label.shape[0]-output_size[0])
    height1 = random.randint(0, label.shape[1]-output_size[1])

    image = image[width1: width1+output_size[0], height1: height1+output_size[1]]
    label = label[width1: width1+output_size[0], height1: height1+output_size[1]]

    return image, label

def rand_crop_unlabeled(image, output_size):
    width1 = random.randint(0, image.shape[0]-output_size[0])
    height1 = random.randint(0, image.shape[1]-output_size[1])

    image = image[width1: width1+output_size[0], height1: height1+output_size[1]]

    return image

class RandomGenerator(object):
    def __init__(self, input_size, data_type='labeled'):
        self.input_size = input_size
        self.data_type = data_type

    def __call__(self, sample):
        if self.data_type == 'labeled':
            image, label = sample['image'], sample['label']
            image, label = random_flip(image, label)

            image, label = rand_crop(image, label, self.input_size)
            image = transforms.ToTensor()(image.astype(np.uint8))
            label = torch.from_numpy(label.astype(np.uint8))

            sample = {'image': image, 'label': label}

        elif self.data_type == 'unlabeled':
            image = sample['image']
            image = random_flip_unlabeled(image)
            image = rand_crop_unlabeled(image, self.input_size)

            img_w, img_s1, img_s2 = deepcopy(image), deepcopy(image), deepcopy(image)
            img_w = transforms.ToTensor()(img_w.astype(np.uint8))
            img_s1 = transforms.ToTensor()(img_s1.astype(np.uint8))
            img_s2 = transforms.ToTensor()(img_s2.astype(np.uint8))

            if random.random() < 0.8:
                img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
            img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
            img_s1 = blur(img_s1, p=0.5)
            cutmix_box1 = obtain_cutmix_box(img_s1.size()[1], p=1)  

            img_w, img_s1, img_s2, cutmix_box1
            sample = {'image': (img_w,img_s1,img_s2), 'cutmix': cutmix_box1}

        return sample


class Resize(object):

    def __init__(self, input_size):
        self.input_size = input_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        image = cv2.resize(image, self.input_size, cv2.INTER_LINEAR)
        label = cv2.resize(label, self.input_size, cv2.INTER_NEAREST)  

        image = transforms.ToTensor()(image.astype(np.uint8))
        label = torch.from_numpy(label.astype(np.uint8))

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2 * self.sigma,
                        2 * self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}



