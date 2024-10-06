import random
import numpy as np
from PIL import Image, ImageOps
import torch
from torchvision import transforms
import cv2

def crop(img, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)
    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask


def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask

def resize(img, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1]))

    if h > w:
        oh = long_side
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        kernel_size = int(6*sigma + 1) if int(6*sigma + 1) % 2 == 1 else int(6*sigma + 1) + 1
        blur = transforms.GaussianBlur(kernel_size, sigma)
        img=blur(img)
    return img

def random_grayscale(img, p=0.2):
    if random.random() < p:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return gray_img
    return img

def obtain_cutmix_box(img_size, p=0.5, size_min=0.1, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask

    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2)
        cutmix_w = int(np.sqrt(size / ratio))
        cutmix_h = int(np.sqrt(size * ratio))
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size:
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1

    return mask

def obtain_cutmix_box_change(img_size, p=0.5, size_min=0.25, size_max=0.5, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size)
    if random.random() > p:
        return mask
    side_length = int(np.sqrt(np.random.uniform(size_min, size_max) * img_size * img_size))
    while True:
        x = np.random.randint(0, img_size)
        y = np.random.randint(0, img_size)
        if x + side_length <= img_size and y + side_length <= img_size:
            break
    
    mask[y:y + side_length, x:x + side_length] = 1
    return mask

def normalize(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img