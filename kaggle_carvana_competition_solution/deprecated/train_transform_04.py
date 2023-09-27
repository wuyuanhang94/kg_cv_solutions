import cv2
import random
import math
import numpy as np
from PIL import Image
from torchvision import transforms

class RandomHorizontalFlip(object):
    def __call__(self, images):
        assert isinstance(images, list)
        if random.random() < 0.5:
            return map(lambda x: x.transpose(Image.FLIP_LEFT_RIGHT), images)
        return images

class Rescale(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image):
        if self.size == image.size:
            return image
        else:
            image = image.resize(self.size, self.interpolation)
        return image

def random_shift_scale_rotate(images, shift_limit=(-0.0625, 0.0625), scale_limit=(1/1.1, 1.1),
                              rotate_limit=(-7, 7), aspect_limit=(-1, 1), borderMode=cv2.BORDER_REFLECT_101,
                              u=0.5):
    assert isinstance(images, list)
    # cv2.BORDER_REFLECT_101  cv2.BORDER_CONSTANT

    # add 3rd channel for mask
    images[1] = images[1].reshape(images[1].shape + (1,))

    if random.random() < u:
        height, width, channel = images[0].shape

        angle = random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = random.uniform(scale_limit[0], scale_limit[1])
        aspect = random.uniform(aspect_limit[0], aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = math.cos(angle / 180 * math.pi) * (sx)
        ss = math.sin(angle / 180 * math.pi) * (sy)
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)

        for i, image in enumerate(images):
            images[i] = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR,
                                     borderMode=borderMode)  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101
    return images

def random_brightness(image, limit=(-0.3, 0.3), u=0.5):
    assert image.max() <= 1.0
    if random.random() < u:
        alpha = 1.0 + random.uniform(limit[0], limit[1])
        image = alpha * image
        image = np.clip(image, 0., 1.)
    return image


def random_contrast(image, limit=(-0.3, 0.3), u=0.5):
    if random.random() < u:
        alpha = 1.0 + random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = image * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        image = alpha * image + gray
        image = np.clip(image, 0., 1.)
    return image


def random_saturation(image, limit=(-0.3, 0.3), u=0.5):
    if random.random() < u:
        alpha = 1.0 + random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])
        gray = image * coef
        gray = np.sum(gray, axis=2, keepdims=True)
        image = alpha * image + (1.0 - alpha) * gray
        image = np.clip(image, 0., 1.)
    return image

def random_hue(image, hue_limit=(-0.1, 0.1), u=0.5):
    if random.random() < u:
        h = int(random.uniform(hue_limit[0], hue_limit[1]) * 180)
        # print(h)

        image = (image * 255).astype(np.uint8)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + h) % 180
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.float32) / 255
    return image

class TrainTransform:
    def __init__(self, new_size):
        self.resize = Rescale(new_size)
        self.flip = RandomHorizontalFlip()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.6982, 0.6909, 0.6840], std=[0.2327, 0.2369, 0.2345])
    
    def __call__(self, image, mask):
        image = self.resize(image)
        mask = self.resize(mask)
        image, mask = self.flip([image, mask])
        image = np.array(image, dtype=np.float32)
        image = image / 255.0
        image = random_brightness(image, limit=(-0.5, 0.5), u=0.5)
        image = random_contrast(image, limit=(-0.5, 0.5), u=0.5)
        image = random_saturation(image, limit=(-0.3, 0.3), u=0.5)
        image = np.asarray(image * 255, dtype=np.uint8)
        image, mask = self.to_tensor(image), self.to_tensor(mask)
        image = self.normalize(image)
        return image, mask