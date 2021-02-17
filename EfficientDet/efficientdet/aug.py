import os
import cv2
import torch
import math
import numpy as np

class RandomCrop(object):
    """Crop a random region of the input image"""
    def __init__(self, crop_height, crop_width):
        self.crop_width = crop_width
        self.crop_height = crop_height

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width = image.shape[:2]
        assert height >= self.crop_height, \
            f"Crop height: {self.crop_height}, Image height: {height}"
        assert width >= self.crop_width, \
            f"Crop width: {self.crop_width}, Image width: {width}"
        
        start_h = np.random.rand()
        start_w = np.random.rand()

        y_min = int((height - self.crop_height) * start_h)
        x_min = int((width - self.crop_width) * start_w)
        y_max = y_min + self.crop_height
        x_max = x_min + self.crop_width

        image = image[y_min:y_max, x_min:x_max]  # crop

        # Adjust Bboxes
        annots[0] -= x_min
        annots[2] -= x_min
        annots[1] -= y_min
        annots[3] -= y_min

        sample = {'img': image, 'annot': annots}
        return sample

class HorizontalFlip(object):
    """Flip the image horizontally with assigned probability"""
    def __init__(self, flip_x=0.5):
        self.flip_x = flip_x

    def __call__(self, sample):
        if np.random.rand() < self.flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x_min = annots[:, 0].copy()
            x_max = annots[:, 2].copy()

            x_tmp = x_min.copy()

            annots[:, 0] = cols - x_max
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class RandomPerspective(object):
    """Randomly adjust the image"""
    def __init__(self, degrees=10, translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border

    def _box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
        # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot'] # annots: [x_min, y_min, x_max, y_max, class]
        height = image.shape[0] + self.border[0] * 2  # shape(h,w,c)
        width = image.shape[1] + self.border[1] * 2

        # Center
        C = np.eye(3)
        C[0, 2] = -image.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -image.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * width  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * height  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        if (self.border[0] != 0) or (self.border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                image = cv2.warpPerspective(image, M, dsize=(width, height), borderValue=(114, 114, 114))
            else:  # affine
                image = cv2.warpAffine(image, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

        n = len(annots)
        if n:
            new = np.zeros((n, 4))

            # warp boxes
            xy = np.ones((n * 4, 3))
            xy[:, :2] = annots[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = xy @ M.T  # transform
            xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = _box_candidates(box1=annots[:, 0:4].T * s, box2=new.T, area_thr=0.10)
        annots = annots[i]
        annots[:, 0:4] = new[i]

        sample = {'img': image, 'annot': annots}
        return sample