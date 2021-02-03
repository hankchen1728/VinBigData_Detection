import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset
# from torch.utils.data import DataLoader


class VinBigDataset(Dataset):
    def __init__(
        self,
        img_dir="/work/VinBigData/preprocessed/img_npz/",
        ann_dir="/work/VinBigData/preprocessed/bbox_txt/",
        img_ext="npz",  # filetype, filename extensions
        class_names=[],
        dset="train",
        image_ids=None,
        transform=None
    ):

        self.img_dir = img_dir  # folder for storing *.npz
        self.ann_dir = ann_dir  # folder for storing *.txt
        self.set_name = dset
        self.img_ext = img_ext
        self.class_names = class_names
        self.transform = transform

        # Get the image id for current Dataset
        self.image_ids = image_ids
        if self.image_ids is None:
            self.image_ids = [
                img_fname.split('.')[0]
                for img_fname in os.listdir(self.img_dir)
                if img_fname.split('.')[-1] == img_ext
            ]

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        self.classes = {}
        for i, c in enumerate(self.class_names):
            self.classes[c] = i

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index: int) -> np.ndarray:
        img_fname = self.image_ids[image_index] + "." + self.img_ext
        path = os.path.join(self.img_dir, img_fname)
        # Read npz (compressed) data
        img = np.load(path)["img"]
        if img.ndim == 2:  # one channel gray image
            img = img[..., np.newaxis]  # add channel dim

        return img.astype(np.float32)

    def load_annotations(self, image_index: int) -> np.ndarray:
        # get ground truth annotations
        annot_fpath = os.path.join(
            self.ann_dir,
            self.image_ids[image_index] + ".txt"
        )
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if not os.path.isfile(annot_fpath):
            return annotations

        # parse annotations
        # the bbox info are normalized by original image shape (w, h)
        # hence all [x1, y1, x2, y2] are in the range [0, 1)
        with open(annot_fpath, 'r') as f:
            annot = np.array(
                [x.split() for x in f.read().strip().splitlines()],
                dtype=np.float32
            )
            # to [x1, y1, x2, y2, class_id]
            annot = np.roll(annot, shift=-1)
            annot = annot.reshape((-1, 5))  # TODO

        return annot


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    # tensor to channel first format
    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, channels = image.shape
        # print(f"Got image with shape ({height}, {width})")
        # print("Got image with shape", image.shape)
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        # compute padding
        padh = max(int((self.img_size - resized_height) / 2.), 0)
        padw = max(int((self.img_size - resized_width) / 2.), 0)
        # print(f"padh: {padh}, padw: {padw}")
        # print(f"re_h: {resized_height}, re_w: {resized_width}")

        image = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR
        )
        if image.ndim == 2:  # grayscale image
            image = image[..., np.newaxis]

        # put image in center (padding)
        new_image = np.zeros((self.img_size, self.img_size, channels))
        assert padh + resized_height <= self.img_size, \
            f"padh: {padh}, re_h: {resized_height}"
        assert padw + resized_width <= self.img_size, \
            f"padw: {padw}, re_w: {resized_width}"
        new_image[padh: padh+resized_height, padw: padw+resized_width] = image

        # annots[:, :4] *= self.img_size
        if annots.size > 0:
            annots[:, [0, 2]] = annots[:, [0, 2]] * resized_height + padh
            annots[:, [1, 3]] = annots[:, [1, 3]] * resized_width + padw

        return {
            "img": torch.from_numpy(new_image).to(torch.float32),
            "annot": torch.from_numpy(annots),
            "scale": scale
        }


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, mean=[0], std=[1]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {
            'img': ((image.astype(np.float32) - self.mean) / self.std),
            'annot': annots}
