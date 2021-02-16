import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset
from load_dicom import read_xray
# from torch.utils.data import DataLoader


class VinBigDicomDataset(Dataset):
    def __init__(
        self,
        img_dir="/work/VinBigData/raw_data/test/",
        img_ext="dicom",
        img_size=1024,
        class_names=[],
        transform=None
    ):
        self.img_dir = img_dir
        self.img_ext = img_ext
        self.img_size = img_size
        self.class_names = class_names
        self.transform = transform
        # Get all dicom filenames
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
        img_id = self.image_ids[idx]
        dcm_fpath = os.path.join(
            self.img_dir,
            self.image_ids[idx] + '.' + self.img_ext
        )
        # Load the dicom Images
        # Normalization has done in reading function
        img = read_xray(
            dcm_fpath,
            voi_lut=False,
            fix_monochrome=True,
            normalization=True,
            apply_window=True
        )
        if img.ndim == 2:
            img = img[..., np.newaxis]  # add channel dim
        # h0, w0, _ = img.shape

        # Resize and padding
        img, scale, padding = letterbox(img, img_size=self.img_size)

        if self.transform:
            img = self.transform(img)
        return {"img": img, "id": img_id, "scale": scale, "padding": padding}


def infer_collater(data):
    imgs = [s["img"] for s in data]
    img_ids = [s["id"] for s in data]
    scales = [s["scale"] for s in data]
    paddings = [s["padding"] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))
    return {"img": imgs, "id": img_ids, "scale": scales, "padding": paddings}


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


def letterbox(image, img_size=1024):
    # TODO: use `cv2.copyMakeBorder` to do padding
    height, width, channels = image.shape
    if height > width:
        scale = img_size / height
        resized_height = img_size
        resized_width = int(width * scale)
    else:
        scale = img_size / width
        resized_height = int(height * scale)
        resized_width = img_size

    # compute padding
    padh = max(int((img_size - resized_height) / 2.), 0)
    padw = max(int((img_size - resized_width) / 2.), 0)
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
    new_image = np.zeros((img_size, img_size, channels))
    # assert padh + resized_height <= img_size, \
    #     f"padh: {padh}, re_h: {resized_height}"
    # assert padw + resized_width <= img_size, \
    #     f"padw: {padw}, re_w: {resized_width}"
    new_image[padh: padh+resized_height, padw: padw+resized_width] = image
    return new_image, scale, (padh, padw)


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        h0, w0, _ = image.shape
        new_image, scale, (padh, padw) = letterbox(
            image, img_size=self.img_size
        )

        # annots[:, :4] *= self.img_size
        if annots.size > 0:
            annots[:, [0, 2]] = annots[:, [0, 2]] * w0 * scale + padw
            annots[:, [1, 3]] = annots[:, [1, 3]] * h0 * scale + padh

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

            x_min = annots[:, 0].copy()
            x_max = annots[:, 2].copy()

            x_tmp = x_min.copy()

            annots[:, 0] = cols - x_max
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
