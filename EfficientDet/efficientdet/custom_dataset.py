import os
from multiprocessing.pool import ThreadPool

import cv2
import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset
from efficientdet.load_dicom import read_xray


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
        orig_img = read_xray(
            dcm_fpath,
            voi_lut=False,
            fix_monochrome=True,
            normalization=True,
            apply_window=True
        )
        if orig_img.ndim == 2:
            orig_img = orig_img[..., np.newaxis]  # add channel dim
        # h0, w0, _ = img.shape

        # Resize and padding
        img, scale, padding = letterbox(orig_img, img_size=self.img_size)

        if self.transform:
            img = self.transform(img)

        # To channel first
        img = torch.from_numpy(img).to(torch.float32).permute(2, 0, 1)
        return {
            "img": img,
            "orig": orig_img,
            "id": img_id,
            "scale": scale,
            "padding": padding
        }


def infer_collater(data):
    """
    collate_fn for inference used dataloader
    """
    imgs = [s["img"] for s in data]
    orig_imgs = [s["orig"] for s in data]
    img_ids = [s["id"] for s in data]
    scales = [s["scale"] for s in data]
    paddings = [s["padding"] for s in data]

    imgs = torch.stack(imgs)

    # tensor to channel first format
    return {
        "img": imgs,
        "orig": orig_imgs,
        "id": img_ids,
        "scale": scales,
        "padding": paddings
    }


def zip_collater(data):
    return tuple(zip(*data))


class VinBigDataset(Dataset):
    def __init__(
        self,
        img_dir="/work/VinBigData/preprocessed/img_npz/",
        ann_dir="/work/VinBigData/preprocessed/bbox_txt/",
        img_ext="npz",  # filetype, filename extensions
        img_size=1024,
        class_names=[],
        dset="train",
        image_ids=None,
        img_normalizer=None,
        transform=None,
        cache_images=False
    ):

        self.img_dir = img_dir  # folder for storing *.npz
        self.ann_dir = ann_dir  # folder for storing *.txt
        self.img_size = img_size
        self.set_name = dset
        self.img_ext = img_ext
        self.class_names = class_names
        self.img_normalizer = img_normalizer
        self.transform = transform

        # Get the image id for current Dataset
        self.image_ids = image_ids
        if self.image_ids is None:
            self.image_ids = [
                img_fname.split('.')[0]
                for img_fname in os.listdir(self.img_dir)
                if img_fname.split('.')[-1] == img_ext
            ]
        self.num_imgs = len(self)

        # Cache labels
        self.annots = [None] * self.num_imgs
        ne = 0  # empty label
        annot_samples = ThreadPool(20).imap(
            self.load_annotations, list(range(self.num_imgs))
        )
        pbar = tqdm(enumerate(annot_samples), total=self.num_imgs)
        for i, sample in pbar:
            self.annots[i] = sample
            if len(sample) == 0:
                ne += 1
            pbar.desc = f"Caching labels... {ne} empty, {i+1} loaded."

        # Cache images for faster training
        self.imgs = [None] * len(self)
        self.img_hw0, self.scales = [None] * len(self), [None] * len(self)
        self.paddings = [None] * len(self)
        if cache_images:
            img_nbytes = 0
            img_samples = ThreadPool(8).imap(
                self.load_image, list(range(self.num_imgs))
            )
            pbar = tqdm(enumerate(img_samples), total=self.num_imgs)
            for i, sample in pbar:
                self.imgs[i] = sample[0]
                self.scales[i] = sample[1]
                self.img_hw0[i] = sample[2]
                self.paddings[i] = sample[3]

                img_nbytes += self.imgs[i].nbytes
                pbar.desc = "Caching images (%.2fGB)" % (img_nbytes / 1e9)

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
        img, scale, img_hw0, padding = self.load_image(idx)
        (h0, w0), (padh, padw) = img_hw0, padding
        annot = self.load_annotations(idx).copy()

        if annot.size > 0:
            annot[:, [0, 2]] = annot[:, [0, 2]] * w0 * scale + padw
            annot[:, [1, 3]] = annot[:, [1, 3]] * h0 * scale + padh

        # sample = {"img": img, "annot": annot}
        sample = {"image": img, "bboxes": annot[:, :4], "labels": annot[:, 4]}
        if self.transform:
            sample = self.transform(**sample)

        # Merge bboxes and labels to annots
        if len(sample["bboxes"]) > 0:
            annot = np.hstack([
                sample["bboxes"],
                np.expand_dims(sample["labels"], -1)
            ])
        else:
            annot = np.zeros((0, 5), dtype=np.float32)

        img = sample["image"]
        # Normalization
        if self.img_normalizer is not None:
            img = self.img_normalizer(img)

        return {"img": img, "annot": annot, "scale": scale}

    def load_image(self, idx: int) -> np.ndarray:
        img = self.imgs[idx]
        if img is None:  # image not cached
            img_fname = self.image_ids[idx] + "." + self.img_ext
            path = os.path.join(self.img_dir, img_fname)
            # Read npz (compressed) data
            if self.img_ext == "npz":
                img = np.load(path)["img"]
            elif self.img_ext == "jpg":
                img = cv2.imread(path)

            if img.ndim == 2:  # one channel gray image
                img = img[..., np.newaxis]  # add channel dim
            h0, w0, _ = img.shape

            # Normalization
            # if self.img_normalizer is not None:
            #     img = self.img_normalizer(img)

            # Resize and padding
            img, scale, padding = letterbox(img, img_size=self.img_size)
            return img, scale, (h0, w0), padding
        else:
            return img, self.scales[idx], self.img_hw0[idx], self.paddings[idx]

    def load_annotations(self, idx: int) -> np.ndarray:
        annot = self.annots[idx]
        if annot is None:
            # get ground truth annotations
            annot_fpath = os.path.join(
                self.ann_dir,
                self.image_ids[idx] + ".txt"
            )
            annot = np.zeros((0, 5))

            # some images appear to miss annotations
            if not os.path.isfile(annot_fpath):
                return annot

            # parse annotations
            # the bbox info are normalized by original image shape (w, h)
            # hence all [x1, y1, x2, y2] are in the range [0, 1)
            with open(annot_fpath, 'r') as f:
                annot = np.array(
                    [x.split() for x in f.read().strip().splitlines()],
                    dtype=np.float32
                )
                # to [x1, y1, x2, y2, class_id]
                # annot = annot[:, [1, 2, 3, 4, 0]]
                if len(annot) > 0:
                    annot = np.roll(annot, shift=-1, axis=1)
                annot = annot.reshape((-1, 5))  # TODO

        return annot


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    # imgs = torch.from_numpy(np.stack(imgs, axis=0)).to(torch.float32)
    # tensor to channel first format
    imgs = torch.stack(imgs).permute(0, 3, 1, 2)

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = torch.from_numpy(annot)
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


def letterbox(image, img_size=1024, pad_color=0):
    # TODO: use `cv2.copyMakeBorder` to do padding
    height, width, channels = image.shape
    if height > width:
        scale = img_size / height
        resized_height = img_size
        resized_width = int(round(width * scale))
    else:
        scale = img_size / width
        resized_height = int(round(height * scale))
        resized_width = img_size

    if scale != 1:
        image = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR
        )
        if image.ndim == 2:  # grayscale image
            image = image[..., np.newaxis]

    if image.shape[0] != img_size or image.shape[1] != img_size:
        # compute padding
        padh = int((img_size - resized_height) / 2.)
        padw = int((img_size - resized_width) / 2.)
        # put image in center (padding)
        image = cv2.copyMakeBorder(
            image,
            padh,  # top
            img_size - resized_height - padh,  # bottom
            padw,  # left
            img_size - resized_width - padw,  # right
            cv2.BORDER_CONSTANT,
            value=pad_color
        )
        if image.ndim == 2:  # grayscale image
            image = image[..., np.newaxis]
        # new_image = np.zeros((img_size, img_size, channels), dtype=np.uint8)
        # new_image[padh: padh+resized_height, padw: padw+resized_width] = image
    else:
        padh, padw = 0, 0
        # new_image = image
    return image, scale, (padh, padw)


class RandomHorizontalFlip(object):
    """Convert ndarrays in sample to Tensors."""
    # This will be removed at next version
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        if np.random.rand() < self.prob:
            image, annots = sample['img'], sample['annot']
            scale = sample["scale"]
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x_min = annots[:, 0].copy()
            x_max = annots[:, 2].copy()

            x_tmp = x_min.copy()

            annots[:, 0] = cols - x_max
            annots[:, 2] = cols - x_tmp

            # sample["img"] = image
            # sample["annot"] = annots
            sample = {"img": image, "annot": annots, "scale": scale}

        return sample


class ImageNormalizer(object):

    def __init__(self, mean=[0], std=[1]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, image):
        return ((image.astype(np.float32) - self.mean) / self.std)
