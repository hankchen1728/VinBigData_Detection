import os
import argparse
# import datetime
# import traceback

import numpy as np
import pandas as pd
import yaml
import torch
from torch.utils.data import DataLoader
# from tqdm.autonotebook import tqdm
from tqdm import tqdm

from backbone import EfficientDetBackbone
from efficientdet.custom_dataset import VinBigDicomDataset, infer_collater
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import CustomDataParallel, init_weights, postprocess


input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]


class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def get_args():
    parser = argparse.ArgumentParser(
        "EfficientDet Prediction"
    )
    parser.add_argument(
        '-p', '--project',
        type=str,
        default='coco',
        help='project file that contains parameters'
    )
    parser.add_argument(
        '-c', '--compound-coef',
        type=int,
        default=0,
        help='coefficients of efficientdet'
    )
    parser.add_argument(
        '-n', '--num-workers',
        type=int,
        default=12,
        help='num-workers of dataloader'
    )
    parser.add_argument(
        '--cuda-devices',
        type=str,
        default="",
        help="CUDA devices, i.e. 0 or 0,1,2,3. None specified for using cpu"
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='The number of images per batch among all devices'
    )
    parser.add_argument(
        '--iou-thres',
        type=float,
        default=0.3,
    )
    parser.add_argument(
        '--score-thres',
        type=float,
        default=0.1,
    )
    parser.add_argument(
        '--pred-csv-path',
        type=str,
        default='submission.csv'
    )
    parser.add_argument(
        '-w', '--load_weights',
        type=str,
        default=None,
        help="whether to load weights from a checkpoint, "
             "set None to initialize, set \'last\' to load last checkpoint"
    )
    parser.add_argument(
        '--visualization',
        action="store_true",
        default=False,
        help="whether visualize the predicted boxes, "
             "the output images will be in test/"
    )

    args = parser.parse_args()
    return args


def predict(opt):
    params = Params(f'projects/{opt.project}.yml')

    num_gpus = len(opt.cuda_devices.split(','))
    if num_gpus == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.cuda_devices

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # opt.saved_path = opt.saved_path + f'/{params.project_name}/'
    # opt.log_path = opt.log_path + f'/{params.project_name}/tensorboard/'
    # os.makedirs(opt.log_path, exist_ok=True)
    # os.makedirs(opt.saved_path, exist_ok=True)

    test_params = {
        'batch_size': opt.batch_size,
        'shuffle': False,
        'drop_last': False,
        'num_workers': opt.num_workers,
        "collate_fn": infer_collater
    }

    testing_set = VinBigDicomDataset(
        img_dir=params.image_dir,
        img_size=input_sizes[opt.compound_coef]
    )
    test_dataloader = DataLoader(testing_set, **test_params)

    model = EfficientDetBackbone(
        in_channels=int(params.in_channels),
        num_classes=len(params.obj_list),
        compound_coef=opt.compound_coef,
        ratios=eval(params.anchors_ratios),
        scales=eval(params.anchors_scales)
    )

    # TODO: can we given specific score-threshold for each class
    threshold = opt.score_thres
    iou_threshold = opt.iou_thres

    # load last weights
    if opt.load_weights is not None:
        assert opt.load_weights.endswith('.pth'), f"{opt.load_weights}"
        weights_path = opt.load_weights

        try:
            model.load_state_dict(torch.load(weights_path), strict=True)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                "[Warning] Don\'t panic if you see this, "
                "this might be because you load a pretrained weights "
                "with different number of classes. The rest of the weights "
                "should be loaded already."
            )

        print(f"[Info] loaded weights: {os.path.basename(weights_path)}, "
              "resuming checkpoint from step: {last_step}")
    else:
        print('[Info] initializing weights...')
        init_weights(model)

    # Mapping model to gpu device
    # TODO: Handling "half-precision" case (float16)
    if num_gpus > 0:
        model = model.cuda()
        if num_gpus > 1:
            # TODO: Modify the `CustomDataParallel`
            # for loading the last batch
            # which size is smaller than given `batch_size`
            model = CustomDataParallel(model, num_gpus)

    model.eval()

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    predictions = []
    for i_batch, data in tqdm(enumerate(test_dataloader)):
        with torch.no_grad():
            imgs = data["img"].to(device)
            img_ids = data["id"]
            scales = data["scale"]
            paddings = data["padding"]
            _, regression, classification, anchors = model(imgs)
            out = postprocess(
                imgs,
                anchors,
                regression,
                classification,
                regressBoxes,
                clipBoxes,
                threshold,
                iou_threshold
            )
            out = invert_affine(out, scales, paddings)

            # Convert to PredictionString format
            for i_img in range(len(imgs)):
                pred = out[i_img]
                pred_str = ""
                if len(pred["rois"]) > 0:
                    pred_str = format_prediction_string(
                        pred["class_ids"],
                        pred["scores"],
                        pred["rois"].astype(np.int)
                    )
                else:
                    pred_str = "14 1.0 0 0 1 1"
                predictions.append({
                    "image_id": img_ids[i_img],
                    "PredictionString": pred_str
                })
    pred_df = pd.DataFrame(
        predictions,
        columns=['image_id', 'PredictionString']
    )
    pred_df.to_csv(opt.pred_csv_path, index=False)


def invert_affine(preds, scales, paddings):
    """
    TODO: Mapping the predicted boxes to original coordination
    """
    for i_img in range(len(preds)):
        if len(preds[i_img]["rois"]) == 0:
            continue
        else:
            # x-axis (width)
            preds[i_img]["rois"][:, [0, 2]] = \
                (preds[i_img]["rois"][:, [0, 2]] - paddings[i_img][1]) / \
                scales[i_img]
            # y-axis (height)
            preds[i_img]["rois"][:, [1, 3]] = \
                (preds[i_img]["rois"][:, [1, 3]] - paddings[i_img][0]) / \
                scales[i_img]
    return preds


def format_prediction_string(labels, scores, boxes):
    """
    https://www.kaggle.com/basu369victor/chest-x-ray-abnormalities-detection-submission/output
    """
    pred_strings = []
    for j in zip(labels, scores, boxes):
        pred_strings.append("{0} {1:.4f} {2} {3} {4} {5}".format(
            j[0], j[1], j[2][0], j[2][1], j[2][2], j[2][3]))

    return " ".join(pred_strings)


if __name__ == '__main__':
    opt = get_args()
    predict(opt)
