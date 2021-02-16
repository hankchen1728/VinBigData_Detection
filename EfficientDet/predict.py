import os
import argparse
# import datetime
# import traceback

import cv2
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
# from tqdm.autonotebook import tqdm

from backbone import EfficientDetBackbone
from efficientdet.custom_dataset import VinBigDicomDataset, infer_collater
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import CustomDataParallel, init_weights, postprocess


input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]


label2color = [
    [59, 238, 119],
    [222, 21, 229],
    [94, 49, 164],
    [206, 221, 133],
    [117, 75, 3],
    [210, 224, 119],
    [211, 176, 166],
    [63, 7, 197],
    [102, 65, 77],
    [194, 134, 175],
    [209, 219, 50],
    [255, 44, 47],
    [89, 125, 149],
    [110, 27, 100]
]


def draw_one_bbox(image, box, label, color, thickness=10):
    alpha, alpha_box = 0.1, 0.4
    font_scale, font_thick = 1.5, 2
    overlay_bbox = image.copy()
    overlay_text = image.copy()
    output = image.copy()

    text_width, text_height = cv2.getTextSize(
        text=label.upper(),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        thickness=font_thick
    )[0]
    cv2.rectangle(overlay_bbox, (box[0], box[1]), (box[2], box[3]), color, -1)
    cv2.addWeighted(overlay_bbox, alpha, output, 1 - alpha, 0, output)
    cv2.rectangle(
        overlay_text,
        (box[0], box[1] - 10 - text_height),
        (box[0] + text_width + 5, box[1]),
        (0, 0, 0),
        -1
    )
    cv2.addWeighted(overlay_text, alpha_box, output, 1 - alpha_box, 0, output)
    cv2.rectangle(
        output,
        (box[0], box[1]),
        (box[2], box[3]),
        color,
        thickness
    )
    cv2.putText(
        img=output,
        text=label.upper(),
        org=(box[0], box[1] - 5),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=font_scale,
        color=(255, 255, 255),
        thickness=font_thick,
        lineType=cv2.LINE_AA
    )
    return output


def display_bboxes(img, class_ids, label_names, box_rois):
    # to uint8
    img = (img * 255.).astype(np.uint8)
    rgb_img = np.repeat(img[..., [0]], axis=-1, repeats=3)
    if len(box_rois) > 0:
        box_rois = box_rois.astype(np.int)
        for i_box, bbox in enumerate(box_rois):
            label_id = class_ids[i_box]
            label_name = label_names[i_box]
            rgb_img = draw_one_bbox(
                rgb_img,
                box=list(bbox),
                label=label_name,
                color=label2color[label_id]
            )

    return rgb_img


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
    # TODO: move these two threshold setting to config file
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
    parser.add_argument(
        "--visual-path",
        type=str,
        default="./pred_display"
    )

    args = parser.parse_args()
    return args


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
    #
    if opt.visualization:
        os.makedirs(opt.visual_path, exist_ok=True)

    print("Configuring dataloader...")
    test_params = {
        'batch_size': opt.batch_size,
        'shuffle': False,
        'drop_last': False,
        'num_workers': opt.num_workers,
        "collate_fn": infer_collater
    }

    testing_set = VinBigDicomDataset(
        img_dir=params.test_dicom,
        img_size=input_sizes[opt.compound_coef]
    )
    test_dataloader = DataLoader(testing_set, **test_params)

    print("Building model...")
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

        print(f"[Info] loaded weights: {os.path.basename(weights_path)}")
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
    num_iters = len(test_dataloader)
    obj_list = params.obj_list
    for i_batch, data in tqdm(enumerate(test_dataloader), total=num_iters):
        with torch.no_grad():
            imgs = data["img"].to(device)
            orig_imgs = data["orig"]
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

                if opt.visualization:
                    # To channel last and convert to numpy array
                    # img = imgs[i_img].permute(1, 2, 0).cpu().numpy()
                    img = orig_imgs[i_img]
                    class_ids = pred["class_ids"]
                    plotted_img = display_bboxes(
                        img=img,
                        class_ids=class_ids,
                        label_names=[obj_list[int(c)] for c in class_ids],
                        box_rois=pred["rois"].astype(np.int)
                    )
                    plt.imsave(
                        os.path.join(opt.visual_path, img_ids[i_img] + ".jpg"),
                        plotted_img
                    )
    pred_df = pd.DataFrame(
        predictions,
        columns=['image_id', 'PredictionString']
    )
    pred_df.to_csv(opt.pred_csv_path, index=False)


if __name__ == '__main__':
    opt = get_args()
    predict(opt)
