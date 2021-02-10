import os
# import sys
import argparse
import multiprocessing

import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ensemble_boxes import weighted_boxes_fusion as wbf

from load_dicom import save_dcm_to_npz
from utils import draw_bbox


lesion_id_to_name = {
    0: "Aortic enlargement",
    1: "Atelectasis",
    2: "Calcification",
    3: "Cardiomegaly",
    4: "Consolidation",
    5: "ILD",
    6: "Infiltration",
    7: "Lung Opacity",
    8: "Nodule/Mass",
    9: "Other lesion",
    10: "Pleural effusion",
    11: "Pleural thickening",
    12: "Pneumothorax",
    13: "Pulmonary fibrosis",
    14: "No finding"
}

# color list for display bounding boxes
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


def bboxes_fusion(img_shape, img_bbox_df):
    bbox_coord_col = ["x_min", "y_min", "x_max", "y_max"]
    # bboxes fusion (WBF)
    class_ids = img_bbox_df["class_id"].values
    class_id_cnt = img_bbox_df["class_id"].value_counts().to_dict()
    orig_bboxes = img_bbox_df[bbox_coord_col].values
    # TODO: check which axis is x-axis or y-axis
    # Normalize the bboxes coordinate
    orig_bboxes /= [img_shape[1], img_shape[0], img_shape[1], img_shape[0]]
    orig_bboxes = np.clip(orig_bboxes, 0, 1)
    fused_bboxes = []
    fused_id = []

    for c_id in class_id_cnt.keys():
        if c_id == 14:
            continue
        if class_id_cnt[c_id] == 1:  # only one bboxes
            selected_bbox = np.array(orig_bboxes[class_ids == c_id])
            # nothing to do with only single bbox
            assert selected_bbox.shape[0] == 1, "Allowed only one bbox"
            fused_bboxes.append(selected_bbox)
            fused_id.append([c_id])

        else:  # more than two bboxes
            selected_bboxes = np.array(orig_bboxes[class_ids == c_id])
            assert selected_bboxes.shape[0] == class_id_cnt[c_id]
            # Use weighted boxes fusion to fuse bboxes
            wbf_boxes, _, labels = wbf(
                boxes_list=[selected_bboxes],
                scores_list=np.ones((1, class_id_cnt[c_id])),
                labels_list=np.full((1, class_id_cnt[c_id]), fill_value=c_id),
                iou_thr=0.5,
                skip_box_thr=1e-3
            )
            fused_bboxes.append(wbf_boxes)
            fused_id.append(labels)

    if len(fused_bboxes) == 0:
        return fused_bboxes

    # Concate the bboxes together and save to txt file
    fused_bboxes = np.concatenate(fused_bboxes)
    fused_id = np.concatenate(fused_id).astype(np.int)
    fused_labels = np.hstack([np.expand_dims(fused_id, -1), fused_bboxes])
    return fused_labels


def processing_case(img_info_dict):
    train_dcm_dir = img_info_dict["train_dcm_dir"]
    save_base_dir = img_info_dict["save_base_dir"]
    npz_dir = os.path.join(save_base_dir, img_info_dict["npz_dirname"])
    bbox_dir = os.path.join(save_base_dir, img_info_dict["txt_dirname"])

    img_id = img_info_dict["img_id"]
    img_bbox_df = img_info_dict["img_df"]
    img_bbox_df = img_bbox_df.reset_index(drop=True)

    # Save pixel data to npz
    img_shape, pixel_data = save_dcm_to_npz(
        dcm_path=os.path.join(train_dcm_dir, img_id + ".dicom"),
        save_dir=npz_dir,
        return_pixel_data=True
    )

    fused_labels = bboxes_fusion(img_shape, img_bbox_df)
    txt_fpath = os.path.join(bbox_dir, img_id + ".txt")
    save_fmt = "%d" + " %.7f" * 4 if len(fused_labels) > 0 else "%d"
    np.savetxt(
        fname=txt_fpath,
        X=fused_labels,
        fmt=save_fmt
    )

    # Display and draw bounding boxes
    if "display_dirname" in img_info_dict.keys():
        dis_dir = os.path.join(save_base_dir, img_info_dict["display_dirname"])
        pixel_data = (pixel_data * 255.).astype(np.uint8)
        rgb_img = np.repeat(pixel_data[..., np.newaxis], axis=-1, repeats=3)

        if len(fused_labels) > 0:
            fused_labels[:, [1, 3]] *= img_shape[1]
            fused_labels[:, [2, 4]] *= img_shape[0]
            fused_labels = fused_labels.astype(np.int)
            for bbox in fused_labels:
                label_id = bbox[0]
                color = label2color[label_id]
                label_name = lesion_id_to_name[label_id]
                rgb_img = draw_bbox(
                    rgb_img,
                    box=list(bbox[1:]),
                    label=label_name,
                    color=color
                )

        plt.imsave(os.path.join(dis_dir, img_id + ".jpg"), rgb_img)
        # end


def main(args):
    train_df = pd.read_csv(args.train_csv)

    # try on local
    # train_df = train_df[
    #     train_df["image_id"] == "ff00fd8abeb1fb6646fc3943f802cb8d"
    # ]

    npz_dirname = "img_npz"
    txt_dirname = "bbox_txt"

    print("Preparing argment list for running function...")
    info_dict_list = [
        {
            "img_id": img_id,
            "img_df": img_df,
            "train_dcm_dir": args.train_dicom_dir,
            "save_base_dir": args.save_base_dir,
            "npz_dirname": npz_dirname,
            "txt_dirname": txt_dirname
        }
        for img_id, img_df in train_df.groupby(by="image_id")
    ]
    os.makedirs(os.path.join(args.save_base_dir, npz_dirname), exist_ok=True)
    os.makedirs(os.path.join(args.save_base_dir, txt_dirname), exist_ok=True)

    # Whether to save display *.jpg
    if args.save_display:
        display_dirname = "display"
        os.makedirs(
            os.path.join(args.save_base_dir, display_dirname),
            exist_ok=True
        )
        for info_dict in info_dict_list:
            info_dict["display_dirname"] = display_dirname

    print("Now running preprocessing part")
    with multiprocessing.Pool(args.workers) as pool:
        gen = pool.imap(processing_case, info_dict_list)
        for _ in tqdm.tqdm(gen, total=len(info_dict_list)):
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parsing raw data to processed data, "
                    "npz for images and txt for bounding boxes labels"
    )

    parser.add_argument(
        "--train-csv",
        type=str,
        default="/work/VinBigData/raw_data/train.csv",
        help="Path of training data label csv file"
    )

    parser.add_argument(
        "--train-dicom-dir",
        type=str,
        default="/work/VinBigData/raw_data/train/",
        help="Path of train dicom data directory"
    )

    parser.add_argument(
        "--save-base-dir",
        type=str,
        default="/work/VinBigData/preprocessed/",
        help="Path to store preprocessed *.npz and *.txt files"
    )

    parser.add_argument(
        "--save-display",
        action="store_true",
        default=False,
        help="Saving image with displaying bounding boxes "
             "to <SAVE_BASE_DIR>/display"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8
    )
    args = parser.parse_args()

    main(args)
