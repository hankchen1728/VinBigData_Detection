import os
# import sys
import argparse

import numpy as np
import pandas as pd
from ensemble_boxes import weighted_boxes_fusion as wbf

from load_dicom import save_dcm_to_npz


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


def processing_case(img_info_dict):
    # ==== tmp variables
    bbox_coord_col = ["x_min", "y_min", "x_max", "y_max"]
    train_dcm_dir = "../tmp_data/"
    save_base_dir = "./saved_processed"
    npz_dir = os.path.join(save_base_dir, "img_npz")
    bbox_dir = os.path.join(save_base_dir, "bbox_txt")
    os.makedirs(npz_dir, exist_ok=True)
    os.makedirs(bbox_dir, exist_ok=True)
    # ==================
    img_id = img_info_dict["img_id"]
    img_bbox_df = img_info_dict["img_df"]
    img_bbox_df = img_bbox_df.reset_index(drop=True)

    # Save pixel data to npz
    img_shape = save_dcm_to_npz(
        dcm_path=os.path.join(train_dcm_dir, img_id + ".dicom"),
        save_dir=npz_dir
    )

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

    # Concate the bboxes together and save to txt file
    fused_bboxes = np.concatenate(fused_bboxes)
    fused_id = np.concatenate(fused_id)
    fused_labels = np.hstack([np.expand_dims(fused_id, -1), fused_bboxes])
    np.savetxt(
        fname=os.path.join(bbox_dir, img_id + ".txt"),
        X=fused_labels,
        fmt="%.7f"
    )

    # end


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
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_csv)
    # train_df = pd.read_csv("../train.csv")
    img_df_list = [
        (img_id, img_df) for img_id, img_df in train_df.groupby(by="image_id")
    ]

    # img_df_dict = {
    #     img_id: img_df for img_id, img_df in train_df.groupby(by="image_id")
    # }
    # print("end grouping dataframes")

    # img_id = "0005e8e3701dfb1dd93d53e2ff537b6e"
    # case_dict = {"img_id": img_id, "img_df": img_df_dict[img_id]}
    # processing_case(case_dict)
