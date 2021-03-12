import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
sns.set(rc={"font.size":9,"axes.titlesize":15,"axes.labelsize":9,
            "axes.titlepad":11, "axes.labelpad":9, "legend.fontsize":7,
            "legend.title_fontsize":7, 'axes.grid' : False})
import argparse
import cv2
import json
import pandas as pd
import glob
import os.path as osp
from path import Path
import datetime
import numpy as np
from tqdm.auto import tqdm
import random
import shutil
from sklearn.model_selection import train_test_split

from ensemble_boxes import *
import warnings
from collections import Counter

labels =  [
            "Aortic_enlargement",
            "Atelectasis",
            "Calcification",
            "Cardiomegaly",
            "Consolidation",
            "ILD",
            "Infiltration",
            "Lung_Opacity",
            "Nodule/Mass",
            "Other_lesion",
            "Pleural_effusion",
            "Pleural_thickening",
            "Pneumothorax",
            "Pulmonary_fibrosis",
            "No_finding"
            ]

label2color = [[59, 238, 119], [222, 21, 229], [94, 49, 164], [206, 221, 133], [117, 75, 3],
                 [210, 224, 119], [211, 176, 166], [63, 7, 197], [102, 65, 77], [194, 134, 175],
                 [209, 219, 50], [255, 44, 47], [89, 125, 149], [110, 27, 100], [50, 50, 50]]
thickness = 3

def draw_bbox(image, box, label, color):   
    alpha = 0.1
    alpha_box = 0.4
    overlay_bbox = image.copy()
    overlay_text = image.copy()
    output = image.copy()

    text_width, text_height = cv2.getTextSize(label.upper(), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
    cv2.rectangle(overlay_bbox, (box[0], box[1]), (box[2], box[3]),
                color, -1)
    cv2.addWeighted(overlay_bbox, alpha, output, 1 - alpha, 0, output)
    cv2.rectangle(overlay_text, (box[0], box[1]-7-text_height), (box[0]+text_width+2, box[1]),
                (0, 0, 0), -1)
    cv2.addWeighted(overlay_text, alpha_box, output, 1 - alpha_box, 0, output)
    cv2.rectangle(output, (box[0], box[1]), (box[2], box[3]),
                    color, thickness)
    cv2.putText(output, label.upper(), (box[0], box[1]-5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return output

def main(args):
    csv_file = args.csv_file
    image_dir = args.image_dir
    mode = args.mode
    saved_coco_path = "./"

    output_dir = "./vinbigdata_coco_chest_xray_normal/"+mode+"_images"

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
        print('Coco Train Image Directory:', output_dir)

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime('%Y-%m-%d %H:%M:%S.%f'),
        ),
        licenses=[dict(
            url=None,
            id=0,
            name=None,
        )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type='instances',
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    out_file = './vinbigdata_coco_chest_xray_normal/{}_annotations.json'.format(mode)

    data_train = data.copy()
    data_train['images'] = []
    data_train['annotations'] = []

    for i in range(len(labels)):
        data_train['categories'].append({'id': i, 'name': labels[i]})

    train_annotations = pd.read_csv(csv_file)
    #train_annotations = train_annotations[train_annotations['class_id']!=14]
    if mode != "test":
        train_annotations = train_annotations[train_annotations['fold']==mode]

    train_annotations['image_path'] = train_annotations['image_id'].map(lambda x:os.path.join(image_dir, str(x)+'.npz'))
    imagepaths = train_annotations['image_path'].unique()

    iou_thr = 0.35
    skip_box_thr = 0.0001
    sigma = 0.1
    viz_labels = labels
    viz_images=[]

    for i, path in tqdm(enumerate(imagepaths)):
        img_array  = np.load(path)['img']
        image_basename = Path(path).stem
    #     print(f"(\'{image_basename}\', \'{path}\')")
        ## Add Images to annotation
        data_train['images'].append(dict(
            license=0,
            url=None,
            file_name=os.path.join(image_dir, path),
            height=img_array.shape[0],
            width=img_array.shape[1],
            date_captured=None,
            id=i
        ))

        if mode == "test":
            continue
        else:
            img_annotations = train_annotations[train_annotations.image_id==image_basename]
            labels_viz = img_annotations['class_id'].to_numpy()
            boxes_viz = img_annotations[['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()

            boxes_viz[np.where(labels_viz==14)[0]]=[0.,0.,img_array.shape[1],img_array.shape[0]]
            boxes_viz = boxes_viz.astype(np.float).tolist()
            labels_viz = labels_viz.tolist()
            
            ## Visualize Original Bboxes every 500th
            if (i%1000==0):
                img_before = img_array.copy()
                for box, label in zip(boxes_viz, labels_viz):
                    x_min, y_min, x_max, y_max = (box[0], box[1], box[2], box[3])
                    color = label2color[int(label)]
                    img_before = draw_bbox(img_before, list(np.int_(box)), viz_labels[label], color)
                plt.imsave(os.path.join(output_dir, image_basename + "_before.jpg"), img_before)
            
            boxes_list = []
            scores_list = []
            labels_list = []
            weights = []
            
            boxes_single = []
            labels_single = []

            cls_ids = img_annotations['class_id'].unique().tolist()
            
            count_dict = Counter(img_annotations['class_id'].tolist())
            for cid in cls_ids:
                ## Performing Fusing operation only for multiple bboxes with the same label
                if count_dict[cid]==1:
                    labels_single.append(cid)
                    boxes_single.append(img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy().squeeze().tolist())
                elif cid==14:
                    labels_single.append(cid)
                    boxes_single.append(np.array([0.,0.,img_array.shape[1],img_array.shape[0]]).squeeze().tolist())
                else:
                    cls_list =img_annotations[img_annotations.class_id==cid]['class_id'].tolist()
                    labels_list.append(cls_list)
                    bbox = img_annotations[img_annotations.class_id==cid][['x_min', 'y_min', 'x_max', 'y_max']].to_numpy()
                    
                    ## Normalizing Bbox by Image Width and Height
                    bbox = bbox/(img_array.shape[1], img_array.shape[0], img_array.shape[1], img_array.shape[0])
                    bbox = np.clip(bbox, 0, 1)
                    boxes_list.append(bbox.tolist())
                    scores_list.append(np.ones(len(cls_list)).tolist())
                    weights.append(1)
                    
            ## Perform WBF
            boxes, scores, box_labels = weighted_boxes_fusion(boxes_list=boxes_list, scores_list=scores_list,
                                                          labels_list=labels_list, weights=weights,
                                                          iou_thr=iou_thr, skip_box_thr=skip_box_thr)
            
            boxes = boxes*(img_array.shape[1], img_array.shape[0], img_array.shape[1], img_array.shape[0])
            boxes = boxes.round(1).tolist()
            box_labels = box_labels.astype(int).tolist()
            boxes.extend(boxes_single)
            box_labels.extend(labels_single)
            
            img_after = img_array.copy()
            for box, label in zip(boxes, box_labels):
                x_min, y_min, x_max, y_max = (box[0], box[1], box[2], box[3])
                area = round((x_max-x_min)*(y_max-y_min),1)
                bbox =[
                        round(x_min, 1),
                        round(y_min, 1),
                        round((x_max-x_min), 1),
                        round((y_max-y_min), 1)
                        ]
                
                data_train['annotations'].append(dict( id=len(data_train['annotations']), image_id=i,
                                                    category_id=int(label), area=area, bbox=bbox,
                                                    iscrowd=0))
                
            ## Visualize Bboxes after operation
            if (i%1000==0):
                img_after = img_array.copy()
                for box, label in zip(boxes, box_labels):
                    color = label2color[int(label)]
                    img_after = draw_bbox(img_after, list(np.int_(box)), viz_labels[label], color)
                viz_images.append(img_after)
                plt.imsave(os.path.join(output_dir, image_basename + "_after.jpg"), img_after)

                   
    with open(out_file, 'w') as f:
        json.dump(data_train, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv_file", type = str, default = "/work/VinBigData/raw_data/train.csv")
    parser.add_argument("--image_dir", type = str, default = "/work/VinBigData/preprocessed/img_npz")
    parser.add_argument("--mode", type = str, default = "train")

    args = parser.parse_args()

    main(args)
