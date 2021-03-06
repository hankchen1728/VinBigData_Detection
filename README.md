# VinBigData Chest X-ray Abnormalities Detection
Competition [Link](https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection/overview)  
Resources [doc](https://docs.google.com/document/d/1fAKkW82ShSpERiUP_TmLPaP-cw_PTVhSRulDIjG6jWg/edit#heading=h.zfdbsf1l98sg)

# Preprocessing
After running the following
```sh
python3 preprocessing/preprocessing_mp.py \
    --train-csv /work/VinBigData/raw_data/train.csv \
    --train-dicom-dir /work/VinBigData/raw_data/train/ \
    --save-base-dir SAVE_BASE_DIR \
    --workers 32
```
The processed image `*.npz` files will be stored in `$SAVE_BASE_DIR/img_npz`,  
and bounding box `*.txt` file will be in `$SAVE_BASE_DIR/bbox_txt`

### Bounding Boxes in `txt` files
The `*.txt` file specifications are:
- One row per object
- Each row is `class x_min y_min x_max y_max` format
- Box coordinates are in **normalized** format (from 0 - 1). If your boxes are in pixels, divide `x_min` and `x_max` by image width, and `y_min` and `y_max` by image height.)  
For example,  
<p align="center">
    <img src="./figures/YOLO_bbox_fmt.png" alt="Example of annotation format" height="100">
</p>

## Update log
[2021-02-01] First version of preprocessing, including converting image to `*.npz` files and bounding boxes info. to `*.txt` files  
[2021-02-03] Clone the [Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch.git), create custom dataset (see `EfficientDet/efficientdet/custom_dataset.py`) for loading VinBigData, and modified the input channels setting for entering gray-scale images  
[2021-02-09] Add drawing box feature to `preprocessing_mp.py`, saving the plotted figures
<p align="center">
    <img src="./figures/0005e8e3701dfb1dd93d53e2ff537b6e.jpg" alt="Display of drawing bounding boxes" height="400">
</p>

[2021-02-17] Commit first version of `aug.py` for augmentation.  
[2021-02-18] Add prediction script `predict.py` for VinBigData test prediction. Also add function of saving prediction plotted figures.
