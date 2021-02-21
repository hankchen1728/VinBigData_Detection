import albumentations as A


def get_train_transforms(aug_cfg: dict):
    # Setting params with bounding boxes
    # See https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/#step-2-define-an-augmentation-pipeline
    bbox_params = A.BboxParams(
        format='pascal_voc',
        min_area=0,
        min_visibility=0,
        label_fields=['labels']
    )

    T_tags = [
        "HorizontalFlip",
        "VerticalFlip",
        "CLAHE",
        "ShiftScaleRotate",
        "Cutout"
    ]

    transform = []
    for tag in T_tags:
        if tag in aug_cfg:
            if isinstance(aug_cfg[tag], list):
                for aug_kwarg in aug_cfg[tag]:
                    aug = decode_aug(tag, aug_kwarg)
                    if aug is not None:
                        transform.append(aug)
            else:
                aug = decode_aug(tag, aug_cfg[tag])
                if aug is not None:
                    transform.append(aug)

    return A.Compose(transform, p=1.0, bbox_params=bbox_params)


def decode_aug(tag: str, tf_dict: dict):
    rand_prob = float(tf_dict["p"])
    if rand_prob > 0:
        return getattr(A, tag)(**tf_dict)
    return None
