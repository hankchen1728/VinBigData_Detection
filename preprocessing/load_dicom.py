import os

# import cv2
import pydicom
import numpy as np
from pydicom.pixel_data_handlers.util import apply_voi_lut

# import matplotlib.pyplot as plt


def read_xray(
    dcm_path,
    voi_lut=False,
    fix_monochrome=True,
    normalization=False,
    apply_window=False
) -> np.ndarray:
    dicom = pydicom.read_file(dcm_path)
    # For ignoring the UserWarning: "Bits Stored" value (14-bit)...
    elem = dicom[0x0028, 0x0101]
    elem.value = 16

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM
    # data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    if normalization:
        if apply_window and "WindowCenter" in dicom and "WindowWidth" in dicom:
            window_center = float(dicom.WindowCenter)
            window_width = float(dicom.WindowWidth)
            y_min = (window_center - 0.5 * window_width)
            y_max = (window_center + 0.5 * window_width)
        else:
            y_min = data.min()
            y_max = data.max()
        data = (data - y_min) / (y_max - y_min)
        data = np.clip(data, 0, 1)

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    return data


def save_dcm_to_npz(
    dcm_path,
    save_dir="/work/VinBigData/preprocessed",
    force_replace=False,
    return_pixel_data=False
):
    npz_fname = os.path.basename(dcm_path).replace("dicom", "npz")
    npz_fpath = os.path.join(save_dir, npz_fname)
    # dtype_max = 65535
    dtype_max = 255

    if not force_replace and os.path.isfile(npz_fpath):
        data = np.load(npz_fpath)["img"]
    else:
        data = read_xray(
            dcm_path=dcm_path,
            voi_lut=False,
            fix_monochrome=True,
            normalization=True,
            apply_window=True
        )
        # TODO: Convert to uint16 type
        data = (data * dtype_max).astype(np.uint8)
        # Save to numpy file
        np.savez_compressed(npz_fpath, img=data)

    shape = data.shape

    if return_pixel_data:
        data = data.astype(np.float32) / dtype_max
        return shape, data
    return shape
    # end


if __name__ == "__main__":
    save_dcm_to_npz(
        dcm_path="../tmp_data/0005e8e3701dfb1dd93d53e2ff537b6e.dicom",
        save_dir="."
    )
