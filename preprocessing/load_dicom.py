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
    normalization=False
) -> np.ndarray:
    dicom = pydicom.read_file(dcm_path)
    # For ignoring the UserWarning: "Bits Stored" value (14-bit)...
    # elem = dicom[0x0028, 0x0101]
    # elem.value = 16

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM
    # data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    if normalization:
        if "WindowCenter" in dicom and "WindowWidth" in dicom:
            window_center = float(dicom.WindowCenter)
            window_width = float(dicom.WindowWidth)
            y_min = (window_center - 0.5 * window_width)
            y_max = (window_center + 0.5 * window_width)
        else:
            y_min = data.min()
            y_max = data.max()
        data = (data - y_min) / (y_max - y_min)
        data = np.clip(data, 0, 1)

    return data


def save_dcm_to_npz(
    dcm_path,
    save_dir="/work/VinBigData/preprocessed",
    return_pixel_data=False
):
    data = read_xray(
        dcm_path=dcm_path,
        voi_lut=False,
        fix_monochrome=True,
        normalization=True
    )

    # Convert to float type
    # data = data.astype(np.float32)
    data = (data * 65535).astype(np.uint16)
    shape = data.shape

    # Save to numpy file
    npz_fname = os.path.basename(dcm_path).replace("dicom", "npz")
    np.savez_compressed(os.path.join(save_dir, npz_fname), img=data)
    if return_pixel_data:
        data = data.astype(np.float32) / 65535.
        return shape, data
    return shape
    # end


if __name__ == "__main__":
    save_dcm_to_npz(
        dcm_path="../tmp_data/0005e8e3701dfb1dd93d53e2ff537b6e.dicom",
        save_dir="."
    )
