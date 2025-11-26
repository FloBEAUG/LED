import numpy as np
import os
import exifread
import subprocess
import json

Sony_A7S2_CCM = np.array([[ 1.9712269,-0.6789218, -0.29230508],
                          [-0.29104823, 1.748401 , -0.45735288],
                          [ 0.02051281,-0.5380369,  1.5175241 ]],
                         dtype='float32')


def pack_raw_bayer(raw: np.ndarray, raw_pattern: np.ndarray):
    #pack Bayer image to 4 channels
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)

    raw = raw.astype(np.uint16)
    H, W = raw.shape
    if H % 2 == 1:
        raw = raw[:-1]
    if W % 2 == 1:
        raw = raw[:, :-1]
    out = np.stack((raw[R[0][0]::2,  R[1][0]::2], #RGBG
                    raw[G1[0][0]::2, G1[1][0]::2],
                    raw[B[0][0]::2,  B[1][0]::2],
                    raw[G2[0][0]::2, G2[1][0]::2]), axis=0).astype(np.uint16)

    return out


def depack_raw_bayer(raw: np.ndarray, raw_pattern: np.ndarray):
    _, H, W = raw.shape
    raw = raw.astype(np.uint16)

    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)

    raw_flatten = np.zeros((H * 2, W * 2))
    raw_flatten[R[0][0]::2,  R[1][0]::2] = raw[0]
    raw_flatten[G1[0][0]::2,  G1[1][0]::2] = raw[1]
    raw_flatten[B[0][0]::2,  B[1][0]::2] = raw[2]
    raw_flatten[G2[0][0]::2,  G2[1][0]::2] = raw[3]

    raw_flatten = raw_flatten.astype(np.uint16)
    return raw_flatten


def metainfo(rawpath):
    with open(rawpath, 'rb') as f:

        result = subprocess.run(
            ["exiftool", "-json", rawpath], capture_output=True, text=True, check=True
        )
        tags = json.loads(result.stdout)[0]

        keys = [key for key in tags.keys() if "ExposureTime" in key or "ISO" in key]
        if len(keys) >= 2:
            expo = eval(str((tags[keys[0]])))
            iso = eval(str(tags[keys[1]]))
        else:
            return None, None

    return iso, expo
