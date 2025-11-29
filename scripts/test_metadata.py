from itertools import count

from led.archs import build_network
from led.utils.options import yaml_load
from led.data.raw_utils import metainfo, pack_raw_bayer, depack_raw_bayer
import rawpy
import torch
from copy import deepcopy
import argparse
import glob
import numpy as np
import os
from tqdm import tqdm
from pidng.core import RAW2DNG, DNGTags, Tag
from pidng.defs import *
import pathlib
import exiftool
import cv2

def read_img(raw, raw_path):
    with exiftool.ExifToolHelper() as et:
        metadata = et.get_metadata(raw_path)


    black_level = np.array(raw.black_level_per_channel,
                                    dtype='float32').reshape(1,4, 1, 1)

    if raw.camera_white_level_per_channel is None:
        white_level = np.array([raw.white_level] * 4, dtype='float32').reshape(4, 1, 1)
    else:
        white_level = np.array(raw.camera_white_level_per_channel,
                                    dtype='float32').reshape(1,4, 1, 1)


    black_level = torch.from_numpy(black_level).contiguous()
    white_level = torch.from_numpy(white_level).contiguous()

    print(raw.camera_whitebalance)

    r, g1, b, g2 = np.array(np.array(raw.camera_whitebalance, dtype='float32') * 10000, dtype=np.uint)
    print(f"{r}, {g1}, {b}, {g2}")
    if g2 != 0:
        white_balance = [[g1, r], [g1, g2], [g2, b]]
    else:
        white_balance = [[g1, r], [g1, g1], [g1, b]]

    rawImage = raw.raw_image
    cfa_pattern_size = list(raw.raw_pattern.shape)
    if cfa_pattern_size == [2, 2]:
        cfa_pattern = [int(x) if x != 3 else 1 for x in raw.raw_pattern.flatten()]
        raw_packed = torch.from_numpy(np.float32(pack_raw_bayer(raw.raw_image_visible, raw.raw_pattern))[np.newaxis]).contiguous()
    else:
        cfa_pattern = None
        raw_packed = None


    color_matrices = []
    if np.array(raw.rgb_xyz_matrix).any():
        color_matrices.append(np.array(raw.rgb_xyz_matrix[:3]).flatten())
    else:
        color_matrices.append(np.fromstring(metadata[0]['EXIF:ColorMatrix1'], dtype='float32', sep= ' '))
        color_matrices.append(np.fromstring(metadata[0]['EXIF:ColorMatrix2'], dtype='float32', sep= ' '))

    for i in range(len(color_matrices)):
        color_matrices[i] = [[int(round(x * 10000)), 10000] for x in color_matrices[i]]


    return raw_packed, black_level, white_level, white_balance, color_matrices, cfa_pattern, cfa_pattern_size



@torch.no_grad()
def image_process():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='the folder where contains only your raw images.')
    args = parser.parse_args()

    raw_paths = []
    for path, subdirs, files in os.walk(args.data_path):
        for name in files:
            raw_paths.append(os.path.join(path, name))

    for raw_path in raw_paths:
        print(f"File : {raw_path}")
        iso, exp_time = metainfo(raw_path)
        with rawpy.imread(raw_path) as raw:
            im, bl, wl, wb, ccms, cfa, cfa_size = read_img(raw, raw_path)

#         print(f"""ISO : {iso}
#         Exposure : {exp_time}
#         Black Level : {bl}
#         White Level : {wl}
#         White Balance : {wb}
#         Color Matrices : {ccms}
#         CFA Pattern : {cfa}
#         Pattern Size : {cfa_size}
#         {'-' * 30}
# """)

if __name__ == '__main__':
    image_process()
