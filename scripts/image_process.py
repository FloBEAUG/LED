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

def load_network(net, load_path, strict=True, param_key='params'):
    """Load network.

    Args:
        load_path (str): The path of networks to be loaded.
        net (nn.Module): Network.
        strict (bool): Whether strictly loaded.
        param_key (str): The parameter key of loaded network. If set to
            None, use the root 'path'.
            Default: 'params'.
    """
    load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
    if param_key is not None:
        if param_key not in load_net and 'params' in load_net:
            param_key = 'params'
            print('Loading: params_ema does not exist, use params.')
        load_net = load_net[param_key]
    print(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    net.load_state_dict(load_net, strict=strict)


def get_available_device():
    # if torch.cuda.is_available():
    #     return torch.device('cuda')
    # if torch.backends.mps.is_available():
    #     return torch.device('mps')
    return torch.device('cpu')


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

    white_level = np.array([12735] * 4, dtype='float32').reshape(4, 1, 1)

    black_level = torch.from_numpy(black_level).contiguous()
    white_level = torch.from_numpy(white_level).contiguous()

    r, g1, b, g2 = np.array(np.array(raw.camera_whitebalance, dtype='float32') * 10000, dtype=np.uint)
    if g2 != 0:
        white_balance = [[int(g1/r * 10000), 10000], [int(g1/g2 * 10000), 10000], [int(g2/b * 10000), 10000]]
    else:
        white_balance = [[int(g1/r * 10000), 10000], [int(g1/g1 * 10000), 10000], [int(g2/b * 10000), 10000]]

    rawImage = raw.raw_image
    cfa_pattern_size = list(raw.raw_pattern.shape)
    cfa_pattern = [int(x) if x != 3 else 1 for x in raw.raw_pattern.flatten()]
    raw_packed = torch.from_numpy(np.float32(pack_raw_bayer(raw.raw_image_visible, raw.raw_pattern))[np.newaxis]).contiguous()


    color_matrices = []
    if np.array(raw.rgb_xyz_matrix).any():
        color_matrices.append(np.array(raw.rgb_xyz_matrix[:3]).flatten())
    else:
        color_matrices.append(np.fromstring(metadata[0]['EXIF:ColorMatrix1'], dtype='float32', sep= ' '))
        color_matrices.append(np.fromstring(metadata[0]['EXIF:ColorMatrix2'], dtype='float32', sep= ' '))

    for i in range(len(color_matrices)):
        color_matrices[i] = [[int(round(x * 10000)), 10000] for x in color_matrices[i]]


    return raw_packed, black_level, white_level, white_balance, color_matrices, cfa_pattern, cfa_pattern_size


def save_as_dng(rawImage, filename, bpp, bl, wl, wb, ccms, cfa, cfa_size):
    height, width = rawImage.shape

    ccms = []
    ccms.append([[12649, 10000], [-6460, 10000], [-13, 10000], [-3506, 10000], [10855, 10000], [3061, 10000], [-100, 10000], [741, 10000], [7311, 10000]])
    ccms.append([[10424, 10000], [-3138, 10000], [-1300, 10000], [-4221, 10000], [11938, 10000], [2584, 10000], [-547, 10000], [1658, 10000], [6183, 10000]])

    # set DNG tags.
    t = DNGTags()
    t.set(Tag.ImageWidth, width)
    t.set(Tag.ImageLength, height)
    t.set(Tag.TileWidth, width)
    t.set(Tag.TileLength, height)
    t.set(Tag.Orientation, Orientation.Horizontal)
    t.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Color_Filter_Array)
    t.set(Tag.SamplesPerPixel, 1)
    t.set(Tag.BitsPerSample, bpp)
    t.set(Tag.CFARepeatPatternDim, cfa_size)
    t.set(Tag.CFAPattern, cfa)
    t.set(Tag.BlackLevelRepeatDim, [2, 2])
    t.set(Tag.BlackLevel, [2046, 2047, 2047, 2047])
    t.set(Tag.WhiteLevel, int(torch.mean(wl)))
    t.set(Tag.ColorMatrix1, ccms[0])
    t.set(Tag.CalibrationIlluminant1, CalibrationIlluminant.Standard_Light_A)
    t.set(Tag.ColorMatrix2, ccms[1])
    t.set(Tag.CalibrationIlluminant2, CalibrationIlluminant.D65)
    t.set(Tag.AsShotNeutral, wb)
    t.set(Tag.BaselineExposure, [[1, 1]])
    t.set(Tag.Make, "Canon")
    t.set(Tag.Model, "EOS R7")
    t.set(Tag.DNGVersion, DNGVersion.V1_4)
    t.set(Tag.DNGBackwardVersion, DNGVersion.V1_2)
    t.set(Tag.PreviewColorSpace, PreviewColorSpace.sRGB)

    # save to dng file.
    r = RAW2DNG()
    r.options(t, path="", compress=False)
    r.convert(rawImage, filename)


def  postprocess(raw, im, bl, wl, output_bps=16):
    im = im * (wl - bl) + bl
    im = im.numpy()[0]
    im = depack_raw_bayer(im, raw.raw_pattern)
    H, W = im.shape
    raw.raw_image_visible[:H, :W] = im
    return im

@torch.no_grad()
def image_process():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pretrained_network', type=str, required=True, help='the pretrained network path.')
    parser.add_argument('--data_path', type=str, required=True, help='the folder where contains only your raw images.')
    parser.add_argument('--save_path', type=str, default='inference/image_process', help='the folder where to save the processed images (in rgb), DEFAULT: \'inference/image_process\'')
    parser.add_argument('-opt', '--network_options', default='options/base/network_g/unet.yaml', help='the arch options of the pretrained network, DEFAULT: \'options/base/network_g/unet.yaml\'')
    parser.add_argument('--ratio', '--dgain', type=float, default=1.0, help='the ratio/additional digital gain you would like to add on the image, DEFAULT: 1.0.')
    parser.add_argument('--target_exposure', type=float, help='Target exposure, activate this will deactivate ratio.')
    parser.add_argument('--bps', '--output_bps', type=int, default=16, help='the bit depth for the output png file, DEFAULT: 16.')
    parser.add_argument('--led', action='store_true', help='if you are using a checkpoint fine-tuned by our led.')
    args = parser.parse_args()

    print('Building network...')
    network_g = build_network(yaml_load(args.network_options)['network_g'])
    print('Loading checkpoint...')
    load_network(network_g, args.pretrained_network, param_key='params' if not args.led else 'params_deploy')
    device = get_available_device()
    network_g = network_g.to(device)
    raw_paths = [os.path.join(args.data_path, file) for file in os.listdir(args.data_path) if os.path.isfile(os.path.join(args.data_path, file))]
    ratio = args.ratio
    os.makedirs(args.save_path, exist_ok=True)

    for raw_path in tqdm(raw_paths):
        if args.target_exposure is not None:
            iso, exp_time = metainfo(raw_path)
            ratio = args.target_exposure / (iso * exp_time)
        with rawpy.imread(raw_path) as raw:
            im, bl, wl, wb, ccms, cfa, cfa_size = read_img(raw, raw_path)
            im = (im - bl) / (raw.white_level - bl)
            im = (im * ratio).clip(max=torch.tensor(1.0))

            # margin = 32
            # im_part = []
            # im_part.append(im[:, :, :im.shape[2]//2 + margin, :im.shape[3]//2 + margin])
            # im_part.append(im[:, :, :im.shape[2]//2 + margin, im.shape[3]//2 - margin:])
            # im_part.append(im[:, :, im.shape[2]//2 - margin:, :im.shape[3]//2 + margin])
            # im_part.append(im[:, :, im.shape[2]//2 - margin:, im.shape[3]//2 - margin:])
            #
            # result_part = []
            # for image in im_part:
            #     image = image.to(device)
            #     result_part.append(network_g(image).clip(0, 1).cpu())
            #
            # result_top = torch.cat([result_part[0][:, :, 0:-margin, 0:-margin], result_part[1][:, :, 0:-margin, margin:]], dim=3)
            # result_bottom = torch.cat([result_part[2][:, :, margin:, 0:-margin], result_part[3][:, :, margin:, margin:]], dim=3)
            # result = torch.cat([result_top, result_bottom], dim=2)

            #exit(0)
            im = im.to(device)
            result = network_g(im)
            result = result.clip(0, 1).cpu()

            bayer_result = postprocess(raw, result, bl, raw.white_level, args.bps)
            #cv2.imwrite(os.path.join(args.save_path, pathlib.Path(raw_path).stem + "_bayer.png"), bayer_result.astype(np.uint16))
            output_name = os.path.join(args.save_path, pathlib.Path(raw_path).stem + "_denoised" + ".dng")
            save_as_dng(bayer_result, output_name, args.bps, bl, wl, wb, ccms, cfa, cfa_size)
            with exiftool.ExifTool() as et:
                et.execute(b"-overwrite_original",
                            f"-tagsFromFile={raw_path}".encode("utf-8"),
                            b"-all:all",
                            output_name.encode("utf-8"))

if __name__ == '__main__':
    image_process()
