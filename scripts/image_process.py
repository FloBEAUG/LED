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


def get_available_device(force_cpu=False):
    if not force_cpu and torch.cuda.is_available():
        return torch.device('cuda')
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

    black_level = torch.from_numpy(black_level).contiguous()
    white_level = torch.from_numpy(white_level).contiguous()

    r, g1, b, g2 = np.array(np.array(raw.camera_whitebalance, dtype='float32') * 10000, dtype=np.uint)
    if g2 != 0:
        white_balance = [[int(g1/r * 10000), 10000], [int(g1/g2 * 10000), 10000], [int(g1/b * 10000), 10000]]
    else:
        white_balance = [[int(g1/r * 10000), 10000], [int(g1/g1 * 10000), 10000], [int(g1/b * 10000), 10000]]

    cfa_pattern_size = list(raw.raw_pattern.shape)
    cfa_pattern = [int(x) if x != 3 else 1 for x in raw.raw_pattern.flatten()]
    raw_packed = torch.from_numpy(np.float32(pack_raw_bayer(raw.raw_image_visible, raw.raw_pattern))[np.newaxis]).contiguous()
    color_matrices, forward_matrices = _extract_ccms(raw, metadata)
    active_area = _get_active_area(raw)
    make = metadata[0].get('EXIF:Make', 'Unknown')
    model = metadata[0].get('EXIF:Model', 'Unknown')

    return raw_packed, black_level, white_level, white_balance, color_matrices, cfa_pattern, cfa_pattern_size, active_area, make, model, forward_matrices


def _get_active_area(raw):
    """Find where raw.raw_image_visible sits within raw.raw_image.

    Returns [top, left, bottom, right] for the DNG ActiveArea tag.
    The full raw_image (including black border) is written to the DNG,
    but ActiveArea tells decoders which region is the visible image.
    """
    ri = raw.raw_image
    riv = raw.raw_image_visible
    ri_h, ri_w = ri.shape
    riv_h, riv_w = riv.shape

    # Try memory-address approach first (rawpy returns views)
    try:
        ri_data = ri.ctypes.data
        riv_data = riv.ctypes.data
        byte_offset = riv_data - ri_data
        itemsize = ri.dtype.itemsize
        elem_offset = byte_offset // itemsize
        left = elem_offset % ri_w
        top = elem_offset // ri_w
        if 0 <= top < ri_h and 0 <= left < ri_w:
            # Verify
            if np.array_equal(ri[top:top+riv_h, left:left+riv_w], riv):
                return [top, left, top + riv_h, left + riv_w]
    except Exception:
        pass

    # Fallback: search for the visible array within the full image
    target_first = riv[0, 0]
    for row in range(ri_h - riv_h + 1):
        for col in range(ri_w - riv_w + 1):
            if ri[row, col] == target_first:
                if np.array_equal(ri[row:row+riv_h, col:col+riv_w], riv):
                    return [row, col, row + riv_h, col + riv_w]

    # Last resort: assume visible starts at 0,0
    return [0, 0, riv_h, riv_w]


def _matrix_to_pairs(mat):
    """Convert a 3x3 numpy matrix to DNG rational pairs format [[num, denom], ...]."""
    pairs = []
    for row in mat:
        for v in row:
            # Convert to rational with denominator 10000
            val = float(v)
            num = int(round(val * 10000))
            pairs.append([num, 10000])
    return pairs


def _matrices_from_rawpy(raw):
    """Extract color matrices from rawpy rawpy object.

    LibRaw's rgb_xyz_matrix = XYZ -> camera RGB (DNG ColorMatrix).
    ForwardMatrix = camera RGB -> XYZ = inverse of ColorMatrix.

    Returns:
        (color_matrices, forward_matrices): each is a list of 2 matrices
        (for illuminant 1 and illuminant 2).
    """
    DEFAULT_CCM1 = [[12649, 10000], [-6460, 10000], [-13, 10000], [-3506, 10000],
                     [10855, 10000], [3061, 10000], [-100, 10000], [741, 10000], [7311, 10000]]
    DEFAULT_CCM2 = [[10424, 10000], [-3138, 10000], [-1300, 10000], [-4221, 10000],
                     [11938, 10000], [2584, 10000], [-547, 10000], [1658, 10000], [6183, 10000]]

    cm1 = None
    cm2 = None

    # Try EXIF ColorMatrix1/2 first (e.g., from DNG source files like RN9P.dng)
    cm1_raw = None
    cm2_raw = None

    # rawpy.rgb_xyz_matrix = XYZ -> camera RGB = ColorMatrix
    if hasattr(raw, 'rgb_xyz_matrix') and raw.rgb_xyz_matrix is not None:
        rxyz = np.array(raw.rgb_xyz_matrix[:3, :3], dtype=np.float64)
        if np.any(rxyz):
            cm1 = rxyz
            cm2 = rxyz.copy()  # Same matrix for both illuminants if only one available

    # Compute ForwardMatrix = inverse of ColorMatrix
    if cm1 is not None:
        try:
            fm1 = np.linalg.inv(cm1)
        except np.linalg.LinAlgError:
            fm1 = np.eye(3)
        fm2 = fm1.copy()
    else:
        fm1 = None
        fm2 = None

    # Convert to DNG rational pairs
    if cm1 is not None:
        ccms = [_matrix_to_pairs(cm1), _matrix_to_pairs(cm2)]
    else:
        ccms = [DEFAULT_CCM1, DEFAULT_CCM2]

    if fm1 is not None:
        fwms = [_matrix_to_pairs(fm1), _matrix_to_pairs(fm2)]
    else:
        # Default forward matrices (inverse of default CCM)
        fwms = [[[4475, 10000], [3644, 10000], [1524, 10000], [2027, 10000], [7486, 10000], [486, 10000], [725, 10000], [14, 10000], [7513, 10000]],
                [[4590, 10000], [2857, 10000], [2196, 10000], [2642, 10000], [6660, 10000], [698, 10000], [1219, 10000], [39, 10000], [6994, 10000]]]

    return ccms, fwms


def _extract_ccms(raw, metadata):
    """Extract Color Correction Matrices from source file metadata.

    Priority:
      1. EXIF:ColorMatrix1 / EXIF:ColorMatrix2 from exiftool metadata
      2. rawpy.rgb_xyz_matrix (LibRaw XYZ->camera RGB matrix)
      3. Hardcoded defaults

    Returns:
        (ccms, fwms): each is a list of 2 matrices
        ccms[i] = ColorMatrix[i+1] (XYZ -> camera RGB)
        fwms[i] = ForwardMatrix[i+1] (camera RGB -> XYZ)
    """
    DEFAULT_CCM1 = [[12649, 10000], [-6460, 10000], [-13, 10000], [-3506, 10000],
                     [10855, 10000], [3061, 10000], [-100, 10000], [741, 10000], [7311, 10000]]
    DEFAULT_CCM2 = [[10424, 10000], [-3138, 10000], [-1300, 10000], [-4221, 10000],
                     [11938, 10000], [2584, 10000], [-547, 10000], [1658, 10000], [6183, 10000]]

    cm1 = None
    cm2 = None

    # 1. Try EXIF ColorMatrix1/2 first (e.g., from DNG source files like RN9P.dng)
    cm1_raw = metadata[0].get('EXIF:ColorMatrix1')
    cm2_raw = metadata[0].get('EXIF:ColorMatrix2')

    if cm1_raw is not None:
        cm1_vals = np.fromstring(cm1_raw, dtype=np.float64, sep=' ')
        if len(cm1_vals) >= 9:
            cm1 = cm1_vals[:9].reshape(3, 3)

    if cm2_raw is not None:
        cm2_vals = np.fromstring(cm2_raw, dtype=np.float64, sep=' ')
        if len(cm2_vals) >= 9:
            cm2 = cm2_vals[:9].reshape(3, 3)

    # 2. Fall back to rawpy.rgb_xyz_matrix (LibRaw XYZ->camera RGB matrix)
    if cm1 is None:
        if hasattr(raw, 'rgb_xyz_matrix') and raw.rgb_xyz_matrix is not None:
            rxyz = np.array(raw.rgb_xyz_matrix[:3, :3], dtype=np.float64)
            if np.any(rxyz):
                cm1 = rxyz
                cm2 = rxyz.copy()  # Same matrix for both illuminants if only one available

    # 3. Compute ForwardMatrix = inverse of ColorMatrix
    if cm1 is not None:
        try:
            fm1 = np.linalg.inv(cm1)
        except np.linalg.LinAlgError:
            fm1 = np.eye(3)
        fm2 = np.linalg.inv(cm2) if cm2 is not None else fm1.copy()
    else:
        fm1 = None
        fm2 = None

    # Convert to DNG rational pairs
    if cm1 is not None:
        ccms = [_matrix_to_pairs(cm1), _matrix_to_pairs(cm2)]
    else:
        ccms = [DEFAULT_CCM1, DEFAULT_CCM2]

    if fm1 is not None:
        fwms = [_matrix_to_pairs(fm1), _matrix_to_pairs(fm2)]
    else:
        # Default forward matrices (inverse of default CCM)
        fwms = [[[4475, 10000], [3644, 10000], [1524, 10000], [2027, 10000], [7486, 10000], [486, 10000], [725, 10000], [14, 10000], [7513, 10000]],
                [[4590, 10000], [2857, 10000], [2196, 10000], [2642, 10000], [6660, 10000], [698, 10000], [1219, 10000], [39, 10000], [6994, 10000]]]

    return ccms, fwms


def save_as_dng(rawImage, filename, bpp, bl, wl, wb, ccms, cfa, cfa_size, active_area,
                make='Unknown', model='Unknown', fwms=None):
    height, width = rawImage.shape

    # set DNG tags.
    t = DNGTags()
    t.set(Tag.ImageWidth, width)
    t.set(Tag.ImageLength, height)
    t.set(Tag.ActiveArea, active_area)
    t.set(Tag.TileWidth, width)
    t.set(Tag.TileLength, height)
    t.set(Tag.Orientation, Orientation.Horizontal)
    t.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Color_Filter_Array)
    t.set(Tag.SamplesPerPixel, 1)
    t.set(Tag.BitsPerSample, bpp)
    t.set(Tag.CFARepeatPatternDim, cfa_size)
    t.set(Tag.CFAPattern, cfa)
    if fwms is not None:
        t.set(Tag.ForwardMatrix1, fwms[0])
        t.set(Tag.ForwardMatrix2, fwms[1])
    t.set(Tag.BlackLevelRepeatDim, [2, 2])
    t.set(Tag.BlackLevel, torch.reshape(bl, (-1,)).type(torch.int32).tolist())
    t.set(Tag.WhiteLevel, int(torch.mean(wl)))
    t.set(Tag.ColorMatrix1, ccms[0])
    t.set(Tag.CalibrationIlluminant1, CalibrationIlluminant.Standard_Light_A)
    t.set(Tag.ColorMatrix2, ccms[1])
    t.set(Tag.CalibrationIlluminant2, CalibrationIlluminant.D65)
    t.set(Tag.AsShotNeutral, wb)
    #t.set(Tag.BaselineExposure, [[35, 100]])
    t.set(Tag.Make, str(make))
    t.set(Tag.Model, str(model))
    t.set(Tag.DNGVersion, DNGVersion.V1_6)
    t.set(Tag.DNGBackwardVersion, DNGVersion.V1_4)
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
    parser.add_argument('--cpu', action='store_true', help='force CPU inference (no GPU tiling). Default: use GPU if available.')
    args = parser.parse_args()

    print('Building network...')
    network_g = build_network(yaml_load(args.network_options)['network_g'])
    print('Loading checkpoint...')
    load_network(network_g, args.pretrained_network, param_key='params' if not args.led else 'params_deploy')
    device = get_available_device(force_cpu=args.cpu)
    network_g = network_g.to(device)
    network_g.eval()
    raw_paths = [os.path.join(args.data_path, file) for file in os.listdir(args.data_path) if os.path.isfile(os.path.join(args.data_path, file))]
    ratio = args.ratio
    os.makedirs(args.save_path, exist_ok=True)

    suffix = '_cpu' if args.cpu else '_gpu'

    for raw_path in tqdm(raw_paths):
        if args.target_exposure is not None:
            iso, exp_time = metainfo(raw_path)
            ratio = args.target_exposure / (iso * exp_time)
        with rawpy.imread(raw_path) as raw:
            im, bl, wl, wb, ccms, cfa, cfa_size, active_area, make, model, fwms = read_img(raw, raw_path)
            im = (im - bl) / (raw.white_level - bl)
            im = (im * ratio).clip(max=torch.tensor(1.0))

            im = im.to(device)
            result = network_g(im)
            result = result.clip(0, 1).cpu()

            postprocess(raw, result, bl, raw.white_level, args.bps)
            #cv2.imwrite(os.path.join(args.save_path, pathlib.Path(raw_path).stem + "_bayer.png"), bayer_result.astype(np.uint16))
            output_name = os.path.join(args.save_path, pathlib.Path(raw_path).stem + f'_denoised{suffix}.dng')
            
            save_as_dng(raw.raw_image, output_name, args.bps, bl, wl, wb, ccms, cfa, cfa_size,
                        active_area, make=make, model=model, fwms=fwms)
            with exiftool.ExifTool() as et:
                et.execute(b"-overwrite_original",
                            f"-tagsFromFile={raw_path}".encode("utf-8"),
                            b"-all:all",
                            output_name.encode("utf-8"))

            print(f'Saved: {os.path.basename(output_name)}')


if __name__ == '__main__':
    image_process()
