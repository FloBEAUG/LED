import cv2
import math
import numpy as np
import os
import torch
from torchvision.utils import make_grid


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError(f'Only support 4D, 3D or 2D tensor. But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def tensor2img_fast(tensor, rgb2bgr=True, min_max=(0, 1)):
    """This implementation is slightly faster than tensor2img.
    It now only supports torch tensor with shape (1, c, h, w).

    Args:
        tensor (Tensor): Now only support torch tensor with (1, c, h, w).
        rgb2bgr (bool): Whether to change rgb to bgr. Default: True.
        min_max (tuple[int]): min and max values for clamp.
    """
    output = tensor.squeeze(0).detach().clamp_(*min_max).permute(1, 2, 0)
    output = (output - min_max[0]) / (min_max[1] - min_max[0]) * 255
    output = output.type(torch.uint8).cpu().numpy()
    if rgb2bgr:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output


def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img


def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    ok = cv2.imwrite(file_path, img, params)
    if not ok:
        raise IOError('Failed in writing images.')


def crop_border(imgs, crop_border):
    """Crop borders of images.

    Args:
        imgs (list[ndarray] | ndarray): Images with shape (h, w, c).
        crop_border (int): Crop border for each end of height and weight.

    Returns:
        list[ndarray]: Cropped images.
    """
    if crop_border == 0:
        return imgs
    else:
        if isinstance(imgs, list):
            return [v[crop_border:-crop_border, crop_border:-crop_border, ...] for v in imgs]
        else:
            return imgs[crop_border:-crop_border, crop_border:-crop_border, ...]


def get_dng_metadata(raw_path):
    """Extract DNG-relevant camera metadata dynamically from a source RAW file.

    Camera metadata is read directly from the source RAW so the output DNG works
    with any camera model, not just the one it was developed on.

    Args:
        raw_path (str): Path to the source RAW image.

    Returns:
        dict: Keys: cfa_pattern (list), cfa_size (list), height (int), width (int),
            black_level (np.ndarray shape (4,)), white_level (float),
            wb (np.ndarray shape (4,)), color_matrix1 (np.ndarray shape (9,)),
            color_matrix2 (np.ndarray shape (9,)), make (str), model (str).
    """
    import rawpy
    import exiftool

    metadata = {}

    with rawpy.imread(raw_path) as raw:
        raw_pattern = raw.raw_pattern
        metadata['cfa_pattern'] = [int(x) for x in raw_pattern.flatten()]
        metadata['cfa_size'] = list(raw_pattern.shape)
        metadata['height'], metadata['width'] = raw.raw_image_visible.shape
        metadata['black_level'] = np.array(raw.black_level_per_channel, dtype='float32').reshape(4)

        if raw.camera_white_level_per_channel is not None:
            metadata['white_level'] = float(np.mean(raw.camera_white_level_per_channel))
        else:
            metadata['white_level'] = float(raw.white_level)

        metadata['wb'] = np.array(raw.camera_whitebalance, dtype='float32')

        if np.array(raw.rgb_xyz_matrix).any():
            metadata['color_matrix1'] = np.array(raw.rgb_xyz_matrix[:3], dtype='float32').flatten()
            metadata['color_matrix2'] = np.array(raw.rgb_xyz_matrix[:3], dtype='float32').flatten()
        else:
            metadata['color_matrix1'] = None
            metadata['color_matrix2'] = None

    with exiftool.ExifToolHelper() as et:
        tags = et.get_metadata(raw_path)[0]
        metadata['make'] = tags.get('EXIF:Make', 'Unknown')
        metadata['model'] = tags.get('EXIF:Model', 'Unknown')

        if metadata['color_matrix1'] is None:
            cm1 = tags.get('EXIF:ColorMatrix1')
            if cm1 is not None:
                metadata['color_matrix1'] = np.fromstring(cm1, dtype='float32', sep=' ')
        if metadata['color_matrix2'] is None:
            cm2 = tags.get('EXIF:ColorMatrix2')
            if cm2 is not None:
                metadata['color_matrix2'] = np.fromstring(cm2, dtype='float32', sep=' ')

    return metadata


def _to_rational_pairs(mat):
    """Convert a color matrix or black/white level array to pidng [[num, denom], ...] pairs."""
    mat = np.asarray(mat, dtype='float32')
    if mat.ndim == 1:
        # flat list of values -> pair each with denominator 10000
        mat = np.stack([mat, np.full_like(mat, 10000)], axis=1)
    elif mat.ndim == 2 and mat.shape[1] == 1:
        mat = np.stack([mat.reshape(-1), np.full(mat.shape[0], 10000)], axis=1)
    return [[int(round(float(v[0]))), int(v[1])] for v in mat]


def save_dng(raw_image, file_path, bpp=16, black_level=None, white_level=None, wb=None,
             ccms=None, metadata=None, auto_mkdir=True, compress=False):
    """Write a single-channel Bayer RAW image as a DNG file.

    Camera metadata (Make, Model, ColorMatrix, CFA pattern, black/white level)
    is taken from ``metadata`` when provided; otherwise sensible defaults are
    used.

    Args:
        raw_image (ndarray): 2D Bayer array of shape (H, W), uint16.
        file_path (str): Output DNG file path.
        bpp (int): Bits per sample.
        black_level (list|ndarray): Black level per channel, shape (4,).
        white_level (float|list|ndarray): White level.
        wb (list|ndarray): White balance, either flat shape (4,) or pair-format
            shape (3, 2) AsShotNeutral style [[r/g,denom],[g/g,denom],[b/g,denom]].
        ccms (list): Color matrices, either a single matrix of 9 pairs or a list
            [matrix1, matrix2] of two 9-pair matrices.
        metadata (dict): Dynamic metadata from :func:`get_dng_metadata`.
        auto_mkdir (bool): Create parent folder if missing.
        compress (bool): Whether to compress the DNG (JPEG).
    """
    from pidng.core import RAW2DNG, DNGTags, Tag
    from pidng.defs import (DNGVersion, Orientation, PhotometricInterpretation,
                            CalibrationIlluminant, PreviewColorSpace)

    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)

    height, width = raw_image.shape

    if metadata is not None:
        cfa_pattern = metadata['cfa_pattern']
        cfa_size = metadata['cfa_size']
        bl = metadata['black_level'] if black_level is None else np.asarray(black_level, dtype='float32')
        wl = metadata['white_level'] if white_level is None else np.asarray(white_level, dtype='float32')
        wb_vals = metadata['wb'] if wb is None else np.asarray(wb, dtype='float32')
        make = metadata.get('make', 'Unknown')
        model = metadata.get('model', 'Unknown')
        cm1 = metadata.get('color_matrix1')
        cm2 = metadata.get('color_matrix2')
    else:
        cfa_pattern = [0, 1, 1, 2]
        cfa_size = [2, 2]
        bl = np.zeros(4, dtype='float32') if black_level is None else np.asarray(black_level, dtype='float32')
        wl = np.array([65535.0], dtype='float32') if white_level is None else np.asarray(white_level, dtype='float32')
        wb_vals = np.array([1.0, 1.0, 1.0], dtype='float32') if wb is None else np.asarray(wb, dtype='float32')
        make = 'Unknown'
        model = 'Unknown'
        cm1 = None
        cm2 = None

    bl = np.asarray(bl, dtype='float32').reshape(-1)
    wl_arr = np.asarray(wl, dtype='float32').reshape(-1)
    wl = float(wl_arr.mean())
    wb_vals = np.asarray(wb_vals, dtype='float32')

    # ccms: single matrix (9 pairs) or list [matrix1, matrix2]
    ccms2 = _to_rational_pairs(cm2) if cm2 is not None else _to_rational_pairs(
        np.array([10424, 10000, -3138, 10000, -1300, 10000, -4221, 10000, 11938, 10000,
                  2584, 10000, -547, 10000, 1658, 10000, 6183, 10000], dtype='float32'))
    if ccms is not None:
        if isinstance(ccms, (list, tuple)) and len(ccms) == 2:
            ccms = _to_rational_pairs(ccms[0])
            ccms2 = _to_rational_pairs(ccms[1])
        elif isinstance(ccms, (list, tuple)) and len(ccms) == 1:
            ccms = _to_rational_pairs(ccms[0])
        else:
            ccms = _to_rational_pairs(ccms)
    else:
        ccms = _to_rational_pairs(cm1) if cm1 is not None else _to_rational_pairs(
            np.array([12649, 10000, -6460, 10000, -13, 10000, -3506, 10000, 10855, 10000,
                      3061, 10000, -100, 10000, 741, 10000, 7311, 10000], dtype='float32'))

    # AsShotNeutral: [r/g, 10000, b/g, 10000] style pairs (3 pairs).
    if wb_vals.ndim == 2 and wb_vals.shape == (3, 2):
        # already in [[num, denom], ...] AsShotNeutral pair format
        as_shot_neutral = [[int(round(float(v[0]))), int(v[1])] for v in wb_vals]
    else:
        wb_flat = wb_vals.reshape(-1)
        g = wb_flat[1] if wb_flat.size > 1 and wb_flat[1] != 0 else 1.0
        r = wb_flat[0] if wb_flat.size > 0 else 1.0
        b = wb_flat[2] if wb_flat.size > 2 else 1.0
        as_shot_neutral = [[int(round(r / g * 10000)), 10000], [10000, 10000],
                           [int(round(b / g * 10000)), 10000]]

    # Black level as list of 4 ints.
    black_level_list = [int(round(float(v))) for v in bl[:4]]

    t = DNGTags()
    t.set(Tag.ImageWidth, width)
    t.set(Tag.ImageLength, height)
    t.set(Tag.ActiveArea, [0, 0, height, width])
    t.set(Tag.TileWidth, width)
    t.set(Tag.TileLength, height)
    t.set(Tag.Orientation, Orientation.Horizontal)
    t.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Color_Filter_Array)
    t.set(Tag.SamplesPerPixel, 1)
    t.set(Tag.BitsPerSample, bpp)
    t.set(Tag.CFARepeatPatternDim, cfa_size)
    t.set(Tag.CFAPattern, cfa_pattern)
    t.set(Tag.BlackLevelRepeatDim, [2, 2])
    t.set(Tag.BlackLevel, black_level_list)
    t.set(Tag.WhiteLevel, int(round(wl)))
    t.set(Tag.ColorMatrix1, ccms)
    t.set(Tag.CalibrationIlluminant1, CalibrationIlluminant.Standard_Light_A)
    t.set(Tag.ColorMatrix2, ccms2)
    t.set(Tag.CalibrationIlluminant2, CalibrationIlluminant.D65)
    t.set(Tag.AsShotNeutral, as_shot_neutral)
    t.set(Tag.Make, str(make))
    t.set(Tag.Model, str(model))
    t.set(Tag.DNGVersion, DNGVersion.V1_6)
    t.set(Tag.DNGBackwardVersion, DNGVersion.V1_4)
    t.set(Tag.PreviewColorSpace, PreviewColorSpace.sRGB)

    r = RAW2DNG()
    r.options(t, path="", compress=compress)
    raw_image = np.clip(raw_image, 0, 65535).astype(np.uint16)
    r.convert(raw_image, file_path)


def tensor_to_bayer_dng(tensor, black_level, white_level, cfa_pattern):
    """Convert a normalized packed-bayer tensor (4, H, W) to a full Bayer DNG array.

    Args:
        tensor (Tensor): Packed bayer of shape (4, H, W) in range [0, 1].
        black_level (ndarray): Black level per channel, shape (4,).
        white_level (float): White level.
        cfa_pattern (list): CFA pattern list for depacking.

    Returns:
        ndarray: Full Bayer array of shape (H*2, W*2).
    """
    from led.data.raw_utils import depack_raw_bayer

    bl = np.asarray(black_level, dtype='float32').reshape(4, 1, 1)
    wl = float(white_level)
    im = tensor.numpy() * (wl - bl) + bl
    full_bayer = depack_raw_bayer(im, np.array(cfa_pattern).reshape(2, 2))
    return full_bayer
