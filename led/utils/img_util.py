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


def save_as_dng(rawImage, filename, bpp, bl, wl, wb, ccms, cfa, cfa_size):
    """Write a Bayer RAW array as a DNG file.

    All DNG tags are kept exactly as in the original scripts/image_process.py
    output (hardcoded Make/Model, color matrices, ActiveArea, etc.).

    Args:
        rawImage (ndarray): 2D Bayer array of shape (H, W).
        filename (str): Output DNG file path.
        bpp (int): Bits per sample.
        bl (Tensor|ndarray): Black level per channel, shape (1, 4, 1, 1).
        wl (Tensor|ndarray): White level per channel, shape (1, 4, 1, 1).
        wb (list): White balance pairs.
        ccms (list): Color correction matrices.
        cfa (list): CFA pattern.
        cfa_size (list): CFA pattern dimensions.
    """
    from pidng.core import RAW2DNG, DNGTags, Tag
    from pidng.defs import (DNGVersion, Orientation, PhotometricInterpretation,
                            CalibrationIlluminant, PreviewColorSpace)

    height, width = rawImage.shape

    ccms = []
    ccms.append([[12649, 10000], [-6460, 10000], [-13, 10000], [-3506, 10000], [10855, 10000], [3061, 10000], [-100, 10000], [741, 10000], [7311, 10000]])
    ccms.append([[10424, 10000], [-3138, 10000], [-1300, 10000], [-4221, 10000], [11938, 10000], [2584, 10000], [-547, 10000], [1658, 10000], [6183, 10000]])

    fwms = []
    fwms.append([[4475, 10000], [3644, 10000], [1524, 10000], [2027, 10000], [7486, 10000], [486, 10000], [725, 10000], [14, 10000], [7513, 10000]])
    fwms.append([[4590, 10000], [2857, 10000], [2196, 10000], [2642, 10000], [6660, 10000], [698, 10000], [1219, 10000], [39, 10000], [6994, 10000]])

    ccs = []
    ccs.append([[10122, 10000], [0, 10000], [0, 10000], [0, 10000], [10000, 10000], [0, 10000], [0, 10000], [0, 10000], [9747, 10000]])
    ccs.append([[10122, 10000], [0, 10000], [0, 10000], [0, 10000], [10000, 10000], [0, 10000], [0, 10000], [0, 10000], [9747, 10000]])

    # set DNG tags.
    t = DNGTags()
    t.set(Tag.ImageWidth, width)
    t.set(Tag.ImageLength, height)
    t.set(Tag.ActiveArea, [72, 144, 4732, 7128])
    t.set(Tag.TileWidth, width)
    t.set(Tag.TileLength, height)
    t.set(Tag.Orientation, Orientation.Horizontal)
    t.set(Tag.PhotometricInterpretation, PhotometricInterpretation.Color_Filter_Array)
    t.set(Tag.SamplesPerPixel, 1)
    t.set(Tag.BitsPerSample, bpp)
    t.set(Tag.CFARepeatPatternDim, cfa_size)
    t.set(Tag.CFAPattern, cfa)
    #t.set(Tag.CFAPlaneColor, [0, 1, 2])
    #t.set(Tag.CFALayout, [1])
    #t.set(Tag.ForwardMatrix1, fwms[0])
    #t.set(Tag.ForwardMatrix2, fwms[1])
    #t.set(Tag.CameraCalibration1, ccs[0])
    #t.set(Tag.CameraCalibration2, ccs[1])
    t.set(Tag.BlackLevelRepeatDim, [2, 2])
    t.set(Tag.BlackLevel, torch.reshape(bl, (-1,)).type(torch.int32).tolist())
    t.set(Tag.WhiteLevel, int(torch.mean(wl)))
    t.set(Tag.ColorMatrix1, ccms[0])
    t.set(Tag.CalibrationIlluminant1, CalibrationIlluminant.Standard_Light_A)
    t.set(Tag.ColorMatrix2, ccms[1])
    t.set(Tag.CalibrationIlluminant2, CalibrationIlluminant.D65)
    t.set(Tag.AsShotNeutral, wb)
    #t.set(Tag.BaselineExposure, [[35, 100]])
    t.set(Tag.Make, "Canon")
    t.set(Tag.Model, "EOS R7")
    t.set(Tag.DNGVersion, DNGVersion.V1_6)
    t.set(Tag.DNGBackwardVersion, DNGVersion.V1_4)
    t.set(Tag.PreviewColorSpace, PreviewColorSpace.sRGB)

    # save to dng file.
    r = RAW2DNG()
    r.options(t, path="", compress=False)
    r.convert(rawImage, filename)
