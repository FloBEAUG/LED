from led.archs import build_network
from led.utils.options import yaml_load
from led.utils import save_as_dng
from led.data.raw_utils import metainfo, pack_raw_bayer, depack_raw_bayer
import rawpy
import torch
from copy import deepcopy
import argparse
import numpy as np
import os
from tqdm import tqdm
import pathlib
import exiftool


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

    black_level = np.array(raw.black_level_per_channel, dtype='float32').reshape(1, 4, 1, 1)

    if raw.camera_white_level_per_channel is None:
        white_level = np.array([raw.white_level] * 4, dtype='float32').reshape(4, 1, 1)
    else:
        white_level = np.array(raw.camera_white_level_per_channel,
                                    dtype='float32').reshape(1, 4, 1, 1)

    black_level = torch.from_numpy(black_level).contiguous()
    white_level = torch.from_numpy(white_level).contiguous()

    r, g1, b, g2 = np.array(np.array(raw.camera_whitebalance, dtype='float32') * 10000, dtype=np.uint)
    if g2 != 0:
        white_balance = [[int(g1 / r * 10000), 10000], [int(g1 / g2 * 10000), 10000], [int(g2 / b * 10000), 10000]]
    else:
        white_balance = [[int(g1 / r * 10000), 10000], [int(g1 / g1 * 10000), 10000], [int(g2 / b * 10000), 10000]]

    cfa_pattern_size = list(raw.raw_pattern.shape)
    cfa_pattern = [int(x) if x != 3 else 1 for x in raw.raw_pattern.flatten()]
    raw_packed = torch.from_numpy(np.float32(pack_raw_bayer(raw.raw_image_visible, raw.raw_pattern))[np.newaxis]).contiguous()

    color_matrices = []
    if np.array(raw.rgb_xyz_matrix).any():
        color_matrices.append(np.array(raw.rgb_xyz_matrix[:3]).flatten())
    else:
        color_matrices.append(np.fromstring(metadata[0]['EXIF:ColorMatrix1'], dtype='float32', sep=' '))
        color_matrices.append(np.fromstring(metadata[0]['EXIF:ColorMatrix2'], dtype='float32', sep=' '))

    for i in range(len(color_matrices)):
        color_matrices[i] = [[int(round(x * 10000)), 10000] for x in color_matrices[i]]

    return raw_packed, black_level, white_level, white_balance, color_matrices, cfa_pattern, cfa_pattern_size


def postprocess_to_full_bayer(result_tensor, bl, wl, raw_pattern):
    """Convert denoised packed tensor (1, 4, H/2, W/2) to full Bayer uint16 (H, W).

    Mirrors the denormalize + depack logic of postprocess() but returns
    a numpy array instead of writing into raw.raw_image_visible.

    raw_pattern: the raw Bayer pattern array from rawpy (values 0=R,1=G1,2=B,3=G2).
    """
    im = result_tensor * (wl - bl) + bl
    im = im.numpy()[0]
    full = depack_raw_bayer(im, raw_pattern)
    return full


def run_inference_tiled(network_g, im_norm, device, margin=16):
    """Run inference with quadrant tiling + margin to avoid OOM on large images.

    im_norm: packed bayer tensor (1, 4, H, W) at half resolution.
    Returns: denoised packed tensor (1, 4, H, W) on CPU.
    """
    h, w = im_norm.shape[2], im_norm.shape[3]
    h_half, w_half = h // 2, w // 2

    # If image is small enough, run directly on GPU
    if h <= 2048 and w <= 2048:
        with torch.no_grad():
            result = network_g(im_norm.to(device))
        return result.clip(0, 1).cpu()

    out = torch.zeros_like(im_norm)

    # Top-left quadrant: output [0:h_half, 0:w_half] from tile [0:h_half+margin, 0:w_half+margin]
    tile = im_norm[:, :, 0:h_half + margin, 0:w_half + margin].to(device)
    with torch.no_grad():
        res = network_g(tile).clip(0, 1).cpu()
    out[:, :, 0:h_half, 0:w_half] = res[:, :, 0:h_half, 0:w_half]

    # Top-right quadrant: output [0:h_half, w_half:w] from tile [0:h_half+margin, w_half-margin:w]
    tile = im_norm[:, :, 0:h_half + margin, w_half - margin:w].to(device)
    with torch.no_grad():
        res = network_g(tile).clip(0, 1).cpu()
    out[:, :, 0:h_half, w_half:w] = res[:, :, 0:h_half, margin:]

    # Bottom-left quadrant: output [h_half:h, 0:w_half] from tile [h_half-margin:h, 0:w_half+margin]
    tile = im_norm[:, :, h_half - margin:h, 0:w_half + margin].to(device)
    with torch.no_grad():
        res = network_g(tile).clip(0, 1).cpu()
    out[:, :, h_half:h, 0:w_half] = res[:, :, margin:, 0:w_half]

    # Bottom-right quadrant: output [h_half:h, w_half:w] from tile [h_half-margin:h, w_half-margin:w]
    tile = im_norm[:, :, h_half - margin:h, w_half - margin:w].to(device)
    with torch.no_grad():
        res = network_g(tile).clip(0, 1).cpu()
    out[:, :, h_half:h, w_half:w] = res[:, :, margin:, margin:]

    return out


@torch.no_grad()
def image_process():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pretrained_network', type=str, required=True, help='the pretrained network path.')
    parser.add_argument('--data_path', type=str, required=True, help='the folder where contains only your raw images.')
    parser.add_argument('--save_path', type=str, default='inference/image_process', help='the folder where to save the processed images (in dng), DEFAULT: \'inference/image_process\'')
    parser.add_argument('-opt', '--network_options', default='options/base/network_g/unet.yaml', help='the arch options of the pretrained network, DEFAULT: \'options/base/network_g/unet.yaml\'')
    parser.add_argument('--ratio', '--dgain', type=float, default=1.0, help='the ratio/additional digital gain you would like to add on the image, DEFAULT: 1.0.')
    parser.add_argument('--target_exposure', type=float, help='Target exposure, activate this will deactivate ratio.')
    parser.add_argument('--bps', '--output_bps', type=int, default=16, help='the bit depth for the output dng file, DEFAULT: 16.')
    parser.add_argument('--led', action='store_true', help='if you are using a checkpoint fine-tuned by our led.')
    parser.add_argument('--cpu', action='store_true', help='force CPU inference (no GPU). Default: use GPU if available.')
    parser.add_argument('--margin', type=int, default=32, help='tile margin in pixels for GPU tiled inference, DEFAULT: 32.')
    args = parser.parse_args()

    print('Building network...')
    network_g = build_network(yaml_load(args.network_options)['network_g'])
    print('Loading checkpoint...')
    load_network(network_g, args.pretrained_network, param_key='params' if not args.led else 'params_deploy')

    device = get_available_device(force_cpu=args.cpu)
    print(f'Using device: {device}')
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
            im, bl, wl, wb, ccms, cfa, cfa_size = read_img(raw, raw_path)
            im_norm = (im - bl) / (raw.white_level - bl)
            im_norm = im_norm.clip(max=torch.tensor(1.0))

            if args.cpu:
                # CPU inference (full image, no tiling)
                network_g_cpu = network_g.cpu()
                result_cpu = network_g_cpu(im_norm)
                result_cpu = result_cpu.clip(0, 1)

                full_bayer = postprocess_to_full_bayer(result_cpu, bl, wl, raw.raw_pattern)
                raw.raw_image_visible[:full_bayer.shape[0], :full_bayer.shape[1]] = full_bayer

                output_name = os.path.join(args.save_path,
                                          pathlib.Path(raw_path).stem + f'_denoised{suffix}.dng')
                save_as_dng(raw.raw_image, output_name, args.bps, bl, wl,
                            wb, ccms, cfa, cfa_size)
            else:
                # GPU inference with tiling + margin
                result_gpu = run_inference_tiled(network_g, im_norm, device, margin=args.margin)
                result_gpu = result_gpu.clip(0, 1)

                full_bayer = postprocess_to_full_bayer(result_gpu, bl, wl, raw.raw_pattern)
                raw.raw_image_visible[:full_bayer.shape[0], :full_bayer.shape[1]] = full_bayer

                output_name = os.path.join(args.save_path,
                                          pathlib.Path(raw_path).stem + f'_denoised{suffix}.dng')
                save_as_dng(raw.raw_image, output_name, args.bps, bl, wl,
                            wb, ccms, cfa, cfa_size)

            # Copy tags from source to DNG
            with exiftool.ExifTool() as et:
                et.execute(b"-overwrite_original",
                            f"-tagsFromFile={raw_path}".encode("utf-8"),
                            b"-all:all",
                            output_name.encode("utf-8"))

            print(f'Saved: {os.path.basename(output_name)}')


if __name__ == '__main__':
    image_process()
