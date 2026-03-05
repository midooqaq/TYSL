'''
# --------------------------------------------------------
# Iterative Inference Script for DA Model
# Based on test_SwinFuSR.py
# 
# Usage:
#   python test_da_iterative.py --opt options/test_da_x8.json --iterations 3
#
# This script performs iterative inference:
#   1. Loads initial LR and Guide images from dataset
#   2. Extracts Y channel from Guide (YUV HR's Y component)
#   3. Upsamples LR to HR size
#   4. Iteratively: 
#      - Input: upsampled LR + Guide Y channel
#      - Output: refined result
#      - Use output as new upsampled LR for next iteration
#   5. Saves final result
# --------------------------------------------------------
'''
import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
import torch.nn.functional as F
import cv2

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model
import warnings
warnings.filterwarnings("ignore")

DEFAULT_DA12_RANDOM_CHECKPOINT = "/data/AI/cyj/SwinPaste/Model/SR_competition26_da13/Guided SR/models/100000_E.pth"

class _DistributedNonPaddingSampler(torch.utils.data.Sampler):
    """
    A simple distributed sampler for evaluation/inference that DOES NOT pad or repeat samples.
    Each rank gets a disjoint subset: indices[rank::world_size].
    """
    def __init__(self, dataset, world_size: int, rank: int):
        self.dataset = dataset
        self.world_size = int(world_size)
        self.rank = int(rank)

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        return iter(indices[self.rank::self.world_size])

    def __len__(self):
        # ceil((N-rank)/world_size) for rank < N%world_size, else floor
        n = len(self.dataset)
        return (n - self.rank + self.world_size - 1) // self.world_size


def upsample_lr_to_hr_size(lr_img, scale_factor, target_h=None, target_w=None, interpolation_mode='bicubic'):
    """
    Upsample LR image to HR size using specified interpolation
    
    Args:
        lr_img: numpy array, HxWxC (uint8, 0-255)
        scale_factor: upsampling scale factor
        target_h, target_w: target size (if None, use scale_factor)
        interpolation_mode: 'bilinear' or 'bicubic'
    
    Returns:
        upsampled_img: numpy array, upsampled to HR size
    """
    h, w = lr_img.shape[:2]
    if target_h is None or target_w is None:
        target_h = int(h * scale_factor)
        target_w = int(w * scale_factor)
    
    # Convert to float for interpolation
    lr_float = lr_img.astype(np.float32) / 255.0
    # Select interpolation method
    if interpolation_mode.lower() == 'bilinear':
        interpolation = cv2.INTER_LINEAR
    else:
        interpolation = cv2.INTER_CUBIC
    upsampled_float = cv2.resize(lr_float, (target_w, target_h), interpolation=interpolation)
    upsampled_img = np.clip(upsampled_float * 255.0, 0, 255).astype(np.uint8)
    
    return upsampled_img


def get_blend_mask(patch_size):
    """
    生成一个 2D 的金字塔形/帐篷形权重 Mask（Bartlett 窗）。
    中心权重为 1，越靠近边缘权重越接近 0，线性衰减。
    """
    # 生成 -1 到 1 的坐标
    coords = torch.linspace(-1.0, 1.0, patch_size)
    # 1D 线性衰减 (Bartlett window): 1 - |x|
    weight_1d = 1.0 - torch.abs(coords)
    # 2D 权重是两个 1D 权重的外积
    weight_2d = weight_1d.unsqueeze(1) * weight_1d.unsqueeze(0)
    # 增加一点极小值防止除以 0
    weight_2d = weight_2d + 1e-5
    return weight_2d  # Shape: [patch_size, patch_size]


def get_hann_mask(patch_size):
    """
    生成 2D 汉宁窗 (Hann Window) 余弦权重 Mask。
    使用 sin^2(x)，x in [0, pi]，边缘和中心过渡平滑。
    注意：与 Bartlett (linear) 形状不同，不应产生相同结果。
    """
    # 生成 0 到 pi 的坐标（与 linear 的 [-1,1] 不同）
    coords = torch.linspace(0, math.pi, patch_size)
    # 1D 汉宁窗: sin^2(x)，在 0 和 pi 处为 0，在 pi/2 处为 1
    weight_1d = torch.sin(coords) ** 2
    # 外积生成 2D 权重
    weight_2d = weight_1d.unsqueeze(1) * weight_1d.unsqueeze(0)
    weight_2d = weight_2d + 1e-5
    return weight_2d  # Shape: [patch_size, patch_size]


def _verify_mask_difference_once():
    """一次性验证 linear 与 hann mask 是否不同（调试用）"""
    if getattr(_verify_mask_difference_once, '_done', False):
        return
    p = 128
    m_lin = get_blend_mask(p)
    m_hann = get_hann_mask(p)
    diff = (m_lin - m_hann).abs().max().item()
    c = p // 2
    print(f'[Mask verify] linear vs hann: max_diff={diff:.6f}')
    print(f'[Mask verify] linear center={m_lin[c,c].item():.4f} edge={m_lin[0,c].item():.4f}')
    print(f'[Mask verify] hann  center={m_hann[c,c].item():.4f} edge={m_hann[0,c].item():.4f}')
    if diff < 1e-5:
        print('[Mask verify] WARNING: linear and hann masks are nearly identical!')
    _verify_mask_difference_once._done = True


def get_gaussian_mask(patch_size, sigma=0.3):
    """
    生成 2D 高斯分布权重 Mask。
    sigma 越小，权重越集中在中心；sigma 越大，越平缓。0.25~0.3 是很好的经验值。
    """
    # 生成 -1 到 1 的坐标
    coords = torch.linspace(-1.0, 1.0, patch_size)
    
    # 1D 高斯分布
    weight_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    
    # 外积生成 2D 权重
    weight_2d = weight_1d.unsqueeze(1) * weight_1d.unsqueeze(0)
    
    # 将最大值归一化到 1.0
    weight_2d = weight_2d / weight_2d.max()
    
    weight_2d = weight_2d + 1e-5 
    
    return weight_2d  # Shape: [patch_size, patch_size]


def get_tta_transforms():
    """
    定义 8 种 TTA 变换：原图、上下翻转、左右翻转、旋转 90/180/270° 以及组合，
    同时返回各自的逆变换，便于把输出变回原始方向做平均。
    返回值：[(forward_fn, inverse_fn, name), ...]
    """
    def identity(x):
        return x

    def vflip(x):
        # 上下翻转：沿 H 维翻转（C, H, W）
        return torch.flip(x, [1])

    def hflip(x):
        # 左右翻转：沿 W 维翻转（C, H, W）
        return torch.flip(x, [2])

    def rot90(x):
        return torch.rot90(x, 1, [1, 2])

    def rot180(x):
        return torch.rot90(x, 2, [1, 2])

    def rot270(x):
        return torch.rot90(x, 3, [1, 2])

    transforms = []
    # 1) 原图
    transforms.append((identity, identity, 'orig'))
    # 2) 上下翻转
    transforms.append((vflip, vflip, 'vflip'))
    # 3) 左右翻转
    transforms.append((hflip, hflip, 'hflip'))
    # 4) 旋转 90°
    transforms.append((rot90, rot270, 'rot90'))
    # 5) 旋转 180°
    transforms.append((rot180, rot180, 'rot180'))
    # 6) 旋转 270°
    transforms.append((rot270, rot90, 'rot270'))
    # 7) 旋转 90° 后再上下翻转
    transforms.append(
        (lambda x: vflip(rot90(x)),
         lambda x: rot270(vflip(x)),
         'rot90_vflip')
    )
    # 8) 旋转 270° 后再上下翻转
    transforms.append(
        (lambda x: vflip(rot270(x)),
         lambda x: rot90(vflip(x)),
         'rot270_vflip')
    )

    return transforms


def run_model_on_patches_forward_only(model, lr_tensor, guide_tensor, test_data, patch_size=128, stride=32):
    """
    只做 patch 切分和模型前向推理，返回所有 patch 的输出和坐标信息。
    这个函数可以被多次调用以复用 patch 推理结果。

    Args:
        model: 已经加载好权重的模型（define_Model 的返回值）
        lr_tensor: CxHxW 的 LR tensor（float, [0,1]）
        guide_tensor: CxHxW 的 Guide tensor（float, [0,1]），与 lr_tensor 空间尺寸一致
        test_data: 原始 dataloader 提供的字典，用于复用 path 信息
        patch_size: patch 的空间尺寸（默认为 128）
        stride: 滑窗步长（默认为 patch_size/4=32）

    Returns:
        patch_outputs: List of tensors, 每个元素是 CxPxP 的 patch 输出
        coords: List of tuples, 每个元素是 (h_start, h_end, w_start, w_end)
        H, W: 原始图像的高度和宽度
        device: 输出所在的设备
    """
    assert lr_tensor.dim() == 3 and guide_tensor.dim() == 3, "期望 lr_tensor 和 guide_tensor 都是 CxHxW"

    # IMPORTANT: keep everything on the same device
    device = getattr(model, "device", lr_tensor.device)
    lr_tensor = lr_tensor.to(device, non_blocking=True)
    guide_tensor = guide_tensor.to(device, non_blocking=True)

    _, H, W = lr_tensor.shape
    _, Hg, Wg = guide_tensor.shape
    assert H == Hg and W == Wg, "LR 与 Guide 的空间尺寸必须一致"

    # Patch batching: accumulate patches and run one forward for many patches at once.
    patch_batch = getattr(run_model_on_patches_forward_only, "patch_batch", 16)
    lr_patches = []
    guide_patches = []
    coords = []          # 当前 batch 内的坐标
    all_coords = []      # 累积所有 batch 的坐标
    all_patch_outputs = []

    def _flush_batch():
        nonlocal lr_patches, guide_patches, coords, all_patch_outputs, all_coords
        if not lr_patches:
            return

        lr_batch = torch.stack(lr_patches, dim=0)        # NxCxPxP
        guide_batch = torch.stack(guide_patches, dim=0)  # NxCxPxP

        iter_data = {
            'Lr': lr_batch,
            'Guide': guide_batch,
            'Lr_path': test_data['Lr_path'],
            'Guide_path': test_data['Guide_path']
        }

        model.feed_data(iter_data, phase='test', need_GT=False)
        model.test()

        # model.output is NxCxPxP (tensor on device)
        out_batch = model.output  # keep on GPU

        # 保存每个 patch 的输出
        for i in range(out_batch.shape[0]):
            all_patch_outputs.append(out_batch[i])  # CxPxP
        # 保存对应坐标
        all_coords.extend(coords)

        # reset buffers（coords 只用于当前 batch）
        lr_patches = []
        guide_patches = []
        coords = []

    # 逐 patch 滑窗推理
    for h_idx in range(0, H, stride):
        for w_idx in range(0, W, stride):
            # 处理边界：保证切出来的 patch 恰好是 patch_size x patch_size
            h_start = min(h_idx, max(0, H - patch_size))
            w_start = min(w_idx, max(0, W - patch_size))
            h_end = h_start + patch_size
            w_end = w_start + patch_size

            lr_patch = lr_tensor[:, h_start:h_end, w_start:w_end]
            guide_patch = guide_tensor[:, h_start:h_end, w_start:w_end]

            lr_patches.append(lr_patch)
            guide_patches.append(guide_patch)
            coords.append((h_start, h_end, w_start, w_end))

            if len(lr_patches) >= patch_batch:
                _flush_batch()

    # flush remaining patches
    _flush_batch()

    # 返回累积的坐标 all_coords，而不是最后一个 batch 的 coords
    return all_patch_outputs, all_coords, H, W, device


def blend_patches_to_canvas(patch_outputs, coords, H, W, device, patch_size=128, mask_type='linear'):
    """
    将 patch 输出使用指定的 mask 融合到完整图像画布上。

    Args:
        patch_outputs: List of tensors, 每个元素是 CxPxP 的 patch 输出
        coords: List of tuples, 每个元素是 (h_start, h_end, w_start, w_end)
        H, W: 输出图像的高度和宽度
        device: 输出所在的设备
        patch_size: patch 的空间尺寸（默认为 128）
        mask_type: 融合掩码类型，可选 'linear', 'hann', 'gaussian'（默认 'linear'）

    Returns:
        output: CxHxW 的重建结果 tensor（重叠区域做加权平均）
    """
    # 根据 mask_type 选择不同的融合掩码
    if mask_type == 'hann':
        mask_2d = get_hann_mask(patch_size).to(device)  # [P, P]
    elif mask_type == 'gaussian':
        mask_2d = get_gaussian_mask(patch_size).to(device)  # [P, P]
    else:  # 'linear' or default
        mask_2d = get_blend_mask(patch_size).to(device)  # [P, P]

    # 获取通道数（从第一个 patch 输出）
    C = patch_outputs[0].shape[0]

    # 输出画布 & 权重画布（用于重叠区域的加权平均）
    output_canvas = torch.zeros(C, H, W, device=device)
    weight_canvas = torch.zeros(1, H, W, device=device)

    # 将每个 patch 加权累加到画布上
    for out_patch, (h_start, h_end, w_start, w_end) in zip(patch_outputs, coords):
        output_canvas[:, h_start:h_end, w_start:w_end] += out_patch * mask_2d.unsqueeze(0)
        weight_canvas[:, h_start:h_end, w_start:w_end] += mask_2d.unsqueeze(0)

    # 归一化：按累计权重做加权平均，得到最终平滑图像
    # weight_canvas: [1, H, W] 会在通道维度上自动广播到 [C, H, W]
    output = output_canvas / weight_canvas

    return output


def run_model_on_patches(model, lr_tensor, guide_tensor, test_data, patch_size=128, stride=32, mask_type='linear'):
    """
    使用小 patch（patch_size x patch_size）进行推理，并将结果拼接回完整图像。
    这是向后兼容的包装函数，内部调用 run_model_on_patches_forward_only + blend_patches_to_canvas。

    Args:
        model: 已经加载好权重的模型（define_Model 的返回值）
        lr_tensor: CxHxW 的 LR tensor（float, [0,1]）
        guide_tensor: CxHxW 的 Guide tensor（float, [0,1]），与 lr_tensor 空间尺寸一致
        test_data: 原始 dataloader 提供的字典，用于复用 path 信息
        patch_size: patch 的空间尺寸（默认为 128）
        stride: 滑窗步长（默认为 patch_size/4=32）
        mask_type: 融合掩码类型，可选 'linear', 'hann', 'gaussian'（默认 'linear'）

    Returns:
        output: CxHxW 的重建结果 tensor（重叠区域做加权平均）
    """
    patch_outputs, coords, H, W, device = run_model_on_patches_forward_only(
        model, lr_tensor, guide_tensor, test_data, patch_size, stride
    )
    output = blend_patches_to_canvas(patch_outputs, coords, H, W, device, patch_size, mask_type)
    return output


def main(json_path='options/test_da_x8.json', num_iterations=1):
    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', action='store_true', help='Enable distributed inference (torchrun).')
    parser.add_argument('--iterations', type=int, default=num_iterations,
                        help='Number of iterative inference steps')
    parser.add_argument('--checkpoint', type=str,
                        default=None,
                        help=('Path to checkpoint file (G or E model). '
                              'If not specified, prefer model path in JSON opt; '
                              'fallback to a hard-coded default only when JSON has no model.'))
    parser.add_argument('--patch_batch', type=int, default=16,
                        help='How many patches to run in one forward pass (bigger = faster, uses more VRAM).')

    args = parser.parse_args()
    # 纯推理
    opt = option.parse(args.opt, is_train=False)
    opt['dist'] = args.dist
    num_iterations = args.iterations
    # set patch-batch on the function (simple way to thread it through without refactor)
    run_model_on_patches.patch_batch = int(args.patch_batch)
    run_model_on_patches_forward_only.patch_batch = int(args.patch_batch)

    # Handle checkpoint loading - support both G and E models
    # 优先级：JSON 配置中的 pretrained_netE/G > 命令行 --checkpoint > 默认模型
    if opt['path'].get('pretrained_netE') or opt['path'].get('pretrained_netG'):
        # JSON 已经指定了预训练模型，直接使用
        if opt['path'].get('pretrained_netE'):
            print(f'Using EMA model (E) from JSON config: {opt["path"]["pretrained_netE"]}')
        if opt['path'].get('pretrained_netG'):
            print(f'Using generator model (G) from JSON config: {opt["path"]["pretrained_netG"]}')
    elif args.checkpoint:
        # 当 JSON 没有指定模型时，再考虑命令行传入的 checkpoint
        checkpoint_path = args.checkpoint
        if not os.path.isabs(checkpoint_path):
            # If relative path, join with models directory
            models_dir = opt['path'].get('models', '')
            checkpoint_path = os.path.join(models_dir, args.checkpoint)
        
        if os.path.exists(checkpoint_path):
            # Determine if it's G or E model
            if checkpoint_path.endswith('_E.pth'):
                opt['path']['pretrained_netE'] = checkpoint_path
                print(f'Using EMA model (E) from command line: {checkpoint_path}')
            elif checkpoint_path.endswith('_G.pth'):
                opt['path']['pretrained_netG'] = checkpoint_path
                print(f'Using generator model (G) from command line: {checkpoint_path}')
            else:
                # Try to determine from filename（默认按 E 处理）
                opt['path']['pretrained_netE'] = checkpoint_path
                print(f'Using model from command line: {checkpoint_path}')
        else:
            print(f'Error: Checkpoint not found: {checkpoint_path}')
            exit(1)
    else:
        # 最后兜底：如果 JSON 和命令行都没给，就使用一个硬编码的默认模型（兼容老实验）
        default_model = "Model/SR_competition26_da8_bicubic/Guided SR/models/70000_E.pth"
        if os.path.exists(default_model):
            opt['path']['pretrained_netE'] = default_model
            print(f'Using default EMA model: {default_model}')
        else:
            print(f'Error: Default model not found: {default_model}')
            print('Please specify a checkpoint in JSON opt.path or via --checkpoint option')
            exit(1)

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()
    rank = opt['rank']
    world_size = opt['world_size']
    if rank == 0:
        print(f"Distributed inference: world_size={world_size}")
    
    if opt['rank'] == 0:
        for key, path in opt['path'].items():
            print(path)
    # Ensure output dirs exist on ALL ranks (each rank will save its own subset of images)
    for key, path in opt['path'].items():
        if 'pretrained' not in key and path is not None:
            os.makedirs(path, exist_ok=True)

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # For pure inference, we rely on pretrained_netG / pretrained_netE
    # specified in the JSON config (no checkpoint auto-search here).
    current_step = 0

    border = opt['scale']
    scale_factor = opt['scale']

    # ----------------------------------------
    # save opt to a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)
    
    seed = 60
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    '''
    # ----------------------------------------
    # Step--2 (create dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) create_dataloader for test
    # ----------------------------------------
    
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'test':
            test_set = define_Dataset(dataset_opt)
            if opt['dist']:
                # Non-padding sampler to avoid duplicated last samples across ranks
                sampler = _DistributedNonPaddingSampler(test_set, world_size=world_size, rank=rank)
                test_dataset = DataLoader(
                    test_set,
                    batch_size=1,
                    shuffle=False,
                    sampler=sampler,
                    num_workers=1,
                    drop_last=False,
                    pin_memory=True,
                )
            else:
                test_dataset = DataLoader(
                    test_set,
                    batch_size=1,
                    shuffle=False,
                    num_workers=1,
                    drop_last=False,
                    pin_memory=True,
                )
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.load()  # load the model

    '''
    # ----------------------------------------
    # Step--4 (iterative inference)
    # ----------------------------------------
    '''

    print(f'Starting inference on full test set (iterations: {num_iterations})...')
    
    # 只使用线性 Bartlett 掩码（linear_mask）
    mask_configs = [
        ('linear', 'linear_mask'),
    ]
    
    # 为每种掩码方法创建输出文件夹（普通平均 + 去极值平均）
    base_images_dir = opt['path']['images']
    output_dirs = {}
    output_dirs_trimmed = {}
    for mask_type, dir_suffix in mask_configs:
        output_dir = os.path.join(base_images_dir, dir_suffix)
        os.makedirs(output_dir, exist_ok=True)
        output_dirs[mask_type] = output_dir
        output_dir_trimmed = os.path.join(base_images_dir, dir_suffix + '_trimmed')
        os.makedirs(output_dir_trimmed, exist_ok=True)
        output_dirs_trimmed[mask_type] = output_dir_trimmed
        if rank == 0:
            print(f'Output directory for {mask_type} mask: {output_dir}')
            print(f'Output directory for {mask_type} mask (trimmed): {output_dir_trimmed}')
    
    # 一次性验证 linear 与 hann mask 是否不同
    if rank == 0:
        _verify_mask_difference_once()
    
    # Process all test images
    for idx, test_data in enumerate(test_dataset):
        # In distributed mode, idx is local to this rank's subset.
        # The sampler guarantees different ranks process different samples (no repeats/padding).
        print(f'\nProcessing image {idx+1}/{len(test_dataset)}')
        print(f'Guide path: {test_data["Guide_path"][0]}')
        print(f'LR path: {test_data["Lr_path"][0]}')
        
        image_name_ext = os.path.basename(test_data['Guide_path'][0])
        image_name = os.path.splitext(image_name_ext)[0]  # without extension
        
        # Get data from dataset - test dataset provides Lr and Guide at same size
        # DataLoader adds batch dimension, so [0] removes it: BxCxHxW -> CxHxW
        lr_tensor = test_data['Lr'][0]  # CxHxW, float [0,1]
        guide_tensor = test_data['Guide'][0]  # CxHxW, float [0,1]
        
        print(f'LR tensor shape: {lr_tensor.shape}')
        print(f'Guide tensor shape: {guide_tensor.shape}')

        # da12_random 模型期望 RGB guide（3 通道）
        if guide_tensor.dim() == 3 and guide_tensor.shape[0] != 3:
            raise RuntimeError(
                f"Guide 通道数异常：期望 3 通道 RGB，但得到 {guide_tensor.shape[0]} 通道。"
                f"请检查 option 中 datasets.test.convert_rgb_to_ycrcb 是否为 false，以及输入图是否为 RGB。"
            )
        
        # -------------------------
        # 8x TTA：整图翻转/旋转 -> 重叠 patch 推理 -> 反变换回原方向
        # 关键优化：对每个 TTA 变换和每次迭代，只调用一次模型前向，然后用三种 mask 分别融合
        # -------------------------
        tta_transforms = get_tta_transforms()
        
        # 为三种 mask 分别准备输出列表
        outputs_inv_by_mask = {mask_type: [] for mask_type, _ in mask_configs}

        for tta_idx, (forward_t, inverse_t, t_name) in enumerate(tta_transforms):
            print(f'    TTA {tta_idx+1}/{len(tta_transforms)}: {t_name}')

            # 对整张 LR 和 Guide 做 TTA 变换
            lr_aug = forward_t(lr_tensor)
            guide_aug = forward_t(guide_tensor)

            # 迭代推理（每次都是"整图 + 重叠切块 + 权重平滑融合"）
            current_lr = lr_aug
            output_tensors_by_mask = None  # 初始化，用于最后的 TTA 反变换
            
            for iter_idx in range(num_iterations):
                if num_iterations > 1:
                    print(f'      Iteration {iter_idx+1}/{num_iterations}...')

                # 【关键优化】只调用一次模型前向，获取所有 patch 的输出
                patch_outputs, coords, H, W, device = run_model_on_patches_forward_only(
                    model=model,
                    lr_tensor=current_lr,
                    guide_tensor=guide_aug,
                    test_data=test_data,
                    patch_size=128,
                    stride=32  # 每次平移 1/4 patch
                )

                # 【关键优化】用三种不同的 mask 分别融合，复用同一批 patch 输出
                output_tensors_by_mask = {}
                for mask_type, _ in mask_configs:
                    output_tensor = blend_patches_to_canvas(
                        patch_outputs, coords, H, W, device,
                        patch_size=128, mask_type=mask_type
                    )
                    output_tensors_by_mask[mask_type] = output_tensor

                # 下一次迭代的输入（使用 linear mask 的结果作为下一轮输入，保持一致性）
                current_lr = output_tensors_by_mask['linear']
            
            # 如果 num_iterations == 0，需要对原始 lr_aug 也做一次推理
            if num_iterations == 0:
                patch_outputs, coords, H, W, device = run_model_on_patches_forward_only(
                    model=model,
                    lr_tensor=current_lr,
                    guide_tensor=guide_aug,
                    test_data=test_data,
                    patch_size=128,
                    stride=32  # 每次平移 1/4 patch
                )
                output_tensors_by_mask = {}
                for mask_type, _ in mask_configs:
                    output_tensor = blend_patches_to_canvas(
                        patch_outputs, coords, H, W, device,
                        patch_size=128, mask_type=mask_type
                    )
                    output_tensors_by_mask[mask_type] = output_tensor
            
            # 把当前 TTA 的三种 mask 结果分别反变换回原始方向，存入各自的列表
            for mask_type, _ in mask_configs:
                output_inv = inverse_t(output_tensors_by_mask[mask_type])
                outputs_inv_by_mask[mask_type].append(output_inv)

        # 对每种 mask，汇聚 8 个 TTA 结果：逐像素平均 + 去极值平均
        final_outputs_by_mask = {}
        final_outputs_by_mask_trimmed = {}
        for mask_type, _ in mask_configs:
            if outputs_inv_by_mask[mask_type]:
                stacked = torch.stack(outputs_inv_by_mask[mask_type], dim=0)  # 8xCxHxW
                final_outputs_by_mask[mask_type] = stacked.mean(dim=0)
                # 去极值平均：去掉每个像素在 8 个 TTA 中的最大、最小值，对剩余 6 个求平均
                sorted_stack, _ = torch.sort(stacked, dim=0)
                trimmed = sorted_stack[1:-1]  # 6xCxHxW
                final_outputs_by_mask_trimmed[mask_type] = trimmed.mean(dim=0)
            else:
                raise RuntimeError(f"No TTA outputs collected for {mask_type} mask")

        # 保存三种 mask 的结果到各自的目录（普通平均 + 去极值平均）
        for mask_type, output_dir in output_dirs.items():
            print(f'\n  === Saving {mask_type} mask result ===')
            final_output_tensor = final_outputs_by_mask[mask_type]
            output_np = util.tensor2uint(final_output_tensor.unsqueeze(0))
            save_img_path = os.path.join(output_dir, '{:s}'.format(image_name_ext))
            print(f'    Mean-of-8 -> {save_img_path}')
            util.imsave(output_np, save_img_path)

            # 去极值平均结果
            trimmed_tensor = final_outputs_by_mask_trimmed[mask_type]
            trimmed_np = util.tensor2uint(trimmed_tensor.unsqueeze(0))
            save_trimmed_path = os.path.join(output_dirs_trimmed[mask_type], '{:s}'.format(image_name_ext))
            print(f'    Trimmed-mean (drop max/min) -> {save_trimmed_path}')
            util.imsave(trimmed_np, save_trimmed_path)
        
        print(f'Completed image {idx+1}/{len(test_dataset)}')


if __name__ == '__main__':
    main()

