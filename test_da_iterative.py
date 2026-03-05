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
import cv2

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist

from data.select_dataset import define_Dataset
from models.select_model import define_Model
import warnings
warnings.filterwarnings("ignore")


def upsample_lr_to_hr_size(lr_img, scale_factor, target_h=None, target_w=None, interpolation_mode='bicubic'):
    """
    Upsample LR image to HR size using specified interpolation
    
    Args:
        lr_img: numpy array, HxWxC (uint8, 0-255)
        scale_factor: upsampling scale factor
        target_h, target_w: target size (if None, use scale_factor)
        interpolation_mode: 'bicubic' or 'bilinear'
    
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


def main(json_path='options/test_da8.json', num_iterations=1):
    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)
    parser.add_argument('--iterations', type=int, default=num_iterations, 
                       help='Number of iterative inference steps')
    parser.add_argument('--checkpoint', type=str, 
                       default=None,
                       help='Path to checkpoint file (G or E model). If not specified, uses model from JSON config.')
    parser.add_argument('--use_g_model', action='store_true',
                       help='Force using G model (non-EMA) instead of E model for inference')

    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)
    opt['dist'] = args.dist
    num_iterations = args.iterations
    use_g_model = args.use_g_model  # Flag to force using G model
    
    # IMPORTANT: Set E_decay to 0 BEFORE model initialization to prevent netE creation
    if use_g_model:
        if 'train' in opt and 'E_decay' in opt['train']:
            original_e_decay = opt['train']['E_decay']
            opt['train']['E_decay'] = 0
            print(f'\n=== Force Using G Model ===')
            print(f'E_decay changed from {original_e_decay} to 0 BEFORE model initialization')
            print(f'This prevents netE from being created')
            print(f'===========================\n')
    
    # Handle checkpoint loading - support both G and E models
    # Priority: JSON config > command line argument > default
    if opt['path'].get('pretrained_netE') or opt['path'].get('pretrained_netG'):
        # JSON config already has model path, use it
        pretrained_netE = opt['path'].get('pretrained_netE')
        # Special case: if pretrained_netE points to a G model file
        if pretrained_netE and pretrained_netE.endswith('_G.pth'):
            if use_g_model:
                # User wants G model, and pretrained_netE points to G model file
                if not opt['path'].get('pretrained_netG'):
                    opt['path']['pretrained_netG'] = pretrained_netE
                    print(f'pretrained_netE points to G model file, copying to pretrained_netG: {pretrained_netE}')
                opt['path']['pretrained_netE'] = None
                print(f'Using generator model (G) from JSON config: {opt["path"]["pretrained_netG"]}')
            else:
                print(f'Warning: pretrained_netE points to G model file: {pretrained_netE}')
                print(f'This may cause loading errors. Consider using --use_g_model flag.')
        elif pretrained_netE and not use_g_model:
            print(f'Using EMA model (E) from JSON config: {pretrained_netE}')
        if opt['path'].get('pretrained_netG'):
            if use_g_model:
                if not (pretrained_netE and pretrained_netE.endswith('_G.pth')):
                    # Only print if we didn't already handle it above
                    print(f'Using generator model (G) from JSON config: {opt["path"]["pretrained_netG"]}')
                # Clear E model path to force using G model
                opt['path']['pretrained_netE'] = None
            else:
                print(f'Generator model (G) available but will use E model: {opt["path"]["pretrained_netG"]}')
    elif args.checkpoint:
        # Use command line argument if JSON doesn't have it
        checkpoint_path = args.checkpoint
        if not os.path.isabs(checkpoint_path):
            # If relative path, join with models directory
            models_dir = opt['path'].get('models', '')
            checkpoint_path = os.path.join(models_dir, args.checkpoint)
        
        if os.path.exists(checkpoint_path):
            # Determine if it's G or E model
            if checkpoint_path.endswith('_E.pth'):
                if use_g_model:
                    # User wants G model, but provided E model - try to find corresponding G model
                    g_model_path = checkpoint_path.replace('_E.pth', '_G.pth')
                    if os.path.exists(g_model_path):
                        opt['path']['pretrained_netG'] = g_model_path
                        opt['path']['pretrained_netE'] = None
                        print(f'Using generator model (G) from command line: {g_model_path}')
                        print(f'  (Found corresponding G model for E model: {checkpoint_path})')
                    else:
                        print(f'Warning: Requested G model but provided E model, and G model not found: {g_model_path}')
                        opt['path']['pretrained_netE'] = checkpoint_path
                        print(f'Using EMA model (E) from command line: {checkpoint_path}')
                else:
                    opt['path']['pretrained_netE'] = checkpoint_path
                    print(f'Using EMA model (E) from command line: {checkpoint_path}')
            elif checkpoint_path.endswith('_G.pth'):
                opt['path']['pretrained_netG'] = checkpoint_path
                if use_g_model:
                    opt['path']['pretrained_netE'] = None
                    print(f'Using generator model (G) from command line: {checkpoint_path}')
                else:
                    print(f'Using generator model (G) from command line: {checkpoint_path}')
                    print(f'  (Note: Will use E model if available. Use --use_g_model to force G model)')
            else:
                # Try to determine from filename
                if use_g_model:
                    # Try to find G model
                    g_model_path = checkpoint_path.replace('_E.pth', '_G.pth')
                    if os.path.exists(g_model_path):
                        opt['path']['pretrained_netG'] = g_model_path
                        opt['path']['pretrained_netE'] = None
                        print(f'Using generator model (G) from command line: {g_model_path}')
                    else:
                        opt['path']['pretrained_netE'] = checkpoint_path
                        print(f'Using model from command line: {checkpoint_path}')
                else:
                    opt['path']['pretrained_netE'] = checkpoint_path
                    print(f'Using model from command line: {checkpoint_path}')
        else:
            print(f'Error: Checkpoint not found: {checkpoint_path}')
            exit(1)
    else:
        # No model specified, use default (but warn)
        if use_g_model:
            # Try to find G model in the scale folder
            default_g_model = "Model/SR_competition26_da13_scale/Guided SR/models/180000_G.pth"
            if os.path.exists(default_g_model):
                opt['path']['pretrained_netG'] = default_g_model
                opt['path']['pretrained_netE'] = None
                print(f'Warning: Using default G model: {default_g_model}')
                print('Please specify model in JSON config or use --checkpoint option')
            else:
                print(f'Error: Default G model not found: {default_g_model}')
                print('Please specify a checkpoint in JSON config or use --checkpoint option')
                exit(1)
        else:
            default_model = "Model/SR_competition26_da13_scale/Guided SR/models/180000_E.pth"
            if os.path.exists(default_model):
                opt['path']['pretrained_netE'] = default_model
                print(f'Warning: Using default EMA model: {default_model}')
                print('Please specify model in JSON config or use --checkpoint option')
            else:
                print(f'Error: Default model not found: {default_model}')
                print('Please specify a checkpoint in JSON config or use --checkpoint option')
                exit(1)

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()
    
    if opt['rank'] == 0:
        for key, path in opt['path'].items():
            print(path)
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

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
            test_dataset = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.load()  # load the model
    
    # Print device information and model status
    print(f'\n=== Device Information ===')
    print(f'Model device: {model.device}')
    print(f'GPU IDs from config: {opt.get("gpu_ids", "Not specified")}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'Current CUDA device: {torch.cuda.current_device()}')
        print(f'CUDA device name: {torch.cuda.get_device_name(torch.cuda.current_device())}')
    print(f'==========================\n')
    
    # Verify which model will be used for inference
    print(f'\n=== Model Status for Inference ===')
    print(f'E_decay in opt: {opt["train"]["E_decay"]}')
    print(f'Has netE: {hasattr(model, "netE")}')
    if hasattr(model, "netE"):
        # Check if netE will be used (based on netG_forward logic in model_plain.py line 457)
        will_use_netE = (opt["train"]["E_decay"] > 0)
        print(f'netE exists, but will use: {"netE (EMA)" if will_use_netE else "netG (Generator)"}')
    else:
        print(f'netE does not exist (E_decay=0), will use: netG (Generator)')
    print(f'pretrained_netG: {opt["path"].get("pretrained_netG")}')
    print(f'pretrained_netE: {opt["path"].get("pretrained_netE")}')
    print(f'==========================\n')

    '''
    # ----------------------------------------
    # Step--4 (iterative inference)
    # ----------------------------------------
    '''

    print(f'Starting inference on full test set (iterations: {num_iterations})...')
    
    # 只跑测试集前两张图片，方便快速检查
    for idx, test_data in enumerate(test_dataset):
        if idx >= 100:
            print('已处理前两张测试图像，提前结束。')
            break
        print(f'\nProcessing image {idx+1}/{len(test_dataset)}')
        print(f'Guide path: {test_data["Guide_path"][0]}')
        print(f'LR path: {test_data["Lr_path"][0]}')
        
        image_name_ext = os.path.basename(test_data['Guide_path'][0])
        image_name = os.path.splitext(image_name_ext)[0]  # without extension
        save_img_path = os.path.join(opt['path']['images'], '{:s}'.format(image_name_ext))
        
        # Get data from dataset - test dataset already provides Lr and Guide at same size
        # DataLoader adds batch dimension, so [0] removes it: BxCxHxW -> CxHxW
        lr_tensor = test_data['Lr'][0]  # CxHxW, float [0,1]
        guide_tensor = test_data['Guide'][0]  # CxHxW, float [0,1]
        
        print(f'LR tensor shape: {lr_tensor.shape}')
        print(f'Guide tensor shape: {guide_tensor.shape}')
        
        # -------------------------
        # TTA x8：做 8 种变换 + 平均
        # -------------------------
        tta_transforms = get_tta_transforms()
        outputs_inv_list = []

        for tta_idx, (forward_t, inverse_t, t_name) in enumerate(tta_transforms):
            print(f'  TTA {tta_idx + 1}/8: {t_name}')

            # 对 LR 和 Guide 同步做变换
            lr_aug = forward_t(lr_tensor)
            guide_aug = forward_t(guide_tensor)

            # Iterative inference on augmented data（通常 iterations=1）
            current_lr = lr_aug
            for iter_idx in range(num_iterations):
                if num_iterations > 1:
                    print(f'    Iteration {iter_idx+1}/{num_iterations}...')

                iter_data = {
                    'Lr': current_lr.unsqueeze(0) if current_lr.dim() == 3 else current_lr,  # 1xCxHxW
                    'Guide': guide_aug.unsqueeze(0),  # 1xCxHxW
                    'Lr_path': test_data['Lr_path'],
                    'Guide_path': test_data['Guide_path']
                }

                # Feed data and test
                model.feed_data(iter_data, phase='test', need_GT=False)
                model.test()

                # Get output
                visuals = model.current_visuals(need_H=False)
                output_tensor = visuals['Output']  # CxHxW, float [0,1]

                # 下一次迭代继续用当前输出
                current_lr = output_tensor

            # 为了做平均，需要把输出反变换回“原始方向”
            output_tensor_inv = inverse_t(output_tensor)
            outputs_inv_list.append(output_tensor_inv)

            # 同时保存每个 TTA 的结果（保持变换后的方向，便于肉眼对比）
            output_np = util.tensor2uint(output_tensor.unsqueeze(0))  # HxWxC, uint8
            tta_save_path = os.path.join(
                opt['path']['images'],
                f'{image_name}_tta{tta_idx + 1}_{t_name}{os.path.splitext(image_name_ext)[1]}'
            )
            print(f'  Saving TTA result to: {tta_save_path}')
            util.imsave(output_np, tta_save_path)

        # 计算 8 张图的平均（在原始方向上）并保存到独立文件夹
        if outputs_inv_list:
            stacked = torch.stack(outputs_inv_list, dim=0)  # 8xCxHxW
            avg_output_tensor = stacked.mean(dim=0)
            avg_output_np = util.tensor2uint(avg_output_tensor.unsqueeze(0))

            # 独立的平均结果目录：与 images 同级，名为 tta_avg_images
            images_dir = opt['path']['images']
            avg_root = os.path.join(os.path.dirname(images_dir), 'tta_avg_images')
            util.mkdir(avg_root)
            avg_save_path = os.path.join(avg_root, image_name_ext)  # 文件名与原始 vis 一致，方便提交

            print(f'Saving TTA average result to: {avg_save_path}')
            util.imsave(avg_output_np, avg_save_path)

        print(f'Completed image {idx+1}/{len(test_dataset)}')


if __name__ == '__main__':
    main()

