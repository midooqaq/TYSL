# TYSL
code for PBVS@CVPR 2026Thermal Images Super-resolution challenge-Track2
# Set up 
```
conda create -n tysl python=3.10
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install mmcv-full==1.7.2 --no-cache-dir --no-build-isolation

```
You may have encountered issues installing mmcv, or noticed the following warnings during training:
```
Warning: mmcv not available, using fallback for multi_scale_deformable_attn
Warning: multi_scale_deform_attn not available
```
This is because the PyTorch version used in this project is relatively new, so you will need to compile the wheel yourself. Please uninstall mmcv-full, and then install it following the method provided in this project.Additionally, if you are running this within a conda virtual environment, please ensure that CUDA 12.6 is installed on your host machine. This is because mmcv-full relies on the host's CUDA toolkit to compile during the installation process.

```
```
# Testing script
If you need to test images, run the script:
```
./run_test_da_x8.sh
./run_test_da_x16.sh
```
If you need to train images, run the script:
```
./run_train_da_x8.sh
./run_train_da_x16.sh
```

# Acknowledgement
Most of the code is based on the work of [SwinPaste](https://github.com/zoniazhong/SwinPaste).thanks to the team for their inspiration!
