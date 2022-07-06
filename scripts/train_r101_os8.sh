#!/usr/bin/env bash
# Example on Cityscapes
# conda activate sml
export CUDA_VISIBLE_DEVICES=1
python -m torch.distributed.launch --nproc_per_node=1 train.py \
   --dataset selfgen \
   --val_dataset selfgen \
   --arch network.deepv3.DeepR101V3PlusD_OS8 \
   --city_mode train \
   --lr_schedule poly \
   --lr 0.01 \
   --poly_exp 0.9 \
   --max_cu_epoch 10000 \
   --class_uniform_pct 0.5 \
   --class_uniform_tile 720 \
   --crop_size 540 \
   --scale_min 0.5 \
   --scale_max 2.0 \
   --rrotate 0 \
   --max_iter 60000 \
   --bs_mult 2 \
   --gblur \
   --color_aug 0.5 \
   --date 0705 \
   --exp r101_os8_base_60K \
   --ckpt c_validation/ckpt/  \
   --tb_path c_validation/tb \
   --snapshot /DATA2/gaoha/liumd/sml/sml/c_selfgen/ckpt/0705/r101_os8_base_60K/07_05_15/last_None_epoch_168_mean-iu_0.00000.pth
