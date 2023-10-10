#/bin/bash
phase='train'
local_rank=0
config='config/train_stage1.toml'

# Stage 1
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master-port=12355 train_hb.py \
    --phase=$phase \
    --config=$config

# # Stage 1
# CUDA_VISIBLE_DEVICES=0 python train_hb.py \
#     --phase=$phase \
#     --config=$config


# # Stage 2
# python train.py \
#     --model-variant mobilenetv3 \
#     --dataset videomatte \
#     --resolution-lr 512 \
#     --seq-length-lr 50 \
#     --learning-rate-backbone 0.00005 \
#     --learning-rate-aspp 0.0001 \
#     --learning-rate-decoder 0.0001 \
#     --learning-rate-refiner 0 \
#     --checkpoint checkpoint/stage1/epoch-19.pth \
#     --checkpoint-dir checkpoint/stage2 \
#     --log-dir log/stage2 \
#     --epoch-start 20 \
#     --epoch-end 22