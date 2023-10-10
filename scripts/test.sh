#/bin/bash
phase='train'
config='config/train_stage1.toml'

# # Video
# root=./logs/stage1/230928_004215/
# python inference_hb_video.py \
#     --checkpoint $root"checkpoints/epoch-19.pth" \
#     --output-type "video" \
#     --input-folder "/home/jhb/dataset/FM_dataset/EF0" \
#     --output-folder $root"/test_images" 
#     # --inpaint= 

# # Image
# root=./logs/stage1/230928_004215/
# python inference_hb_img.py \
#     --checkpoint $root"checkpoints/epoch-19.pth" \
#     --input-folder "/home/jhb/dataset/FM_dataset/test_benchmark_02" \
#     --output-type "png_sequence" \
#     --output-folder $root"/test_images" \
#     --output-alpha "True" 
#     # --inpaint


# # Image
# root=./logs/stage1/231008_232647/
# python inference_hb.py \
#     --checkpoint $root"checkpoints/epoch-9.pth" \
#     --input-folder "/home/jhb/dataset/FM_dataset/test_benchmark_02" \
#     --input-resize 512 512 \
#     --output-type "png_sequence" \
#     --output-folder $root"/test_images" \
#     --output-alpha "True" \
#     --nofg 
#     # --inpaint


# Video
root=./logs/stage1/231008_232647/
python inference_hb.py \
    --checkpoint $root"checkpoints/epoch-9.pth" \
    --input-folder "/home/jhb/dataset/FM_dataset/EF0" \
    --output-type "video" \
    --output-folder $root"/test_images_pha9/EF0" \
    --nofg 
    # --input-resize 512 512 \
    # --inpaint