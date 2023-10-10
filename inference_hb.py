"""
python inference.py \
    --variant mobilenetv3 \
    --checkpoint "CHECKPOINT" \
    --device cuda \
    --input-source "input.mp4" \
    --output-type video \
    --output-composition "composition.mp4" \
    --output-alpha "alpha.mp4" \
    --output-foreground "foreground.mp4" \
    --output-video-mbps 4 \
    --seq-chunk 1
"""

import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Optional, Tuple
from tqdm.auto import tqdm

from inference_utils import VideoReader, VideoWriter, ImageSequenceReader, ImageSequenceWriter
import cv2
import numpy as np

# Get face mask
import sys
sys.path.append('/home/jhb/base/DIFAI')
from difai import DIFAI
from def_argu import define_argus

sys.path.append('/home/jhb/base/packages')
from face_aligner import FaceAligner
FA = FaceAligner(lmk_type='3D', size=512)

def convert_video(model,
                  input_source: str,
                  input_resize: Optional[Tuple[int, int]] = None,
                  downsample_ratio: Optional[float] = None,
                  output_type: str = 'video',
                  output_composition: Optional[str] = None,
                  output_alpha: Optional[str] = None,
                  output_foreground: Optional[str] = None,
                  output_source: Optional[str] = None,
                  output_video_mbps: Optional[float] = None,
                  seq_chunk: int = 1,
                  num_workers: int = 0,
                  progress: bool = True,
                  device: Optional[str] = None,
                  dtype: Optional[torch.dtype] = None,
                  inpaint: bool = False,
                  nofg: bool = False):
    
    """
    Args:
        input_source:A video file, or an image sequence directory. Images must be sorted in accending order, support png and jpg.
        input_resize: If provided, the input are first resized to (w, h).
        downsample_ratio: The model's downsample_ratio hyperparameter. If not provided, model automatically set one.
        output_type: Options: ["video", "png_sequence"].
        output_composition:
            The composition output path. File path if output_type == 'video'. Directory path if output_type == 'png_sequence'.
            If output_type == 'video', the composition has green screen background.
            If output_type == 'png_sequence'. the composition is RGBA png images.
        output_alpha: The alpha output from the model.
        output_foreground: The foreground output from the model.
        seq_chunk: Number of frames to process at once. Increase it for better parallelism.
        num_workers: PyTorch's DataLoader workers. Only use >0 for image input.
        progress: Show progress bar.
        device: Only need to manually provide if model is a TorchScript freezed model.
        dtype: Only need to manually provide if model is a TorchScript freezed model.
    """
    
    assert downsample_ratio is None or (downsample_ratio > 0 and downsample_ratio <= 1), 'Downsample ratio must be between 0 (exclusive) and 1 (inclusive).'
    assert any([output_composition, output_alpha, output_foreground, output_source]), 'Must provide at least one output.'
    assert output_type in ['video', 'png_sequence'], 'Only support "video" and "png_sequence" output modes.'
    assert seq_chunk >= 1, 'Sequence chunk must be >= 1'
    assert num_workers >= 0, 'Number of workers must be >= 0'
    
    # Initialize transform
    if input_resize is not None:
        transform = transforms.Compose([
            transforms.Resize(input_resize[::-1]),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()

    # Initialize reader
    if os.path.isfile(input_source):
        source = VideoReader(input_source, transform)
    else:
        source = ImageSequenceReader(input_source, transform)
    reader = DataLoader(source, batch_size=seq_chunk, pin_memory=True, num_workers=num_workers)
    
    # Initialize writers
    if output_type == 'video':
        frame_rate = source.frame_rate if isinstance(source, VideoReader) else 30
        output_video_mbps = 1 if output_video_mbps is None else output_video_mbps
        if output_composition is not None:
            writer_com = VideoWriter(
                path=output_composition,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
        if output_alpha is not None:
            writer_pha = VideoWriter(
                path=output_alpha,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
        if output_foreground is not None:
            writer_fgr = VideoWriter(
                path=output_foreground,
                frame_rate=frame_rate,
                bit_rate=int(output_video_mbps * 1000000))
        if output_source is not None:
            # writer_hb = VideoWriter(
            #     path=output_source,
            #     frame_rate=frame_rate,
            #     bit_rate=int(output_video_mbps * 1000000))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = frame_rate
            frame_width, frame_height = source.video.frame_shape[1], source.video.frame_shape[0]*2
            writer_hb = cv2.VideoWriter(output_source, fourcc, fps, (frame_width, frame_height))


    else:
        if output_composition is not None:
            writer_com = ImageSequenceWriter(output_composition, 'png')
        if output_alpha is not None:
            writer_pha = ImageSequenceWriter(output_source, 'png')
        if output_foreground is not None:
            writer_fgr = ImageSequenceWriter(output_foreground, 'png')

    # Inference
    model = model.eval()
    if device is None or dtype is None:
        param = next(model.parameters())
        dtype = param.dtype
        device = param.device
    
    if (output_composition is not None) and (output_type == 'video'):
        bgr = torch.tensor([120, 255, 155], device=device, dtype=dtype).div(255).view(1, 1, 3, 1, 1)
    
    try:
        with torch.no_grad():
            bar = tqdm(total=len(source), disable=not progress, dynamic_ncols=True)
            rec = [None] * 4
            for idx, src in enumerate(reader):

                if downsample_ratio is None:
                    downsample_ratio = 1
                    # downsample_ratio = auto_downsample_ratio(*src.shape[2:])
                if output_type == 'video': # for EF0

                    ''' crop and align the face '''
                    # crop_frame = frame[pt1[1]:pt2[1], pt1[0]:pt2[0]] #@# alignment
                    frame = src.clone()
                    tmp = np.array(src[0]*255)
                    tmp = tmp.transpose((1,2,0))
                    FA_dict = FA.get_face(tmp) #@# 얼굴 영역을 찾아서 가져옵니다
                    tmp = FA_dict['aligned_face'] #@# 얼굴 주변 영역을 1024x1024 크기로 잘라옵니다. 
                    tfm_inv = FA_dict['tfm_inv'] #@# 잘라온 얼굴을 원래 얼굴 위치로 되돌려주는 transform matrix 입니다.
                    # print(tfm_inv)

                    tmp = tmp.transpose(2,0,1)
                    src = torch.Tensor(tmp/255.0)

                src = src.to(device, dtype, non_blocking=True).unsqueeze(0) # [B, T, C, H, W]
                if nofg:
                    pha, *rec = model(src, *rec, downsample_ratio)
                else:
                    fgr, pha, *rec = model(src, *rec, downsample_ratio)

                if output_foreground is not None:
                    writer_fgr.write(fgr[0])
                elif output_alpha is not None: # for test-benchmark-02
                    writer_pha.write(pha[0], source.get_name(idx))
                elif output_composition is not None:
                    if output_type == 'recvideo':
                        com = fgr * pha + bgr * (1 - pha)
                    else:
                        fgr = fgr * pha.gt(0)
                        com = torch.cat([fgr, pha], dim=-3)
                    writer_com.write(com[0])
                elif output_source is not None:
                    # 1) src, 2) fgr, 3) src*pha, 4) inpaint
                    if output_type == 'video' and not inpaint:
                        pha = np.array(pha[0].detach().cpu())
                        pha = pha.transpose((1,2,0))
                        pha = cv2.warpAffine(pha, tfm_inv, (frame.shape[3], frame.shape[2]))
                        pha[pha<=0] = 0
                        pha[pha>=1] = 1
                        pha = torch.Tensor(np.array([[pha]]))

                        
                        com = frame * pha * 255
                        pha = pha.repeat((1,3,1,1)) * 255
                        com = torch.cat([pha, com], dim=-2)
                    elif output_type == 'video' and inpaint:
                        com = src * pha
                        inpaint = fgr # ??
                        com = torch.cat([src, fgr, com, inpaint], dim=-2)
                    else: # if image & com
                        if nofg:
                            fgr = src * pha.gt(0)
                        else:
                            fgr = fgr * pha.gt(0)
                        com = torch.cat([fgr, pha], dim=-3)
                    com = np.array(com).astype('uint8')
                    com = com[0].transpose((1,2,0))
                    # writer_hb.write(com)
                    writer_hb.write(com[:,:,::-1])
                bar.update(src.size(1))


    finally:
        # Clean up
        if output_composition is not None:
            writer_com.close()
        if output_alpha is not None:
            writer_pha.close()
        if output_foreground is not None:
            writer_fgr.close()
        if output_source is not None and output_type == 'video':
            writer_hb.release()


def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)


class Converter:
    def __init__(self, variant: str, checkpoint: str, device: str):
        self.model = MattingNetwork(variant).eval().to(device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
        self.model = torch.jit.script(self.model)
        self.model = torch.jit.freeze(self.model)
        self.device = device
    
    def convert(self, *args, **kwargs):
        convert_video(self.model, device=self.device, dtype=torch.float32, *args, **kwargs)
    
if __name__ == '__main__':
    import argparse
    from model import MattingNetwork
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, default='mobilenetv3', choices=['mobilenetv3', 'resnet50'])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--input-folder', type=str, required=True)
    parser.add_argument('--input-resize', type=int, default=None, nargs=2)
    parser.add_argument('--downsample-ratio', type=float)
    parser.add_argument('--output-composition', type=str, default=None)
    parser.add_argument('--output-alpha', type=str, default=None)
    parser.add_argument('--output-foreground', type=str, default=None)
    parser.add_argument('--output-type', type=str, required=True, choices=['video', 'png_sequence'])
    parser.add_argument('--output-video-mbps', type=int, default=1)
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--seq-chunk', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--disable-progress', action='store_true')
    parser.add_argument('--inpaint', action='store_true')
    parser.add_argument('--nofg', action='store_true')
    args = parser.parse_args()
    
    converter = Converter(args.variant, args.checkpoint, args.device)
    # get the input_source from args.input_folder
    # output will be include 1) origin 2) estimated foreground = skin 3) origin * estimated matte 4) inpaint

    if args.output_type == 'png_sequence':

        # args.output_alpha should be True
        args.output_alpha = "True" if args.output_alpha == None else args.output_alpha

        # hiu, rand, sim
        all_folders = [item for item in os.listdir(args.input_folder) if not os.path.isfile(os.path.join(args.input_folder, item))]

        for folder in all_folders:

            folder_dir = os.path.join(args.input_folder, folder)
            img_dir = os.path.join(folder_dir, 'img')

            output_fold = os.path.join(args.output_folder, folder)
            os.makedirs(output_fold, exist_ok=True)

            converter.convert(
                input_source=img_dir,
                input_resize=args.input_resize,
                downsample_ratio=args.downsample_ratio,
                output_type=args.output_type,
                output_composition=args.output_composition,
                output_alpha=args.output_alpha,
                output_foreground=args.output_foreground,
                output_video_mbps=args.output_video_mbps,
                output_source=output_fold,
                seq_chunk=args.seq_chunk,
                num_workers=args.num_workers,
                progress=not args.disable_progress,
                inpaint=args.inpaint,
                nofg=args.nofg
            )

    else:    
        os.makedirs(args.output_folder, exist_ok=True)

        # EF0
        for input_source in tqdm(os.listdir(args.input_folder)):
            if input_source.endswith('.mov'): #  and '22' in video:
                input_path = os.path.join(args.input_folder, input_source)
                output_path = os.path.join(args.output_folder, input_source)
                converter.convert(
                    input_source=input_path,
                    input_resize=args.input_resize,
                    downsample_ratio=args.downsample_ratio,
                    output_type=args.output_type,
                    output_composition=args.output_composition,
                    output_alpha=args.output_alpha,
                    output_foreground=args.output_foreground,
                    output_video_mbps=args.output_video_mbps,
                    output_source=output_path,
                    seq_chunk=args.seq_chunk,
                    num_workers=args.num_workers,
                    progress=not args.disable_progress,
                    inpaint=args.inpaint,
                    nofg=args.nofg
                )

                    
        
