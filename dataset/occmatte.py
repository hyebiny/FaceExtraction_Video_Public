import os
import random
from torch.utils.data import Dataset
from PIL import Image
import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F

from .augmentation import MotionAugmentation
from config import CONFIG
import time
import imutils
import math

interp_list = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4]

def maybe_random_interp(cv2_interp, random=True):
    if random:
        return np.random.choice(interp_list)
    else:
        return cv2_interp

def lerp(a, b, percentage):
    return a * (1 - percentage) + b * percentage

class OccMatteDataset(Dataset):
    def __init__(self,
                 rand_dir,
                 occmatte_dir,
                 occmatte_list,
                 bg_txt,
                 # background_video_dir,
                 size,
                 seq_length,
                 seq_sampler,
                 transform,
                 phase,
                 occ_ratio = 0.25,
                 occ_repeat = 1,
                 bg_image=True):
        
        self.rand_dir = rand_dir
        self.occmatte_dir = occmatte_dir
        self.occmatte_list = occmatte_list # os.listdir(os.path.join(occmatte_dir, 'fgr'))

        if phase == 'train':
            self.bg_txt = bg_txt+'train.txt'
        else:
            self.bg_txt = bg_txt+'test.txt'
        
        self.seq_length = seq_length
        self.seq_sampler = seq_sampler # 얼마나 빨리 Frame을 진행시킬 것인가?

        self.size = size
        self.transform = transform

        self.phase = phase # train or val
        self.occ_size_ratio = (0.3, 0.7)
        self.occ_translate = None # 'brown' 'line' 
        self.occ_ratio = occ_ratio
        self.occ_repeat = occ_repeat
        self.bg_image = bg_image

        self.occmatte_imgs = []      
        init_img_list, init_mask_list = self._get_img_pair_list(self.bg_txt)
        init_occ_list, init_occ_mask_list, init_rand_list = self._get_occ_pair_list(self.occmatte_dir, self.occmatte_list, self.rand_dir, phase)
        
        # sampling by face
        occ_num = len(init_occ_list)
        self.img_list, self.mask_list, self.occ_list, self.occ_mask_list, self.rand_list = [], [], [], [], []
        for i in range(len(init_img_list)):
            # sample the face imgs per occ
            img_index = np.random.choice(occ_num, occ_repeat)
            for j in range(occ_repeat):
                self.img_list.append(init_img_list[i])
                self.mask_list.append(init_mask_list[i])
                self.occ_list.append(init_occ_list[img_index[j]])
                self.occ_mask_list.append(init_occ_mask_list[img_index[j]])
                self.rand_list.append(init_rand_list[img_index[j]])

        self.img_num = len(self.img_list)
        
    def __len__(self):
        return len(self.img_list) * self.seq_length
        # return max(len(self.imagematte_files), len(self.background_image_files) + len(self.background_video_clips))
    
    def __getitem__(self, idx):
        # if CONFIG.gpu == 0:
        #     past = time.time()
        if self.bg_image:
            bgrs, bgrs_mask = self._get_random_image_background(idx) # face
        else:
            bgrs, bgrs_mask = self._get_random_video_background(idx) # face
        
        fgrs, phas = self._get_imagematte(idx)                # occluder
        # if CONFIG.gpu == 0:
        #     print("load", time.time()-past)
        #     past = time.time()
        
        if self.transform is not None:
            fgrs, phas, bgrs, bgrs_phas = self.transform(fgrs, phas, bgrs, bgrs_mask)
            # if CONFIG.gpu == 0:
            #     print("transform", time.time()-past)

        return fgrs, phas, bgrs, bgrs_phas
            
        # return fgrs, phas, bgrs
    

    
    def _get_img_pair_list(self, bg_txt):

        img_list, mask_list = [], []
        img_path = '/home/jhb/dataset/source/face/CelebAMask-HQ/CelebA-HQ-img-512'
        mask_path = '/home/jhb/dataset/source/face/CelebAMask-HQ-masks_corrected-512'

        with open(bg_txt, 'r') as f:
            for line in f:
                file = line.strip()
                name = file.split('.')[0]
                img_list.append(os.path.join(img_path, name+'.jpg'))
                mask_list.append(os.path.join(mask_path, name+'.png'))

        return img_list, mask_list
    

    def _get_occ_pair_list(self, fg_dir, folder_list, rand_dir, phase):

        img_list, mask_list, rand_list = [], [], []

        for folder in folder_list:
            path = os.path.join(fg_dir, folder)
            if folder == 'rand':
                rdir = os.path.join(rand_dir + phase, 'rand')
                img_path = os.path.join(rdir, 'occlusion_img')
                mask_path = os.path.join(rdir, 'occlusion_mask')

                rand_sample = np.random.choice(os.listdir(img_path), 200, False)
                count = 0
                for file in rand_sample:
                    name = file.split('.')[0]
                    img_list.append(os.path.join(img_path, name+'.png'))
                    mask_list.append(os.path.join(mask_path, name+'.png'))
                    rand_list.append(True)
                    count += 1
                print(phase, folder, ':', count)
            elif folder == '11k':
                img_path = os.path.join(path, 'Hands')
                mask_path = os.path.join(path, '11k-hands_masks')
                for file in os.listdir(mask_path):
                    name = file.split('.')[0]
                    img_list.append(os.path.join(img_path, name+'.jpg'))
                    mask_list.append(os.path.join(mask_path, name+'.png'))
                    rand_list.append(False)
                    count += 1
                print(phase, folder, ':', count)
            elif "sim" in folder:
                path = os.path.join(path, phase)
                count = 0
                for fo in os.listdir(path):
                    path_folder = os.path.join(path, fo)
                    if not os.path.isdir(path_folder):
                        continue
                    if fo in CONFIG.data.sim_list1: 
                        print(fo, CONFIG.data.sim_list1)
                        continue
                    img_path = os.path.join(path_folder, 'fg')
                    mask_path = os.path.join(path_folder, 'alpha')
                    for file in os.listdir(img_path):
                        if file.endswith(('jpg', 'png')):
                            name = file.split('.')[0]
                            img_list.append(os.path.join(img_path, name+'.jpg'))
                            mask_list.append(os.path.join(mask_path, name+'.jpg'))
                            rand_list.append(True if fo in CONFIG.data.sim_list2 else False)
                            count += 1
                print(phase, folder, ':', count)

            elif folder == 'hiu':
                txt_file = os.path.join(path, 'hiu_'+phase+'.txt')
                file_list = []
                with open(txt_file, 'r') as file:
                    for line in file.readlines():
                        file_list.append(line.strip())
                img_path = os.path.join(path, 'fg')
                mask_path = os.path.join(path, 'alpha')

                count = 0
                # randomly select 200 for train, 100 for test like 11k
                train_num = np.random.choice(len(file_list), 200, False)
                for i in train_num: # range(len(file_list)): # train_num:
                    img_list.append(os.path.join(img_path, file_list[i]+'.jpg'))
                    mask_list.append(os.path.join(mask_path, file_list[i]+'_mask.png'))
                    rand_list.append(False)
                    count += 1
                print(phase, folder, ':', count)

            elif folder == 'am2k':
                img_path = os.path.join(path, phase+'/fg')
                mask_path = os.path.join(path, phase+'/mask')

                rand_sample = os.listdir(img_path)
                if phase == 'train':
                    rand_sample = np.random.choice(rand_sample, 1000, False)
                count = 0
                for file in rand_sample:
                    if file.endswith(('jpg', 'png')):
                        if phase == 'test':
                            name = file.split('.')[0]
                            img_list.append(os.path.join(img_path, name+'.jpg'))
                            mask_list.append(os.path.join(mask_path, name+'.png'))
                        else:
                            img_list.append(os.path.join(img_path, file))
                            mask_list.append(os.path.join(mask_path, file))
                        rand_list.append(False)
                        count +=1 
                print(phase, "AM2K : ", count)
            elif folder == 'test':
                # test_benchmark_02
                root_path = '/home/jhb/dataset/FM_dataset/test_benchmark_02'
                folders = os.listdir(root_path)
                count = 0
                for folder in folders:
                    path = os.path.join(root_path, folder)

                    img_path = os.path.join(path,'img')
                    mask_path = os.path.join(path, 'mask')
                    for img in os.listdir(img_path):
                        if img.endswith(('jpg', 'png')):
                            name = img.split('.')[0]
                            img_list.append(os.path.join(img_path, name+'.jpg'))
                            mask_list.append(os.path.join(mask_path, name+'.png'))
                            rand_list.append(False)
                            count +=1 
                print(phase, "TEST : ", count)

            else:
                raise ValueError(" -- Folder ERROR --  : ", folder)
        
        return img_list, mask_list, rand_list

    
    def _get_imagematte(self, idx):

        # with cv2.imread(os.path.join(self.imagematte_dir, 'fgr', self.imagematte_files[idx % len(self.imagematte_files)])) as fgr, \
        #      cv2.imread(os.path.join(self.imagematte_dir, 'pha', self.imagematte_files[idx % len(self.imagematte_files)])) as pha:

        occ = cv2.imread(self.occ_list[idx % self.img_num])
        occ_mask = cv2.imread(self.occ_mask_list[idx % self.img_num], 0) # .astype(np.float32)/255
        
        # crop the occlusion
        occluder_rect = cv2.boundingRect(occ_mask)
        crop_occ = occ[occluder_rect[1]:(occluder_rect[1]+occluder_rect[3]),occluder_rect[0]:(occluder_rect[0]+occluder_rect[2])]
        crop_occ_mask = occ_mask[occluder_rect[1]:(occluder_rect[1]+occluder_rect[3]),occluder_rect[0]:(occluder_rect[0]+occluder_rect[2])] 
    
        # fgr = self._downsample_if_needed(fgr.convert('RGB'))
        # pha = self._downsample_if_needed(pha.convert('L'))

        # expand the border line
        h, w = crop_occ_mask.shape
        scale_factor = (((self.size*self.size))/(h*w))*np.random.uniform(self.occ_size_ratio[0], self.occ_size_ratio[1])
        scale_factor=np.sqrt(scale_factor)
        new_size = tuple(np.round(np.array([w, h]) * scale_factor).astype(int))
        crop_occ = cv2.resize(crop_occ, new_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
        crop_occ_mask = cv2.resize(crop_occ_mask, new_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
        h, w = crop_occ_mask.shape
        # if h <= self.size and w <= self.size:
        #     break
        
        h, w = crop_occ_mask.shape
        empty = np.zeros((self.size, self.size, 3), dtype=np.float32)
        empty_mask = np.zeros((self.size, self.size), dtype=np.float32)
        if self.occ_translate is None:
            if h > self.size or w > self.size:
                # random crop to make self.size
                if h > self.size:
                    diff = random.randint(0, h - self.size)
                    crop_occ = crop_occ[diff:diff+self.size, :, :] 
                    crop_occ_mask = crop_occ_mask[diff:diff+self.size, :] 
                    spoint = (0, random.randint(0, self.size-w))
                if w > self.size:
                    diff = random.randint(0, w - self.size)
                    crop_occ = crop_occ[:, diff:diff+self.size,:] 
                    crop_occ_mask = crop_occ_mask[:, diff:diff+self.size] 
                    spoint = (random.randint(0, self.size-h), 0)
                h, w = crop_occ_mask.shape
            else:
                spoint = (random.randint(0, self.size-h), random.randint(0, self.size-w))

            try:
                empty[spoint[0]:spoint[0]+h, spoint[1]:spoint[1]+w, :] = crop_occ
                empty_mask[spoint[0]:spoint[0]+h, spoint[1]:spoint[1]+w] = crop_occ_mask
            except:
                print(h,w, crop_occ.shape)
            
        # convert to PIL shape
        empty = empty.astype(np.uint8)
        crop_occ = cv2.cvtColor(empty, cv2.COLOR_BGR2RGB)
        crop_occ = Image.fromarray(crop_occ)
        crop_occ_mask = Image.fromarray(empty_mask)
        crop_occ_mask = crop_occ_mask.convert("L")

        fgrs = [crop_occ] * self.seq_length
        phas = [crop_occ_mask] * self.seq_length

        return fgrs, phas
    
    
    def _get_random_image_background(self, idx):
        # with Image.open(os.path.join(self.background_image_dir, self.background_image_files[random.choice(range(len(self.background_image_files)))])) as bgr:
        #     bgr = self._downsample_if_needed(bgr.convert('RGB'))

        # self.img_list, self.mask_list, self.occ_list, self.occ_mask_list, self.rand_list 
        
        bgr = Image.open(self.img_list[idx % self.img_num])
        pha =  Image.open(self.mask_list[idx % self.img_num])
        pha = pha.convert("L")

        # img = cv2.imread(self.img_list[idx % self.img_num])
        # mask = cv2.imread(self.mask_list[idx % self.img_num], 0) # .astype(np.float32)/255
        
        # # convert to PIL shape
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(img)
        # mask = Image.fromarray(mask)

        bgrs = [bgr] * self.seq_length
        bgrs_mask = [pha] * self.seq_length

        return bgrs, bgrs_mask

    def _get_random_video_background(self, idx):
        # not implemented : hyebin
        clip_idx = random.choice(range(len(self.background_video_clips)))
        frame_count = len(self.background_video_frames[clip_idx])
        frame_idx = random.choice(range(max(1, frame_count - self.seq_length)))
        clip = self.background_video_clips[clip_idx]
        bgrs = []
        for i in self.seq_sampler(self.seq_length):
            frame_idx_t = frame_idx + i
            frame = self.background_video_frames[clip_idx][frame_idx_t % frame_count]
            with Image.open(os.path.join(self.background_video_dir, clip, frame)) as bgr:
                bgr = self._downsample_if_needed(bgr.convert('RGB'))
            bgrs.append(bgr)
        return bgrs
    
    def _downsample_if_needed(self, img):
        w, h = img.size
        if min(w, h) > self.size:
            scale = self.size / min(w, h)
            w = int(scale * w)
            h = int(scale * h)
            img = img.resize((w, h))
        return img

class OccMatteAugmentation(MotionAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=0.95,
            prob_bgr_affine=0.3,
            prob_noise=0.05,
            prob_color_jitter=0.3,
            prob_grayscale=0.03,
            prob_sharpness=0.05,
            prob_blur=0.02,
            prob_hflip=0.5,
            prob_pause=0.03,
        )

    def __call__(self, fgrs, phas, bgrs, bgrs_phas): # occ, occ_mask, face, skin_mask

        '''

        ***  steps  ***

        * crop occlusion 
        * extend the border line -> done when img load
        * motion affine fg, bg
        *--- convert to tensor ---* 
        * resize fg, bg
        * flip fg, bg
        * motion color jitter fg, bg
        * pause fg, bg
        *--- right before train ---*
        * composite gt alpha matte

        '''


        if CONFIG.gpu == 0:
            past = time.time()
        
        # Foreground affine > occlusio = hand
        if random.random() < self.prob_fgr_affine:
            fgrs, phas = self._motion_affine(fgrs, phas) #, varT=15)

        # Background affine  > face
        if random.random() < self.prob_bgr_affine / 2:
            bgrs, bgrs_phas = self._motion_affine(bgrs, bgrs_phas) #, varT=15)

        # both
        if random.random() < self.prob_bgr_affine / 2:
            fgrs, phas, bgrs, bgrs_phas = self._motion_affine(fgrs, phas, bgrs, bgrs_phas) #, varT=15)
                
        # Still Affine
        if self.static_affine:
            fgrs, phas = self._static_affine(fgrs, phas, scale_ranges=(0.5, 1))
            bgrs, bgrs_phas = self._static_affine(bgrs, bgrs_phas, scale_ranges=(1, 1.5))

        # if CONFIG.gpu == 0:
        #     print("Affine", time.time() - past)
        #     past = time.time()
        

        fgrs = [fgr.resize((512, 512)) for fgr in fgrs]
        phas = [pha.resize((512, 512)) for pha in phas]
        bgrs = [bgr.resize((512, 512)) for bgr in bgrs]
        bgrs_phas = [bgr_pha.resize((512, 512)) for bgr_pha in bgrs_phas]


        # To tensor
        fgrs = torch.stack([F.to_tensor(fgr) for fgr in fgrs])
        phas = torch.stack([F.to_tensor(pha) for pha in phas])
        bgrs = torch.stack([F.to_tensor(bgr) for bgr in bgrs])
        bgrs_phas = torch.stack([F.to_tensor(bgr_pha) for bgr_pha in bgrs_phas])

        # return fgrs, phas, bgrs, bgrs_phas

        
        # Resize & Crop 
        params = transforms.RandomResizedCrop.get_params(fgrs, scale=(1, 1), ratio=self.aspect_ratio_range)
        fgrs = F.resized_crop(fgrs, *params, self.size, interpolation=F.InterpolationMode.BILINEAR, antialias=True)
        phas = F.resized_crop(phas, *params, self.size, interpolation=F.InterpolationMode.BILINEAR, antialias=True)
        params = transforms.RandomResizedCrop.get_params(bgrs, scale=(1, 1), ratio=self.aspect_ratio_range)
        bgrs = F.resized_crop(bgrs, *params, self.size, interpolation=F.InterpolationMode.BILINEAR, antialias=True)
        bgrs_phas = F.resized_crop(bgrs_phas, *params, self.size, interpolation=F.InterpolationMode.BILINEAR, antialias=True)


        # Horizontal flip
        if random.random() < self.prob_hflip:
            fgrs = F.hflip(fgrs)
            phas = F.hflip(phas)
        if random.random() < self.prob_hflip:
            bgrs = F.hflip(bgrs)
            bgrs_phas = F.hflip(bgrs_phas)

        # Color jitter
        # 기존 _motion_color_jitter은 tensor로 변환 후 진행된다.
        if random.random() < self.prob_color_jitter:
            fgrs = self._motion_color_jitter(fgrs, varT=15)
        if random.random() < self.prob_color_jitter:
            bgrs = self._motion_color_jitter(bgrs, varT=15)

        # # Grayscale
        # if random.random() < self.prob_grayscale:
        #     fgrs = F.rgb_to_grayscale(fgrs, num_output_channels=3).contiguous()
        #     bgrs = F.rgb_to_grayscale(bgrs, num_output_channels=3).contiguous()
            
        # # Sharpen
        # if random.random() < self.prob_sharpness:
        #     sharpness = random.random() * 8
        #     fgrs = F.adjust_sharpness(fgrs, sharpness)
        #     phas = F.adjust_sharpness(phas, sharpness)
        #     bgrs = F.adjust_sharpness(bgrs, sharpness)
        
        # # Blur
        # if random.random() < self.prob_blur / 3:
        #     fgrs, phas = self._motion_blur(fgrs, phas)
        # if random.random() < self.prob_blur / 3:
        #     bgrs = self._motion_blur(bgrs)
        # if random.random() < self.prob_blur / 3:
        #     fgrs, phas, bgrs = self._motion_blur(fgrs, phas, bgrs)

        # Pause
        if random.random() < self.prob_pause:
            fgrs, phas, bgrs, bgrs_phas = self._motion_pause(fgrs, phas, bgrs, bgrs_phas)

        return fgrs, phas, bgrs, bgrs_phas


class OccMatteDataset2(Dataset):
    def __init__(self,
                 rand_dir,
                 occmatte_dir,
                 occmatte_list,
                 bg_txt,
                 # background_video_dir,
                 size,
                 seq_length,
                 seq_sampler,
                 transform,
                 phase,
                 occ_ratio = 0.25,
                 occ_repeat = 1,
                 bg_image=True):
        
        self.rand_dir = rand_dir
        self.occmatte_dir = occmatte_dir
        self.occmatte_list = occmatte_list # os.listdir(os.path.join(occmatte_dir, 'fgr'))

        if phase == 'train':
            self.bg_txt = bg_txt+'train.txt'
        else:
            self.bg_txt = bg_txt+'test.txt'
        
        self.seq_length = seq_length
        self.seq_sampler = seq_sampler # 얼마나 빨리 Frame을 진행시킬 것인가?

        self.size = size
        self.transform = transform

        self.phase = phase # train or val
        self.occ_size_ratio = (0.3, 0.7)
        self.occ_translate = None # 'brown' 'line' 
        self.occ_ratio = occ_ratio
        self.occ_repeat = occ_repeat
        self.bg_image = bg_image

        self.occmatte_imgs = []      
        init_img_list, init_mask_list = self._get_img_pair_list(self.bg_txt)
        init_occ_list, init_occ_mask_list, init_rand_list = self._get_occ_pair_list(self.occmatte_dir, self.occmatte_list, self.rand_dir, phase)
        
        # sampling by face
        occ_num = len(init_occ_list)
        self.img_list, self.mask_list, self.occ_list, self.occ_mask_list, self.rand_list = [], [], [], [], []
        for i in range(len(init_img_list)):
            # sample the face imgs per occ
            img_index = np.random.choice(occ_num, occ_repeat)
            for j in range(occ_repeat):
                self.img_list.append(init_img_list[i])
                self.mask_list.append(init_mask_list[i])
                self.occ_list.append(init_occ_list[img_index[j]])
                self.occ_mask_list.append(init_occ_mask_list[img_index[j]])
                self.rand_list.append(init_rand_list[img_index[j]])

        self.img_num = len(self.img_list)
        
    def __len__(self):
        return len(self.img_list) * self.seq_length
        # return max(len(self.imagematte_files), len(self.background_image_files) + len(self.background_video_clips))
    
    def __getitem__(self, idx):
        # if CONFIG.gpu == 0:
        #     past = time.time()
        if self.bg_image:
            bgrs, bgrs_mask = self._get_random_image_background(idx) # face
        else:
            bgrs, bgrs_mask = self._get_random_video_background(idx) # face
        
        fgrs, phas = self._get_imagematte(idx)                # occluder
        # if CONFIG.gpu == 0:
        #     print("load", time.time()-past)
        #     past = time.time()
        
        if self.transform is not None:
            if CONFIG.data.trimap == True:
                fgrs, phas, trimaps = self.transform(fgrs, phas, bgrs, bgrs_mask)
                return fgrs, phas, trimaps
            else:
                fgrs, phas, bgrs, bgrs_mask = self.transform(fgrs, phas, bgrs, bgrs_mask)
                # if CONFIG.gpu == 0:
                #     print("transform", time.time()-past)

                return fgrs, phas, bgrs, bgrs_mask
            
        # return fgrs, phas, bgrs
    

    
    def _get_img_pair_list(self, bg_txt):

        img_list, mask_list = [], []
        img_path = '/home/jhb/dataset/source/face/CelebAMask-HQ/CelebA-HQ-img-512'
        mask_path = '/home/jhb/dataset/source/face/CelebAMask-HQ-masks_corrected-512'

        with open(bg_txt, 'r') as f:
            for line in f:
                file = line.strip()
                name = file.split('.')[0]
                img_list.append(os.path.join(img_path, name+'.jpg'))
                mask_list.append(os.path.join(mask_path, name+'.png'))

        return img_list, mask_list
    

    def _get_occ_pair_list(self, fg_dir, folder_list, rand_dir, phase):

        img_list, mask_list, rand_list = [], [], []

        for folder in folder_list:
            path = os.path.join(fg_dir, folder)
            if folder == 'rand':
                rdir = os.path.join(rand_dir + phase, 'rand')
                img_path = os.path.join(rdir, 'occlusion_img')
                mask_path = os.path.join(rdir, 'occlusion_mask')

                rand_sample = np.random.choice(os.listdir(img_path), 200, False)
                count = 0
                for file in rand_sample:
                    name = file.split('.')[0]
                    img_list.append(os.path.join(img_path, name+'.png'))
                    mask_list.append(os.path.join(mask_path, name+'.png'))
                    rand_list.append(True)
                    count += 1
                print(phase, folder, ':', count)
            elif folder == '11k':
                img_path = os.path.join(path, 'Hands')
                mask_path = os.path.join(path, '11k-hands_masks')
                for file in os.listdir(mask_path):
                    name = file.split('.')[0]
                    img_list.append(os.path.join(img_path, name+'.jpg'))
                    mask_list.append(os.path.join(mask_path, name+'.png'))
                    rand_list.append(False)
                    count += 1
                print(phase, folder, ':', count)
            elif "sim" in folder:
                path = os.path.join(path, phase)
                count = 0
                for fo in os.listdir(path):
                    path_folder = os.path.join(path, fo)
                    if not os.path.isdir(path_folder):
                        continue
                    if fo in CONFIG.data.sim_list1: 
                        print(fo, CONFIG.data.sim_list1)
                        continue
                    img_path = os.path.join(path_folder, 'fg')
                    mask_path = os.path.join(path_folder, 'alpha')
                    for file in os.listdir(img_path):
                        if file.endswith(('jpg', 'png')):
                            for _ in range(1):
                                name = file.split('.')[0]
                                img_list.append(os.path.join(img_path, name+'.jpg'))
                                mask_list.append(os.path.join(mask_path, name+'.jpg'))
                                rand_list.append(True if fo in CONFIG.data.sim_list2 else False)
                                count += 1
                print(phase, folder, ':', count)

            elif folder == 'hiu':
                txt_file = os.path.join(path, 'hiu_'+phase+'.txt')
                file_list = []
                with open(txt_file, 'r') as file:
                    for line in file.readlines():
                        file_list.append(line.strip())
                img_path = os.path.join(path, 'fg')
                mask_path = os.path.join(path, 'alpha')

                count = 0
                # randomly select 200 for train, 100 for test like 11k
                train_num = np.random.choice(len(file_list), 200, False)
                for i in train_num: # range(len(file_list)): # train_num:
                    img_list.append(os.path.join(img_path, file_list[i]+'.jpg'))
                    mask_list.append(os.path.join(mask_path, file_list[i]+'_mask.png'))
                    rand_list.append(False)
                    count += 1
                print(phase, folder, ':', count)

            elif folder == 'am2k':
                img_path = os.path.join(path, phase+'/fg')
                mask_path = os.path.join(path, phase+'/mask')

                rand_sample = os.listdir(img_path)
                if phase == 'train':
                    rand_sample = np.random.choice(rand_sample, 1000, False)
                count = 0
                for file in rand_sample:
                    if file.endswith(('jpg', 'png')):
                        for _ in range(1):
                            if phase == 'test':
                                name = file.split('.')[0]
                                img_list.append(os.path.join(img_path, name+'.jpg'))
                                mask_list.append(os.path.join(mask_path, name+'.png'))
                            else:
                                img_list.append(os.path.join(img_path, file))
                                mask_list.append(os.path.join(mask_path, file))
                            rand_list.append(False)
                            count +=1 
                print(phase, "AM2K : ", count)
            elif folder == 'test':
                # test_benchmark_02
                root_path = '/home/jhb/dataset/FM_dataset/test_benchmark_02'
                folders = os.listdir(root_path)
                count = 0
                for folder in folders:
                    path = os.path.join(root_path, folder)

                    img_path = os.path.join(path,'img')
                    mask_path = os.path.join(path, 'mask')
                    for img in os.listdir(img_path):
                        if img.endswith(('jpg', 'png')):
                            name = img.split('.')[0]
                            img_list.append(os.path.join(img_path, name+'.jpg'))
                            mask_list.append(os.path.join(mask_path, name+'.png'))
                            rand_list.append(False)
                            count +=1 
                print(phase, "TEST : ", count)

            else:
                raise ValueError(" -- Folder ERROR --  : ", folder)
        
        return img_list, mask_list, rand_list

    
    def _get_imagematte(self, idx):

        # with cv2.imread(os.path.join(self.imagematte_dir, 'fgr', self.imagematte_files[idx % len(self.imagematte_files)])) as fgr, \
        #      cv2.imread(os.path.join(self.imagematte_dir, 'pha', self.imagematte_files[idx % len(self.imagematte_files)])) as pha:

        occ = cv2.imread(self.occ_list[idx % self.img_num])
        occ_mask = cv2.imread(self.occ_mask_list[idx % self.img_num], 0) # .astype(np.float32)/255
        
        # crop the occlusion
        occluder_rect = cv2.boundingRect(occ_mask)
        crop_occ = occ[occluder_rect[1]:(occluder_rect[1]+occluder_rect[3]),occluder_rect[0]:(occluder_rect[0]+occluder_rect[2])]
        crop_occ_mask = occ_mask[occluder_rect[1]:(occluder_rect[1]+occluder_rect[3]),occluder_rect[0]:(occluder_rect[0]+occluder_rect[2])] 
    
        # fgr = self._downsample_if_needed(fgr.convert('RGB'))
        # pha = self._downsample_if_needed(pha.convert('L'))

        # expand the border line
        h, w = crop_occ_mask.shape
        scale_factor = (((self.size*self.size))/(h*w))*np.random.uniform(self.occ_size_ratio[0], self.occ_size_ratio[1])
        scale_factor=np.sqrt(scale_factor)
        new_size = tuple(np.round(np.array([w, h]) * scale_factor).astype(int))
        crop_occ = cv2.resize(crop_occ, new_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
        crop_occ_mask = cv2.resize(crop_occ_mask, new_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
        h, w = crop_occ_mask.shape
        # if h <= self.size and w <= self.size:
        #     break
        
        h, w = crop_occ_mask.shape
        empty = np.zeros((self.size, self.size, 3), dtype=np.float32)
        empty_mask = np.zeros((self.size, self.size), dtype=np.float32)
        if self.occ_translate is None:
            if h > self.size or w > self.size:
                # random crop to make self.size
                if h > self.size:
                    diff = random.randint(0, h - self.size)
                    crop_occ = crop_occ[diff:diff+self.size, :, :] 
                    crop_occ_mask = crop_occ_mask[diff:diff+self.size, :] 
                    spoint = (0, random.randint(0, self.size-w))
                if w > self.size:
                    diff = random.randint(0, w - self.size)
                    crop_occ = crop_occ[:, diff:diff+self.size,:] 
                    crop_occ_mask = crop_occ_mask[:, diff:diff+self.size] 
                    spoint = (random.randint(0, self.size-h), 0)
                h, w = crop_occ_mask.shape
            else:
                spoint = (random.randint(0, self.size-h), random.randint(0, self.size-w))

            try:
                empty[spoint[0]:spoint[0]+h, spoint[1]:spoint[1]+w, :] = crop_occ
                empty_mask[spoint[0]:spoint[0]+h, spoint[1]:spoint[1]+w] = crop_occ_mask
            except:
                print(h,w, crop_occ.shape)
            
        # convert to PIL shape
        empty = empty.astype(np.uint8)
        crop_occ = cv2.cvtColor(empty, cv2.COLOR_BGR2RGB)
        crop_occ = Image.fromarray(crop_occ)
        crop_occ_mask = Image.fromarray(empty_mask)
        crop_occ_mask = crop_occ_mask.convert("L")

        fgrs = [crop_occ] * self.seq_length
        phas = [crop_occ_mask] * self.seq_length

        return fgrs, phas
    
    
    def _get_random_image_background(self, idx):
        # with Image.open(os.path.join(self.background_image_dir, self.background_image_files[random.choice(range(len(self.background_image_files)))])) as bgr:
        #     bgr = self._downsample_if_needed(bgr.convert('RGB'))

        # self.img_list, self.mask_list, self.occ_list, self.occ_mask_list, self.rand_list 
        
        bgr = Image.open(self.img_list[idx % self.img_num])
        pha =  Image.open(self.mask_list[idx % self.img_num])
        pha = pha.convert("L")

        # img = cv2.imread(self.img_list[idx % self.img_num])
        # mask = cv2.imread(self.mask_list[idx % self.img_num], 0) # .astype(np.float32)/255
        
        # # convert to PIL shape
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = Image.fromarray(img)
        # mask = Image.fromarray(mask)

        bgrs = [bgr] * self.seq_length
        bgrs_mask = [pha] * self.seq_length

        return bgrs, bgrs_mask

    def _get_random_video_background(self, idx):
        # not implemented : hyebin
        clip_idx = random.choice(range(len(self.background_video_clips)))
        frame_count = len(self.background_video_frames[clip_idx])
        frame_idx = random.choice(range(max(1, frame_count - self.seq_length)))
        clip = self.background_video_clips[clip_idx]
        bgrs = []
        for i in self.seq_sampler(self.seq_length):
            frame_idx_t = frame_idx + i
            frame = self.background_video_frames[clip_idx][frame_idx_t % frame_count]
            with Image.open(os.path.join(self.background_video_dir, clip, frame)) as bgr:
                bgr = self._downsample_if_needed(bgr.convert('RGB'))
            bgrs.append(bgr)
        return bgrs
    
    def _downsample_if_needed(self, img):
        w, h = img.size
        if min(w, h) > self.size:
            scale = self.size / min(w, h)
            w = int(scale * w)
            h = int(scale * h)
            img = img.resize((w, h))
        return img

class OccMatteAugmentation2(MotionAugmentation):
    def __init__(self, size):
        super().__init__(
            size=size,
            prob_fgr_affine=0.95,
            prob_bgr_affine=0.3,
            prob_noise=0.05,
            prob_color_jitter=0.3,
            prob_grayscale=0.03,
            prob_sharpness=0.05,
            prob_blur=0.02,
            prob_hflip=0.5,
            prob_pause=0.03,
        )
        self.erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,30)]
        self.varT = CONFIG.data.varT


    def __call__(self, fgrs, phas, bgrs, bgrs_phas): # occ, occ_mask, face, skin_mask

        '''

        ***  steps  ***

        * crop occlusion 
        * extend the border line -> done when img load
        * motion affine fg, bg
        *--- convert to tensor ---* 
        * resize fg, bg
        * flip fg, bg
        * motion color jitter fg, bg
        * pause fg, bg
        *--- convert to np array ---*
        * composite gt alpha matte & trimap
        *--- convert to tensor ---* 
        '''


        if CONFIG.gpu == 0:
            past = time.time()
        
        # Foreground affine > occlusion = hand
        if random.random() < self.prob_fgr_affine:
            fgrs, phas = self._motion_affine(fgrs, phas, varT=self.varT)

        # Background affine  > face
        if random.random() < self.prob_bgr_affine / 2:
            bgrs, bgrs_phas = self._motion_affine(bgrs, bgrs_phas, varT=self.varT)

        # both
        if random.random() < self.prob_bgr_affine / 2:
            fgrs, phas, bgrs, bgrs_phas = self._motion_affine(fgrs, phas, bgrs, bgrs_phas, varT=self.varT)
                
        # Still Affine
        if self.static_affine:
            fgrs, phas = self._static_affine(fgrs, phas, scale_ranges=(0.5, 1))
            bgrs, bgrs_phas = self._static_affine(bgrs, bgrs_phas, scale_ranges=(1, 1.5))

        # if CONFIG.gpu == 0:
        #     print("Affine", time.time() - past)
        #     past = time.time()
        
        # fgrs = [fgr.resize((512, 512)) for fgr in fgrs]
        # phas = [pha.resize((512, 512)) for pha in phas]
        # bgrs = [bgr.resize((512, 512)) for bgr in bgrs]
        # bgrs_phas = [bgr_pha.resize((512, 512)) for bgr_pha in bgrs_phas]


        # To tensor
        fgrs = torch.stack([F.to_tensor(fgr) for fgr in fgrs])
        phas = torch.stack([F.to_tensor(pha) for pha in phas])
        bgrs = torch.stack([F.to_tensor(bgr) for bgr in bgrs])
        bgrs_phas = torch.stack([F.to_tensor(bgr_pha) for bgr_pha in bgrs_phas])

        # return fgrs, phas, bgrs, bgrs_phas

        
        # Resize & Crop 
        # params = transforms.RandomResizedCrop.get_params(fgrs, scale=(1, 1), ratio=self.aspect_ratio_range)
        # fgrs = F.resized_crop(fgrs, *params, self.size, interpolation=F.InterpolationMode.BILINEAR, antialias=True)
        # phas = F.resized_crop(phas, *params, self.size, interpolation=F.InterpolationMode.BILINEAR, antialias=True)
        params = transforms.RandomResizedCrop.get_params(bgrs, scale=(1, 1), ratio=self.aspect_ratio_range)
        bgrs = F.resized_crop(bgrs, *params, self.size, interpolation=F.InterpolationMode.BILINEAR, antialias=True)
        bgrs_phas = F.resized_crop(bgrs_phas, *params, self.size, interpolation=F.InterpolationMode.BILINEAR, antialias=True)


        # Horizontal flip
        if random.random() < self.prob_hflip:
            fgrs = F.hflip(fgrs)
            phas = F.hflip(phas)
        if random.random() < self.prob_hflip:
            bgrs = F.hflip(bgrs)
            bgrs_phas = F.hflip(bgrs_phas)

        # Color jitter
        # 기존 _motion_color_jitter은 tensor로 변환 후 진행된다.
        if random.random() < self.prob_color_jitter:
            fgrs = self._motion_color_jitter(fgrs, varT=self.varT)
        if random.random() < self.prob_color_jitter:
            bgrs = self._motion_color_jitter(bgrs, varT=self.varT)

        # # Grayscale
        # if random.random() < self.prob_grayscale:
        #     fgrs = F.rgb_to_grayscale(fgrs, num_output_channels=3).contiguous()
        #     bgrs = F.rgb_to_grayscale(bgrs, num_output_channels=3).contiguous()
            
        # # Sharpen
        # if random.random() < self.prob_sharpness:
        #     sharpness = random.random() * 8
        #     fgrs = F.adjust_sharpness(fgrs, sharpness)
        #     phas = F.adjust_sharpness(phas, sharpness)
        #     bgrs = F.adjust_sharpness(bgrs, sharpness)
        
        # # Blur
        # if random.random() < self.prob_blur / 3:
        #     fgrs, phas = self._motion_blur(fgrs, phas)
        # if random.random() < self.prob_blur / 3:
        #     bgrs = self._motion_blur(bgrs)
        # if random.random() < self.prob_blur / 3:
        #     fgrs, phas, bgrs = self._motion_blur(fgrs, phas, bgrs)

        # Pause
        if random.random() < self.prob_pause:
            fgrs, phas, bgrs, bgrs_phas = self._motion_pause(fgrs, phas, bgrs, bgrs_phas)



        # convert to numpy array 
        trimaps = []
        if random.random() < CONFIG.data.occlusion:

            ##### random resize fg(occluder)
            tmp1 = np.array(bgrs_phas[0]*255).transpose((1,2,0)).astype('uint8')
            tmp2 = np.array(phas[0]*255).transpose((1,2,0)).astype('uint8')
            src_rect = cv2.boundingRect(tmp1)
            occ_rect = cv2.boundingRect(tmp2)

            # src_rect = [0,0,bgrs[0].shape[2], bgrs[0].shape[1]] 
            # occ_rect = [0,0,fgrs[0].shape[2], fgrs[0].shape[1]]
            # src_rect = [0,0,512,512] 
            # occ_rect = [0,0,512,512] 
            
            try:
                scale_factor = (((src_rect[2]*src_rect[3]))/(occ_rect[2]*occ_rect[3]) )*np.random.uniform(0.3, 0.7)
                scale_factor=np.sqrt(scale_factor)
            except Exception as e:
                print(e)
                scale_factor=1

            # Add rotate around center
            src_center=(src_rect[0]+(src_rect[2]/2),(src_rect[1]+src_rect[3]/2))
            occ_coord = np.random.uniform([src_rect[0],src_rect[1]], [src_rect[0]+src_rect[2],src_rect[1]+src_rect[3]])
            rotation= self.angle3pt((src_center[0],occ_coord[1]),src_center,occ_coord)
            if occ_coord[1]>src_center[1]:
                rotation=rotation+180

            comps, comps_phas = [], []
            for i in range(bgrs.shape[0]):
                fgr = np.array(fgrs[i])*255
                pha = np.array(phas[i])*255
                bgr = np.array(bgrs[i])*255
                bgr_pha = np.array(bgrs_phas[i])*255
                ##### Resize and Composit the image
                bgr[bgr>255] = 255
                bgr[bgr<0] = 0
                bgr_pha[bgr_pha>255] = 255
                bgr_pha[bgr_pha<0] = 0
                fgr[fgr>255] = 255
                fgr[fgr<0] = 0
                pha[pha>255] = 255
                pha[pha<0] = 0
                    
                occ, occ_mask = fgr.astype(np.uint8), pha.astype(np.uint8)
                occ = occ.transpose((1,2,0))
                occ_mask = occ_mask.transpose((1,2,0))
                h, w, _ = occ_mask.shape
                try:
                    new_size = tuple(np.round(np.array([w, h]) * scale_factor).astype(int))
                    if new_size[0]<=0 and new_size[1]<=0:
                        print("============", new_size, h, w, scale_factor)
                        new_size = occ.shape[:2]
                    occ = cv2.resize(occ, new_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                except:
                    print(new_size)
                    import sys
                    sys.exit()
                # new_size = tuple(np.round(np.array([w, h]) * scale_factor).astype(int))
                # occ = cv2.resize(occ, new_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                occ_mask = cv2.resize(occ_mask, new_size, interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                occ, occ_mask = occ.astype(np.float32), occ_mask.astype(np.float32)

                alpha = np.array(bgrs_phas[i])
                occ  = imutils.rotate_bound(occ,rotation)
                occ_mask = imutils.rotate_bound(occ_mask,rotation)

                # make occluder mask's shape to image_mask's shape
                occlusion_mask=np.zeros(bgr_pha[0].shape, np.float32)
                occlusion_mask[(occlusion_mask>0) & (occlusion_mask<255)] = 255
                bgr = bgr.transpose((1,2,0))
                comp, comp_pha, occlusion_mask, occ  = self.paste_over(occ,occ_mask,bgr,bgr_pha[0],occ_coord,occlusion_mask, False)
                comp = comp.transpose((2,0,1))/255.0
                comps.append(comp)
                comps_phas.append(np.array([comp_pha/255.0]))
                # tmp = np.expand_dims(comp_pha, 2)
                # tmp = np.repeat(tmp, 3, axis=2)
                # tmp2 = np.expand_dims(occlusion_mask, 2)
                # tmp2 = np.repeat(tmp2, 3, axis=2)
                # show = np.concatenate((comp, tmp, occ, tmp2), axis=0)
                # cv2.imwrite('show.png', show[:,:,::-1])


                # generate trimap with comp_pha
                h, w = comp_pha.shape
                alpha = cv2.resize(comp_pha/255.0, (640,640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                max_kernel_size = 30
                fg_mask = (alpha + 1e-5).astype(np.int).astype(np.uint8)
                bg_mask = (1 - alpha + 1e-5).astype(np.int).astype(np.uint8)
                fg_mask = cv2.erode(fg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
                bg_mask = cv2.erode(bg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])

                # fg_width = np.random.randint(1, 30)
                # bg_width = np.random.randint(1, 30)
                # fg_mask = (alpha + 1e-5).astype(np.int).astype(np.uint8)
                # bg_mask = (1 - alpha + 1e-5).astype(np.int).astype(np.uint8)
                # fg_mask = cv2.erode(fg_mask, self.erosion_kernels[fg_width])
                # bg_mask = cv2.erode(bg_mask, self.erosion_kernels[bg_width])

                trimap = np.ones_like(alpha) * 128
                trimap[fg_mask == 1] = 255
                trimap[bg_mask == 1] = 0

                trimap = cv2.resize(trimap, (w,h), interpolation=cv2.INTER_NEAREST)
                trimaps.append(np.array([trimap]))
                
                # # generate trimap with occlusion_mask
                # h, w = comp_pha.shape
                # alpha = cv2.resize(occlusion_mask/255.0, (640,640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                # max_kernel_size = 30
                # fg_mask = (alpha + 1e-5).astype(np.int).astype(np.uint8)
                # bg_mask = (1 - alpha + 1e-5).astype(np.int).astype(np.uint8)
                # fg_mask = cv2.erode(fg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
                # bg_mask = cv2.erode(bg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])

                # trimap_occ = np.ones_like(alpha) * 128
                # trimap_occ[fg_mask == 1] = 255
                # trimap_occ[bg_mask == 1] = 0

                # trimap_occ = cv2.resize(trimap_occ, (w,h), interpolation=cv2.INTER_NEAREST)
                # trimap_occ[trimap==0] = 0
                
                # trimaps.append(np.array([trimap_occ]))

            comps = torch.stack([torch.Tensor(comp) for comp in comps])
            comps_phas = torch.stack([torch.Tensor(comp_pha) for comp_pha in comps_phas])
            trimaps = torch.stack([torch.Tensor(trimap) for trimap in trimaps])

            return comps, comps_phas, trimaps

        else:
            
            # generate trimap with bgrs_phas = skin mask
            for i in range(bgrs.shape[0]):
                alpha = np.array(bgrs_phas[i]*255)[0]
                h, w = alpha.shape
                alpha = cv2.resize(alpha, (640,640), interpolation=maybe_random_interp(cv2.INTER_NEAREST))
                max_kernel_size = 30
                fg_mask = (alpha + 1e-5).astype(np.int).astype(np.uint8)
                bg_mask = (1 - alpha + 1e-5).astype(np.int).astype(np.uint8)
                fg_mask = cv2.erode(fg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])
                bg_mask = cv2.erode(bg_mask, self.erosion_kernels[np.random.randint(1, max_kernel_size)])

                fg_width = np.random.randint(1, 30)
                bg_width = np.random.randint(1, 30)
                fg_mask = (alpha + 1e-5).astype(np.int).astype(np.uint8)
                bg_mask = (1 - alpha + 1e-5).astype(np.int).astype(np.uint8)
                fg_mask = cv2.erode(fg_mask, self.erosion_kernels[fg_width])
                bg_mask = cv2.erode(bg_mask, self.erosion_kernels[bg_width])

                trimap = np.ones_like(alpha) * 128
                trimap[fg_mask == 1] = 255
                trimap[bg_mask == 1] = 0

                trimap = cv2.resize(trimap, (w,h), interpolation=cv2.INTER_NEAREST)
                trimaps.append(np.array([trimap]))
            trimaps = torch.stack([torch.Tensor(trimap) for trimap in trimaps])
            return bgrs, bgrs_phas, trimaps


    
    def angle3pt(self, a, b, c):
        """Counterclockwise angle in degrees by turning from a to c around b
            Returns a float between 0.0 and 360.0"""
    
        ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
        return ang + 360 if ang < 0 else ang


    # https://github.com/isarandi/synthetic-occlusion
    def paste_over(self, im_src, occluder_mask, im_dst, dst_mask, center, occlusion_mask, randOcc):
        """Pastes `im_src` onto `im_dst` at a specified position, with alpha blending, in place.
        Locations outside the bounds of `im_dst` are handled as expected (only a part or none of
        `im_src` becomes visie).
        Args:
            im_src: The RGBA image to be pasted onto `im_dst`. Its size can be arbitrary.
            im_dst: The target image.
            alpha: A float (0.0-1.0) array of the same size as `im_src` controlling the alpha blending
                at each pixel. Large values mean more visibility for `im_src`.
            center: coordinates in `im_dst` where the center of `im_src` should be placed.
        """

        width_height_src = np.asarray([im_src.shape[1], im_src.shape[0]])
        width_height_dst = np.asarray([im_dst.shape[1], im_dst.shape[0]])

        center = np.round(center).astype(np.int32)
        raw_start_dst = center - width_height_src // 2
        raw_end_dst = raw_start_dst + width_height_src

        start_dst = np.clip(raw_start_dst, 0, width_height_dst)
        end_dst = np.clip(raw_end_dst, 0, width_height_dst)
        region_dst = im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]

        start_src = start_dst - raw_start_dst
        end_src = width_height_src + (end_dst - raw_end_dst)
        occluder_mask = occluder_mask[start_src[1]:end_src[1], start_src[0]:end_src[0]]
        region_src = im_src[start_src[1]:end_src[1], start_src[0]:end_src[0]]
        color_src = region_src[..., 0:3]


        # alpha = (region_src[..., 3:].astype(np.float32)/255)
        alpha = (occluder_mask.astype(np.float32)/255)
        if randOcc:
            if np.random.rand()<0.3:
                trans = np.random.uniform(0.4, 0.7)
                alpha *= trans
            else:
                trans = 1

        # kernel = np.ones((3,3),np.uint8)
        # alpha = cv2.erode(alpha,kernel,iterations = 1)
        # alpha = cv2.GaussianBlur(alpha,(5,5),0)
        alpha = np.expand_dims(alpha, axis=2)
        alpha = np.repeat(alpha, 3, axis=2)

        im_dst_cp = im_dst.copy()
        dst_mask_cp = dst_mask.copy()
        occ_mask_cp = occlusion_mask.copy()
        color = np.zeros(im_dst.shape, dtype=np.uint8)

        if randOcc:
            if np.random.rand()<0.3:
                occ_mask_cp = occ_mask_cp.astype(np.float32)
                occ_mask_cp *= trans
                occ_mask_cp = occ_mask_cp.astype(np.uint8)
                occluder_mask = occluder_mask.astype(np.float32)
                occluder_mask *= trans
                occluder_mask = occluder_mask.astype(np.float32)
        
        occ_mask_cp[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = cv2.add(occlusion_mask[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]],occluder_mask)
        dst_mask_cp[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = cv2.subtract(dst_mask[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]],occluder_mask)
        im_dst_cp[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = (alpha * color_src + (1 - alpha) * region_dst)
        color[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = color_src

        dst_mask_cp[dst_mask_cp<0] = 0
        dst_mask_cp[dst_mask_cp>255] = 255

        return im_dst_cp,dst_mask_cp,occ_mask_cp,color

    def paste_over1(self, im_src, occluder_mask, im_dst, dst_mask, center, occlusion_mask):
        """Pastes `im_src` onto `im_dst` at a specified position, with alpha blending, in place.
        Locations outside the bounds of `im_dst` are handled as expected (only a part or none of
        `im_src` becomes visie).
        Args:
            im_src: The RGBA image to be pasted onto `im_dst`. Its size can be arbitrary.
            im_dst: The target image.
            alpha: A float (0.0-1.0) array of the same size as `im_src` controlling the alpha blending
                at each pixel. Large values mean more visibility for `im_src`.
            center: coordinates in `im_dst` where the center of `im_src` should be placed.
        """

        width_height_src = np.asarray([im_src.shape[2], im_src.shape[1]])
        width_height_dst = np.asarray([im_dst.shape[2], im_dst.shape[1]])

        center = np.round(center).astype(np.int32)
        raw_start_dst = center - width_height_src // 2
        raw_end_dst = raw_start_dst + width_height_src

        start_dst = np.clip(raw_start_dst, 0, width_height_dst)
        end_dst = np.clip(raw_end_dst, 0, width_height_dst)
        region_dst = im_dst[:, start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]

        start_src = start_dst - raw_start_dst
        end_src = width_height_src + (end_dst - raw_end_dst)
        occluder_mask =occluder_mask[:, start_src[1]:end_src[1], start_src[0]:end_src[0]]
        region_src = im_src[:, start_src[1]:end_src[1], start_src[0]:end_src[0]]
        # color_src = region_src[..., 0:3]


        # alpha = (region_src[..., 3:].astype(np.float32)/255)
        alpha = (occluder_mask.astype(np.float32)/255)
        # if randOcc:
        #     if np.random.rand()<0.3:
        #         trans = np.random.uniform(0.4, 0.7)
        #         alpha *= trans
        #     else:
        #         trans = 1

        # kernel = np.ones((3,3),np.uint8)
        # alpha = cv2.erode(alpha,kernel,iterations = 1)
        # alpha = cv2.GaussianBlur(alpha,(5,5),0)
        # alpha = np.expand_dims(alpha, axis=2)
        # alpha = np.repeat(alpha, 3, axis=2)

        im_dst_cp = im_dst.copy()
        dst_mask_cp = dst_mask.copy()
        occ_mask_cp = occlusion_mask.copy()
        color = np.zeros(im_dst.shape, dtype=np.uint8)

        # if randOcc:
        #     if np.random.rand()<0.3:
        #         occ_mask_cp = occ_mask_cp.astype(np.float32)
        #         occ_mask_cp *= trans
        #         occ_mask_cp = occ_mask_cp.astype(np.uint8)
        #         occluder_mask = occluder_mask.astype(np.float32)
        #         occluder_mask *= trans
        #         occluder_mask = occluder_mask.astype(np.float32)
        
        occ_mask_cp[:, start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = cv2.add(occlusion_mask[:, start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]], occluder_mask)
        dst_mask_cp[:, start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = cv2.subtract(dst_mask[:, start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]], occluder_mask)
        im_dst_cp[:, start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = (alpha * region_src + (1 - alpha) * region_dst)
        # color[:, start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = region_src

        return im_dst_cp, dst_mask_cp, occlusion_mask # ,occ_mask_cp,color