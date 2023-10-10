import os
import cv2
import numpy as np
import argparse

######
# From matteformer
# https://github.com/webtoon/matteformer/blob/master/evaluation.py
######

def compute_mse_loss(pred, target, trimap):
    error_map = (pred - target) / 255.0
    loss = np.sum((error_map ** 2) * (trimap == 128)) / (np.sum(trimap == 128) + 1e-8)

    return loss


def compute_sad_loss(pred, target, trimap):
    error_map = np.abs((pred - target) / 255.0)
    loss = np.sum(error_map * (trimap == 128))

    return loss / 1000, np.sum(trimap == 128) / 1000


def evaluate(args):
    img_names = []
    mse_loss_unknown = []
    sad_loss_unknown = []

    folders = [item for item in os.listdir(args.label_dir) if not os.path.isfile(os.path.join(args.label_dir, item))]
    print(folders)
    for folder in folders:

        if folder == '11k-sot':
            continue

        tmp_mse_loss_unknown = []
        tmp_sad_loss_unknown = []

        label_dir = os.path.join(args.label_dir, folder)
        label_dir = os.path.join(label_dir, 'mask')
        pred_dir = os.path.join(args.pred_dir, folder)
        for i, img in enumerate(os.listdir(label_dir)):

            img_name = img[:-4]+'.jpg'
            mask_name = img[:-4]+'.png'
            if not((os.path.isfile(os.path.join(pred_dir, img_name)) and
                    os.path.isfile(os.path.join(label_dir, mask_name)))):
                print(os.path.join(pred_dir, img_name))
                print(os.path.isfile(os.path.join(pred_dir, img_name)))
                print(os.path.join(label_dir, mask_name))
                print(os.path.isfile(os.path.join(label_dir, mask_name)))
                print('[{}/{}] "{}" skipping'.format(i, len(os.listdir(label_dir)), mask_name))
                import sys
                sys.exit()
                continue

            pred = cv2.imread(os.path.join(pred_dir, img_name), 0).astype(np.float32)
            label = cv2.imread(os.path.join(label_dir, mask_name), 0).astype(np.float32)
            # trimap = cv2.imread(os.path.join(args.trimap_dir, img), 0).astype(np.float32)

            # resize
            if label.shape != pred.shape:
                label = cv2.resize(label, pred.shape)

            # generate trimap
            max_kernel_size = 30
            erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1,30)]
            fg_mask = (label + 1e-5).astype(np.int).astype(np.uint8)
            bg_mask = (1 - label + 1e-5).astype(np.int).astype(np.uint8)
            fg_mask = cv2.erode(fg_mask, erosion_kernels[np.random.randint(1, max_kernel_size)])
            bg_mask = cv2.erode(bg_mask, erosion_kernels[np.random.randint(1, max_kernel_size)])

            trimap = np.ones_like(label) * 128
            trimap[fg_mask == 1] = 255
            trimap[bg_mask == 1] = 0

            # calculate loss
            mse_loss_unknown_ = compute_mse_loss(pred, label, trimap)
            sad_loss_unknown_ = compute_sad_loss(pred, label, trimap)[0]
            # print('Unknown Region: MSE:', mse_loss_unknown_, ' SAD:', sad_loss_unknown_)

            # save for average
            img_names.append(img)

            tmp_mse_loss_unknown.append(mse_loss_unknown_)  # mean l2 loss per unknown pixel
            tmp_sad_loss_unknown.append(sad_loss_unknown_)  # l1 loss on unknown area

            # print('[{}/{}] "{}" processed'.format(i, len(os.listdir(args.label_dir)), img))
        print('*', folder, '*')
        print('* Unknown Region: MSE:', np.array(tmp_mse_loss_unknown).mean(), ' SAD:', np.array(tmp_sad_loss_unknown).mean())
        mse_loss_unknown.append(np.array(tmp_mse_loss_unknown).mean())
        sad_loss_unknown.append(np.array(tmp_sad_loss_unknown).mean())

    print('* TOTAL *')
    print('* Unknown Region: MSE:', np.array(mse_loss_unknown).mean(), ' SAD:', np.array(sad_loss_unknown).mean())
    print('* if you want to report scores in your paper, please use the official matlab codes for evaluation.')
    print(args.pred_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-dir', type=str, default='predDIM/pred_alpha/', help="output dir")
    parser.add_argument('--label-dir', type=str, default='Composition-1k-testset/alpha_copy/', help="GT alpha dir")
    parser.add_argument('--trimap-dir', type=str, default='Composition-1k-testset/trimaps/', help="trimap dir")

    args = parser.parse_args()

    evaluate(args)