from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import cv2
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
import argparse
import os


def make_mask(A_sam_masks, B_sam_masks, bool_diff_map, mask_thre, overlap_thre):
    w,h = bool_diff_map.shape 
    refine_mask = torch.tensor(np.zeros((w,h))).cuda()
    thre_size = w * h * mask_thre
    overlap_size = w * h * overlap_thre
    bool_diff_map = torch.tensor(bool_diff_map).cuda()

    for B_sam_mask_dict in B_sam_masks:
        B_sam_mask = torch.tensor(B_sam_mask_dict['segmentation']).cuda()
        overlap_trigger = 0
        _, b_sam_mask_count = torch.unique(B_sam_mask,return_counts=True)
        if b_sam_mask_count[-1] > thre_size:
            continue
        for A_sam_mask_dict in A_sam_masks:
            A_sam_mask = torch.tensor(A_sam_mask_dict['segmentation']).cuda()
            _, a_sam_mask_count = torch.unique(A_sam_mask,return_counts=True)
            if a_sam_mask_count[-1] > thre_size:
                continue
            overlap_ab = B_sam_mask * A_sam_mask
            _, overlap_ab_count = torch.unique(overlap_ab,return_counts=True)
            if len(overlap_ab_count) == 2:
                if overlap_ab_count[-1] > (b_sam_mask_count[-1] * 0.95) and overlap_ab_count[-1] > (b_sam_mask_count[-1] * 1.05):
                    if overlap_ab_count[-1] > (a_sam_mask_count[-1] * 0.95) and overlap_ab_count[-1] > (a_sam_mask_count[-1] * 1.05):
                        overlap_trigger = 1
                        break
        if overlap_trigger == 0:
            overlap_diff = bool_diff_map * B_sam_mask
            _, overlap_diff_count = torch.unique(overlap_diff,return_counts=True)
            if len(overlap_diff_count) == 2 and (b_sam_mask_count[-1] * overlap_thre) <= overlap_diff_count[-1]:
                refine_mask[B_sam_mask==True] = 255
    return refine_mask


def main():
    parser = argparse.ArgumentParser(description='Make test sam image')
    parser.add_argument('--root_path', default='./datasets/', type=str)
    parser.add_argument('--dataset_name', default='LEVIR-CD', type=str)
    parser.add_argument('--save_name', default='levir', type=str)
    parser.add_argument('--mode', default='test', type=str)
    parser.add_argument('--mask_thre', default=0.05, type=int)
    parser.add_argument('--overlap_thre', default=0.1, type=int)
    args = parser.parse_args()

    save_dir = './sam_refine_mask/' + args.save_name + '/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    A_img_dir = args.root_path + args.dataset_name + '/' + args.mode + '/A/'
    B_img_dir = args.root_path + args.dataset_name + '/' + args.mode + '/B/'
    
    # CDRL output directory
    diff_map_path = './pixel_img_morpho/' + args.save_name + '/'

    A_img_paths = glob(A_img_dir + '*')
    A_img_paths.sort()
    
    sam = sam_model_registry["default"](checkpoint="./pretrain_weight/sam_vit_h_4b8939.pth")
    sam.to(device='cuda')
    mask_generator = SamAutomaticMaskGenerator(sam)

    for A_img_path in tqdm(A_img_paths[:2]):
        img_name = A_img_path.split('/')[-1]
        A_img = cv2.imread(A_img_path)
        B_img = cv2.imread(B_img_dir + img_name)
        ori_w, ori_h, _ = B_img.shape

        diff_map = cv2.imread(diff_map_path + img_name, 0)
        diff_map = cv2.resize(diff_map,(ori_w,ori_h))
        bool_diff_map = np.where(diff_map>0,True,False)

        A_masks = mask_generator.generate(A_img)
        B_masks = mask_generator.generate(B_img)

        refine_mask = make_mask(A_masks, B_masks, bool_diff_map, args.mask_thre, args.overlap_thre)
        refine_mask = refine_mask.cpu().numpy()
        cv2.imwrite(save_dir + img_name, refine_mask)

if __name__ == '__main__':
    main()
