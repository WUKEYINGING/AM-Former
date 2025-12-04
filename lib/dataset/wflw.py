# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
#
# MODIFIED TO SUPPORT SUBSET-BASED EVALUATION
# ------------------------------------------------------------------------------

import os
import random
import logging

import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image
import numpy as np

from ..utils.transforms import fliplr_joints, crop, transform_pixel_float

logger = logging.getLogger(__name__)


class WFLW(data.Dataset):
    def __init__(self, cfg, is_train=True, transform=None, subset=None):
        self.is_train = is_train
        self.transform = transform
        self.data_root = cfg.DATASET_WFLW.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.scale_factor = cfg.DATASET_WFLW.SCALE_FACTOR
        self.rot_factor = cfg.DATASET_WFLW.ROT_FACTOR
        self.flip = cfg.DATASET_WFLW.FLIP

        if is_train:
            self.csv_file = os.path.join(self.data_root, cfg.DATASET_WFLW.TRAINSET)
            self.landmarks_frame = pd.read_csv(self.csv_file)
            if cfg.TRAIN.LARGEST_NUM > 0:
                extended_df = pd.concat([self.landmarks_frame] * (cfg.TRAIN.LARGEST_NUM // len(self.landmarks_frame) + 1),
                                        ignore_index=True)
                self.landmarks_frame = extended_df.iloc[:cfg.TRAIN.LARGEST_NUM]
        else:
            self.csv_file = os.path.join(self.data_root, cfg.DATASET_WFLW.TESTSET)
            self.landmarks_frame = pd.read_csv(self.csv_file)

        if not is_train and subset:
            logger.info(f"=> Preparing WFLW subset: '{subset}'")

            attr_map = {'Pose': 0, 'Expression': 1, 'Illumination': 2, 'Makeup': 3, 'Occlusion': 4, 'Blur': 5}
            if subset not in attr_map:
                raise ValueError(f"Invalid WFLW subset name '{subset}'. Must be one of {list(attr_map.keys())}")
            subset_attr_index = attr_map[subset]

            attr_file_path = os.path.join(self.data_root, 'WFLW_annotations', 'list_98pt_rect_attr_test.txt')
            if not os.path.exists(attr_file_path):
                raise FileNotFoundError(f"Official WFLW annotation file not found at: {attr_file_path}. "
                                        "Please ensure it exists in the 'WFLW_annotations' subdirectory.")

            subset_image_names = []
            with open(attr_file_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    attributes = parts[-7:-1]
                    image_name = parts[-1]
                    if int(attributes[subset_attr_index]) == 1:
                        subset_image_names.append(image_name)

            logger.info(f"=> Found {len(subset_image_names)} images for subset '{subset}'. Filtering the dataset.")

            self.landmarks_frame = self.landmarks_frame[self.landmarks_frame['image_name'].isin(subset_image_names)]
            self.landmarks_frame = self.landmarks_frame.reset_index(drop=True)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if is_train:
            mode_str = "training"
        else:
            mode_str = f"test subset '{subset}'" if subset else "full test set"

        logger.info(f'=> loaded {len(self.landmarks_frame)} samples for {mode_str}')
    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_root, self.landmarks_frame.iloc[idx, 0])
        scale = self.landmarks_frame.iloc[idx, 1]
        center_w = self.landmarks_frame.iloc[idx, 2]
        center_h = self.landmarks_frame.iloc[idx, 3]
        center = torch.Tensor([center_w, center_h])
        pts = self.landmarks_frame.iloc[idx, 4:].values
        pts = pts.astype('float').reshape(-1, 2)
        scale *= 1.25
        nparts = pts.shape[0]
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)
        r = 0
        if self.is_train:
            scale = scale * (random.uniform(1 - self.scale_factor, 1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor) if random.random() <= 0.6 else 0
            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)
                pts = fliplr_joints(pts, width=img.shape[1], dataset='WFLW')
                center[0] = img.shape[1] - center[0]
        img = crop(img, center, scale, self.input_size, rot=r)
        tpts = pts.copy()
        for i in range(nparts):
            if tpts[i, 1] > 0:
                tpts[i, 0:2] = transform_pixel_float(tpts[i, 0:2] + 1, center, scale, self.input_size, rot=r)
        img = img.astype(np.float32)
        img = (img / 255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)
        target = tpts[:, 0:2] / self.input_size[0]
        target_weight = np.ones((nparts, 1), dtype=np.float32)
        target_weight = torch.from_numpy(target_weight)
        meta = {'index': idx, 'center': center, 'scale': scale, 'rotate': r, 'pts': torch.Tensor(pts), 'tpts': tpts}
        return img, target, target_weight, meta


if __name__ == '__main__':
    pass
