import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class RoadDatasetSegFormer(Dataset):
    def __init__(self, csv_path, image_dir, mask_dir, split='train', augment=True):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, row['filename'])
        mask_path = os.path.join(self.mask_dir, row['maskname'])


        # Read image & mask
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Augment
        if self.augment:
            image = self.randomHueSaturationValue(image)
            image, mask = self.randomShiftScaleRotate(image, mask)
            image, mask = self.randomHorizontalFlip(image, mask)
            image, mask = self.randomVerticalFlip(image, mask)
            image, mask = self.randomRotate90(image, mask)

        # Normalize image
        image = image.astype(np.float32) / 255.0
        image = image * 2 - 1
        image = np.transpose(image, (2, 0, 1))  # HWC → CHW

        # Normalize mask
        mask = mask.astype(np.float32) / 255.0
        mask = np.expand_dims(mask, axis=0)  # (1, H, W)
        mask = (mask > 0.5).astype(np.float32)

        # print(f"Loading sample {idx}")

        return torch.tensor(image), torch.tensor(mask)
    # {
    #     "pixel_values": torch.tensor(image),
    #     "labels": torch.tensor(mask)
    # }

    ## ---------- AUGMENTATION METHODS BELOW ----------
    def randomHueSaturationValue(self, image, hue_shift_limit=(-30, 30),
                                  sat_shift_limit=(-5, 5),
                                  val_shift_limit=(-15, 15), u=0.5):
        if np.random.rand() < u:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h, s, v = cv2.split(image)
            h = h.astype(np.float32)
            s = s.astype(np.float32)
            v = v.astype(np.float32)


            h += np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
            s += np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
            v += np.random.uniform(val_shift_limit[0], val_shift_limit[1])

            h = np.clip(h, 0, 179).astype(np.uint8)
            s = np.clip(s, 0, 255).astype(np.uint8)
            v = np.clip(v, 0, 255).astype(np.uint8)

            image = cv2.merge((h, s, v))
            image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        return image

    def randomShiftScaleRotate(self, image, mask,
                               shift_limit=(-0.1, 0.1),
                               scale_limit=(-0.1, 0.1),
                               rotate_limit=(0, 0),
                               aspect_limit=(-0.1, 0.1),
                               borderMode=cv2.BORDER_CONSTANT, u=0.5):
        if np.random.rand() < u:
            height, width = image.shape[:2]
            angle = np.random.uniform(*rotate_limit)
            scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
            aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
            sx = scale * aspect / (aspect ** 0.5)
            sy = scale / (aspect ** 0.5)
            dx = int(np.random.uniform(*shift_limit) * width)
            dy = int(np.random.uniform(*shift_limit) * height)

            cc = np.cos(np.radians(angle)) * sx
            ss = np.sin(np.radians(angle)) * sy
            rotation = np.array([[cc, -ss], [ss, cc]])

            box = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
            box -= np.array([width / 2, height / 2], dtype=np.float32)
            box = np.dot(box, rotation.T) + np.array([width / 2 + dx, height / 2 + dy], dtype=np.float32)

            mat = cv2.getPerspectiveTransform(box.astype(np.float32), np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32))
            image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode, borderValue=(0, 0, 0))
            mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_NEAREST, borderMode=borderMode, borderValue=(0,))

        return image, mask

    def randomHorizontalFlip(self, image, mask, u=0.5):
        if np.random.rand() < u:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        return image, mask

    def randomVerticalFlip(self, image, mask, u=0.5):
        if np.random.rand() < u:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        return image, mask

    def randomRotate90(self, image, mask, u=0.5):
        if np.random.rand() < u:
            image = np.rot90(image).copy()
            mask = np.rot90(mask).copy()
        return image, mask