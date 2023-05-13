from torch.utils.data import Dataset
from PIL import Image
import cv2
import io
import torch
import numpy as np

class SegmentationDataset(Dataset):
    def __init__(self, img_paths, mask_paths):
        super().__init__()
        # store image and mask paths
        self.imgPaths = img_paths
        self.maskPaths = mask_paths

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, index):
        # grab image path from current index
        imagePath = self.imgPaths[index]

        # load image from disk, swap channels from BGR to RGM if loading with cv2
        image = cv2.imread(imagePath)
        print(imagePath, len(self.maskPaths), len(self.imgPaths))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.maskPaths[index])

        return image, mask
