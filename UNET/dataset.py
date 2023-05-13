import torch
import cv2
import numpy as np
from torchvision.transforms.functional import to_tensor


class SegmentationDataset(torch.utils.data.Dataset):
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        mask = cv2.imread(self.maskPaths[index], cv2.IMREAD_GRAYSCALE)

        return to_tensor(image), to_tensor(mask)
