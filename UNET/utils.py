import numpy as np
import os
import torch
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor
from .unet import Unet
import matplotlib
import json

matplotlib.use("Agg")


def get_sorted_PNGs(path):
    fpaths = [
        os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith(".png")
    ]
    fpaths.sort()
    return fpaths


def load_config_file(config_file):
    all_params = json.load(open(config_file))
    return all_params


def prepare_plot(origImage, predMask, threshold, save_path):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    overlay = origImage.copy()
    for i in range(overlay.shape[0]):
        for j in range(overlay.shape[1]):
            if predMask[i][j] < threshold:
                overlay[i][j] += [255, 0, 0]
    overlay = np.clip(overlay, None, 255)

    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage)
    ax[1].imshow(predMask)
    ax[2].imshow(overlay)
    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Predicted Mask")
    ax[2].set_title("Ovelayed")
    # set the layout of the figure and display it
    figure.tight_layout()
    plt.savefig(save_path)
    plt.close()


def predict_and_plot(model, id, image, save_dir, threshold, device="cuda"):
    # set model to evaluation mode
    model.eval()

    # turn off gradient tracking
    with torch.no_grad():
        input = to_tensor(image).to(device=device)
        input = input.unsqueeze(0)

        pred_mask = model(input).squeeze()
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = pred_mask.cpu().numpy()

        # visualization
        prepare_plot(
            origImage=image,
            predMask=pred_mask,
            threshold=threshold,
            save_path=os.path.join(save_dir, "plot{}.png".format(id)),
        )
