import torch
from trainer import Trainer
from dataset import SegmentationDataset
import os

def get_sorted_PNGs(path):
    fpaths = [path + fname for fname in os.listdir(path) if fname.endswith(".png")]
    fpaths.sort()
    return fpaths

if __name__ == "__main__":
    print("Hello World!")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network_params = {
        "enc_channels": (3, 16, 32, 64),
        "dec_channels": (64, 32, 16),
        "num_classes": 1,
        "out_size": (500, 500),
    }

    train_params = {
        "num_epochs": 100,
        "batch_size": 16,
        "learning_rate": 1e-3,
    }

    #img_paths = get_sorted_PNGs("/workspaces/hackathon_bern_23/grid/")
    mask_paths=get_sorted_PNGs("/workspaces/hackathon_bern_23/mask/")
    img_paths = [os.path.join("/workspaces/hackathon_bern_23/grid/", os.path.basename(mask_path).replace("_mask","")) for mask_path in mask_paths]

    dataset = SegmentationDataset(img_paths=img_paths, mask_paths=mask_paths)


    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    trainer = Trainer(network_params=network_params, device=device)
    trainer.train(train_dataset=train_dataset, test_dataset=test_dataset, train_params=train_params)
    print("Done!")
