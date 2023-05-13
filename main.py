import torch
from UNET.trainer import Trainer
from UNET.dataset import SegmentationDataset
from UNET.utils import load_config_file, get_sorted_PNGs
import os


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Model will be running on {}.".format(device))

    project_dir = os.path.dirname(os.path.realpath(__file__))
    unet_dir = os.path.join(project_dir, "UNET")

    print(project_dir)
    print(unet_dir)


    # load configuration from json file
    json_fpath = os.path.join(unet_dir, "config.json")
    params = load_config_file(json_fpath)
    network_params = params["network_params"]
    train_params = params["train_params"]

    # define data directory paths
    image_dir = os.path.join(project_dir, params["image_dir"])
    mask_dir = os.path.join(project_dir, params["mask_dir"])

    # path to save trained model and the loss plot
    train_params["result_dir"] = os.path.join(project_dir, train_params["result_dir"])
    os.makedirs(train_params["result_dir"], exist_ok=True)

    mask_paths = get_sorted_PNGs(mask_dir)
    img_paths = [
        os.path.join(
            image_dir,
            os.path.basename(mask_path).replace("_mask", ""),
        )
        for mask_path in mask_paths
    ]

    # Prepare dataset
    dataset = SegmentationDataset(img_paths=img_paths, mask_paths=mask_paths)

    # split into training and test datasets
    train_size = int(params["train_split"] * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # Prepare training
    pretrained_model_path = os.path.join(
        project_dir, train_params["result_dir"], params["pretrained_model_path"]
    )

    print(pretrained_model_path)

    trainer = Trainer(
        network_params=network_params,
        pretrained_model_path=pretrained_model_path,
        device=device,
    )
    trainer.train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_params=train_params,
    )

    # Prepare prediction
    save_dir = params["pred_plots"]
    os.makedirs(save_dir, exist_ok=True)

    # Predict and plot on test set
    trainer.predict(
        pretrained_model_path=pretrained_model_path,
        test_dataset=test_dataset,
        threshold=params["pred_threshold"],
        save_dir=save_dir,
    )
