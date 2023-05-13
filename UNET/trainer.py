import torch
from .unet import Unet
from .utils import predict_and_plot
from torch.utils.data import DataLoader
import time
import os
from datetime import timedelta, datetime
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


class Trainer:
    def __init__(self, network_params, pretrained_model_path=None, device="cuda"):
        enc_channels = network_params["enc_channels"]
        dec_channels = network_params["dec_channels"]
        num_classes = network_params["num_classes"]
        out_size = network_params["out_size"]

        self.model = Unet(
            enc_channels=enc_channels,
            dec_channels=dec_channels,
            num_classes=num_classes,
            retain_dim=True,
            out_size=out_size,
        )

        self.device = device
        self.start_epoch = 0

        if pretrained_model_path:
            print("Starting from scratch.")
        elif os.path.exists(pretrained_model_path):
            self.model.load_state_dict(
                torch.load(pretrained_model_path, map_location=device)
            )
            state_dict_name = os.path.basename(pretrained_model_path)
            print("Load weights from {}.".format(state_dict_name))
        else:
            print(f"Starting from scratch because no model was found at {pretrained_model_path}")

        self.model.to(device=self.device)

    def train(self, train_dataset, test_dataset, train_params):
        num_epochs = train_params["num_epochs"]
        batch_size = train_params["batch_size"]
        learning_rate = train_params["learning_rate"]
        log_freq = train_params["log_freq"]
        result_dir = train_params["result_dir"]

        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        loss_function = torch.nn.BCEWithLogitsLoss()
        lossHistory = {"train_loss": [], "test_loss": []}

        trainSteps = len(train_dataset) // batch_size
        testSteps = len(test_dataset) // batch_size

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        startTime = time.time()

        for epoch in range(num_epochs):
            # set model into training mode
            self.model.train()

            # initialize total training and validation loss
            totalTrainLoss, totalTestLoss = 0, 0

            # print epoch start info
            print(
                "\n[INFO] EPOCH: {}/{} started at {}.".format(
                    epoch + 1, num_epochs, datetime.now().strftime("%H:%M:%S")
                )
            )

            # loop over training set
            for x, y in train_dataloader:
                # send input tensors to device
                x, y = x.to(self.device), y.to(self.device)

                # perform forward pass and calculate trainig loss
                pred = self.model(x)
                loss = loss_function(pred, y)

                # zero out previously accumulated gradients
                # perform backpropagation, update model parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # add the loss to the total training loss so far
                totalTrainLoss += loss

            # switch off autograd
            with torch.no_grad():
                # set model in evaluation mode
                self.model.eval()

                # loop over validation set
                for x, y in test_dataloader:
                    # send input to device
                    x, y = x.to(self.device), y.to(self.device)

                    # make predictions and calculate validation loss
                    pred = self.model(x)
                    totalTestLoss += loss_function(pred, y)

            # calculate average training and validation loss
            avgTrainLoss = totalTrainLoss / trainSteps
            avgTestLoss = totalTestLoss / testSteps

            lossHistory["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
            lossHistory["test_loss"].append(avgTestLoss.cpu().detach().numpy())

            # print the model training and validation information
            print(
                "Train loss: {:.4f}, Test loss: {:.4f}".format(
                    avgTrainLoss, avgTestLoss
                )
            )

            if (epoch + 1) % log_freq == 0 or (epoch == num_epochs-1):
                # save model
                torch.save(
                    self.model.state_dict(),
                    f"{result_dir}/epoch-{epoch+1}.model",
                )
                print("Model saved in {}".format(result_dir))

        endTime = time.time()
        print("\n[INFO] Total training time:", timedelta(seconds=endTime - startTime))

        train_loss = lossHistory["train_loss"]
        test_loss = lossHistory["test_loss"]
        x = range(num_epochs)
        plt.plot(x, train_loss, "r", label="training loss")
        plt.plot(x, test_loss, "b", label="validation loss")
        plt.title("Binary Cross-Entropy Loss")
        plt.xlabel("epochs")
        plt.savefig(os.path.join(result_dir, "loss_plot.png"))
        plt.close()

    def predict(self, pretrained_model_path, test_dataset, threshold, save_dir):
        self.model.load_state_dict(
            torch.load(pretrained_model_path, map_location=self.device)
        )
        self.model.to(self.device)

        test_dataloader = DataLoader(test_dataset)
        for i, (x, _) in enumerate(test_dataloader):
            predict_and_plot(
                model=self.model,
                id=i,
                image=x.numpy().squeeze().transpose(),
                save_dir=save_dir,
                threshold=threshold,
                device=self.device,
            )
