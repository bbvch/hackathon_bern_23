import torch
from unet import Unet
from torch.utils.data import DataLoader
import tqdm
import time


class Trainer:
    def __init__(self, network_params, pretrained_model_path=None, device="cuda"):
        enc_channels = network_params["enc_channels"]
        dec_channels = network_params["dec_channels"]
        num_classes = network_params["num_classes"]
        out_size = network_params["out_size"]

        self.model = Unet(
            encChannels=enc_channels,
            decChannels=dec_channels,
            numClasses=num_classes,
            retainDim=True,
            outSize=out_size,
        )

        self.device = device

        if pretrained_model_path:
            self.model.load_state_dict(torch.load(pretrained_model_path))
        self.model.to(device=self.device)

    def train(self, train_dataset, test_dataset, train_params):
        num_epochs = train_params["num_epochs"]
        batch_size = train_params["batch_size"]
        learning_rate = train_params["learning_rate"]

        optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=learning_rate
        )
        optimizer.zero_grad()
        loss = torch.nn.BCEWithLogitsLoss()

        trainSteps = len(train_dataset) // batch_size
        testSteps = len(test_dataset) // batch_size

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        startTime = time.time()
        print("[INFO] Training started: {:.2f}s".format(startTime))

        for epoch in range(num_epochs):
            # set model into training mode
            self.model.train()

            # initialize total training and validation loss
            totalTrainLoss, totalTestLoss = 0, 0

            # loop over training set
            for x, y in train_dataloader:
                # send input to device
                x, y = x.to(self.device), y.to(self.device)

                # perform forward pass and calculate trainig loss
                pred = self.model(x)
                loss = loss(pred, y)

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
                    totalTestLoss += loss(pred, y)

            # calculate average training and validation loss
            avgTrainLoss = totalTrainLoss / trainSteps
            avgTestLoss = totalTestLoss / testSteps

            # print the model training and validation information
            print("[INFO] EPOCH: {}/{}".format(epoch + 1, num_epochs))
            print(
                "Train loss: {:.4f}, Test loss: {:.4f}".format(
                    avgTrainLoss, avgTestLoss
                )
            )

        endTime = time.time()
        print("[INFO] Total training time: {:.2f}s".format(endTime - startTime))
