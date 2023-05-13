import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # store convolution and ReLU layers
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3
        )

    def forward(self, x):
        # apply Conv -> ReLU -> Conv block to inputs
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, channels=(3, 16, 32, 64)):
        super().__init__()
        # store encoder blocks and maxpooling layer
        self.enc_blocks = nn.ModuleList(
            [
                Block(in_channels=channels[i], out_channels=channels[i + 1])
                for i in range(len(channels) - 1)
            ]
        )
        self.maxPool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        # initialize empty list to store intermediate outputs
        block_outputs = []

        # loop through the encoder blocks
        for block in self.enc_blocks:
            # pass inputs through current encoder block
            # store outputs and apply maxpooling on output
            x = block(x)
            block_outputs.append(x)
            x = self.maxPool(x)

        return block_outputs


class Decoder(nn.Module):
    def __init__(self, channels=(64, 32, 16)):
        super().__init__()
        # initialize number of channels, upsampler blocks, and decoder blocks
        self.channels = channels
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=2,
                    stride=2,
                )
                for i in range(len(channels) - 1)
            ]
        )
        self.dec_blocks = nn.ModuleList(
            [
                Block(in_channels=channels[i], out_channels=channels[i + 1])
                for i in range(len(channels) - 1)
            ]
        )

    def forward(self, x, enc_features):
        # loop through number of channels
        for i in range(len(self.channels) - 1):
            # pass inputs through upsampler blocks
            x = self.upconvs[i](x)

            # crop the current features from the encoder blocks,
            # concatenate them with the current upsampled features,
            # and pass the concatenated output through the current decoder block
            enc_feat = self.crop(enc_features[i], x)
            x = torch.cat([x, enc_feat], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, enc_features, x):
        # grab the dimensions of the inputs, and crop the encoder
        # features to match the dimensions
        _, _, H, W = x.shape
        enc_features = CenterCrop([H, W])(enc_features)
        # return the cropped features
        return enc_features


class Unet(nn.Module):
    def __init__(
        self,
        enc_channels=(3, 16, 32, 64),
        dec_channels=(64, 32, 16),
        num_classes=1,
        retain_dim=True,
        out_size=(64, 64),
    ):
        super().__init__()
        # initialize the encoder and decoder
        self.encoder = Encoder(enc_channels)
        self.decoder = Decoder(dec_channels)

        # initialize regression head and store class variables
        self.head = nn.Conv2d(
            in_channels=dec_channels[-1], out_channels=num_classes, kernel_size=1
        )
        self.retain_dim = retain_dim
        self.out_size = out_size

    def forward(self, x):
        # grab features from encoder
        enc_features = self.encoder(x)

        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        dec_features = self.decoder(enc_features[::-1][0], enc_features[::-1][1:])

        # pass decoder features through regression head to obtain segmentation mask
        # obtain segmentation mask
        map = self.head(dec_features)

        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        if self.retain_dim:
            map = torch.nn.functional.interpolate(map, self.out_size)

        # return segmentation map
        return map
