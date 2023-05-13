import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop


class Block(nn.Module):
    def __init__(self, inChannels, outChannels):
        super().__init__()
        # store convolution and ReLU layers
        self.conv1 = nn.Conv2d(
            in_channels=inChannels, out_channels=outChannels, kernel_size=3
        )
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=outChannels, out_channels=outChannels, kernel_size=3
        )

    def forward(self, x):
        # apply Conv -> ReLU -> Conv block to inputs
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        return out


class Encoder(nn.Module):
    def __init__(self, channels=(3, 16, 32, 64)):
        super().__init__()
        # store encoder blocks and maxpooling layer
        self.encBlocks = nn.ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)]
        )
        self.maxPool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        # initialize empty list to store intermediate outputs
        blockOutputs = []

        # loop through the encoder blocks
        for block in self.encBlocks:
            # pass inputs through current encoder block
            # store outputs and apply maxpooling on output
            x = block(x)
            blockOutputs.append(x)
            x = self.maxPool(x)

        return blockOutputs


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
        self.decBlocks = nn.ModuleList(
            [
                Block(inChannels=channels[i], outChannels=channels[i + 1])
                for i in range(len(channels) - 1)
            ]
        )

    def forward(self, x, encFeatures):
        # loop through number of channels
        for i in range(len(self.channels) - 1):
            # pass inputs through upsampler blocks
            x = self.upconvs[i](x)

            # crop the current features from the encoder blocks,
            # concatenate them with the current upsampled features,
            # and pass the concatenated output through the current decoder block
            encFeat = self.crop(encFeatures[i], x)
            x = torch.cat([x, encFeat], dim=1)
            x = self.dec_blocks[i](x)
        return x

    def crop(self, encFeatures, x):
        # grab the dimensions of the inputs, and crop the encoder
        # features to match the dimensions
        _, _, H, W = x.shape
        encFeatures = CenterCrop([H, W])(encFeatures)
        # return the cropped features
        return encFeatures


class Unet(nn.Module):
    def __init__(
        self,
        encChannels=(3, 16, 32, 64),
        decChannels=(64, 32, 16),
        numClasses=1,
        retainDim=True,
        outSize=(64, 64),
    ):
        super().__init__()
        # initialize the encoder and decoder
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)

        # initialize regression head and store class variables
        self.head = nn.Conv2d(decChannels[-1], numClasses, 1)
        self.retainDim = retainDim
        self.outSize = outSize

    def forward(self, x):
        # grab features from encoder
        encFeatures = self.encoder(x)

        # pass the encoder features through decoder making sure that
        # their dimensions are suited for concatenation
        decFeatures = self.decoder(encFeatures[::-1][0], encFeatures[::-1][1:])

        # pass decoder features through regression head to obtain segmentation mask
        # obtain segmentation mask
        map = self.head(decFeatures)

        # check to see if we are retaining the original output
        # dimensions and if so, then resize the output to match them
        if self.retainDim:
            map = torch.functional.interpolate(map, self.outSize)

        # return segmentation map
        return map