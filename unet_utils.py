import torch


class DownsampleBlock:
    def __init__(self, conv_in_channels, conv_out_channels, conv_kernel_size):
        self.maxpool = torch.nn.MaxPool2d(kernel_size=(2, 1))
        self.conv_layer1 = torch.nn.Conv2d(in_channels=conv_in_channels,
                                          out_channels=conv_out_channels,
                                          kernel_size=conv_kernel_size,
                                          stride=(1, 1),
                                          padding="same")
        self.conv_layer2 = torch.nn.Conv2d(in_channels=conv_out_channels,
                                          out_channels=conv_out_channels,
                                          kernel_size=conv_kernel_size,
                                          stride=(1, 1),
                                          padding="same")
    
    def __call__(self, x):
        x = self.maxpool(x)
        x = torch.nn.ReLU()(self.conv_layer1(x))
        x = torch.nn.ReLU()(self.conv_layer2(x))
        return x


class UpsampleBlock:
    def __init__(self, conv_in_channels, conv_out_channels, conv_kernel_size):        
        self.upsample = torch.nn.Upsample(scale_factor=(2, 1))
        self.conv_layer1 = torch.nn.Conv2d(in_channels=conv_in_channels,
                                           out_channels=conv_out_channels,
                                           kernel_size=conv_kernel_size,
                                           stride=(1, 1),
                                           padding="same")
        self.conv_layer2 = torch.nn.Conv2d(in_channels=conv_out_channels,
                                           out_channels=conv_out_channels,
                                           kernel_size=conv_kernel_size,
                                           stride=(1, 1),
                                           padding="same")

    def __call__(self, x , y):
        x = self.upsample(x)
        x = torch.concatenate((x, y), dim=1)
        x = torch.nn.ReLU()(self.conv_layer1(x))
        x = torch.nn.ReLU()(self.conv_layer2(x))
        return x

