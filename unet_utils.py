import torch


class DownsampleBlock:
    def __init__(self, conv_in_channels, conv_out_channels, conv_type="basic"):
        self.maxpool = torch.nn.MaxPool2d(kernel_size=(2, 1))
        if conv_type == "basic":
            self.conv_layer1 = torch.nn.Conv2d(in_channels=conv_in_channels,
                                               out_channels=conv_out_channels,
                                               kernel_size=(3, 3),
                                               stride=(1, 1),
                                               padding="same")
            self.conv_layer2 = torch.nn.Conv2d(in_channels=conv_out_channels,
                                               out_channels=conv_out_channels,
                                               kernel_size=(3, 3),
                                               stride=(1, 1),
                                               padding="same")
        elif conv_type == "time_shifted":
            self.conv_layer1 = TSConv(input_channels=conv_in_channels,
                                      output_channels=conv_out_channels)
            self.conv_layer2 = TSConv(input_channels=conv_out_channels,
                                      output_channels=conv_out_channels)
        else:
            raise NotImplementedError("""Right now only two types of convolution layers 
                                      are supported 'basic' and 'time_shifted'.
                                      Read the class docstring for more info.""")

    def __call__(self, x):
        x = self.maxpool(x)
        x = torch.nn.ReLU()(self.conv_layer1(x))
        x = torch.nn.ReLU()(self.conv_layer2(x))
        return x


class UpsampleBlock:
    def __init__(self, conv_in_channels, conv_out_channels, conv_type="basic"):        
        self.upsample = torch.nn.Upsample(scale_factor=(2, 1))
        if conv_type == "basic":
            self.conv_layer1 = torch.nn.Conv2d(in_channels=conv_in_channels,
                                               out_channels=conv_out_channels,
                                               kernel_size=(3, 3),
                                               stride=(1, 1),
                                               padding="same")
            self.conv_layer2 = torch.nn.Conv2d(in_channels=conv_out_channels,
                                               out_channels=conv_out_channels,
                                               kernel_size=(3, 3),
                                               stride=(1, 1),
                                               padding="same")
        elif conv_type == "time_shifted":
            self.conv_layer1 = TSConv(input_channels=conv_in_channels,
                                      output_channels=conv_out_channels)
            self.conv_layer2 = TSConv(input_channels=conv_out_channels,
                                      output_channels=conv_out_channels)
        else:
            raise NotImplementedError("""Right now only two types of convolution layers 
                                      are supported 'basic' and 'time_shifted'.
                                      Read the class docstring for more info.""")

    def __call__(self, x , y):
        x = self.upsample(x)
        x = torch.concatenate((x, y), dim=1)
        x = torch.nn.ReLU()(self.conv_layer1(x))
        x = torch.nn.ReLU()(self.conv_layer2(x))
        return x


class TSConv(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        super(TSConv, self).__init__()
        self.conv_layer1 = torch.nn.Conv2d(in_channels=input_channels,
                                           out_channels=output_channels,
                                           kernel_size=(3, 1),
                                           stride=(1, 1),
                                           padding="same")
        
    def forward(self, x):
        x_dynamic = x[:, :x.shape[1] // 2, :, :]
        x_dynamic_forward = torch.concat([torch.zeros(1, (x_dynamic.shape[1] // 2), x_dynamic.shape[2], 1), x_dynamic[:, :(x_dynamic.shape[1] // 2), :, :-1]], dim=-1)
        x_dynamic_backward = torch.concat([x_dynamic[:, (x_dynamic.shape[1] // 2):, :, 1:], torch.zeros(1, (x_dynamic.shape[1] // 2), x_dynamic.shape[2], 1)], dim=-1)
        x = torch.concat([x_dynamic_forward, x_dynamic_backward, x[:, (x.shape[1] // 2):, :, :]], dim=1)
        x = self.conv_layer1(x)
        return x 
    