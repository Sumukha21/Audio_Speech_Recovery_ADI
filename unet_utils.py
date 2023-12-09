import torch


class DownsampleBlock:
    def __init__(self, conv_in_channels, conv_out_channels, conv_type="basic"):
        """
        Downsampling block of the encoder in the Unet Model. Currently the convolution operation
        is performed with a fixed kernel size, stride and padding type. Downsampling scale is fixed at (2, 1).
        :param conv_in_channels
               type: int
               Number of channels in the input.
        :param conv_out_channels
               type: int
               Number of channels required in the output.
        :param conv_type 
               type: string
               Currently supports two values ("basic", "time_shifted"). It indicates the type of convolution 
               operation to use in the model.
               "basic" uses the default pytorch Conv2D class for convoution operation. "time_shifted" uses a 
               custom convolution layer which shifts the input channels and uses a 1D kernel while maintaining 
               the same receptive field. 
        """
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
        """
        Forward propoagation of input through the encoder downsampling block.
        :param x
               type: 4D Tensor (bs, ch1, h, w)
               4D tensor input to the model for convolution and downsampling operation.
        :return 
            The convolved and downsampled 4D tensor (bs, ch2, h//2, w).
        """
        x = self.maxpool(x)
        x = torch.nn.ReLU()(self.conv_layer1(x))
        x = torch.nn.ReLU()(self.conv_layer2(x))
        return x


class UpsampleBlock:
    def __init__(self, conv_in_channels, conv_out_channels, conv_type="basic"):     
        """
        Upsampling block of the decoder in the Unet Model. Currently the convolution operation
        is performed with a fixed kernel size, stride and padding type. Upsampling scale is fixed at (2, 1).
        :param conv_in_channels
               type: int
               Number of channels in the input.
        :param conv_out_channels
               type: int
               Number of channels required in the output.
        :param conv_type 
               type: string
               Currently supports two values ("basic", "time_shifted"). It indicates the type of convolution 
               operation to use in the model.
               "basic" uses the default pytorch Conv2D class for convoution operation. "time_shifted" uses a 
               custom convolution layer which shifts the input channels and uses a 1D kernel while maintaining 
               the same receptive field. 
        """   
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
        """
        Forward propoagation of input through the decoder upsampling block.
        :param x
               type: 4D Tensor (bs, ch1, h, w)
               4D tensor input to the model for convolution and upsampling operation.
        :param y
               type: 4D Tensor (bs, ch2, h*2, w)
               4D tensor output of convolution and upsampling operations.
        """
        x = self.upsample(x)
        x = torch.concatenate((x, y), dim=1)
        x = torch.nn.ReLU()(self.conv_layer1(x))
        x = torch.nn.ReLU()(self.conv_layer2(x))
        return x


class TSConv(torch.nn.Module):
    def __init__(self, input_channels, output_channels):
        """
        Custom layer for performing time shifted convolution operation. The following operations are 
        performed for the provided input:
        1. Split the input along the channels into two equal parts (dynamic and static)
        2. Shift the first half of the dynamic channels forward by 1 timestep, pad the start of the timesteps
        and truncate the rolled out timestep (dynamic_forward).
        3. Shift the second half of the dynamic channels backward by 1 timestep, pad the end of the timesteps
        and truncate the rolled out timestep (dynamic_backward).
        4. Concatenate [dynamic_forward, dynamic_backard, static] to obtain the tensor back to original input
        shape.
        5. Perform 1D convolution using a kernel size (3, 1), fixed stride and padding type.
        """
        super(TSConv, self).__init__()
        self.conv_layer1 = torch.nn.Conv2d(in_channels=input_channels,
                                           out_channels=output_channels,
                                           kernel_size=(3, 1),
                                           stride=(1, 1),
                                           padding="same")
        
    def forward(self, x):
        """
        Performs time shifted colvolution on the input x providing a receptive field of 3x3 pixels/input array 
        using a 1D kernel for convoluition.
        :param x
               type: 4D tensor (bs, ch1, h, w)
               4D tensor input on which time shifted convolution needs to be performed.
        :return 
            Returns the time shifted and convolved 4D tensor (bs, ch2, h, w).
        """
        x_dynamic = x[:, :x.shape[1] // 2, :, :]
        x_dynamic_forward = torch.concat([torch.zeros(1, (x_dynamic.shape[1] // 2), x_dynamic.shape[2], 1), x_dynamic[:, :(x_dynamic.shape[1] // 2), :, :-1]], dim=-1)
        x_dynamic_backward = torch.concat([x_dynamic[:, (x_dynamic.shape[1] // 2):, :, 1:], torch.zeros(1, (x_dynamic.shape[1] // 2), x_dynamic.shape[2], 1)], dim=-1)
        x = torch.concat([x_dynamic_forward, x_dynamic_backward, x[:, (x.shape[1] // 2):, :, :]], dim=1)
        x = self.conv_layer1(x)
        return x 
    