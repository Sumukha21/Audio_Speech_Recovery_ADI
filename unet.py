import torch
from unet_utils import DownsampleBlock, UpsampleBlock, TSConv


class Unet(torch.nn.Module):
    def __init__(self, encoder_channels_list, decoder_channels_list, conv_type="basic"):
        """
        Define the layer blocks since structure is going to remain the same
        """
        super(Unet, self).__init__()
        self.conv_input1 = torch.nn.Conv2d(in_channels=1,
                                           out_channels=encoder_channels_list[0],
                                           kernel_size=(3, 3),
                                           stride=(1, 1),
                                           padding="same")
        if conv_type == "basic":
            self.conv_input2 = torch.nn.Conv2d(in_channels=encoder_channels_list[0],
                                               out_channels=encoder_channels_list[0],
                                               kernel_size=(3, 3),
                                               stride=(1, 1),
                                               padding="same")
            self.final_conv = torch.nn.Conv2d(in_channels=decoder_channels_list[-1],
                                              out_channels=1,
                                              kernel_size=(3, 3),
                                              stride=(1, 1),
                                              padding="same")
        
        elif conv_type == "time_shifted":
            self.conv_input2 = TSConv(input_channels=encoder_channels_list[0],
                                      output_channels=encoder_channels_list[0])
            self.final_conv = TSConv(input_channels=decoder_channels_list[-1],
                                      output_channels=1)
        else:
            raise NotImplementedError("""Right now only two types of convolution layers 
                                      are supported 'basic' and 'time_shifted'.
                                      Read the class docstring for more info.""")
        self.encoder_blocks = []
        input_conv_channel_size = encoder_channels_list[0]
        for i in encoder_channels_list[1:]:
            self.encoder_blocks.append(DownsampleBlock(conv_in_channels=input_conv_channel_size,
                                                       conv_out_channels=i, conv_type=conv_type))
            input_conv_channel_size = i
        self.decoder_blocks = []
        input_conv_channel_size = encoder_channels_list[-1]
        for i in range(len(decoder_channels_list)):
            self.decoder_blocks.append(UpsampleBlock(conv_in_channels=input_conv_channel_size + encoder_channels_list[~(i+1)],
                                                     conv_out_channels=decoder_channels_list[i], conv_type=conv_type))
            input_conv_channel_size = decoder_channels_list[i]
    
    def forward(self, x):
        """
        Define the actual flow here
        """
        x = torch.nn.ReLU()(self.conv_input1(x))
        x = torch.nn.ReLU()(self.conv_input2(x))
        encoder_outputs = []
        encoder_outputs.append(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            encoder_outputs.append(x)
        for i in range(len(self.decoder_blocks)):
            encoder_output_to_concat = encoder_outputs[~(i + 1)]
            x = self.decoder_blocks[i](x, encoder_output_to_concat)
        x = self.final_conv(x)
        return x

"""
ToDO:
1. Add assert statements to verify length of encoder and decoder channel lists
2. Remove input_channel_size and kernel_size as parameters - Done
3. Add assert statement to verify the values of encoder, decoder channel values
"""

