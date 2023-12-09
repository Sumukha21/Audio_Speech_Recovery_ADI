import torch
from unet_utils import DownsampleBlock, UpsampleBlock


class Unet(torch.nn.Module):
    def __init__(self, input_channel_size, conv_kernel_size, encoder_channels_list, decoder_channels_list):
        """
        Define the layer blocks since structure is going to remain the same
        """
        super(Unet, self).__init__()
        self.conv_input1 = torch.nn.Conv2d(in_channels=input_channel_size,
                                           out_channels=encoder_channels_list[0],
                                           kernel_size=conv_kernel_size,
                                           stride=(1, 1),
                                           padding="same")
        self.conv_input2 = torch.nn.Conv2d(in_channels=encoder_channels_list[0],
                                           out_channels=encoder_channels_list[0],
                                           kernel_size=conv_kernel_size,
                                           stride=(1, 1),
                                           padding="same")
        self.encoder_blocks = []
        input_conv_channel_size = encoder_channels_list[0]
        for i in encoder_channels_list[1:]:
            self.encoder_blocks.append(DownsampleBlock(conv_in_channels=input_conv_channel_size,
                                                       conv_out_channels=i,
                                                       conv_kernel_size=conv_kernel_size))
            input_conv_channel_size = i
        self.decoder_blocks = []
        input_conv_channel_size = encoder_channels_list[-1]
        for i in range(len(decoder_channels_list)):
            self.decoder_blocks.append(UpsampleBlock(conv_in_channels=input_conv_channel_size + encoder_channels_list[~(i+1)],
                                                     conv_out_channels=decoder_channels_list[i],
                                                     conv_kernel_size=conv_kernel_size))
            input_conv_channel_size = decoder_channels_list[i]
        self.final_conv = torch.nn.Conv2d(in_channels=decoder_channels_list[-1],
                                          out_channels=1,
                                          kernel_size=conv_kernel_size,
                                          stride=(1, 1),
                                          padding="same")
        
    
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

