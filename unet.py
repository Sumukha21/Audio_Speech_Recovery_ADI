import torch
from unet_utils import DownsampleBlock, UpsampleBlock, TSConv


class Unet(torch.nn.Module):
    def __init__(self, encoder_channels_list, decoder_channels_list, conv_type="basic"):
        """
        Class for creating a unet model. 
        :param encoder_channels_list
               type: List
               It indicates the number of channels for each layer in the encoder with the length of the list 
               indicating the number of layers needed in the encoder.
        :param decoder_channels_list
               type: List
               It indicates the number of channels for each layer in the decoder with the length of the list 
               indicating the number of layers needed in the decoder.
        :param conv_type
               type: string
               Currently supports two values ("basic", "time_shifted"). It indicates the type of convolution 
               operation to use in the model.
               "basic" uses the default pytorch Conv2D class for convoution operation. "time_shifted" uses a 
               custom convolution layer which shifts the input channels and uses a 1D kernel while maintaining 
               the same receptive field.
        """
        super(Unet, self).__init__()
        assert len(encoder_channels_list) == len(decoder_channels_list) + 1, f"""The channels
          list of encoder and decoder should differ by only 1 in a Unet architecture. For example: 
          encoder_channels_list = [4, 4, 4, 8, 8, 16] and decoder_channels_list = [8, 8, 4, 4, 4].
          The lists provided by user: {encoder_channels_list} and {decoder_channels_list}"""
        assert not (False in [True if i % 4 == 0 else False for i in encoder_channels_list[1:]] and conv_type=="time_shifted"), f"""
        When using time_shifted convolution operation, the encoder channels shoud be divisible by 4. Provided encoder
        list: {encoder_channels_list} """
        assert not (False in [True if i % 4 == 0 else False for i in decoder_channels_list] and conv_type=="time_shifted"), f"""
        When using time_shifted convolution operation, the decoder channels shoud be divisible by 4. Provided encoder
        list: {encoder_channels_list} """
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
        Forward propagation of input through the encoder-decoder structure of the unet model.
        :param x
               type: 4D Tensor (bs, ch, h, w)
               4D tensor input to the model for encoding and decoding.
        :return 
            The decoded 4D tensor (bs, ch, h, w) from the model. 
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
1. Add assert statements to verify length of encoder and decoder channel lists - Done
2. Add assert statement to verify the values of encoder, decoder channel values - Done
3. Verification of why TSConv works - Done
4. Comments and docstrings in code - Done
5. Testing
6. Documentation
"""

