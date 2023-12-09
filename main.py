import torch
from unet import Unet


if __name__ == "__main__":
    encoder_channels_list = [4, 4, 4, 8, 8, 16]
    decoder_channels_list = [8, 8, 4, 4, 4]
    conv_type = "time_shifted"  # "basic" 
    T = 10
    sample_input = torch.rand((1, 1, 256, T))
    model_basic = Unet(encoder_channels_list=encoder_channels_list,
                       decoder_channels_list=decoder_channels_list,
                       conv_type=conv_type)
    model_basic_trainable_params = sum(p.numel() for p in model_basic.parameters() if p.requires_grad)
    print("Number of model parameters in basic Unet: ", model_basic_trainable_params)
    output_basic = model_basic(x=sample_input)
    print("Output shape: ", output_basic.shape)
