import torch
import argparse
from unet import Unet

def channels_list(input_list):
    return list(map(int, input_list.split(",")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Unet instantiater")
    parser.add_argument("--encoder_channels_list", required=True, 
                        type=channels_list, help="""It indicates the number of 
                        channels for each layer in the encoder with the length of the list
                        indicating the number of layers needed in the encoder.""")
    parser.add_argument("--decoder_channels_list", required=True, 
                        type=channels_list, help="""It indicates the number of
                        channels for each layer in the decoder with the length of the list
                        indicating the number of layers needed in the decoder.""")
    parser.add_argument("--conv_type", type=str, default="time_shifted", required=False, help="""It
                        indicates the type of convolution operation to use in the model.""") 
    args = parser.parse_args()
    T = 10
    sample_input = torch.rand((1, 1, 256, T))
    model_basic = Unet(encoder_channels_list=args.encoder_channels_list,
                       decoder_channels_list=args.decoder_channels_list,
                       conv_type=args.conv_type)
    model_basic_trainable_params = sum(p.numel() for p in model_basic.parameters() if p.requires_grad)
    print(f"Number of model parameters in {args.conv_type} Unet: ", model_basic_trainable_params)
    output_basic = model_basic(x=sample_input)
    print("Output shape: ", output_basic.shape)
