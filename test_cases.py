import torch
from unet_utils import TSConv
from unet import Unet


def test_basic_unet():
    encoder_channels_list = [4, 4, 4, 8, 8, 16]
    decoder_channels_list = [8, 8, 4, 4, 4]
    model_basic = Unet(encoder_channels_list=encoder_channels_list,
                       decoder_channels_list=decoder_channels_list,
                       conv_type="basic")
    model_basic_trainable_params = sum(p.numel() for p in model_basic.parameters() if p.requires_grad)
    print("Number of model parameters in basic Unet: ", model_basic_trainable_params)
    T = 10
    sample_input = torch.rand((1, 1, 256, T))
    output_basic = model_basic(x=sample_input)
    assert output_basic.shape == sample_input.shape, """Output shape of the Time shifted Unet model
                                                        should match the shape of the input."""
    

def test_ts_unet():    
    encoder_channels_list = [4, 4, 4, 8, 8, 16]
    decoder_channels_list = [8, 8, 4, 4, 4]
    model_ts = Unet(encoder_channels_list=encoder_channels_list,
                   decoder_channels_list=decoder_channels_list,
                   conv_type="time_shifted")
    model_ts_trainable_params = sum(p.numel() for p in model_ts.parameters() if p.requires_grad)
    print("Number of model parameters in time-shifted Unet: ", model_ts_trainable_params)
    T = 10
    sample_input = torch.rand((1, 1, 256, T))
    output_ts = model_ts(x=sample_input)
    assert output_ts.shape == sample_input.shape, """Output shape of the Time shifted Unet model
                                                     should match the shape of the input."""


def test_tsconv_with_ones_kernel():
    import torch.nn.init as init
    ts_conv = TSConv(4, 4)
    init.ones_(ts_conv.conv_layer1.weight)
    init.zeros_(ts_conv.conv_layer1.bias)
    x = torch.arange(1, 37, 1).reshape(1, 4, 3, 3)
    x = x.to(torch.float32)
    ts_output = ts_conv(x)
    assert not False in ts_output[:, 0, :, :].to(torch.int32) == torch.Tensor([[125, 136, 115],
                                                                               [201, 222, 186],
                                                                               [143, 160, 133]])
