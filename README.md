# Audio_Speech_Recovery_ADI

This repository provides implementation of the popular Unet architecture with the addition of a custom convolution operation.

## Installation
The repository can be cloned using 
`"git clone https://github.com/Sumukha21/Audio_Speech_Recovery_ADI.git"`.

The following steps can be used to create an environment requried to use the repository:
1. `conda env create -f environment.yml`
2. `conda activate audio_reconstruction`
3. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

## Usage
The repository contains Unet implementation with two options for performing the convolution operation -
"basic" and "time_shifted".
- `basic: Uses the standard convolution operation from pytorch with 3x3 kernels`
- `time_shifted: Uses the custom TSConv operation using 3x1 kernels having a 3x3 receptive field of the input`

The Unet model can be run following the below example commands:

1. `python .\main.py --encoder_channels_list 4,4,4,8,8,16 --decoder_channels_list 8,8,4,4,4 --conv_type "time_shifted"`

2. `python .\main.py --encoder_channels_list 4,4,4,8,8,16 --decoder_channels_list 8,8,4,4,4 --conv_type "basic"`

In addition to this, the repository can be tested using the pytest test cases defined in `test_cases.py` using the following command:

`pytest` (Assuming that you are in the project repository)

The coverage of the test cases can be observed using:

`coverage report -m .\test_cases.py`

## Time-shifted convolution vs Standard Convolution operation
The time-shifted convolution operation facilitates convolution with a receptive field of approximately 3x3 when using only a 1D kernel. This reduces the number of  operations performed and consequently the number of trainable parameters as well. Below illustrates an example of the same:

![Standard Convolution vs tsConv](./images/standard_convolution%20vs%20tsConv.png)

