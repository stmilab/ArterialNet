import torch
import torch.nn as nn
import pdb


def conv_calc(
    input_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
):
    """
    conv_calc Calculation of output size of convolutional layer

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d

    Args:
        input_size (int): length of input
        kernel_size (int): Size of the convolving kernel
        stride (int): Stride of the convolution
        padding (int): Padding added to both sides of the input.
        dilation (int): Spacing between kernel elements.

    Returns:
        int: Output shape of the given convolutional layer setup
    """
    return (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


class Feature_Extractor(nn.Module):
    def __init__(
        self,
        input_size,
        num_channels,
        output_size=256,
        norm_layer=nn.BatchNorm1d,
    ):
        super(Feature_Extractor, self).__init__()
        # Always use batchnorm since this is better for training
        self.output_size = output_size
        # Start of waveform dilation
        middle_layer_size = min(4, num_channels * 2)
        self.waveform_dilation = nn.Sequential(
            norm_layer(num_channels),
            nn.Conv1d(
                in_channels=num_channels,
                out_channels=middle_layer_size,
                kernel_size=9,
                stride=1,
                padding=0,
                dilation=2,
            ),
            nn.ReLU(inplace=True),
            norm_layer(middle_layer_size),
            nn.Conv1d(
                in_channels=middle_layer_size,
                out_channels=8,
                kernel_size=3,
                stride=1,
                padding=0,
                dilation=2,
            ),
            nn.ReLU(inplace=True),
        )
        self.convlayers_out_size = conv_calc(
            conv_calc(
                input_size=input_size,
                kernel_size=9,
                stride=1,
                padding=0,
                dilation=2,
            ),
            kernel_size=3,
            stride=1,
            padding=0,
            dilation=2,
        )
        # Start of fclayers -> the MLP part of feature extraction layers
        self.fclayers = nn.Sequential(
            nn.Linear(self.convlayers_out_size * 8, 1024),
            nn.LeakyReLU(inplace=True),
            norm_layer(1024),
            nn.Linear(1024, 512),
            nn.LeakyReLU(inplace=True),
            norm_layer(512),
            nn.Linear(512, 256),
        )

    def forward(self, x):
        # Waveform Dilation
        x = self.waveform_dilation(x)
        x = x.view(-1, self.convlayers_out_size * 8)
        x = self.fclayers(x)
        #
        return x.view(-1, 1, self.output_size)


class ArterialNet(nn.Module):
    # TODO: adding layer and group norm? maybe?
    def __init__(
        self,
        input_size,
        num_channels,
        output_size,
        use_norm="batch",
        add_morph=False,
        trained_model=None,
    ):
        super(ArterialNet, self).__init__()
        # Start of convlayers -> the conv part of feature extraction layers
        self.add_morph = add_morph
        normalization_dict = {
            "batch": nn.BatchNorm1d,
            "instance": nn.InstanceNorm1d,
        }
        if use_norm not in normalization_dict.keys():
            raise ValueError(f"You selected invalid: {use_norm}")
        norm_layer = normalization_dict[use_norm]
        self.norm_layer = norm_layer
        self.feature_extractor = Feature_Extractor(
            input_size=input_size,
            num_channels=num_channels,
            norm_layer=norm_layer,
        )
        if trained_model is None:
            raise ValueError("Must provide a trained seq2seq model!!!")
        self.trained_model = trained_model

        # Concatenating the predicted features with the morphology features
        if self.add_morph:
            self.last_layers = nn.Sequential(
                norm_layer(1),
                nn.Linear(256 + 5, 512),
                nn.ReLU(inplace=True),
                norm_layer(1),
                nn.Linear(512, output_size),
                nn.ReLU(inplace=True),
            )
        else:
            self.last_layers = nn.Sequential(
                norm_layer(1),
                nn.Linear(256, output_size),
                nn.ReLU(inplace=True),
            )

    def forward(self, x, morph=None):
        # bioz to ppg feature extraction layers
        x = self.feature_extractor(x)
        # Now feeding that into the pre-trained model
        x = self.trained_model(x)
        if self.add_morph:
            # Adding morphology features
            morph = morph.view(len(morph), 1, -1)
            morph = self.norm_layer(morph.shape[1])(morph)
            x = torch.cat((x, morph), dim=2)
        x = self.last_layers(x)
        return x.view(len(x), 1, -1)


def main():
    import pdb

    print("This is arterialnet.py, only used for testing!!!")
    # X_data is a 3D torch tensor array of [batch, waveform_channel, waveform_length]
    x_data = torch.load(
        "/home/grads/s/siconghuang/REx_candidate_torch_tensors/mimic_patient_99659_ppg.pt"
    )
    # y_data is a 3D torch tensor array of [batch, waveform_channel, waveform_length]
    y_data = torch.load(
        "/home/grads/s/siconghuang/REx_candidate_torch_tensors/mimic_patient_99659_abp.pt"
    )
    # Only using first 32 samples to test arterialnet dimension
    x_data = x_data[:32, :, :]
    y_data = y_data[:32, :, :]
    # Loading the Model and testing the dimension
    from transformer_model import TransformerModel

    anet = ArterialNet(
        input_size=x_data.shape[2],
        num_channels=x_data.shape[1],
        output_size=y_data.shape[2],
        use_norm="instance",
        trained_model=TransformerModel(),
    )
    pred = anet(x_data)
    print("shape matched") if pred.shape == y_data.shape else print("Shape not matched")
    pdb.set_trace()


if __name__ == "__main__":
    # Test the model
    main()
