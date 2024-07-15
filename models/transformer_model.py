import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size=256,
        output_size=256,
        hidden_size=256,
        num_layers=2,
        num_heads=4,
        dropout_prob=0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

        # Input embedding layer
        self.embedding = nn.Linear(input_size, hidden_size)

        # Transformer encoder and decoder layers
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=num_heads
        )
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads
        )

        # Transformer encoder and decoder
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)

        # Output linear layer
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, src):
        # src: [batch_size, 1, input_size]
        # trg: [batch_size, 1, output_size]
        trg = src
        # Input embedding
        src = self.embedding(src.transpose(0, 1)).transpose(
            0, 1
        )  # [batch_size, 1, hidden_size]
        trg = self.embedding(trg.transpose(0, 1)).transpose(
            0, 1
        )  # [batch_size, 1, hidden_size]

        # Transformer encoding
        enc_output = self.encoder(src)  # [batch_size, 1, hidden_size]

        # Transformer decoding
        dec_output = self.decoder(trg, enc_output)  # [batch_size, 1, hidden_size]

        # Output linear layer
        output = self.linear(dec_output)  # [batch_size, 1, output_size]

        return output
