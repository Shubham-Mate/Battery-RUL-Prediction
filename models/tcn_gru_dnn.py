import torch

# Defining Encoder and its submodules

class DilatedCausalConv1D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(DilatedCausalConv1D, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation

        self.conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            dilation=self.dilation
        )

    def forward(self, x):
        x = self.conv(x)

        # We only want padding on left side, but the actual padding is on both the sides.
        # So we discard last outputs which are present because of right padding
        x = x[:, :, :-self.padding] if self.padding != 0 else x 
        return x

class TCNBlock(torch.nn.Module):
    def __init__(self, in_channels, intermediate_channels, out_channels, kernel_size_1, kernel_size_2, dilation_1=2, dilation_2=3, dropout=0.2):
        super(TCNBlock, self).__init__()
        self.dilated_conv_1 = DilatedCausalConv1D(in_channels=in_channels, out_channels=intermediate_channels, kernel_size=kernel_size_1, dilation=dilation_1)
        self.dilated_conv_2 = DilatedCausalConv1D(in_channels=intermediate_channels, out_channels=out_channels, kernel_size=kernel_size_2, dilation=dilation_2)

        self.norm1 = torch.nn.BatchNorm1d(intermediate_channels)
        self.norm2 = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout)

        self.residual_conv = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels)


    def forward(self, x):
        # Block 1
        out = self.dilated_conv_1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Block 2
        out = self.dilated_conv_2(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Residual Connection
        res = self.residual_conv(x)
        out = out + res

        return out

class FeatureAttention(torch.nn.Module):
    def __init__(self, in_dim):
        super(FeatureAttention, self).__init__()
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax()
        self.map = torch.nn.Linear(in_dim, 1)

    def forward(self, x):
        out = self.tanh(self.map(x))
        out = out.view(out.size(0), -1)
        alpha = self.softmax(out)
        s = alpha * x
        return s

class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.tcn = TCNBlock(1, 8, 16, 3, 4)
        self.feature_attention = FeatureAttention(16)

    def forward(self, x):
        out = self.tcn(x)
        out = self.feature_attention(out)
        return out


# Defining the decoder
