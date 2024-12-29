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

        self.residual_conv = torch.nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)


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
        out = self.tanh(self.map(x.mT))
        out = out.view(out.size(0), -1)

        alpha = self.softmax(out)
        alpha = alpha.unsqueeze(1)

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

class TemporalAttention(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TemporalAttention, self).__init__()
        self.softmax = torch.nn.Softmax()
        self.tanh = torch.nn.Tanh()
        self.linear = torch.nn.Linear(in_dim * 2, out_dim)

    def forward(self, x):
        dot_prods = x @ x.mT

        mask =torch.full((x.size(-2), x.size(-2)), -1e9).to(x.get_device())
        mask = torch.triu(mask, diagonal=1)
        masked_dot_prods = dot_prods + mask

        alpha = self.softmax(masked_dot_prods)
        context_vectors = alpha @ x
        out = self.tanh(self.linear(torch.cat((context_vectors, x), dim=-1)))

        return out

class Decoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Decoder, self).__init__()
        self.gru = torch.nn.GRU(in_dim, hidden_dim, batch_first=True)
        self.temporal_attention = TemporalAttention(hidden_dim, out_dim)

        

    def forward(self, x):
        outputs, hidden = self.gru(x)
        out = self.temporal_attention(outputs)

        return out


# Assemble the Model

class TCN_GRU_DNN_Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(16, 8, 4)

        # Define DNN
        self.dnn = torch.nn.Sequential(
            torch.nn.Linear(40, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1)
        )

    def forward(self, x):
        out = self.encoder(x)
        out = out.mT
        out = self.decoder(out)
        out = out.view(out.size(0), -1)
        out = self.dnn(out)

        return out