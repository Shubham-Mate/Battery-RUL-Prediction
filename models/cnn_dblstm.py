import torch


class CNN_Block(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.leaky_relu = torch.nn.LeakyReLU()

        # Block 1
        self.cnn_1 = torch.nn.Conv1d(in_channels=1, out_channels=hidden_channels[0], padding='same', kernel_size=5)
        self.bn_1 = torch.nn.BatchNorm1d(num_features=hidden_channels[0])

        # Block 2
        self.cnn_2 = torch.nn.Conv1d(in_channels=hidden_channels[0], out_channels=hidden_channels[1], padding='same', kernel_size=3)
        self.bn_2 = torch.nn.BatchNorm1d(num_features=hidden_channels[1])

        # Block 3
        self.cnn_3 = torch.nn.Conv1d(in_channels=hidden_channels[1], out_channels=hidden_channels[2], padding='same', kernel_size=2)
        self.bn_3 = torch.nn.BatchNorm1d(num_features=hidden_channels[2])

    def forward(self, x):
        # Block 1
        x = self.cnn_1(x)
        x = self.bn_1(x)
        x = self.leaky_relu(x)

        # Block 2
        x = self.cnn_2(x)
        x = self.bn_2(x)
        x = self.leaky_relu(x)

        # Block 3
        x = self.cnn_3(x)
        x = self.bn_3(x)
        x = self.leaky_relu(x)

        return x
    
class CNN_DBLSTM_Model(torch.nn.Module):
    def __init__(self, cnn_hidden_dims, lstm_hidden_dims, lstm_layers):
        super().__init__()
        self.cnn_block = CNN_Block(cnn_hidden_dims)
        self.lstm_block = torch.nn.LSTM(cnn_hidden_dims[-1], hidden_size=lstm_hidden_dims, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2*lstm_hidden_dims, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1)
        )

    def forward(self, x):
        x = self.cnn_block(x)
        x = x.mT
        output, (h_n, c_n) = self.lstm_block(x)
        output = output[:, -1, :]
        output = self.fc(output)
        return output