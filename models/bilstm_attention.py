import torch


class Attention(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.w_k = torch.nn.Linear(in_features=input_dim, out_features=input_dim)
        self.softmax = torch.nn.Softmax(dim=-2)
        self.w_s = torch.nn.Linear(in_features=input_dim*2, out_features=output_dim)
        self.tanh = torch.nn.Tanh()

        # Xavier Initialization
        torch.nn.init.xavier_uniform_(self.w_k.weight)
        torch.nn.init.zeros_(self.w_k.bias)
        torch.nn.init.xavier_uniform_(self.w_s.weight)
        torch.nn.init.zeros_(self.w_s.bias)

    def forward(self, x):
        h_t = x[:, -1, :]
        k = self.w_k(x)
        #print(k.shape, h_t.shape, h_t.mT.shape)
        e = torch.bmm(k, h_t.unsqueeze(1).mT)
        alpha = self.softmax(e)
        c = torch.sum(alpha * x, dim=-2)
        a = self.tanh(self.w_s(torch.concat([h_t, c], dim=-1)))
        return a


class BiLSTM_Attention(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.5)
        self.bilstm_1 = torch.nn.LSTM(1, hidden_size=hidden_dim, batch_first=True, bidirectional=True, num_layers=1, dropout=0.5)
        self.bilstm_2 = torch.nn.LSTM(input_size=hidden_dim*2, hidden_size=hidden_dim, batch_first=True, bidirectional=True, num_layers=1, dropout=0.5)
        self.bilstm_3 = torch.nn.LSTM(input_size=hidden_dim*2, hidden_size=hidden_dim, batch_first=True, bidirectional=True, num_layers=1, dropout=0.5)
        self.attn = Attention(hidden_dim*2, hidden_dim)
        self.dnn = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 1)
        )

        # Xavier Initialization
        for layer in self.dnn:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

        for lstm in [self.bilstm_1, self.bilstm_2, self.bilstm_3]:
            for name, param in lstm.named_parameters():
                if "weight_ih" in name or "weight_hh" in name:
                    torch.nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    torch.nn.init.zeros_(param)

    def forward(self, x):
        output, (h, c) = self.bilstm_1(x)
        #output = self.dropout(output)
        output, (h, c) = self.bilstm_2(output)
        #output = self.dropout(output)
        output, (h, c) = self.bilstm_3(output)
        #output = self.dropout(output)
        a = self.attn(output)
        out = self.dnn(a)
        return out
