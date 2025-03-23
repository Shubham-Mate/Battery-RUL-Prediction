import torch
import torch.nn.init as init

class ASTLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, cell_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.w_f = torch.nn.Parameter(init.xavier_uniform_(torch.empty(hidden_size+input_size, cell_size)))
        self.b_f = torch.nn.Parameter(torch.zeros(cell_size))

        self.sig = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

        self.inp_peephole = torch.nn.Parameter(init.uniform_(torch.empty(cell_size), -0.01, 0.01))

        self.w_z = torch.nn.Parameter(init.xavier_uniform_(torch.empty(hidden_size+input_size, cell_size)))
        self.b_z = torch.nn.Parameter(torch.zeros(cell_size))

        self.out_peephole = torch.nn.Parameter(init.uniform_(torch.empty(cell_size), -0.01, 0.01))
        self.w_o = torch.nn.Parameter(init.xavier_uniform_(torch.empty(hidden_size+input_size, cell_size)))
        self.b_o = torch.nn.Parameter(torch.zeros(cell_size))

    def forward(self, x, init_states=None):
        bs, seq_sz, _ = x.size()
        hidden_seq = []
        
        if init_states is None:
            h_t, c_t = (
                torch.zeros(bs, self.hidden_size).to(x.device),
                torch.zeros(bs, self.hidden_size).to(x.device),
            )
        else:
            h_t, c_t = init_states

        
            
        for t in range(seq_sz):
            x_t = x[:, t, :]
            #print(x_t.shape, h_t.shape)
            inp = torch.cat([x_t, h_t], dim=-1)

            z_t = self.tanh(torch.matmul(inp, self.w_z) + self.b_z)
            f_t = self.sig(torch.matmul(inp, self.w_f) + self.b_f)
            i_t = (1 - f_t) * self.sig(c_t * self.inp_peephole)

            c_t = (f_t * c_t) + (i_t * z_t)
            o_t = self.sig(torch.matmul(inp, self.w_o) + (c_t * self.out_peephole) + self.b_o)
            h_t = o_t * self.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)
    

class CNN_ASTLSTM(torch.nn.Module):
    def __init__(self, cnn_channels, hidden_dims):
        super().__init__()

        # CNN Blocks
        self.cnn_1 = torch.nn.Conv1d(in_channels=1, out_channels=cnn_channels, kernel_size=7, padding='same')
        self.relu = torch.nn.ReLU()

        # AST-LSTM Blocks
        self.ast_lstm_1 = ASTLSTM(cnn_channels, hidden_size=hidden_dims[0], cell_size=hidden_dims[0])
        self.ast_lstm_2 = ASTLSTM(hidden_dims[0], hidden_size=hidden_dims[1], cell_size=hidden_dims[1])

        # DNN
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dims[1], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        # CNN Initialization
        init.xavier_normal_(self.cnn_1.weight)
        
        # ASTLSTM Initialization
        for lstm in [self.ast_lstm_1, self.ast_lstm_2]:
            init.xavier_uniform_(lstm.w_f)
            init.xavier_uniform_(lstm.w_z)
            init.xavier_uniform_(lstm.w_o)
            init.uniform_(lstm.inp_peephole, -0.01, 0.01)
            init.uniform_(lstm.out_peephole, -0.01, 0.01)

        # DNN Initialization
        for layer in self.fc:
            if isinstance(layer, torch.nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)  # Zero bias

    def forward(self, x):
        x = self.cnn_1(x)
        x = self.relu(x)

        x = x.mT

        output, (_, c) = self.ast_lstm_1(x)
        output, (h, c) = self.ast_lstm_2(output)

        h = self.fc(h)
        return h

