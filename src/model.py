import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = int(hidden_channels)
        self.conv = nn.Conv2d(
            in_channels=int(input_channels) + int(hidden_channels),
            out_channels=4 * int(hidden_channels),
            kernel_size=int(kernel_size),
            padding=int(padding)
        )

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_state(self, batch, spatial, device, dtype):
        H, W = spatial
        h = torch.zeros(batch, self.hidden_channels, H, W, device=device, dtype=dtype)
        c = torch.zeros(batch, self.hidden_channels, H, W, device=device, dtype=dtype)
        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_layers=1, kernel_size=3):
        super().__init__()
        self.num_layers = int(num_layers)

        if isinstance(hidden_channels, (list, tuple)):
            hidden_list = list(hidden_channels)
        else:
            hidden_list = [int(hidden_channels)] * self.num_layers
        assert len(hidden_list) == self.num_layers

        cells = []
        for i in range(self.num_layers):
            in_ch = int(input_channels) if i == 0 else int(hidden_list[i - 1])
            cells.append(ConvLSTMCell(in_ch, int(hidden_list[i]), kernel_size=kernel_size))
        self.cells = nn.ModuleList(cells)

    def forward(self, x):
        # x: (B,T,C,H,W)
        B, T, C, H, W = x.shape
        device = x.device
        dtype = x.dtype

        layer_input = x
        for cell in self.cells:
            h, c = cell.init_state(B, (H, W), device, dtype)
            outputs = []
            for t in range(T):
                h, c = cell(layer_input[:, t], (h, c))
                outputs.append(h)
            layer_input = torch.stack(outputs, dim=1)  # (B,T,Hc,H,W)

        return layer_input  # last layer sequence




class FirePatchConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden=64, lstm_layers=1, kernel=3, dropout=0.2):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.convlstm = ConvLSTM(
            input_channels=32,
            hidden_channels=[hidden] * int(lstm_layers),
            num_layers=int(lstm_layers),
            kernel_size=int(kernel)
        )

        self.drop = nn.Dropout(float(dropout))

        # Patch head
        self.head = nn.Sequential(
            nn.Conv2d(hidden, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # x: (B,T,C,H,W)
        B, T, C, H, W = x.shape

        x2 = x.reshape(B * T, C, H, W)
        f2 = self.encoder(x2)                  # (B*T,32,H,W)
        feats = f2.reshape(B, T, 32, H, W)     # (B,T,32,H,W)

        seq = self.convlstm(feats)             # (B,T,hidden,H,W)
        h_last = seq[:, -1]                    # (B,hidden,H,W)

        h_last = self.drop(h_last)
        pooled = self.head(h_last).reshape(B, 64)
        logits = self.fc(pooled)               # (B,1)
        return logits