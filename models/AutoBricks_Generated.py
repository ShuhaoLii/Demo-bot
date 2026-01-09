import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
################################################################################
#                         MODULE DEFINITION AREA                               #
#                                                                              #
#  1. Define or Import your 'SpatialModule' here.                              #
#     Requirement: Input [B, C_in, T, H, W] -> Output [B, C_out, T, H, W]      #
#                                                                              #
#  2. Define or Import your 'TemporalModule' here.                             #
#     Requirement: Input [B, C_in, T, H, W] -> Output [B, C_out, T_out, H, W]  #
#                                                                              #
################################################################################
"""


# Example Placeholder (The LLM should replace this with the actual class definition)
# class SpatialModule(nn.Module): ...
# class TemporalModule(nn.Module): ...


class LocalCNN(nn.Module):
    """
    [Spatial Module]
    Implements the 'Spatial View: Local CNN' from DMVST-Net.
    Captures local spatial dependencies using stacked Convolutions.

    Input:  [B, C_in, T, H, W]
    Output: [B, C_out, T, H, W]
    """

    def __init__(self, in_channels, out_channels, height, width, input_len, num_layers=3, kernel_size=3):
        super(LocalCNN, self).__init__()

        layers = []
        padding = kernel_size // 2

        # K Convolutional Layers
        # Reference: "After K convolution layers..." [cite: 1198]
        current_in = in_channels
        for i in range(num_layers):
            # In the paper, filter count is lambda=64 [cite: 1382]
            # We keep it consistent with out_channels for intermediate layers or make it configurable.
            # Here we act as a feature extractor, so we map to out_channels eventually.
            layers.append(nn.Conv2d(current_in, out_channels, kernel_size, padding=padding))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            current_in = out_channels

        self.conv_stack = nn.Sequential(*layers)

        # Dimension Reduction
        # Reference: "At last, we use a fully connected layer to reduce the dimension" [cite: 1199]
        # We use 1x1 Conv to act as FC per node, preserving (H, W).
        self.reduce_dim = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape

        # 1. Fold Time into Batch: [B*T, C, H, W]
        # "Local CNN method... considers spatially nearby regions" [cite: 1184]
        # Processing each timeframe independently.
        x_reshaped = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

        # 2. Convolutional Layers
        x_conv = self.conv_stack(x_reshaped)

        # 3. Dimension Reduction (Simulating the FC layer in spatial view)
        # "transform the output... to a feature vector" [cite: 1198]
        out = self.relu(self.reduce_dim(x_conv))

        # 4. Unfold back: [B, C_out, T, H, W]
        _, C_out, _, _ = out.shape
        out = out.view(B, T, C_out, H, W).permute(0, 2, 1, 3, 4)

        return out


class ConvLSTMCell(nn.Module):
    """
    Standard ConvLSTM Cell as used in ACFM.
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels=input_dim + hidden_dim,
                              out_channels=4 * hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)

        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, height, width):
        device = self.conv.weight.device
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))


class AttentionConvLSTM(nn.Module):
    """
    [Temporal Module]
    Attentive Crowd Flow Machine (ACFM).
    Composed of two progressive ConvLSTM units with an Attention mechanism.

    Reference:
    - "ACFM is composed of two progressive ConvLSTM units connected with a convolutional layer for attention weight prediction" [cite: 1143]

    Input:  [B, C_in, T, H, W]
    Output: [B, C_out, Pred_Len, H, W]
    """

    def __init__(self, in_channels, out_channels, input_len, pred_len, height, width, hidden_dim=32):
        super(AttentionConvLSTM, self).__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim

        # 1. First ConvLSTM (Models temporal dependency)
        self.lstm1 = ConvLSTMCell(in_channels, hidden_dim, kernel_size=3, bias=True)

        # 2. Attention Layer (Spatial weight prediction)
        # Input: Concat(Hidden1, Input), Kernel: 1x1
        self.att_conv = nn.Conv2d(hidden_dim + in_channels, 1, kernel_size=1)

        # 3. Second ConvLSTM (Learning from reweighted features)
        # Input: Input * Attention
        self.lstm2 = ConvLSTMCell(in_channels, hidden_dim, kernel_size=3, bias=True)

        # 4. Output Projection
        # "fed into a following convolution layer... denoted as Sf" [cite: 1183]
        # We project the last hidden state to the prediction horizon.
        self.pred_conv = nn.Conv2d(hidden_dim, out_channels * pred_len, kernel_size=3, padding=1)

    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape

        # Initialize States
        h1, c1 = self.lstm1.init_hidden(B, H, W)
        h2, c2 = self.lstm2.init_hidden(B, H, W)

        # Loop over time steps
        for t in range(T):
            x_t = x[:, :, t, :, :]  # [B, C, H, W]

            # --- First LSTM ---
            # H_i^1 = ConvLSTM(H_{i-1}^1, X_i)
            h1, c1 = self.lstm1(x_t, (h1, c1))

            # --- Attention Mechanism ---
            # W_i = Conv(H_i^1 + X_i)
            att_input = torch.cat([h1, x_t], dim=1)
            att_map = torch.sigmoid(self.att_conv(att_input))  # [B, 1, H, W]

            # --- Reweighting ---
            # X_i * W_i
            x_weighted = x_t * att_map

            # --- Second LSTM ---
            # H_i^2 = ConvLSTM(H_{i-1}^2, X_i * W_i)
            h2, c2 = self.lstm2(x_weighted, (h2, c2))

        # Extract last hidden state of second LSTM as representation
        last_state = h2  # [B, Hidden, H, W]

        # Project to future prediction
        # Output: [B, C_out * Pred_Len, H, W]
        out_flat = self.pred_conv(last_state)

        # Reshape to 5D: [B, C_out, Pred_Len, H, W]
        out = out_flat.view(B, self.out_channels, self.pred_len, H, W).permute(0, 1, 2, 3,
                                                                               4).contiguous()  # Fixed permutation logic
        out = out.view(B, self.pred_len, self.out_channels, H, W).permute(0, 2, 1, 3, 4)

        return out

class STModel(nn.Module):
    def __init__(self, args, meta, hidden_dim=64):
        super().__init__()

        # Configuration
        self.input_seq_length = args.his_len
        self.horizon = args.pred_len
        self.spatial_layers = args.spatial_layers
        self.temporal_layers = args.temporal_layers

        input_dim = meta.get('input_dim', 1)
        output_dim = meta.get('input_dim', 1)

        # ---------------- 1. Spatial Blocks ----------------
        self.spatial_blocks = nn.ModuleList()
        for i in range(self.spatial_layers):
            in_c = input_dim if i == 0 else hidden_dim
            out_c = hidden_dim

            # [LLM INSERTION POINT]
            self.spatial_blocks.append(LocalCNN(in_channels=in_c, out_channels=out_c, height=meta.get('H'), width=meta.get('W'), input_len=self.input_seq_length))

        # ---------------- 2. Temporal Blocks ----------------
        self.temporal_blocks = nn.ModuleList()
        for i in range(self.temporal_layers):
            in_c = hidden_dim
            out_c = hidden_dim

            # [LLM INSERTION POINT]
            self.temporal_blocks.append(AttentionConvLSTM(in_channels=in_c, out_channels=out_c, input_len=self.input_seq_length, pred_len=self.horizon, height=meta.get('H'), width=meta.get('W')))

        # ---------------- 3. Output Projection ----------------
        self.head = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)

    def forward(self, x, past_ts=None, past_x_period=None):
        # ---------------- Spatial Processing ----------------
        for block in self.spatial_blocks:
            x = F.relu(block(x))

        # ---------------- Temporal Processing ----------------
        for block in self.temporal_blocks:
            x = block(x)

        # ---------------- Output Projection ----------------
        if hasattr(self, 'head'):
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
            x = self.head(x)
            _, C_out, _, _ = x.shape
            x = x.view(B, T, C_out, H, W).permute(0, 2, 1, 3, 4)

        return x