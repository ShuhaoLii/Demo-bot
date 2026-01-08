from Bricks.grid_bricks.spatial.ResConv import ResConv
from Bricks.grid_bricks.temporal.PyramidalConvGRU import PyramidalConvGRU

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
            self.spatial_blocks.append(ResConv(in_channels=in_c, out_channels=out_c, height=meta.get('H'), width=meta.get('W'), input_len=self.input_seq_length))

        # ---------------- 2. Temporal Blocks ----------------
        self.temporal_blocks = nn.ModuleList()
        for i in range(self.temporal_layers):
            in_c = hidden_dim
            out_c = hidden_dim

            # [LLM INSERTION POINT]
            self.temporal_blocks.append(PyramidalConvGRU(in_channels=in_c, out_channels=out_c, input_len=self.input_seq_length, pred_len=self.horizon, height=meta.get('H'), width=meta.get('W')))

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
        