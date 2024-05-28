from transformers import PreTrainedModel
from OmicsConfig import OmicsConfig
from transformers import PretrainedConfig, PreTrainedModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Batch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch_geometric.utils import negative_sampling
from torch.nn.functional import cosine_similarity
from torch.optim.lr_scheduler import StepLR


class GATv2EncoderModel(PreTrainedModel):
    config_class = OmicsConfig
    base_model_prefix = "gatv2_encoder"

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([
            GATv2Conv(config.in_channels if i == 0 else config.out_channels, config.out_channels, heads=1, concat=True, edge_dim=config.edge_attr_channels, add_self_loops=False)
            for i in range(config.num_layers)
        ])

    def forward(self, x, edge_index, edge_attr):
        attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, edge_index, edge_attr, return_attention_weights=True)
            attention_weights.append(attn_weights)
        return x, attention_weights