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


class EdgeWeightPredictorModel(PreTrainedModel):
    config_class = OmicsConfig
    base_model_prefix = "edge_weight_predictor"

    def __init__(self, config):
        super().__init__(config)
        layers = []
        input_size = 2 * config.out_channels
        for hidden_size, activation in zip(config.edge_decoder_hidden_sizes, config.edge_decoder_activations):
            layers.append(nn.Linear(input_size, hidden_size))
            if activation == 'ReLU':
                layers.append(nn.ReLU())
            elif activation == 'Sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'Tanh':
                layers.append(nn.Tanh())
            # Add more activations if needed
            input_size = hidden_size
        layers.append(nn.Linear(input_size, 1))
        self.predictor = nn.Sequential(*layers)

    def forward(self, z, edge_index):
        edge_embeddings = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1)
        return self.predictor(edge_embeddings)
