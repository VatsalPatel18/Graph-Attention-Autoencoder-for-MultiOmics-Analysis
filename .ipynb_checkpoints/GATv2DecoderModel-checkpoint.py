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

from EdgeWeightPredictorModel import EdgeWeightPredictorModel

class GATv2DecoderModel(PreTrainedModel):
    config_class = OmicsConfig
    base_model_prefix = "gatv2_decoder"

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([
            nn.Linear(config.out_channels if i == 0 else config.out_channels, config.out_channels)
            for i in range(config.num_layers)
        ])
        self.fc = nn.Linear(config.out_channels, config.original_feature_size)
        self.edge_weight_predictor = EdgeWeightPredictorModel(config)

    def forward(self, z):
        for layer in self.layers:
            z = layer(z)
            z = F.relu(z)
        x_reconstructed = self.fc(z)
        return x_reconstructed

    def predict_edge_weights(self, z, edge_index):
        return self.edge_weight_predictor(z, edge_index)
