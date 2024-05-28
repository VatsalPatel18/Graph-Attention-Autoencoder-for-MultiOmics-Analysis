from transformers import PretrainedConfig
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


class OmicsConfig(PretrainedConfig):
    model_type = "omics-graph-network"

    def __init__(self, in_channels=768, edge_attr_channels=128, out_channels=128, original_feature_size=768, learning_rate=0.01, num_layers=1, edge_decoder_hidden_sizes=[128], edge_decoder_activations=['ReLU'], **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.edge_attr_channels = edge_attr_channels
        self.out_channels = out_channels
        self.original_feature_size = original_feature_size
        self.learning_rate = learning_rate
        self.num_layers = num_layers
        self.edge_decoder_hidden_sizes = edge_decoder_hidden_sizes
        self.edge_decoder_activations = edge_decoder_activations 