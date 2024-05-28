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

from GATv2EncoderModel import GATv2EncoderModel
from GATv2DecoderModel import GATv2DecoderModel
from EdgeWeightPredictorModel import EdgeWeightPredictorModel


class MultiOmicsGraphAttentionAutoencoderModel(PreTrainedModel):
    config_class = OmicsConfig
    base_model_prefix = "graph-attention-autoencoder"

    def __init__(self, config):
        super().__init__(config)
        self.encoder = GATv2EncoderModel(config)
        self.decoder = GATv2DecoderModel(config)
        self.optimizer = AdamW(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=config.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.7)

    def forward(self, x, edge_index, edge_attr):
        z, attention_weights = self.encoder(x, edge_index, edge_attr)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, attention_weights

    def predict_edge_weights(self, z, edge_index):
        return self.decoder.predict_edge_weights(z, edge_index)

    def train_model(self, data_loader, device):
        self.encoder.to(device)
        self.decoder.to(device)
        self.encoder.train()
        self.decoder.train()
        total_loss = 0
        total_cosine_similarity = 0
        loss_weight_node = 1.0
        loss_weight_edge = 2.0
        loss_weight_edge_attr = 2.0

        for data in data_loader:
            data = data.to(device)
            self.optimizer.zero_grad()
            z, attention_weights = self.encoder(data.x, data.edge_index, data.edge_attr)
            x_reconstructed = self.decoder(z)
            node_loss = graph_reconstruction_loss(x_reconstructed, data.x)
            edge_loss = edge_reconstruction_loss(z, data.edge_index)
            cos_sim = cosine_similarity(x_reconstructed, data.x, dim=-1).mean()
            total_cosine_similarity += cos_sim.item()
            pred_edge_weights = self.decoder.predict_edge_weights(z, data.edge_index)
            edge_weight_loss = edge_weight_reconstruction_loss(pred_edge_weights, data.edge_attr)
            loss = (loss_weight_node * node_loss) + (loss_weight_edge * edge_loss) + (loss_weight_edge_attr * edge_weight_loss)
            print(f"node_loss: {node_loss}, edge_loss: {edge_loss:.4f}, edge_weight_loss: {edge_weight_loss:.4f}, cosine_similarity: {cos_sim:.4f}")
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss, avg_cosine_similarity = total_loss / len(data_loader), total_cosine_similarity / len(data_loader)
        return avg_loss, avg_cosine_similarity

    def fit(self, train_loader, validation_loader, epochs, device):
        train_losses = []
        val_losses = []

        for epoch in range(1, epochs + 1):
            train_loss, train_cosine_similarity = self.train_model(train_loader, device)
            torch.cuda.empty_cache()
            val_loss, val_cosine_similarity = self.validate(validation_loader, device)
            print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Cosine Similarity: {train_cosine_similarity:.4f}, Validation Loss: {val_loss:.4f}, Validation Cosine Similarity: {val_cosine_similarity:.4f}")
            self.scheduler.step()

        return train_losses, val_losses

    def validate(self, validation_loader, device):
        self.encoder.to(device)
        self.decoder.to(device)
        self.encoder.eval()
        self.decoder.eval()
        total_loss = 0
        total_cosine_similarity = 0

        with torch.no_grad():
            for data in validation_loader:
                data = data.to(device)
                z, attention_weights = self.encoder(data.x, data.edge_index, data.edge_attr)
                x_reconstructed = self.decoder(z)
                node_loss = graph_reconstruction_loss(x_reconstructed, data.x)
                edge_loss = edge_reconstruction_loss(z, data.edge_index)
                cos_sim = cosine_similarity(x_reconstructed, data.x, dim=-1).mean()
                total_cosine_similarity += cos_sim.item()
                loss = node_loss + edge_loss
                total_loss += loss.item()

        avg_loss = total_loss / len(validation_loader)
        avg_cosine_similarity = total_cosine_similarity / len(validation_loader)
        return avg_loss, avg_cosine_similarity

    def evaluate(self, test_loader, device):
        self.encoder.to(device)
        self.decoder.to(device)
        self.encoder.eval()
        self.decoder.eval()
        total_loss = 0
        total_accuracy = 0

        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                z, attention_weights = self.encoder(data.x, data.edge_index, data.edge_attr)
                x_reconstructed = self.decoder(z)
                node_loss = graph_reconstruction_loss(x_reconstructed, data.x)
                edge_loss = edge_reconstruction_loss(z, data.edge_index)
                loss = node_loss + edge_loss
                total_loss += loss.item()

        avg_loss = total_loss / len(test_loader)
        avg_accuracy = total_accuracy / len(test_loader)
        return avg_loss, avg_accuracy

# Define a collate function for the DataLoader
def collate_graph_data(batch):
    return Batch.from_data_list(batch)

# Define a function to create a DataLoader
def create_data_loader(train_data, batch_size=1, shuffle=True):
    graph_data = list(train_data.values())
    return DataLoader(graph_data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_graph_data)

# Define functions for the losses
def graph_reconstruction_loss(pred_features, true_features):
    return F.mse_loss(pred_features, true_features)

def edge_reconstruction_loss(z, pos_edge_index, neg_edge_index=None):
    pos_logits = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)
    pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
    if neg_edge_index is None:
        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
    neg_logits = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)
    neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
    return pos_loss + neg_loss

def edge_weight_reconstruction_loss(pred_weights, true_weights):
    pred_weights = pred_weights.squeeze(-1)
    return F.mse_loss(pred_weights, true_weights)
