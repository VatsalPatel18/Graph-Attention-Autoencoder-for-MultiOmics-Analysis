import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GAE, GATv2Conv
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch_geometric.utils import negative_sampling
from torch.nn.functional import cosine_similarity
from torch.optim import AdamW  # NEW: Import AdamW
from torch.optim.lr_scheduler import StepLR  

def collate_graph_data(batch):
    return Batch.from_data_list(batch)

@staticmethod
def create_data_loader(train_data, batch_size=1, shuffle=True):
    graph_data = list(train_data.values())
    return DataLoader(graph_data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_graph_data)

class GATv2Encoder(torch.nn.Module):
    def __init__(self, in_channels, edge_attr_channels, out_channels, heads=1, concat=True):
        super(GATv2Encoder, self).__init__()
        self.conv1 = GATv2Conv(in_channels, out_channels, heads=heads, concat=concat, edge_dim=edge_attr_channels, add_self_loops=False)
        self.attention_weights1 = None;
       
    def forward(self, x, edge_index, edge_attr):
        x, _a1_ = self.conv1(x, edge_index, edge_attr, return_attention_weights=True)
        # x = x.relu()
        self.attention_weights1 = _a1_
        return x

class GATv2Decoder(torch.nn.Module):
    def __init__(self, in_channels, original_feature_size):
        super(GATv2Decoder, self).__init__()
        self.edge_weight_predictor = torch.nn.Sequential(
            torch.nn.Linear(2 * in_channels, 128),  # First linear layer
            torch.nn.ReLU(),                         # Activation function
            torch.nn.Linear(128, 1)                  # Output layer
        )
        self.fc = torch.nn.Linear(in_channels, original_feature_size)

    def forward(self, z, sigmoid=True):
        x_reconstructed = self.fc(z)
        return x_reconstructed
   
    def predict_edge_weights(self, z, edge_index):
        edge_embeddings = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=-1)
        return self.edge_weight_predictor(edge_embeddings)


def graph_reconstruction_loss(pred_features, true_features):
    node_loss = F.mse_loss(pred_features, true_features)
    return node_loss

def edge_reconstruction_loss(z, pos_edge_index, neg_edge_index=None):
    # Get the positive edge logits (inner products)
    pos_logits = (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=-1)
    pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
   
    # If negative samples are not provided, generate them
    if neg_edge_index is None:
        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
   
    # Get the negative edge logits (inner products)
    neg_logits = (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=-1)
    neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))

    return pos_loss + neg_loss

def edge_weight_reconstruction_loss(pred_weights, true_weights):
    pred_weights = pred_weights.squeeze(-1)
    return F.mse_loss(pred_weights, true_weights)

class GraphAutoencoder:
    def __init__(self, in_channels, edge_attr_channels, out_channels, original_feature_size, learning_rate=0.01):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.learning_rate = learning_rate

        self.Gencoder = GATv2Encoder(in_channels, edge_attr_channels, out_channels)
        self.Gdecoder = GATv2Decoder(out_channels, original_feature_size)
        self.device = torch.device('cuda')
        self.Gencoder = self.Gencoder.to(self.device)
        self.Gdecoder = self.Gdecoder.to(self.device)
        self.gae = GAE(self.Gencoder, self.Gdecoder)
        self.gae = self.gae.to(self.device)
        self.optimizer = AdamW(list(self.Gencoder.parameters()) + list(self.Gdecoder.parameters()), lr=self.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.7)

    def train(self, data_loader):
        self.gae.train()
        total_loss = 0
        total_cosine_similarity = 0

        # NEW: Loss weights
        loss_weight_node = 1.0
        loss_weight_edge = 2.0
        loss_weight_edge_attr = 2.0 
        for data in data_loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            z = self.gae(data.x, data.edge_index, data.edge_attr)
            x_reconstructed = self.Gdecoder(z)

            # Calculate node loss and edge loss as before
            node_loss = graph_reconstruction_loss(x_reconstructed, data.x)
            edge_loss = edge_reconstruction_loss(z, data.edge_index)

            # Calculate cosine similarity
            cos_sim = cosine_similarity(x_reconstructed, data.x, dim=-1).mean()
            total_cosine_similarity += cos_sim.item()  # Aggregate for all batches

            # Continue as usual
            pred_edge_weights = self.Gdecoder.predict_edge_weights(z, data.edge_index)
            edge_weight_loss = edge_weight_reconstruction_loss(pred_edge_weights, data.edge_attr)
            loss = (loss_weight_node * node_loss) + (loss_weight_edge * edge_loss) + (loss_weight_edge_attr * edge_weight_loss)
            print(f"node_loss: {node_loss}, edge_loss: {edge_loss:.4f}, edge_weight_loss: {edge_weight_loss:.4f}, cosine_similarity: {cos_sim:.4f}")
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss, avg_cosine_similarity = total_loss / len(data_loader), total_cosine_similarity / len(data_loader)
        return avg_loss, avg_cosine_similarity  # Return both the average loss and average cosine similarity


    def fit(self, train_loader, validation_loader, epochs):
        train_losses = []
        val_losses = []

        for epoch in range(1, epochs + 1):
            train_loss, train_cosine_similarity = self.train(train_loader)  # Unpack the tuple
            torch.cuda.empty_cache()
            val_loss, val_cosine_similarity = self.validate(validation_loader)  # Unpack the tuple

            print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Cosine Similarity: {train_cosine_similarity:.4f}, Validation Loss: {val_loss:.4f}, Validation Cosine Similarity: {val_cosine_similarity:.4f}")

            # NEW: Step the learning rate scheduler
            self.scheduler.step()

        return train_losses, val_losses


    def validate(self, validation_loader):
        self.gae.eval()  # set the model to evaluation mode
        total_loss = 0
        total_cosine_similarity = 0  

        with torch.no_grad():  # No gradient computation during validation
            for data in validation_loader:
                data = data.to(self.device)
                z = self.gae(data.x, data.edge_index, data.edge_attr)
                x_reconstructed = self.Gdecoder(z)
                node_loss = graph_reconstruction_loss(x_reconstructed, data.x)
                edge_loss = edge_reconstruction_loss(z, data.edge_index)

                # Calculate cosine similarity as you do in the train method
                cos_sim = cosine_similarity(x_reconstructed, data.x, dim=-1).mean()
                total_cosine_similarity += cos_sim.item()  # Aggregate for all batches

                loss = node_loss + edge_loss
                total_loss += loss.item()

        avg_loss = total_loss / len(validation_loader)
        avg_cosine_similarity = total_cosine_similarity / len(validation_loader)  # Calculate average cosine similarity

        return avg_loss, avg_cosine_similarity  # Return both the average loss and average cosine similarity


    def evaluate(self, test_loader):
        self.gae.eval()  # Set the model to evaluation mode
        total_loss = 0
        total_accuracy = 0
        # torch.cuda.empty_cache()
        with torch.no_grad():  # No gradient computation during evaluation
            for data in test_loader:
                data = data.to(self.device)
                z = self.gae(data.x, data.edge_index, data.edge_attr)
                x_reconstructed = self.Gdecoder(z)
                node_loss = graph_reconstruction_loss(x_reconstructed, data.x)
                edge_loss = edge_reconstruction_loss(z, data.edge_index)

                loss = node_loss + edge_loss
                total_loss += loss.item()

        avg_loss = total_loss / len(test_loader)
        avg_accuracy = total_accuracy / len(test_loader)  # Calculate average accuracy

        return avg_loss, avg_accuracy