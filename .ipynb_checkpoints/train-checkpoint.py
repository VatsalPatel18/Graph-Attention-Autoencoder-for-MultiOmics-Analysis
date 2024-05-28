import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from sklearn.model_selection import train_test_split
import pickle

from OmicsConfig import OmicsConfig
from MultiOmicsGraphAttentionAutoencoderModel import MultiOmicsGraphAttentionAutoencoderModel
from GATv2EncoderModel import GATv2EncoderModel
from GATv2DecoderModel import GATv2DecoderModel
from EdgeWeightPredictorModel import EdgeWeightPredictorModel

def collate_graph_data(batch):
    return Batch.from_data_list(batch)

def create_data_loader(graph_data_dict, batch_size=1, shuffle=True):
    graph_data = list(graph_data_dict.values())
    return DataLoader(graph_data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_graph_data)

# Load your data
graph_data_dict = torch.load('data/graph_data_dictN.pth')

# Split the data
train_data, temp_data = train_test_split(list(graph_data_dict.items()), train_size=0.6, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Convert lists back into dictionaries
train_data = dict(train_data)
val_data = dict(val_data)
test_data = dict(test_data)

# Define the configuration for the model
autoencoder_config = OmicsConfig(
    in_channels=17,
    edge_attr_channels=1,
    out_channels=1,
    original_feature_size=17,
    learning_rate=0.01,
    num_layers=2,
    edge_decoder_hidden_sizes=[128, 64],
    edge_decoder_activations=['ReLU', 'ReLU']
)

# Initialize the model
autoencoder_model = MultiOmicsGraphAttentionAutoencoderModel(autoencoder_config)

# Create data loaders
train_loader = create_data_loader(train_data, batch_size=5, shuffle=True)
val_loader = create_data_loader(val_data, batch_size=2, shuffle=False)
test_loader = create_data_loader(test_data, batch_size=2, shuffle=False)

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training process
def train_autoencoder(autoencoder_model, train_loader, validation_loader, epochs, device):
    autoencoder_model.to(device)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Train
        autoencoder_model.train()
        train_loss, train_cosine_similarity = autoencoder_model.train_model(train_loader, device)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Cosine Similarity: {train_cosine_similarity:.4f}")
        train_losses.append(train_loss)

        # Validate
        autoencoder_model.eval()
        val_loss, val_cosine_similarity = autoencoder_model.validate(validation_loader, device)
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}, Validation Cosine Similarity: {val_cosine_similarity:.4f}")
        val_losses.append(val_loss)

    # Save the trained encoder weights
    trained_encoder_path = "lc_models/MultiOmicsAutoencoder/trained_encoder"
    autoencoder_model.encoder.save_pretrained(trained_encoder_path)

    # Save the trained decoder weights
    trained_decoder_path = "lc_models/MultiOmicsAutoencoder/trained_decoder"
    autoencoder_model.decoder.save_pretrained(trained_decoder_path)

    # Save the trained edge weight predictor weights (if needed separately)
    trained_edge_weight_predictor_path = "lc_models/MultiOmicsAutoencoder/trained_edge_weight_predictor"
    autoencoder_model.decoder.edge_weight_predictor.save_pretrained(trained_edge_weight_predictor_path)

    # Optionally save the entire autoencoder again if you want to have a complete package
    trained_autoencoder_path = "lc_models/MultiOmicsAutoencoder/trained_autoencoder"
    autoencoder_model.save_pretrained(trained_autoencoder_path)

    return train_losses, val_losses

# Train and save the model
train_losses, val_losses = train_autoencoder(autoencoder_model, train_loader, val_loader, epochs=10, device=device)

# Evaluate the model
test_loss, test_accuracy = autoencoder_model.evaluate(test_loader, device)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4%}")

# Save the training and validation losses
with open('./results/train_loss.pkl', 'wb') as f:
    pickle.dump(train_losses, f)

with open('./results/val_loss.pkl', 'wb') as f:
    pickle.dump(val_losses, f)
