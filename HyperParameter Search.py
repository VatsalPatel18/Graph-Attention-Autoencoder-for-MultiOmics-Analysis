import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data, Batch, DataLoader
from v012824graph_autoencoder import GraphAutoencoder 

def create_data_loader(train_data, batch_size=1, shuffle=True):
    graph_data = list(train_data.values())
    return DataLoader(graph_data, batch_size=batch_size, shuffle=shuffle, collate_fn=Batch.from_data_list)

# Load your data
graph_data_dict = torch.load('fdata/hnsc3/graph_data_dictN.pth')
train_data, temp_data = train_test_split(list(graph_data_dict.items()), train_size=0.6, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Convert lists back into dictionaries
train_data = dict(train_data)
val_data = dict(val_data)
test_data = dict(test_data)

# Define hyperparameters for grid search
learning_rates = [0.001, 0.005, 0.01]
weight_decays = [0, 0.0001, 0.001]

# Store results
results = []

# Grid Search
for lr in learning_rates:
    for wd in weight_decays:
        # Initialize model with current set of hyperparameters
        print('running')
        gae = GraphAutoencoder(in_channels=17, edge_attr_channels=1, out_channels=1, original_feature_size=17, learning_rate=lr, weight_decay=wd)
        
        # Data loaders
        train_loader = create_data_loader(train_data, 5)
        val_loader = create_data_loader(val_data, 2)

        # Train the model
        train_losses, val_losses = gae.fit(train_loader, val_loader, epochs=5)

        # Store or log results
        results.append({
            'learning_rate': lr,
            'weight_decay': wd,
            'train_losses': train_losses,
            'val_losses': val_losses
        })

        # Save the model (optional)
        torch.save(gae.gae.state_dict(), f'fdata/hnsc3/models/model_lr{lr}_wd{wd}.pth')

with open('fdata/hnsc3/hyperparam_search_results.pkl', 'wb') as f:
    pickle.dump(results, f)