# import pandas as pd
import numpy as np
import os
import pickle
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch

def collate_graph_data(batch):
    return Batch.from_data_list(batch)

@staticmethod
def create_data_loader(train_data, batch_size=1, shuffle=True):
    graph_data = list(train_data.values())
    return DataLoader(graph_data, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_graph_data)


graph_data_dict = torch.load('data/graph_data_dictN.pth')

train_data, temp_data = train_test_split(list(graph_data_dict .items())[:], train_size=0.6, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Convert lists back into dictionaries
train_data = dict(train_data)
val_data = dict(val_data)
test_data = dict(test_data)

torch.cuda.is_available();

from v830graph_autoencoder import GraphAutoencoder


in_channels = 17
out_channels = 1
# edge_attr

gae = GraphAutoencoder(in_channels=17, edge_attr_channels=1, out_channels=1, original_feature_size=17)


train_loader = create_data_loader(train_data,5)
val_loader = create_data_loader(val_data,2)
test_loader = create_data_loader(test_data,2)



# Train the model and store the training and validation losses
train_losses, val_losses = gae.fit(train_loader, val_loader, epochs=5)

test_loss, test_accuracy = gae.evaluate(test_loader)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4%}")

torch.save(gae.gae.state_dict(), './models/model.pth')

with open('./results/train_loss.pkl','wb') as f:
    pickle.dump(train_losses,f)
    
with open('./results/val_loss.pkl','wb') as f:
    pickle.dump(val_losses,f)