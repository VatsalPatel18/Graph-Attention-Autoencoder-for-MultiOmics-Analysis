import torch
import pickle
import numpy as np
import concurrent.futures

class ALExtracter:
    def __init__(self, graph_data_dict_path, model_object,model, model_path, gpu=False):
        if gpu:
            self.torch_device = 'cuda'
        else:
            self.torch_device = 'cpu'
            
        self.graph_data_dict = torch.load(graph_data_dict_path)
        self.gae = model_object
        self.gae.gae = model.to(self.torch_device)
        self.gae.gae.load_state_dict(torch.load(model_path, map_location=self.torch_device))
        self.gae.gae.eval()
        self.latent_feat_dict , self.attention_scores1 = self.extract_latent_attention_features()

    def extract_latent_attention_features(self):
        latent_features = {}
        attention_scores1 = {}
        
        with torch.no_grad():
            for graph_id, data in self.graph_data_dict.items():
                data = data.to(self.torch_device)
                z = self.gae.Gencoder(data.x, data.edge_index, data.edge_attr)
                latent_features[graph_id] = z.cpu()
                attention_weights1 = self.gae.Gencoder.attention_weights1
                attention_scores1[graph_id] = attention_weights1
        return latent_features, attention_scores1
    
    def load_edge_indices(self, glist_path, edge_matrix_path):
        with open(glist_path, 'rb') as f:
            glist = pickle.load(f)

        edge_matrix = np.load(edge_matrix_path)
        edge_matrix = torch.tensor(edge_matrix, dtype=torch.float)
        edge_index = torch.nonzero(edge_matrix, as_tuple=False).t().contiguous()
        edge_indices_dict = {}

        for i in range(edge_index.shape[1]):
            index1, index2 = edge_index[0, i].item(), edge_index[1, i].item()
            gene1, gene2 = glist[index1], glist[index2]
            edge_indices_dict[(index1, index2)] = (gene1, gene2)

        return edge_indices_dict
