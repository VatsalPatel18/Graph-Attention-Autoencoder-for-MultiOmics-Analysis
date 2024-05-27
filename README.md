# MultiOmics Graph Attention Network

## Overview

This repository contains the codebase for our research project titled "Graph Attention Networks for Biomedical Insights: MultiOmics Integration for Risk Stratification and Biomarker Identification", submitted to ICML 2024. The project explores the application of Graph Attention Networks to integrate and analyze multiomic data types, such as gene expression, mutations, methylation, and copy number alterations, primarily focusing on head and neck squamous cell carcinoma (HNSCC).

## Repository Structure

- `data/`: Folder containing necessary datasets (Note: Large data files like .pth are not tracked due to size constraints).
- `models/`: This directory includes the models developed during the study.
- `results/`: Contains output from the models including figures and result summaries.
- `graph_autoencoder.py`: Implements the graph autoencoder model.
- `HyperParameterSearch.py`: Script for hyperparameter tuning of models.
- `train_GAE.py`: Main training script for the Graph Attention Network.
- `LICENSE`: License file for the project.

## Purpose

The code provided in this repository is intended to support the reproducibility of our research findings. By leveraging graph neural networks, particularly graph attention mechanisms, we aim to derive insightful biomarkers and identify survival groups in cancer genomics. This integration of heterogeneous data sources aims to provide a comprehensive view of the genomic landscape to improve patient stratification and treatment outcomes.

## Model Details

- **Architecture:** GATv2Encoder, GATv2Decoder
- **Training Data:** Multi-omics data types including gene expression, mutations, methylation, and copy number alterations.
- **Usage:** Instructions on how to use the model.

## How to Use

```python
from torch_geometric.data import DataLoader
from model import GraphAutoencoder

# Load model
model_path = 'path_to_model.pth'
gae = GraphAutoencoder(in_channels=17, edge_attr_channels=1, out_channels=1, original_feature_size=17)
gae.gae.load_state_dict(torch.load(model_path))
gae.gae.eval()

# Use the model
# Example usage
## Getting Started

To use this repository:
1. Clone the repository to your local machine.
2. Ensure you have the necessary Python environment and dependencies installed.
3. Explore the data and model scripts provided.

## Contact

For any additional information or queries, please contact:
- Vatsal Pravinbhai Patel (vatsal1804@gmail.com)

## Citation

If you use this work or dataset in your research, please cite it as follows:
Patel, V. P., & Biswas, N. K. (2024). Graph Attention Networks for Biomedical Insights: MultiOmics Integration for Risk Stratification and Biomarker Identification. Submitted to ICML 2024,but not Accepted.
