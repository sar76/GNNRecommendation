# GNNRecommendation

A repository exploring whether Graph Neural Networks (GNNs) can outperform traditional machine learning models in predicting user preferences for movie recommendations ([github.com][1])

## Repository Structure

* **SGJG\_GNN-Recommender.ipynb**
  A Jupyter notebook that walks through data download, preprocessing, graph construction, model definition, training, and evaluation of a GNN-based recommender.
* **Final\_Report-GNN.pdf**
  A detailed project report with problem statement, methodology, experimental setup, and results. ([github.com][1])
* **README.md**
  This file.

## Getting Started

### Prerequisites

* Python 3.8+
* pip

### Install Dependencies

```bash
pip install torch torchvision torchaudio \
    torch-scatter torch-sparse torch-cluster torch-spline-conv \
    torch-geometric pandas numpy scikit-learn scipy matplotlib jupyterlab
```

### Download & Run

1. Clone the repo

   ```bash
   git clone https://github.com/sar76/GNNRecommendation.git
   cd GNNRecommendation
   ```
2. Launch Jupyter

   ```bash
   jupyter notebook SGJG_GNN-Recommender.ipynb
   ```
3. The first cell downloads and extracts the MovieLens 1M dataset into an `ml-1m/` folder automatically.

## Data

Uses the [MovieLens 1M dataset](https://files.grouplens.org/datasets/movielens/ml-1m.zip) (≈6 M ratings by \~6 000 users on \~4 000 movies). The notebook’s data pipeline fetches, parses, and constructs a bipartite user–item graph.

## Model

1. **Graph Construction**
   Users and movies as nodes; edges carry rating values as features.
2. **GNN Architecture**
   Graph Convolutional Network (GCN) / GraphSAGE via PyTorch Geometric.
3. **Training & Evaluation**
   Optimized for RMSE on a held-out test set; additional ranking metrics (Precision\@K, Recall\@K).

## Results

See **Final\_Report-GNN.pdf** for full experimental results. In summary, the GNN model demonstrates improved RMSE and ranking performance compared to traditional baselines.

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

[1]: https://github.com/sar76/GNNRecommendation "GitHub - sar76/GNNRecommendation: Are GNN Recommendation systems for User preference more efficient than traditional ML models?"
