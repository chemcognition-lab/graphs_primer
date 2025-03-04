import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# For GNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem

# Set the global font to Arial
plt.rcParams['font.family'] = 'Arial'

# 🔹 Get the current working directory and set up directories
cwd = os.getcwd()
if cwd.endswith("graphs_primer"):  
    base_dir = cwd
else:
    base_dir = os.path.join(cwd, "graphs_primer")

data_dir = os.path.join(base_dir, "data")
figures_dir = os.path.join(base_dir, "figures", "assets")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)

# 🔹 Load dataset
csv_path = os.path.join(data_dir, "delaney-processed.csv")
df = pd.read_csv(csv_path)

# 🔹 Select features and target variable
X_global = df[["Molecular Weight", "Number of H-Bond Donors"]].values
y = df["measured log solubility in mols per litre"].values
smiles = df["smiles"].values

# 🔹 Split data into training and test sets
X_train_global, X_test_global, y_train, y_test, smiles_train, smiles_test = train_test_split(
    X_global, y, smiles, test_size=0.2, random_state=42
)

# 🔹 Standardize X and y for the models
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_train_global)
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

X_test_scaled = X_scaler.transform(X_test_global)
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

# 🔹 Utility function to convert SMILES to graph data
def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Get atom features (we'll use simple atom types as features)
    atomic_nums = []
    for atom in mol.GetAtoms():
        atomic_nums.append(atom.GetAtomicNum())
    
    # One-hot encode atom types
    atom_features = np.zeros((len(atomic_nums), 100), dtype=np.float32)
    for i, atomic_num in enumerate(atomic_nums):
        if atomic_num < 100:  # Ensure we don't go out of bounds
            atom_features[i, atomic_num] = 1.0
    
    # Get bond information (edges)
    src = []
    dst = []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # Add edges in both directions
        src.extend([start, end])
        dst.extend([end, start])
    
    # Convert to PyTorch Geometric Data object
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Add global molecular features
    global_features = torch.tensor([
        mol.GetNumAtoms(),
        Chem.Descriptors.ExactMolWt(mol),
        Chem.Lipinski.NumHDonors(mol)
    ], dtype=torch.float).view(1, -1)
    
    return Data(x=x, edge_index=edge_index, global_features=global_features)

# 🔹 Define GNN model
class GNN(nn.Module):
    def __init__(self, node_features=100, hidden_channels=64, global_features=3):
        super(GNN, self).__init__()
        # GNN layers
        self.conv1 = GCNConv(node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        
        # Global feature processing
        self.global_nn = nn.Sequential(
            nn.Linear(global_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # Combined processing
        self.combined = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1)
        )
        
    def forward(self, data):
        x, edge_index, global_features = data.x, data.edge_index, data.global_features
        
        # Process graph structure
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        
        # Global pooling
        x = global_mean_pool(x, data.batch)
        
        # Process global features
        global_x = self.global_nn(global_features)
        
        # Combine graph and global features
        combined = torch.cat([x, global_x], dim=1)
        
        # Final prediction
        return self.combined(combined)

# 🔹 Prepare graph data for GNN
print("Converting SMILES to graphs...")
train_graphs = []
for i, smiles_str in enumerate(smiles_train):
    graph = smiles_to_graph(smiles_str)
    if graph is not None:
        # Add global features from our dataset
        graph.global_features = torch.tensor(X_train_scaled[i], dtype=torch.float).view(1, -1)
        graph.y = torch.tensor([y_train_scaled[i]], dtype=torch.float)
        train_graphs.append(graph)

test_graphs = []
for i, smiles_str in enumerate(smiles_test):
    graph = smiles_to_graph(smiles_str)
    if graph is not None:
        graph.global_features = torch.tensor(X_test_scaled[i], dtype=torch.float).view(1, -1)
        graph.y = torch.tensor([y_test_scaled[i]], dtype=torch.float)
        test_graphs.append(graph)

# 🔹 Create data loaders
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

# 🔹 Initialize and train GNN
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gnn_model = GNN().to(device)
optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)
criterion = nn.MSELoss()

print(f"Training GNN model on {device}...")
gnn_model.train()
for epoch in range(100):
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = gnn_model(batch)
        loss = criterion(out.flatten(), batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}: Loss = {total_loss / len(train_graphs):.4f}')

# 🔹 Evaluate GNN
gnn_model.eval()
y_pred_gnn_scaled = []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        pred = gnn_model(batch)
        y_pred_gnn_scaled.extend(pred.cpu().numpy().flatten())

# Convert predictions back to original scale
y_pred_gnn = y_scaler.inverse_transform(np.array(y_pred_gnn_scaled).reshape(-1, 1)).flatten()

# 🔹 Train MLP model (same as in your original code)
mlp_model = MLPRegressor(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=0.0001,
    max_iter=1000,
    random_state=42
)

print("Training MLP model...")
mlp_model.fit(X_train_scaled, y_train_scaled)
y_pred_mlp_scaled = mlp_model.predict(X_test_scaled)
y_pred_mlp = y_scaler.inverse_transform(y_pred_mlp_scaled.reshape(-1, 1)).flatten()

# 🔹 Calculate metrics for both models
gnn_mse = mean_squared_error(y_test[:len(y_pred_gnn)], y_pred_gnn)
gnn_r2 = r2_score(y_test[:len(y_pred_gnn)], y_pred_gnn)

mlp_mse = mean_squared_error(y_test, y_pred_mlp)
mlp_r2 = r2_score(y_test, y_pred_mlp)

print(f"GNN - MSE: {gnn_mse:.3f}, R²: {gnn_r2:.3f}")
print(f"MLP - MSE: {mlp_mse:.3f}, R²: {mlp_r2:.3f}")

# 🔹 Plot results for both models
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# MLP plot
axes[0].scatter(y_test, y_pred_mlp, alpha=0.7)
axes[0].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="red")
axes[0].set_title(f"MLP\nMSE: {mlp_mse:.3f}, R²: {mlp_r2:.3f}", fontsize=16)
axes[0].set_xlabel("Actual logS", fontsize=14)
axes[0].set_ylabel("Predicted logS", fontsize=14)

# GNN plot
axes[1].scatter(y_test[:len(y_pred_gnn)], y_pred_gnn, alpha=0.7)
axes[1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="red")
axes[1].set_title(f"GNN\nMSE: {gnn_mse:.3f}, R²: {gnn_r2:.3f}", fontsize=16)
axes[1].set_xlabel("Actual logS", fontsize=14)
axes[1].set_ylabel("Predicted logS", fontsize=14)

plt.tight_layout()
plt.savefig(f"{figures_dir}/MLP_vs_GNN_comparison.svg", format="svg")
plt.show()

# 🔹 Feature importance analysis
# For MLP, we can't easily get feature importance, but we can visualize weights for the first layer
if hasattr(mlp_model, 'coefs_') and len(mlp_model.coefs_) > 0:
    plt.figure(figsize=(8, 6))
    feature_names = ["Molecular Weight", "H-Bond Donors"]
    weights = mlp_model.coefs_[0]
    
    avg_importance = np.abs(weights).mean(axis=1)
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': avg_importance})
    
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title('MLP Feature Importance (First Layer Weights)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/MLP_feature_importance.svg", format="svg")
    plt.show()

print(f"All SVG plots saved in '{figures_dir}' folder.")