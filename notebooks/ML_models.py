import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor  # Import MLP
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler  # Import scaler

# 🔹 Get the current working directory
cwd = os.getcwd()
# 🔹 Ensure we're in the correct base project directory
if cwd.endswith("graphs_primer"):  
    base_dir = cwd  # Already in graphs_primer
else:
    base_dir = os.path.join(cwd, "graphs_primer")  # Move to the correct directory

# 🔹 Define paths for data and output
data_dir = os.path.join(base_dir, "data")
figures_dir = os.path.join(base_dir, "figures", "assets")
font_dir = os.path.join(data_dir, "misc")

# 🔹 Ensure required directories exist
os.makedirs(data_dir, exist_ok=True)
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(font_dir, exist_ok=True)

# 🔹 Load dataset
csv_path = os.path.join(data_dir, "delaney-processed.csv")
df = pd.read_csv(csv_path)

# 🔹 Select features and target variable
X = df[["Molecular Weight", "Number of H-Bond Donors"]].values
y = df["measured log solubility in mols per litre"].values

# For the distribution plots later
MW = df["Molecular Weight"]
NHBA = df["Number of H-Bond Donors"]
logS = df["measured log solubility in mols per litre"]

# 🔹 Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Standardize X and y for MLP
X_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = X_scaler.fit_transform(X_train)
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

X_test_scaled = X_scaler.transform(X_test)
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

# 🔹 Define models
models = {
    "Linear Regression": LinearRegression(),
    "MLP": MLPRegressor(
        hidden_layer_sizes=(100, 50),  # Two hidden layers with 100 and 50 neurons
        activation='relu',
        solver='adam',
        alpha=0.0001,  # L2 regularization
        max_iter=1000,
        random_state=42
    ),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
}

# 🔹 Load Fira Code font
font_path = os.path.join(font_dir, "FiraCode-Regular.ttf")  # Ensure file exists
print("Font Path:", font_path)

if not os.path.exists(font_path):
    raise FileNotFoundError(f"Font file not found: {font_path}")

fira_code_font = fm.FontProperties(fname=font_path)

# 🔹 Train, predict, and evaluate models
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, (name, model) in enumerate(models.items()):
    if name == "MLP":
        model.fit(X_train_scaled, y_train_scaled)  # Train MLP on normalized data
        y_pred_scaled = model.predict(X_test_scaled)  # Predict in normalized space
        y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()  # Convert back
    else:
        model.fit(X_train, y_train)  # Train normally
        y_pred = model.predict(X_test)  # Predict normally
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 🔹 Plot actual vs. predicted values
    axes[i].scatter(y_test, y_pred, alpha=0.7)
    axes[i].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="red")
    axes[i].set_title(f"{name}\nMSE: {mse:.3f}, R²: {r2:.3f}", fontproperties=fira_code_font, fontsize=16)
    axes[i].set_xlabel("Actual logS", fontproperties=fira_code_font, fontsize=14)
    axes[i].set_ylabel("Predicted logS", fontproperties=fira_code_font, fontsize=14)

plt.tight_layout()
plt.savefig(f"{figures_dir}/ML_models_actual_vs_predicted.svg", format="svg")  # Save full figure
plt.show()


# 🔹 Distribution Plots
sns.set_style("whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# MW Distribution
sns.histplot(MW, kde=True, bins=20, ax=axes[0], color="#fdc422")
axes[0].set_title("Molecular Weight (MW) Distribution", fontproperties=fira_code_font, fontsize=12)

# NHBA Distribution
sns.histplot(NHBA, kde=True, bins=10, ax=axes[1], color="#fdc422")
axes[1].set_title("NHBA Distribution", fontproperties=fira_code_font, fontsize=12)

# logS Distribution
sns.histplot(logS, kde=True, bins=20, ax=axes[2], color="#fc8d62")
axes[2].set_title("logS Distribution", fontproperties=fira_code_font, fontsize=12)

plt.tight_layout()
plt.savefig(f"{figures_dir}/input_distributions.svg", format="svg")  # Save as SVG
plt.show()

print(f"All SVG plots saved in '{figures_dir}' folder.")

# 🔹 Create distribution plots for MW, NHBA, and logS
features = {"MW": MW, "NHBA": NHBA, "logS": logS}

for feature, values in features.items():
    plt.figure(figsize=(6, 4))
    color = "#fdc422" if feature != "logS" else "#fc8d62"
    sns.histplot(values, kde=True, bins=20, color=color)
    plt.title(f"{feature} Distribution", fontproperties=fira_code_font, fontsize=16)
    plt.xlabel(feature, fontproperties=fira_code_font, fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Save distribution plots
    plt.savefig(os.path.join(figures_dir, f"{feature}_distribution.svg"), format="svg", bbox_inches="tight")
    plt.close()