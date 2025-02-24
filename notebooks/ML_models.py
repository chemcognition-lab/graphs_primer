import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 🔹 Define paths
base_dir = os.path.abspath(os.path.join(os.getcwd(), ".", "figures", "assets"))
os.makedirs(base_dir, exist_ok=True)  # Ensure the directory exists

# 🔹 Generate synthetic data
np.random.seed(42)
n_samples = 200

MW = np.random.uniform(150, 600, n_samples)  # Molecular Weight
NHBA = np.random.randint(0, 10, n_samples)   # Number of H-bond Acceptors
logP = 0.02 * MW - 0.5 * NHBA + np.random.normal(0, 1, n_samples)  # Simulated logP

X = np.column_stack((MW, NHBA))
y = logP

# 🔹 Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔹 Define models
models = {
    "Linear Regression": LinearRegression(),
    "Gaussian Process": GaussianProcessRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
}

# 🔹 Load Fira Code font (update with actual font path)
font_path = "C:/Users/Rana/AppData/Local/Microsoft/Windows/Fonts/FiraCode-Regular.ttf"  # Update with correct path
fira_code_font = fm.FontProperties(fname=font_path)


# 🔹 Train, predict, and evaluate models
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, (name, model) in enumerate(models.items()):
    model.fit(X_train, y_train)  # Train model
    y_pred = model.predict(X_test)  # Predict
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # 🔹 Plot actual vs. predicted values
    axes[i].scatter(y_test, y_pred, alpha=0.7)
    axes[i].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle="--", color="red")
    axes[i].set_title(f"{name}\nMSE: {mse:.2f}, R²: {r2:.2f}", fontproperties=fira_code_font, fontsize=16)
    axes[i].set_xlabel("Actual logP", fontproperties=fira_code_font, fontsize=14)
    axes[i].set_ylabel("Predicted logP", fontproperties=fira_code_font, fontsize=14)

    # 🔹 Save each plot as SVG
    plt.savefig(f"{base_dir}/{name.replace(' ', '_')}_actual_vs_predicted.svg", format="svg")

plt.tight_layout()
plt.savefig(f"{base_dir}/ML_models_actual_vs_predicted.svg", format="svg")  # Save full figure
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

# logP Distribution
sns.histplot(logP, kde=True, bins=20, ax=axes[2], color="#fc8d62")
axes[2].set_title("logP Distribution", fontproperties=fira_code_font, fontsize=12)

plt.tight_layout()
plt.savefig(f"{base_dir}/input_distributions.svg", format="svg")  # Save as SVG
plt.show()

print(f"All SVG plots saved in '{base_dir}' folder.")

# 🔹 Create distribution plots for MW, NHBA, and logP
features = {"MW": MW, "NHBA": NHBA, "logP": logP}

for feature, values in features.items():
    if feature != 'logP':
        plt.figure(figsize=(6, 4))
        sns.histplot(values, kde=True, bins=20, color="#fdc422")
        plt.title(f"{feature} Distribution", fontproperties=fira_code_font, fontsize=16)
        plt.xlabel(feature, fontproperties=fira_code_font, fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # Save distribution plots
        plt.savefig(os.path.join(base_dir, f"{feature}_distribution.svg"), format="svg", bbox_inches="tight")
        plt.close()

    else:
        plt.figure(figsize=(6, 4))
        sns.histplot(values, kde=True, bins=20, color="#fc8d62")
        plt.title(f"{feature} Distribution", fontproperties=fira_code_font, fontsize=16)
        plt.xlabel(feature, fontproperties=fira_code_font, fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        # Save distribution plots
        plt.savefig(os.path.join(base_dir, f"{feature}_distribution.svg"), format="svg", bbox_inches="tight")
        plt.close()
